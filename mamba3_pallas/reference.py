"""Pure-JAX references for Mamba-3 SISO. No Pallas, no TPU.

Three things live here:

* `preprocess` -- the cheap per-token algebra that the plan deliberately keeps
  *outside* the kernel: ``lambda``, ``gamma``, ``scale`` (with its ``t+1`` shift) and
  the cumulative rotary angle ``phi``. Autodiff through these few lines replaces two
  hand-written Triton backward kernels upstream.
* `siso_recurrent` -- the literal three-term trapezoidal recurrence from the paper,
  via ``lax.scan``. Slow, obviously correct, the definition of ground truth.
* `siso_chunked` -- the same maths in the chunked/linear-attention form the Pallas
  kernel implements. Differentiated by ``jax.grad``, it is the gradient oracle for
  the hand-written backward pass.

All of these work in the *permuted* (half-split rotary) state layout -- see
`layout.deinterleave_perm`. Inputs are already-normalized ``q``/``k`` (i.e. C and B
after their RMSNorms), matching the kernel boundary.

Reference for the semantics: ``mamba/tests/ops/triton/test_mamba3_siso.py``
(``mamba3_siso_step_ref`` and ``mamba3_siso_fwd_ref``) plus the Triton forward kernel.
"""

from __future__ import annotations

import dataclasses
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import layout as L

TWO_PI = 2.0 * math.pi

#: The references are the oracle, so every matmul here is forced to true f32.
#:
#: This is not paranoia. The TPU MXU has no f32 multiplier: an f32 ``einsum`` at the
#: default precision truncates both operands to bf16 and accumulates in f32, which
#: lands around 1e-2 relative error. CPU *does* have an f32 multiplier, so a reference
#: left at DEFAULT agrees with itself locally and silently degrades to bf16 on TPU --
#: which is exactly how this went unnoticed until the first hardware run.
EXACT = jax.lax.Precision.HIGHEST


class SISOState(NamedTuple):
    """Carry between segments of a sequence (and between prefill and decode).

    Four pieces, because the trapezoidal rule reaches one step back:

    Attributes:
      ssm: ``(B, H, P, N)`` f32 running state.
      k: ``(B, H, N)`` the previous token's rotated-but-**unscaled** key. The kernel
        needs it unscaled because ``scale`` for that position depends on the *next*
        token's ``dt``/``lambda``, which lives in the following segment.
      v: ``(B, H, P)`` the previous token's value.
      phi: ``(B, H, Nr)`` cumulative rotary angle, mod 2*pi.
    """

    ssm: jnp.ndarray
    k: jnp.ndarray
    v: jnp.ndarray
    phi: jnp.ndarray


def zero_state(
    batch: int, nheads: int, headdim: int, state_dim: int, n_angles: int, dtype=jnp.float32
) -> SISOState:
    return SISOState(
        ssm=jnp.zeros((batch, nheads, headdim, state_dim), jnp.float32),
        k=jnp.zeros((batch, nheads, state_dim), dtype),
        v=jnp.zeros((batch, nheads, headdim), dtype),
        phi=jnp.zeros((batch, nheads, n_angles), jnp.float32),
    )


# --------------------------------------------------------------------------------------
# Preprocessing (hoisted out of the kernel on purpose)
# --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Preprocessed:
    """Per-token quantities the kernel consumes.

    Attributes:
      adt: ``(B, H, L)`` ``A_t * dt_t``, already negative. Log-domain decay.
      gamma: ``(B, H, L)`` ``dt_t * lambda_t`` -- weight on the current token.
      scale: ``(B, H, L)`` ``gamma_t + dt_{t+1} (1 - lambda_{t+1})`` -- the single
        coefficient that collapses the three-term recurrence into one term.
      phi: ``(B, H, L, Nr)`` cumulative rotary angle mod 2*pi.
      lam: ``(B, H, L)`` ``sigmoid(trap)``, kept for the decode path.
      dt: ``(B, H, L)`` passthrough, kept for state seeding.
    """

    adt: jnp.ndarray
    gamma: jnp.ndarray
    scale: jnp.ndarray
    phi: jnp.ndarray
    lam: jnp.ndarray
    dt: jnp.ndarray


def trapezoid_coeffs(
    dt: jnp.ndarray, lam: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(gamma, scale)`` from ``dt`` and ``lambda``.

    Why ``scale`` exists at all. The paper's recurrence is three-term,

        h_t = alpha_t h_{t-1} + beta_t (k_{t-1} x v_{t-1}) + gamma_t (k_t x v_t)
        alpha_t = e^{dt A},  beta_t = (1 - lam_t) dt_t alpha_t,  gamma_t = dt_t lam_t

    but the total contribution of position ``s`` to ``h_t`` telescopes:

        gamma_s * alpha_{s+1..t}  +  beta_{s+1} * alpha_{s+2..t}
          = (gamma_s + dt_{s+1}(1 - lam_{s+1})) * alpha_{s+1..t}
          = scale_s * alpha_{s+1..t}

    so an ordinary *single*-term scan on ``scale * k`` reproduces it exactly. The
    trapezoid costs one shifted slice, not a second scan. (Checked numerically at
    5.8e-08 against `siso_recurrent`.)

    The last position has no successor, so its shifted term is zero -- which is also
    why `SISOState` must carry an unscaled ``k``.
    """
    gamma = dt * lam
    dt_next = jnp.concatenate([dt[..., 1:], jnp.zeros_like(dt[..., :1])], axis=-1)
    lam_next = jnp.concatenate([lam[..., 1:], jnp.zeros_like(lam[..., :1])], axis=-1)
    scale = gamma + dt_next * (1.0 - lam_next)
    return gamma, scale


def cumulative_angles(
    angles: jnp.ndarray, dt: jnp.ndarray, phi_init: jnp.ndarray | None = None
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``phi_t = (phi_init + sum_{i<=t} tanh(angles_i) * pi * dt_i) mod 2*pi``.

    Args:
      angles: ``(B, H, L, Nr)`` raw projection output.
      dt: ``(B, H, L)``.
      phi_init: ``(B, H, Nr)`` carry-in, or ``None`` for a fresh sequence.

    Returns:
      ``(phi, phi_final)`` with shapes ``(B, H, L, Nr)`` and ``(B, H, Nr)``.

    Kept out of the kernel for two reasons. It is what upstream does (a dedicated
    Triton kernel pair materializes exactly this tensor), and ``cumsum`` has no
    Pallas TPU lowering at all -- so leaving it to XLA is both faithful and free.
    ``floor`` has zero derivative, so the modulo is transparent to autodiff.
    """
    theta = jnp.tanh(angles) * math.pi * dt[..., None]
    phi = jnp.cumsum(theta.astype(jnp.float32), axis=2)
    if phi_init is not None:
        phi = phi + phi_init[:, :, None, :].astype(jnp.float32)
    phi = phi - TWO_PI * jnp.floor(phi / TWO_PI)
    return phi, phi[:, :, -1, :]


def preprocess(
    dt_raw: jnp.ndarray,
    a_raw: jnp.ndarray,
    trap_raw: jnp.ndarray,
    angles: jnp.ndarray,
    dt_bias: jnp.ndarray,
    a_floor: float = 1e-4,
    phi_init: jnp.ndarray | None = None,
) -> tuple[Preprocessed, jnp.ndarray]:
    """Turn raw ``in_proj`` slices into what the kernel wants.

    Mirrors ``Mamba3.forward``: softplus on ``dt``, the heavy-tail activation on
    ``A`` (negated and floored), sigmoid on the trapezoid weight.

    Args:
      dt_raw: ``(B, L, H)`` pre-softplus dt.
      a_raw: ``(B, L, H)`` pre-activation A.
      trap_raw: ``(B, L, H)`` pre-sigmoid trapezoid weight.
      angles: ``(B, L, Nr)`` raw rotary rates, shared across heads.
      dt_bias: ``(H,)``.
      a_floor: Clamp ``A <= -a_floor`` so the state always decays.
      phi_init: Optional ``(B, H, Nr)`` carry-in.

    Returns:
      ``(Preprocessed, phi_final)``.
    """
    dt = jax.nn.softplus(dt_raw + dt_bias).transpose(0, 2, 1)  # (B, H, L)
    a = -heavy_tail(a_raw.astype(jnp.float32)).transpose(0, 2, 1)
    a = jnp.minimum(a, -a_floor)
    lam = jax.nn.sigmoid(trap_raw.astype(jnp.float32)).transpose(0, 2, 1)

    gamma, scale = trapezoid_coeffs(dt, lam)
    nheads = dt.shape[1]
    # angles are shared across heads (one projection for the whole layer), but dt is
    # per-head, so phi ends up per-head anyway.
    angles_h = jnp.broadcast_to(
        angles[:, None, :, :], (angles.shape[0], nheads, angles.shape[1], angles.shape[2])
    )
    phi, phi_final = cumulative_angles(angles_h, dt, phi_init)
    return Preprocessed(adt=a * dt, gamma=gamma, scale=scale, phi=phi, lam=lam, dt=dt), phi_final


def heavy_tail(x: jnp.ndarray) -> jnp.ndarray:
    """``1 + x`` for ``x >= 0``, ``1 / (1 - x)`` for ``x < 0``.

    Positive, continuous, C1 at zero. Upstream uses it for data-dependent ``A``
    because it improves stability at higher learning rates
    (``mamba/mamba_ssm/modules/mamba3.py``).
    """
    return jnp.maximum(x, 0.0) + jnp.reciprocal(1.0 - jnp.minimum(x, 0.0))


# --------------------------------------------------------------------------------------
# Recurrent reference: the paper's equations, one token at a time
# --------------------------------------------------------------------------------------


def siso_recurrent(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    adt: jnp.ndarray,
    dt: jnp.ndarray,
    lam: jnp.ndarray,
    phi: jnp.ndarray,
    q_bias: jnp.ndarray,
    k_bias: jnp.ndarray,
    d_skip: jnp.ndarray | None = None,
    z: jnp.ndarray | None = None,
    state: SISOState | None = None,
) -> tuple[jnp.ndarray, SISOState]:
    """Ground truth: explicit three-term trapezoidal scan.

    Args:
      q, k: ``(B, L, N)`` shared across heads, already RMSNormed, permuted layout.
      v: ``(B, H, L, P)``.
      adt, dt, lam: ``(B, H, L)``.
      phi: ``(B, H, L, Nr)`` cumulative angles from `cumulative_angles`.
      q_bias, k_bias: ``(H, N)``.
      d_skip: ``(H,)`` or None.
      z: ``(B, H, L, P)`` gate, or None.
      state: Carry-in, or None for a fresh sequence.

    Returns:
      ``(y, final_state)`` with ``y`` shaped ``(B, H, L, P)``.
    """
    batch, seqlen, state_dim = q.shape
    nheads, headdim = v.shape[1], v.shape[3]
    n_angles = phi.shape[-1]

    if state is None:
        state = zero_state(batch, nheads, headdim, state_dim, n_angles, v.dtype)

    def step(carry, t):
        ssm, k_prev, v_prev = carry
        # Bias then rotate. q/k are shared across heads but the bias is per-head, so
        # the broadcast to (B, H, N) happens here.
        u = q[:, t][:, None, :] + q_bias[None]  # (B, H, N)
        w = k[:, t][:, None, :] + k_bias[None]
        cos_full, sin_full = L.rope_cos_sin(phi[:, :, t, :], state_dim)
        q_rot = L.rope_apply(u, cos_full, sin_full)
        k_rot = L.rope_apply(w, cos_full, sin_full)

        alpha = jnp.exp(adt[:, :, t])
        beta = (1.0 - lam[:, :, t]) * dt[:, :, t] * alpha
        gamma = dt[:, :, t] * lam[:, :, t]

        v_t = v[:, :, t, :]
        ssm = (
            alpha[..., None, None] * ssm
            + beta[..., None, None] * (v_prev[..., :, None] * k_prev[..., None, :])
            + gamma[..., None, None] * (v_t[..., :, None] * k_rot[..., None, :])
        )
        y = jnp.einsum(
            "bhpn,bhn->bhp", ssm, q_rot.astype(ssm.dtype), precision=EXACT
        )
        if d_skip is not None:
            y = y + d_skip[None, :, None] * v_t
        if z is not None:
            z_t = z[:, :, t, :]
            y = y * (z_t * jax.nn.sigmoid(z_t))
        return (ssm, k_rot, v_t), y

    (ssm_f, k_f, v_f), ys = jax.lax.scan(
        step, (state.ssm, state.k, state.v), jnp.arange(seqlen)
    )
    y = jnp.transpose(ys, (1, 2, 0, 3))  # (L,B,H,P) -> (B,H,L,P)
    return y, SISOState(ssm=ssm_f, k=k_f, v=v_f, phi=phi[:, :, -1, :])


# --------------------------------------------------------------------------------------
# Chunked reference: the form the Pallas kernel implements
# --------------------------------------------------------------------------------------


def siso_chunked(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    adt: jnp.ndarray,
    gamma: jnp.ndarray,
    scale: jnp.ndarray,
    phi: jnp.ndarray,
    q_bias: jnp.ndarray,
    k_bias: jnp.ndarray,
    d_skip: jnp.ndarray | None = None,
    z: jnp.ndarray | None = None,
    state: SISOState | None = None,
    seed_dt: jnp.ndarray | None = None,
    seed_lam: jnp.ndarray | None = None,
    chunk: int = 128,
    return_debug: bool = False,
):
    """Chunked linear-attention form. Same numbers as `siso_recurrent`.

    This is the gradient oracle: ``jax.grad`` of this function is what the
    hand-written Pallas backward is validated against.

    Structure per chunk, with local ``A_t = sum_{i<=t} adt_i``:

        ybar = (q~ @ S.T) * exp(A)                              inter-chunk
             + tril_strict(q~ @ khat.T * exp(A_i - A_j)) @ v    intra-chunk
             + (D_h + gamma * (u . w)) * v                      diagonal + skip
        S   <- S * exp(A_last) + (v * exp(A_last - A)).T @ khat

    The diagonal is masked *out* of the intra-chunk product (strict lower triangle)
    and added back weighted by ``gamma`` rather than ``scale``, because position
    ``t`` enters its own output through ``gamma_t`` only. Rotations are orthogonal,
    so ``q~ . k~ == u . w`` and the pre-rotation dot is used -- exactly what the
    Triton kernel does.

    Args:
      seed_dt, seed_lam: ``(B, H)`` first-token ``dt``/``lambda``, needed to fold the
        carried ``(k, v)`` into the initial state. Required iff ``state`` is given.
      return_debug: Also return the per-chunk states and rotated tensors, so tests
        can compare against the kernel's saved residuals.

    Note on segmenting a long sequence: ``gamma``/``scale``/``phi`` must be built
    from *that segment's own* raw inputs via `preprocess`, not sliced out of a
    whole-sequence version. ``scale`` reaches one step forward, so a sliced copy
    would give the segment's last token a successor term that belongs to the next
    segment -- and then double-count it, since the carried ``(k, v)`` pair supplies
    it again on the next call. Preprocessing per segment (passing ``phi_init``)
    reproduces a single full-length call to ~5e-07.
    """
    batch, seqlen, state_dim = q.shape
    nheads, headdim = v.shape[1], v.shape[3]
    nc = L.num_chunks(seqlen, chunk)
    n_angles = phi.shape[-1]

    if state is None:
        state = zero_state(batch, nheads, headdim, state_dim, n_angles, v.dtype)
        s0 = state.ssm
    else:
        if seed_dt is None or seed_lam is None:
            raise ValueError("seed_dt and seed_lam are required when passing a state")
        # The carried (k, v) pair owes the current segment its beta term. Its alpha
        # factor arrives through this chunk's decay, so seeding with beta/alpha =
        # dt_0 (1 - lam_0) reproduces beta_0 exactly.
        seed = (seed_dt * (1.0 - seed_lam))[..., None, None]
        s0 = state.ssm + seed * (state.v[..., :, None] * state.k[..., None, :])

    # Bias, rotate, scale -- all of it outside the chunk loop, since none of it is
    # sequential.
    u = q[:, None, :, :] + q_bias[None, :, None, :]  # (B, H, L, N)
    w = k[:, None, :, :] + k_bias[None, :, None, :]
    cos_full, sin_full = L.rope_cos_sin(phi, state_dim)
    q_rot = L.rope_apply(u, cos_full, sin_full)
    k_rot = L.rope_apply(w, cos_full, sin_full)
    k_hat = k_rot * scale[..., None]
    qk_diag = jnp.sum(u * w, axis=-1) * gamma  # (B, H, L)

    def to_chunks(x):
        return x.reshape(*x.shape[:2], nc, chunk, *x.shape[3:])

    qc, kc, vc = to_chunks(q_rot), to_chunks(k_hat), to_chunks(v)
    adtc = adt.reshape(batch, nheads, nc, chunk)
    diagc = qk_diag.reshape(batch, nheads, nc, chunk)

    i, j = L.row_iota(chunk)
    strict = (i > j)[None, None]

    def body(ssm, c):
        a = adtc[:, :, c, :]  # (B, H, chunk)
        A = jnp.cumsum(a, axis=-1)
        A_last = A[..., -1:]

        q_b, k_b, v_b = qc[:, :, c], kc[:, :, c], vc[:, :, c]

        # Contribution of everything before this chunk.
        out = jnp.einsum("bhtn,bhpn->bhtp", q_b, ssm.astype(q_b.dtype), precision=EXACT)
        out = out * jnp.exp(A)[..., None]

        # Within-chunk causal attention, diagonal excluded.
        s = jnp.einsum("bhtn,bhsn->bhts", q_b, k_b, precision=EXACT)
        s = s * jnp.exp(jnp.minimum(A[..., :, None] - A[..., None, :], 0.0))
        s = jnp.where(strict, s, 0.0)
        out = out + jnp.einsum("bhts,bhsp->bhtp", s, v_b, precision=EXACT)

        # Diagonal (gamma-weighted) plus the D skip.
        coeff = diagc[:, :, c]
        if d_skip is not None:
            coeff = coeff + d_skip[None, :, None]
        out = out + coeff[..., None] * v_b

        # Advance the state past this chunk.
        v_scaled = v_b * jnp.exp(A_last - A)[..., None]
        ssm_next = ssm * jnp.exp(A_last)[..., None] + jnp.einsum(
            "bhtp,bhtn->bhpn", v_scaled, k_b, precision=EXACT
        )
        return ssm_next, (out, ssm)

    ssm_final, (outs, states) = jax.lax.scan(body, s0, jnp.arange(nc))
    # scan stacks on axis 0; move the chunk axis back and flatten it into L.
    ybar = jnp.transpose(outs, (1, 2, 0, 3, 4)).reshape(batch, nheads, seqlen, headdim)
    y = ybar * (z * jax.nn.sigmoid(z)) if z is not None else ybar

    final = SISOState(
        ssm=ssm_final,
        k=k_rot[:, :, -1, :],  # unscaled, per SISOState's contract
        v=v[:, :, -1, :],
        phi=phi[:, :, -1, :],
    )
    if not return_debug:
        return y, final
    debug = {
        "ybar": ybar,
        "chunk_states": jnp.transpose(states, (1, 2, 0, 3, 4)),  # (B,H,nc,P,N)
        "q_rot": q_rot,
        "k_rot": k_rot,
        "k_hat": k_hat,
        "qk_diag": qk_diag,
        "seeded_state": s0,
    }
    return y, final, debug


# --------------------------------------------------------------------------------------
# Decode reference
# --------------------------------------------------------------------------------------


def siso_step(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    adt: jnp.ndarray,
    dt: jnp.ndarray,
    lam: jnp.ndarray,
    angles: jnp.ndarray,
    q_bias: jnp.ndarray,
    k_bias: jnp.ndarray,
    state: SISOState,
    d_skip: jnp.ndarray | None = None,
    z: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, SISOState]:
    """One decode token. Three-term form -- there is no chunk to fold beta into.

    Args:
      q, k: ``(B, N)``. v: ``(B, H, P)``. adt, dt, lam: ``(B, H)``.
      angles: ``(B, Nr)`` raw (pre-tanh) rotary rates.
      state: Carry-in.

    Returns:
      ``(y, next_state)`` with ``y`` shaped ``(B, H, P)``.
    """
    state_dim = q.shape[-1]
    phi = state.phi + jnp.tanh(angles)[:, None, :] * math.pi * dt[..., None]
    phi = phi - TWO_PI * jnp.floor(phi / TWO_PI)

    u = q[:, None, :] + q_bias[None]
    w = k[:, None, :] + k_bias[None]
    cos_full, sin_full = L.rope_cos_sin(phi, state_dim)
    q_rot = L.rope_apply(u, cos_full, sin_full)
    k_rot = L.rope_apply(w, cos_full, sin_full)

    alpha = jnp.exp(adt)
    beta = alpha * dt * (1.0 - lam)
    gamma = dt * lam

    ssm = (
        alpha[..., None, None] * state.ssm
        + beta[..., None, None] * (state.v[..., :, None] * state.k[..., None, :])
        + gamma[..., None, None] * (v[..., :, None] * k_rot[..., None, :])
    )
    y = jnp.einsum("bhpn,bhn->bhp", ssm, q_rot.astype(ssm.dtype), precision=EXACT)
    if d_skip is not None:
        y = y + d_skip[None, :, None] * v
    if z is not None:
        y = y * (z * jax.nn.sigmoid(z))
    return y, SISOState(ssm=ssm, k=k_rot, v=v, phi=phi)
