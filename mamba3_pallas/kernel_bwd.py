"""Backward Pallas kernel for Mamba-3 SISO, plus its pure-JAX epilogue.

The kernel walks chunks in reverse. The Pallas grid still counts up, so the reversal
lives in the index maps (`layout.reverse_spec`) -- that keeps the sequential-axis
guarantee that lets the adjoint state sit in VMEM scratch. Grid order is
``(batch, nchunks, nheads)`` with head innermost, so the head-shared ``q``/``k`` are
fetched once and reused across heads; see `layout` for why. The adjoint scratch
therefore spans all heads, ``(H, P, N)``.

Given ``g = dybar`` and the forward's saved per-chunk starting states, one chunk with
local ``A_t = sum_{i<=t} adt_i`` produces

    Wm     = tril_strict( (q~ @ khat.T) * exp(min(A_i - A_j, 0)) )
    dM     = tril_strict( (g  @ v.T)    * exp(min(A_i - A_j, 0)) )
    dq~    = dM  @ khat + exp(A)        * (g    @ S)        # intra + past
    dkhat  = dM.T @ q~  + exp(A_last-A) * (v    @ G)        # intra + future
    dv     = Wm.T @ g   + exp(A_last-A) * (khat @ G.T) + (D_h + qk_diag) * g
    dS     = exp(A_last) * G + (exp(A) * g).T @ q~          # reverse scan
    dA     = rowsum(q~ * dq~) - rowsum(khat * dkhat) + A_last_term

``dA_t = q~_t . dq~_t - khat_t . dkhat_t`` is exact for the attention part: those two
row-dots pick out precisely the ``+A_t`` and ``-A_t`` occurrences in
``M_ts = exp(A_t - A_s) q~_t . khat_s``. But ``A_last`` also appears *positively* in
two places that no per-row dot sees -- the state pass-through ``exp(A_last) * S`` and
the ``exp(A_last - A)`` of the state update. ``A_last`` is ``A`` at the final
position, so that scalar attaches to ``dA[chunk-1]`` and therefore survives into
every ``dadt_t`` after the suffix sum below. Getting it wrong is not subtle: dropping
it leaves ``dadt`` off by ~22%, and adding it *before* the suffix sum (so it is
counted ``chunk - t`` times) is off by 2650%.

Then ``dadt_t = sum_{tau >= t} dA_tau``, a suffix sum -- another triangular matmul,
since ``cumsum`` has no TPU lowering. No log2 conversion factor appears anywhere:
the kernel works in log2 units, and ``d/dA2 exp2(A2)`` contributes ``ln 2`` while
``a2 = adt * log2(e)`` contributes ``log2(e)``, whose product is 1.

**The rotation pullback and the head reduction happen inside the kernel.** They used to
be a pure-JAX epilogue, and that was the largest inefficiency in the implementation:
``dq_rot``/``dk_hat`` are per-head ``(B, H, L, N)`` while ``q``/``k`` are only
``(B, L, N)``, so reducing outside made XLA materialize nine tensors of the larger shape
-- ~11 GiB at ``B=8, H=32, L=8192`` against 0.06 GiB of real ``q``/``k``, which
exhausted a 16 GiB chip outright. Doing it here costs a few elementwise ops on tiles
already in VMEM and shrinks the kernel's output by ``H``: 11.0 GiB of non-kernel traffic
becomes 0.35 GiB. ``dq``/``dk`` accumulate across heads directly into a head-invariant
output window, which is legal because ``h`` is the innermost grid axis so those writes
are consecutive.
"""

from __future__ import annotations

import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import layout as L

LOG2_E = 1.4426950408889634


class KernelGrads(NamedTuple):
    """Per-position gradients. The rotation pullback happens inside the kernel.

    ``dq``/``dk`` come back already summed over heads and already pulled back through
    the rotation, so they are ``(B, L, N)`` -- matching ``q``/``k`` -- rather than
    ``(B, H, L, N)``. That is the point: doing it outside forced XLA to materialize
    nine per-head tensors of that larger shape.

    Attributes:
      dq: ``(B, L, N)`` cotangent of ``q``, summed over heads.
      dk: ``(B, L, N)`` cotangent of ``k``, summed over heads.
      dv: ``(B, H, L, P)``.
      dadt: ``(B, H, L)``.
      dqk_diag: ``(B, H, L)`` cotangent of the ``gamma * (u . w)`` diagonal
        coefficient; feeds both ``dgamma`` and ``dD``.
      dscale: ``(B, H, L)`` cotangent of ``scale``.
      dphi: ``(B, H, L, Nr)`` cotangent of the cumulative rotary angle.
      dz: ``(B, H, L, P)`` gate cotangent, or None when ungated.
      uw: ``(B, H, L)`` the ``u . w`` dot the diagonal term used; ``dgamma`` needs it.
      dq_bias, dk_bias: ``(H, N)`` already reduced over batch and time.
      k_final: ``(B, H, N)`` the last position's rotated-unscaled k, as a *value* --
        needed to differentiate the returned final k w.r.t. ``phi``.
      dseed_ssm: ``(B, H, P, N)`` cotangent of the incoming SSM state.
    """

    dq: jnp.ndarray
    dk: jnp.ndarray
    dv: jnp.ndarray
    dadt: jnp.ndarray
    dqk_diag: jnp.ndarray
    dscale: jnp.ndarray
    dphi: jnp.ndarray
    dz: jnp.ndarray | None
    uw: jnp.ndarray
    dq_bias: jnp.ndarray
    dk_bias: jnp.ndarray
    k_final: jnp.ndarray
    dseed_ssm: jnp.ndarray


def _bwd_kernel(
    # inputs (post-squeeze shapes)
    dy_ref,         # (chunk, P)   cotangent of the GATED output
    ybar_ref,       # (chunk, P)   forward's pre-gate output, for dz
    z_ref,          # (chunk, P)   the gate
    q_ref,          # (chunk, N)
    k_ref,          # (chunk, N)
    v_ref,          # (chunk, P)
    adt_ref,        # (1, chunk)
    gamma_ref,      # (1, chunk)
    scale_ref,      # (1, chunk)
    phi_ref,        # (chunk, Nr)
    qb_ref,         # (1, N)
    kb_ref,         # (1, N)
    d_ref,          # (1, 1)
    state_ref,      # (P, N)       forward's state at this chunk's start
    dfinal_ssm_ref, # (P, N)       cotangent of the final SSM state
    # outputs
    dq_ref,         # (chunk, N)   du, ACCUMULATED over heads -- see the note below
    dk_ref,         # (chunk, N)   dw, likewise
    dv_ref,         # (chunk, P)
    dadt_ref,       # (1, chunk)
    ddiag_ref,      # (1, chunk)
    dscale_ref,     # (1, chunk)
    dphi_ref,       # (chunk, Nr)
    dz_ref,         # (chunk, P)
    uw_ref,         # (1, chunk)   u . w, so the epilogue need not rebuild u and w
    dqb_ref,        # (1, N)       this step's contribution to dq_bias
    dkb_ref,        # (1, N)       this step's contribution to dk_bias
    kfin_ref,       # (H, 1, N)    k~ (the VALUE) at the last position; see below
    dseed_ref,      # (H, P, N)    all heads; see layout.spec_final_state
    # scratch
    adj_ref,        # (H, P, N) f32  the reverse-scan adjoint G, per head
    *,
    state_dim: int,
    headdim: int,
    n_angles: int,
    has_d: bool,
    has_z: bool,
    is_last_chunk_first: bool,
    matmul_dtype: Any,
    precision: Any,
):
    chunk = q_ref.shape[0]
    c = pl.program_id(1)
    h = pl.program_id(2)
    last = pl.num_programs(1) - 1

    # Grid step c handles forward chunk (nc - 1 - c), so c == 0 is the *last* chunk
    # and the incoming adjoint is whatever the caller passed for the final state.
    @pl.when(c == 0)
    def _init():
        adj_ref[h] = dfinal_ssm_ref[...]

    tril = L.tril_ones(chunk)
    i, j = L.row_iota(chunk)
    a_row = adt_ref[...] * LOG2_E
    A = L.prefix_sum(a_row, tril)
    A_last = A[chunk - 1]

    gamma = gamma_ref[0]
    scale = scale_ref[0]

    # ---- recompute the forward's rotated tensors ---------------------------------
    # Cheaper than saving them: two adds, a cos/sin, and a concatenate against
    # (chunk, N) of extra HBM traffic each way.
    u = q_ref[...] + qb_ref[...]
    w = k_ref[...] + kb_ref[...]
    cos_full, sin_full = L.rope_cos_sin(phi_ref[...].astype(jnp.float32), state_dim)
    q_rot = L.rope_apply(u, cos_full.astype(u.dtype), sin_full.astype(u.dtype))
    k_rot = L.rope_apply(w, cos_full.astype(w.dtype), sin_full.astype(w.dtype))
    k_hat = k_rot * scale[:, None].astype(k_rot.dtype)

    v = v_ref[...].astype(jnp.float32)
    S = state_ref[...].astype(jnp.float32)

    # ---- gate, in-kernel ---------------------------------------------------------
    # y = ybar * silu(z), so dybar = dy * silu(z) and
    # dz = dy * ybar * (sigmoid(z) + z sigmoid(z)(1 - sigmoid(z))).
    # Done here rather than outside because `g = dybar` is the kernel's main input: as
    # a separate XLA op it is written to HBM and read straight back, a (B,H,L,P) round
    # trip that exists only to hand a tensor over -- 1 GiB at bench shapes, and the
    # gate was 90% of what remained in the epilogue.
    dy = dy_ref[...].astype(jnp.float32)
    if has_z:
        zf = z_ref[...].astype(jnp.float32)
        sig = jax.nn.sigmoid(zf)
        g = dy * (zf * sig)
        dz_ref[...] = (
            dy * ybar_ref[...].astype(jnp.float32) * sig * (1.0 + zf * (1.0 - sig))
        ).astype(dz_ref.dtype)
    else:
        g = dy
        dz_ref[...] = jnp.zeros_like(dz_ref)
    G = adj_ref[h]

    qm = q_rot.astype(matmul_dtype)
    km = k_hat.astype(matmul_dtype)
    vm = v.astype(matmul_dtype)
    gm = g.astype(matmul_dtype)

    decay = jnp.exp2(jnp.minimum(A[:, None] - A[None, :], 0.0))
    exp_A = jnp.exp2(A)
    exp_rev = jnp.exp2(A_last - A)

    # ---- intra-chunk attention adjoints -----------------------------------------
    dM = jax.lax.dot_general(
        gm, vm, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32,
        precision=precision,
    )
    dM = jnp.where(i > j, dM * decay, 0.0)
    dMm = dM.astype(matmul_dtype)

    Wm = jax.lax.dot_general(
        qm, km, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32,
        precision=precision,
    )
    Wm = jnp.where(i > j, Wm * decay, 0.0)

    dq_rot = jax.lax.dot_general(
        dMm, km, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32,
        precision=precision,
    )
    dk_hat = jax.lax.dot_general(
        dMm, qm, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32,
        precision=precision,
    )
    dv = jax.lax.dot_general(
        Wm.astype(matmul_dtype), gm, (((0,), (0,)), ((), ())),
        preferred_element_type=jnp.float32, precision=precision,
    )

    # ---- inter-chunk terms ------------------------------------------------------
    Gm = G.astype(matmul_dtype)
    # past: ybar picked up exp(A) * (q~ @ S.T)
    dq_rot = dq_rot + exp_A[:, None] * jax.lax.dot_general(
        gm, S.astype(matmul_dtype), (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32, precision=precision,
    )
    # future: the state update contributed (v * exp(A_last - A)).T @ khat
    dk_inter = exp_rev[:, None] * jax.lax.dot_general(
        vm, Gm, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32,
        precision=precision,
    )
    dk_hat = dk_hat + dk_inter
    dv = dv + exp_rev[:, None] * jax.lax.dot_general(
        km, Gm, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32,
        precision=precision,
    )

    # diagonal + skip: ybar += (D_h + qk_diag) * v
    uw = jnp.sum(u.astype(jnp.float32) * w.astype(jnp.float32), axis=-1)
    # Emitted, not recomputed downstream: dgamma needs exactly this, and rebuilding it
    # in the epilogue would mean materializing u and w at (B, H, L, N) -- the very
    # blowup the rest of this fusion removes. As a (B, H, L) stream it is 1/N the size.
    uw_ref[...] = uw[None, :]
    diag_coeff = gamma * uw
    coeff = diag_coeff + d_ref[0, 0] if has_d else diag_coeff
    dv = dv + coeff[:, None] * g
    dqk_diag = jnp.sum(g * v, axis=-1)

    # ---- decay adjoint ----------------------------------------------------------
    # Exact for the attention part: these two row-dots pick out precisely the +A_t
    # and -A_t occurrences in M_ts = exp(A_t - A_s) q~_t . khat_s.
    dA = jnp.sum(q_rot.astype(jnp.float32) * dq_rot, axis=-1) - jnp.sum(
        k_hat.astype(jnp.float32) * dk_hat, axis=-1
    )
    # A_last also appears positively in the state pass-through exp(A_last) * S and in
    # every exp(A_last - A_i) of the state update -- neither is visible to a per-row
    # dot. A_last is A at the final position, so this lands on dA[chunk-1] alone.
    a_last_term = jnp.exp2(A_last) * jnp.sum(G * S) + jnp.sum(
        k_hat.astype(jnp.float32) * dk_inter
    )
    # adt_t feeds A_tau for every tau >= t, so accumulate a suffix sum. Since the
    # A_last term sits at the very end, it survives into every position -- add it
    # after the suffix sum rather than before, where it would be counted (chunk - t)
    # times.
    #
    # No log2/ln2 conversion factor: the kernel works in log2 units, so
    # d/dA2 exp2(A2) contributes ln2 while a2 = adt * log2(e) contributes log2(e),
    # and log2(e) * ln2 == 1.
    dadt = L.suffix_sum(dA[None, :], tril) + a_last_term
    dadt_ref[...] = dadt[None, :]

    ddiag_ref[...] = dqk_diag[None, :]
    dv_ref[...] = dv.astype(dv_ref.dtype)

    # ---- rotation pullback, in-kernel ------------------------------------------
    # This used to live in a pure-JAX epilogue, and that was the single largest
    # inefficiency in the whole implementation. dq_rot/dk_hat are per-head
    # (B, H, L, N), but q/k are (B, L, N) -- shared across heads. Emitting the
    # per-head cotangents and reducing outside meant XLA materialized nine
    # (B, H, L, N) f32 intermediates (u, w, cos, sin, q_rot, k_rot, dk_rot, du, dw):
    # ~11 GiB at B=8 H=32 L=8192 against 0.06 GiB of actual q/k, a 32x blowup that
    # exhausted a 16 GiB chip.
    #
    # Everything needed is already in VMEM here -- q_rot, k_rot, cos, sin -- so the
    # pullback costs a few elementwise ops on tiles we hold anyway, and the output
    # shrinks by H. Accumulating over heads is legal because dq_ref/dk_ref use a
    # head-invariant index map: every head in a (b, c) step writes the same window,
    # and those writes are consecutive under this grid ordering.
    dscale = jnp.sum(dk_hat * k_rot.astype(jnp.float32), axis=-1)
    dk_rot = dk_hat * scale[:, None]

    diag_scaled = (dqk_diag * gamma)[:, None]
    # R is orthogonal, so R.T == R(-phi): negate sin to pull a cotangent back.
    du = L.rope_apply(dq_rot, cos_full, -sin_full) + diag_scaled * w.astype(jnp.float32)
    dw = L.rope_apply(dk_rot, cos_full, -sin_full) + diag_scaled * u.astype(jnp.float32)

    dphi = L.rope_dphi(q_rot.astype(jnp.float32), dq_rot, n_angles) + L.rope_dphi(
        k_rot.astype(jnp.float32), dk_rot, n_angles
    )

    dscale_ref[...] = dscale[None, :]
    dphi_ref[...] = dphi.astype(dphi_ref.dtype)

    # The biases are per-head, so they cannot be recovered from the head-summed dq/dk.
    # Each grid step writes its own slot; the caller sums over (batch, chunk).
    dqb_ref[...] = jnp.sum(du, axis=0)[None, :]
    dkb_ref[...] = jnp.sum(dw, axis=0)[None, :]

    # dq/dk sum over heads. h == 0 initializes the window; the rest add into it.
    @pl.when(h == 0)
    def _init_dqk():
        dq_ref[...] = du.astype(dq_ref.dtype)
        dk_ref[...] = dw.astype(dk_ref.dtype)

    @pl.when(h > 0)
    def _accum_dqk():
        dq_ref[...] += du.astype(dq_ref.dtype)
        dk_ref[...] += dw.astype(dk_ref.dtype)

    # State passing: the forward returned k~ at the last position, so the caller may
    # hand back a cotangent for it. Turning that into a dphi contribution needs the
    # rotated *value* there -- rope_dphi(x_rot, dx_rot) -- so emit k_rot, not its
    # cotangent. Under the reversed index maps, forward chunk nc-1 is grid step 0.
    if is_last_chunk_first:
        @pl.when(c == 0)
        def _emit_kfin():
            kfin_ref[h] = k_rot[chunk - 1][None, :].astype(kfin_ref.dtype)

    # ---- advance the adjoint backwards ------------------------------------------
    # dS for this chunk becomes G for the chunk before it.
    adj_ref[h] = jnp.exp2(A_last) * G + jax.lax.dot_general(
        (exp_A[:, None] * g).astype(matmul_dtype), qm,
        (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32,
        precision=precision,
    )

    @pl.when(c == last)
    def _emit_seed():
        # After the first forward chunk, G is the cotangent of the seeded state.
        dseed_ref[h] = adj_ref[h]


def siso_backward_kernel(
    dy: jnp.ndarray,
    ybar: jnp.ndarray,
    z: jnp.ndarray | None,
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    adt: jnp.ndarray,
    gamma: jnp.ndarray,
    scale: jnp.ndarray,
    phi: jnp.ndarray,
    q_bias: jnp.ndarray,
    k_bias: jnp.ndarray,
    chunk_states: jnp.ndarray,
    d_skip: jnp.ndarray | None = None,
    dfinal_ssm: jnp.ndarray | None = None,
    chunk: int = 128,
    dtype_policy: L.DtypePolicy = L.BF16,
    interpret: Any = False,
    device_kind: str | None = None,
) -> KernelGrads:
    """Reverse chunk scan. Returns the raw per-position gradients.

    Args:
      dy: ``(B, H, L, P)`` cotangent of the *gated* output. The gate split happens
        inside the kernel, so callers pass ``dy``, not ``dybar``.
      ybar: ``(B, H, L, P)`` forward's pre-gate output, needed for ``dz``.
      z: ``(B, H, L, P)`` gate, or None.
      q, k, v, adt, gamma, scale, phi, q_bias, k_bias: as passed to the forward.
      chunk_states: ``(B, H, nc, P, N)`` from `kernel_fwd.ForwardResiduals`.
      d_skip: ``(H,)`` or None.
      dfinal_ssm: ``(B, H, P, N)`` cotangent of the returned final state, or None.
      chunk: Must match the forward's.

    Returns:
      `KernelGrads`. The epilogue in `siso_backward` maps these onto the actual
      layer inputs.
    """
    L.validate_chunk(chunk)
    batch, seqlen, state_dim = q.shape
    nheads, headdim = v.shape[1], v.shape[3]
    n_angles = phi.shape[-1]
    nc = L.num_chunks(seqlen, chunk)
    has_d = d_skip is not None
    has_z = z is not None
    cfg = L.block_config(device_kind, chunk=chunk)
    mm = dtype_policy.matmul

    z_in = z if has_z else jnp.zeros_like(dy)
    adt_b = L.scalars_to_blocks(adt.astype(jnp.float32), chunk)
    gamma_b = L.scalars_to_blocks(gamma.astype(jnp.float32), chunk)
    scale_b = L.scalars_to_blocks(scale.astype(jnp.float32), chunk)
    d_in = L.head_rows(
        d_skip.astype(jnp.float32) if has_d else jnp.zeros((nheads,), jnp.float32)
    )
    dfinal_in = (
        dfinal_ssm.astype(jnp.float32)
        if dfinal_ssm is not None
        else jnp.zeros((batch, nheads, headdim, state_dim), jnp.float32)
    )

    rev = functools.partial(L.reverse_spec, nc=nc)
    in_specs = [
        rev(L.spec_bhl_p(chunk, headdim)),              # dy
        rev(L.spec_bhl_p(chunk, headdim)),              # ybar
        rev(L.spec_bhl_p(chunk, headdim)),              # z
        rev(L.spec_shared_bl_n(chunk, state_dim)),      # q
        rev(L.spec_shared_bl_n(chunk, state_dim)),      # k
        rev(L.spec_bhl_p(chunk, headdim)),              # v
        rev(L.spec_scalar(chunk)),                      # adt
        rev(L.spec_scalar(chunk)),                      # gamma
        rev(L.spec_scalar(chunk)),                      # scale
        rev(L.spec_bhl_nr(chunk, n_angles)),            # phi
        L.spec_head_row(state_dim),                     # q_bias
        L.spec_head_row(state_dim),                     # k_bias
        L.spec_head_row(1),                             # D
        rev(L.spec_chunk_state(headdim, state_dim)),    # saved states
        L.spec_seed_state(headdim, state_dim),          # dfinal_ssm
    ]
    out_specs = [
        rev(L.spec_headsum_bl_n(chunk, state_dim)),     # dq  (B,L,N), head-summed
        rev(L.spec_headsum_bl_n(chunk, state_dim)),     # dk  (B,L,N), head-summed
        rev(L.spec_bhl_p(chunk, headdim)),              # dv
        rev(L.spec_scalar(chunk)),                      # dadt
        rev(L.spec_scalar(chunk)),                      # dqk_diag
        rev(L.spec_scalar(chunk)),                      # dscale
        rev(L.spec_bhl_nr(chunk, n_angles)),            # dphi
        rev(L.spec_bhl_p(chunk, headdim)),              # dz
        rev(L.spec_scalar(chunk)),                      # u . w
        rev(L.spec_per_step_head_row(state_dim)),       # dq_bias, per step
        rev(L.spec_per_step_head_row(state_dim)),       # dk_bias, per step
        L.spec_final_row(nheads, state_dim),            # k_final (value)
        L.spec_final_state(nheads, headdim, state_dim), # dseed_ssm
    ]
    out_shapes = [
        jax.ShapeDtypeStruct((batch, seqlen, state_dim), jnp.float32),
        jax.ShapeDtypeStruct((batch, seqlen, state_dim), jnp.float32),
        jax.ShapeDtypeStruct((batch, nheads, seqlen, headdim), jnp.float32),
        jax.ShapeDtypeStruct((batch, nheads, nc, 1, chunk), jnp.float32),
        jax.ShapeDtypeStruct((batch, nheads, nc, 1, chunk), jnp.float32),
        jax.ShapeDtypeStruct((batch, nheads, nc, 1, chunk), jnp.float32),
        jax.ShapeDtypeStruct((batch, nheads, seqlen, n_angles), jnp.float32),
        jax.ShapeDtypeStruct((batch, nheads, seqlen, headdim), jnp.float32),
        jax.ShapeDtypeStruct((batch, nheads, nc, 1, chunk), jnp.float32),
        jax.ShapeDtypeStruct((batch, nc, nheads, 1, state_dim), jnp.float32),
        jax.ShapeDtypeStruct((batch, nc, nheads, 1, state_dim), jnp.float32),
        jax.ShapeDtypeStruct((batch, nheads, 1, state_dim), jnp.float32),
        jax.ShapeDtypeStruct((batch, nheads, headdim, state_dim), jnp.float32),
    ]

    kernel = functools.partial(
        _bwd_kernel,
        state_dim=state_dim,
        headdim=headdim,
        n_angles=n_angles,
        has_d=has_d,
        has_z=has_z,
        is_last_chunk_first=True,
        matmul_dtype=mm,
        precision=dtype_policy.precision,
    )

    (
        dq, dk, dv, dadt_b, ddiag_b, dscale_b, dphi, dz, uw_b, dqb_b, dkb_b,
        k_final, dseed,
    ) = pl.pallas_call(
        kernel,
        grid=(batch, nc, nheads),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shapes,
        scratch_shapes=(pltpu.VMEM((nheads, headdim, state_dim), jnp.float32),),
        compiler_params=cfg.compiler_params(),
        interpret=interpret,
    )(
        dy, ybar, z_in, q, k, v, adt_b, gamma_b, scale_b, phi.astype(jnp.float32),
        L.head_rows(q_bias), L.head_rows(k_bias), d_in, chunk_states, dfinal_in,
    )

    return KernelGrads(
        dq=dq,
        dk=dk,
        dv=dv,
        dadt=L.blocks_to_scalars(dadt_b),
        dqk_diag=L.blocks_to_scalars(ddiag_b),
        dscale=L.blocks_to_scalars(dscale_b),
        dphi=dphi,
        dz=dz if has_z else None,
        uw=L.blocks_to_scalars(uw_b),
        dq_bias=jnp.sum(dqb_b[:, :, :, 0, :], axis=(0, 1)),
        dk_bias=jnp.sum(dkb_b[:, :, :, 0, :], axis=(0, 1)),
        k_final=k_final[:, :, 0, :],
        dseed_ssm=dseed,
    )


class SISOGrads(NamedTuple):
    """Gradients w.r.t. everything `kernel_fwd.siso_forward` takes."""

    dq: jnp.ndarray
    dk: jnp.ndarray
    dv: jnp.ndarray
    dadt: jnp.ndarray
    dgamma: jnp.ndarray
    dscale: jnp.ndarray
    dphi: jnp.ndarray
    dq_bias: jnp.ndarray
    dk_bias: jnp.ndarray
    dd_skip: jnp.ndarray | None
    dz: jnp.ndarray | None
    dseed_ssm: jnp.ndarray | None
    dseed_k: jnp.ndarray | None
    dseed_v: jnp.ndarray | None
    dseed_coef: jnp.ndarray | None


def siso_backward(
    dy: jnp.ndarray,
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    adt: jnp.ndarray,
    gamma: jnp.ndarray,
    scale: jnp.ndarray,
    phi: jnp.ndarray,
    q_bias: jnp.ndarray,
    k_bias: jnp.ndarray,
    ybar: jnp.ndarray,
    chunk_states: jnp.ndarray,
    d_skip: jnp.ndarray | None = None,
    z: jnp.ndarray | None = None,
    seed_ssm: jnp.ndarray | None = None,
    seed_k: jnp.ndarray | None = None,
    seed_v: jnp.ndarray | None = None,
    seed_coef: jnp.ndarray | None = None,
    dfinal_ssm: jnp.ndarray | None = None,
    dfinal_k: jnp.ndarray | None = None,
    dfinal_v: jnp.ndarray | None = None,
    chunk: int = 128,
    dtype_policy: L.DtypePolicy = L.BF16,
    interpret: Any = False,
    device_kind: str | None = None,
) -> SISOGrads:
    """Full backward: the Pallas reverse scan plus the elementwise epilogue.

    The epilogue is plain JAX because it is all elementwise or a reduction over one
    axis -- memory-bound work that a kernel would not speed up, and keeping it in
    XLA means ``dgamma``/``dscale`` compose with `reference.trapezoid_coeffs` and
    ``dphi`` with `reference.cumulative_angles` by ordinary autodiff.

    Args:
      dy: ``(B, H, L, P)`` cotangent of the gated output.
      ybar, chunk_states: from `kernel_fwd.ForwardResiduals`.
      dfinal_ssm, dfinal_k, dfinal_v: cotangents of the returned final states, for
        the state-passing case. ``dfinal_k`` is w.r.t. the *unscaled* rotated k.

    Returns:
      `SISOGrads`.
    """
    batch, seqlen, state_dim = q.shape
    nheads, headdim = v.shape[1], v.shape[3]
    n_angles = phi.shape[-1]
    has_z = z is not None
    has_seed = seed_ssm is not None

    # ---- the reverse scan -------------------------------------------------------
    # The gate split is inside the kernel now, so `dy` goes in unmodified and `dz`
    # comes back out with everything else.
    kg = siso_backward_kernel(
        dy, ybar, z, q, k, v, adt, gamma, scale, phi, q_bias, k_bias, chunk_states,
        d_skip=d_skip, dfinal_ssm=dfinal_ssm, chunk=chunk,
        dtype_policy=dtype_policy, interpret=interpret, device_kind=device_kind,
    )

    # ---- epilogue ---------------------------------------------------------------
    # Almost nothing left: the kernel did the rotation pullback, the head reduction,
    # and the bias reductions. What remains is dgamma (which needs u . w) and the
    # dfinal_* injections -- all on (B, H, L) tensors or smaller.
    dq, dk = kg.dq, kg.dk
    dscale, dphi = kg.dscale, kg.dphi
    dq_bias, dk_bias = kg.dq_bias, kg.dk_bias
    dz = kg.dz

    # The kernel emitted u . w, so nothing here touches an (B, H, L, N) tensor.
    dgamma = kg.dqk_diag * kg.uw
    dd_skip = jnp.sum(kg.dqk_diag, axis=(0, 2)) if d_skip is not None else None

    # The caller's cotangent of the returned final k lands on the last position. The
    # kernel already emitted dk~ there, so route it through the same rotation pullback
    # the kernel used, then add to dq/dk's last row.
    if dfinal_k is not None:
        phi_last = phi[:, :, -1, :]
        cos_l, sin_l = L.rope_cos_sin(phi_last, state_dim)
        dw_last = jnp.sum(
            L.rope_apply(dfinal_k.astype(jnp.float32), cos_l, -sin_l), axis=1
        )
        pad = jnp.zeros((batch, seqlen - 1, state_dim), dk.dtype)
        dk = dk + jnp.concatenate([pad, dw_last[:, None, :].astype(dk.dtype)], axis=1)
        dk_bias = dk_bias + jnp.sum(
            L.rope_apply(dfinal_k.astype(jnp.float32), cos_l, -sin_l), axis=0
        )
        # phi at the last position also rotated that k, so it takes a share too.
        # kg.dk_final is the rotated-unscaled k the forward emitted there.
        dphi_last = L.rope_dphi(
            kg.k_final, dfinal_k.astype(jnp.float32), n_angles
        )
        dphi = dphi + jnp.concatenate(
            [
                jnp.zeros((batch, nheads, seqlen - 1, n_angles), dphi.dtype),
                dphi_last[:, :, None, :].astype(dphi.dtype),
            ],
            axis=2,
        )

    dv = kg.dv
    if dfinal_v is not None:
        pad = jnp.zeros((batch, nheads, seqlen - 1, headdim), dv.dtype)
        dv = dv + jnp.concatenate(
            [pad, dfinal_v.astype(dv.dtype)[:, :, None, :]], axis=2
        )

    # ---- carried-state gradients ------------------------------------------------
    if has_seed:
        # The forward seeded S0 = seed_ssm + coef * (v_prev (x) k_prev).
        outer_adj = kg.dseed_ssm                                     # (B, H, P, N)
        coef = seed_coef[..., None, None]
        dseed_k = jnp.sum(outer_adj * coef * seed_v[..., :, None], axis=2)
        dseed_v = jnp.sum(outer_adj * coef * seed_k[..., None, :], axis=3)
        dseed_coef = jnp.sum(
            outer_adj * (seed_v[..., :, None] * seed_k[..., None, :]), axis=(2, 3)
        )
        dseed_ssm = kg.dseed_ssm
    else:
        dseed_ssm = dseed_k = dseed_v = dseed_coef = None

    return SISOGrads(
        dq=dq,
        dk=dk,
        dv=dv,
        dadt=kg.dadt,
        dgamma=dgamma,
        dscale=dscale,
        dphi=dphi,
        dq_bias=dq_bias,
        dk_bias=dk_bias,
        dd_skip=dd_skip,
        dz=dz,
        dseed_ssm=dseed_ssm,
        dseed_k=dseed_k,
        dseed_v=dseed_v,
        dseed_coef=dseed_coef,
    )
