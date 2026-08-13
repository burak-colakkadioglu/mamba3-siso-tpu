"""Chunked forward Pallas kernel for Mamba-3 SISO.

Grid is ``(batch, nchunks, nheads)`` -- head **innermost**, which is what makes the
head-shared ``q``/``k`` free: their index maps ignore ``h``, so adjacent grid steps hit
the same window and Pallas elides the HBM copy. With ``h`` outer instead, q/k move ``H``
times (1024 MiB rather than 32 MiB at ``B=8, H=32, L=8192``). See `layout` for the full
argument. The cost is that the state scratch holds all heads, ``(H, P, N)`` f32 -- 1 MiB
at ``H=32`` against 128 MiB of VMEM.

``nchunks`` is ``arbitrary`` because it carries the recurrence; ``nheads`` is too, since
only a prefix of parallel axes is Megacore-splittable and we do not want the scratch
split. Only ``batch`` is parallel.

Per chunk the kernel evaluates the linear-attention form of the recurrence:

    ybar = (q~ @ S.T) * exp(A)                              inter-chunk
         + tril_strict(q~ @ khat.T * exp(A_i - A_j)) @ v     intra-chunk
         + (D_h + gamma * (u . w)) * v                       diagonal + skip
    S   <- S * exp(A_last) + (v * exp(A_last - A)).T @ khat

Two Pallas TPU constraints shape the code (see `layout` for detail): ``cumsum`` has
no lowering, so ``A`` comes from an MXU matmul against a triangular ones matrix; and
per-token scalar streams must arrive as ``(B, H, nc, 1, chunk)``.
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


class ForwardResiduals(NamedTuple):
    """Everything the backward pass needs that it cannot cheaply recompute.

    Attributes:
      ybar: ``(B, H, L, P)`` pre-gate output. Needed for ``dz``.
      chunk_states: ``(B, H, nc, P, N)`` the SSM state at each chunk's *start*.
        Saved rather than recomputed because the reverse scan visits chunks in the
        opposite order.
    """

    ybar: jnp.ndarray
    chunk_states: jnp.ndarray


def _fwd_kernel(
    # inputs. Shapes are *post-squeeze*: a ``None`` in a block_shape drops that axis
    # from the ref, so e.g. ``(None, None, None, 1, chunk)`` arrives as ``(1, chunk)``.
    q_ref,          # (chunk, N)      shared across heads
    k_ref,          # (chunk, N)      shared across heads
    v_ref,          # (chunk, P)
    z_ref,          # (chunk, P)      unread when no gate
    adt_ref,        # (1, chunk)
    gamma_ref,      # (1, chunk)
    scale_ref,      # (1, chunk)
    phi_ref,        # (chunk, Nr)     cumulative rotary angle
    qb_ref,         # (1, N)
    kb_ref,         # (1, N)
    d_ref,          # (1, 1)
    seed_ssm_ref,   # (P, N)          carried state, read on chunk 0 only
    seed_kv_ref,    # (1, N+P)        carried (k | v) row, packed
    seed_coef_ref,  # (1, 1)          dt_0 * (1 - lam_0)
    # outputs
    y_ref,          # (chunk, P)
    ybar_ref,       # (chunk, P)
    states_ref,     # (P, N)          this chunk's starting state
    final_ssm_ref,  # (H, P, N)       all heads: see layout.spec_final_state
    final_kv_ref,   # (H, 1, N+P)
    # scratch
    acc_ref,        # (H, P, N) f32   carried state for every head
    *,
    n_angles: int,
    state_dim: int,
    headdim: int,
    has_z: bool,
    has_d: bool,
    has_seed: bool,
    matmul_dtype: Any,
    precision: Any,
    save_residuals: bool,
):
    chunk = q_ref.shape[0]
    c = pl.program_id(1)
    h = pl.program_id(2)
    last = pl.num_programs(1) - 1

    # ---- state init -------------------------------------------------------------
    @pl.when(c == 0)
    def _init():
        if has_seed:
            # The carried (k, v) pair still owes this segment its beta term. Its
            # alpha factor arrives via this chunk's decay, so seeding with
            # beta/alpha = dt_0 (1 - lam_0) reproduces beta_0 exactly. This mirrors
            # mamba3_siso_fwd.py:354-371.
            kv = seed_kv_ref[0]
            k_prev = kv[:state_dim]
            v_prev = kv[state_dim:]
            acc_ref[h] = seed_ssm_ref[...] + seed_coef_ref[0, 0] * (
                v_prev[:, None] * k_prev[None, :]
            )
        else:
            acc_ref[h] = jnp.zeros_like(acc_ref[h])

    # ---- decay ------------------------------------------------------------------
    # exp2 with a log2(e) prescale, matching the Triton kernel: the hardware has a
    # cheaper exp2 than exp.
    tril = L.tril_ones(chunk)
    i, j = L.row_iota(chunk)
    a_row = adt_ref[...] * LOG2_E                       # (1, chunk)
    A = L.prefix_sum(a_row, tril)                       # (chunk,)
    A_last = A[chunk - 1]

    gamma = gamma_ref[0]
    scale = scale_ref[0]

    # ---- bias, rotate, scale ----------------------------------------------------
    u = q_ref[...] + qb_ref[...]
    w = k_ref[...] + kb_ref[...]

    # Rotation is orthogonal, so q~ . k~ == u . w; dotting pre-rotation is what the
    # Triton kernel does and it saves nothing to do otherwise.
    qk_diag = jnp.sum(u * w, axis=-1) * gamma           # (chunk,)

    cos_full, sin_full = L.rope_cos_sin(phi_ref[...].astype(jnp.float32), state_dim)
    q_rot = L.rope_apply(u, cos_full.astype(u.dtype), sin_full.astype(u.dtype))
    k_rot = L.rope_apply(w, cos_full.astype(w.dtype), sin_full.astype(w.dtype))
    k_hat = k_rot * scale[:, None].astype(k_rot.dtype)

    v = v_ref[...]
    qm = q_rot.astype(matmul_dtype)
    km = k_hat.astype(matmul_dtype)
    vm = v.astype(matmul_dtype)

    # ---- output -----------------------------------------------------------------
    # Everything before this chunk, read out of the carried state.
    state = acc_ref[h]
    out = jax.lax.dot_general(
        qm,
        state.astype(matmul_dtype),
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
        precision=precision,
    )
    out = out * jnp.exp2(A)[:, None]

    # Within-chunk causal attention. The diagonal is masked out (strict >) and added
    # back below weighted by gamma rather than scale: position t enters its own
    # output only through gamma_t.
    s = jax.lax.dot_general(
        qm, km, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32,
        precision=precision,
    )
    s = s * jnp.exp2(jnp.minimum(A[:, None] - A[None, :], 0.0))
    s = jnp.where(i > j, s, 0.0)
    out = out + jax.lax.dot_general(
        s.astype(matmul_dtype), vm, (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32, precision=precision,
    )

    coeff = qk_diag + d_ref[0, 0] if has_d else qk_diag
    out = out + coeff[:, None] * v.astype(jnp.float32)

    # Written unconditionally: Pallas has no optional outputs, and when residuals
    # are not wanted these refs are aliased down to a single block (see the caller).
    states_ref[...] = state.astype(states_ref.dtype)
    ybar_ref[...] = out.astype(ybar_ref.dtype)

    if has_z:
        z = z_ref[...].astype(jnp.float32)
        out = out * (z * jax.nn.sigmoid(z))
    y_ref[...] = out.astype(y_ref.dtype)

    # ---- advance the state ------------------------------------------------------
    v_scaled = (v.astype(jnp.float32) * jnp.exp2(A_last - A)[:, None]).astype(matmul_dtype)
    acc_ref[h] = state * jnp.exp2(A_last) + jax.lax.dot_general(
        v_scaled, km, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32,
        precision=precision,
    )

    @pl.when(c == last)
    def _emit_final():
        final_ssm_ref[h] = acc_ref[h]
        # k is stored *unscaled* -- its scale depends on the next segment's first
        # token, which this call has not seen. Packed with v into one row so the
        # minormost block dim stays a single legal width.
        final_kv_ref[h] = jnp.concatenate(
            [
                k_rot[chunk - 1][None, :].astype(jnp.float32),
                v[chunk - 1][None, :].astype(jnp.float32),
            ],
            axis=-1,
        )


def siso_forward(
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
    seed_ssm: jnp.ndarray | None = None,
    seed_k: jnp.ndarray | None = None,
    seed_v: jnp.ndarray | None = None,
    seed_coef: jnp.ndarray | None = None,
    chunk: int = 128,
    dtype_policy: L.DtypePolicy = L.BF16,
    save_residuals: bool = False,
    interpret: Any = False,
    device_kind: str | None = None,
):
    """Chunked forward pass.

    Args:
      q, k: ``(B, L, N)`` post-RMSNorm C and B projections, permuted layout, shared
        across heads (``ngroups=1``).
      v: ``(B, H, L, P)``.
      adt, gamma, scale: ``(B, H, L)`` f32 from `reference.preprocess`.
      phi: ``(B, H, L, Nr)`` cumulative rotary angles, f32, from `reference.preprocess`.
      q_bias, k_bias: ``(H, N)``.
      d_skip: ``(H,)`` or None.
      z: ``(B, H, L, P)`` gate, or None.
      seed_ssm, seed_k, seed_v, seed_coef: carried state. ``seed_ssm`` is
        ``(B, H, P, N)``, ``seed_k`` is ``(B, H, N)``, ``seed_v`` is ``(B, H, P)``,
        ``seed_coef`` is ``(B, H)`` holding ``dt_0 * (1 - lam_0)``. All or none.
      chunk: Rows per grid step; multiple of 128.
      save_residuals: Also return `ForwardResiduals` for the backward pass.
      interpret: Pass `layout.interpret_mode()` to run on CPU.
      device_kind: Override for block-config selection; defaults to the live device.

    Returns:
      ``(y, final_ssm, final_k, final_v)``, and `ForwardResiduals` when
      ``save_residuals``. ``final_k`` is rotated but **unscaled**.
    """
    L.validate_chunk(chunk)
    batch, seqlen, state_dim = q.shape
    nheads, headdim = v.shape[1], v.shape[3]
    n_angles = phi.shape[-1]
    nc = L.num_chunks(seqlen, chunk)

    if q.shape != k.shape:
        raise ValueError(f"q {q.shape} and k {k.shape} must match")
    if v.shape != (batch, nheads, seqlen, headdim):
        raise ValueError(f"unexpected v shape {v.shape}")
    for name, t in (("adt", adt), ("gamma", gamma), ("scale", scale)):
        if t.shape != (batch, nheads, seqlen):
            raise ValueError(f"{name} must be (B, H, L) = {(batch, nheads, seqlen)}, got {t.shape}")

    seeds = (seed_ssm, seed_k, seed_v, seed_coef)
    has_seed = any(s is not None for s in seeds)
    if has_seed and not all(s is not None for s in seeds):
        raise ValueError("seed_ssm, seed_k, seed_v and seed_coef must be given together")

    has_z = z is not None
    has_d = d_skip is not None
    cfg = L.block_config(device_kind, chunk=chunk)
    mm = dtype_policy.matmul

    if phi.shape != (batch, nheads, seqlen, n_angles):
        raise ValueError(
            f"phi must be (B, H, L, Nr) = {(batch, nheads, seqlen, n_angles)}, "
            f"got {phi.shape}"
        )

    # Scalar streams get their legal 5-D shape; these are free reshapes.
    adt_b = L.scalars_to_blocks(adt.astype(jnp.float32), chunk)
    gamma_b = L.scalars_to_blocks(gamma.astype(jnp.float32), chunk)
    scale_b = L.scalars_to_blocks(scale.astype(jnp.float32), chunk)

    # Placeholders keep the argument list a fixed shape; index maps point them at a
    # single element and the kernel never reads them when the flag is off.
    z_in = z if has_z else jnp.zeros((batch, nheads, seqlen, headdim), v.dtype)
    d_in = L.head_rows(
        d_skip.astype(jnp.float32) if has_d else jnp.zeros((nheads,), jnp.float32)
    )

    if has_seed:
        seed_ssm_in = seed_ssm.astype(jnp.float32)
        # Pack k and v into one row so the minormost dim is a single legal width.
        seed_kv_in = jnp.concatenate(
            [seed_k.astype(jnp.float32), seed_v.astype(jnp.float32)], axis=-1
        )[:, :, None, :]
        seed_coef_in = seed_coef.astype(jnp.float32)[:, :, None, None]
    else:
        seed_ssm_in = jnp.zeros((batch, nheads, headdim, state_dim), jnp.float32)
        seed_kv_in = jnp.zeros((batch, nheads, 1, state_dim + headdim), jnp.float32)
        seed_coef_in = jnp.zeros((batch, nheads, 1, 1), jnp.float32)

    in_specs = [
        L.spec_shared_bl_n(chunk, state_dim),                        # q
        L.spec_shared_bl_n(chunk, state_dim),                        # k
        L.spec_bhl_p(chunk, headdim),                                # v
        L.spec_bhl_p(chunk, headdim),                                # z
        L.spec_scalar(chunk),                                        # adt
        L.spec_scalar(chunk),                                        # gamma
        L.spec_scalar(chunk),                                        # scale
        L.spec_bhl_nr(chunk, n_angles),                              # phi
        L.spec_head_row(state_dim),                                  # q_bias
        L.spec_head_row(state_dim),                                  # k_bias
        L.spec_head_row(1),                                          # D
        L.spec_seed_state(headdim, state_dim),                       # seed ssm
        L.spec_seed_row(state_dim + headdim),                        # seed kv
        pl.BlockSpec((None, None, 1, 1), lambda b, c, h: (b, h, 0, 0)),  # seed coef
    ]

    y_shape = jax.ShapeDtypeStruct((batch, nheads, seqlen, headdim), v.dtype)
    # Pallas has no optional outputs. When residuals are not wanted, keep the buffers
    # but collapse them to a single block, so every grid step overwrites the same
    # window and the cost is one block rather than the whole sequence.
    if save_residuals:
        ybar_shape = jax.ShapeDtypeStruct((batch, nheads, seqlen, headdim), jnp.float32)
        states_shape = jax.ShapeDtypeStruct(
            (batch, nheads, nc, headdim, state_dim), dtype_policy.saved_state
        )
        ybar_spec = L.spec_bhl_p(chunk, headdim)
        states_spec = L.spec_chunk_state(headdim, state_dim)
    else:
        ybar_shape = jax.ShapeDtypeStruct((batch, nc, chunk, headdim), jnp.float32)
        states_shape = jax.ShapeDtypeStruct(
            (batch, nc, headdim, state_dim), dtype_policy.saved_state
        )
        ybar_spec = L.spec_scratch_bl_p(chunk, headdim)
        states_spec = L.spec_scratch_state(headdim, state_dim)

    out_shapes = [
        y_shape,
        ybar_shape,
        states_shape,
        jax.ShapeDtypeStruct((batch, nheads, headdim, state_dim), jnp.float32),
        jax.ShapeDtypeStruct((batch, nheads, 1, state_dim + headdim), jnp.float32),
    ]
    out_specs = [
        L.spec_bhl_p(chunk, headdim),
        ybar_spec,
        states_spec,
        L.spec_final_state(nheads, headdim, state_dim),
        L.spec_final_row(nheads, state_dim + headdim),
    ]

    kernel = functools.partial(
        _fwd_kernel,
        n_angles=n_angles,
        state_dim=state_dim,
        headdim=headdim,
        has_z=has_z,
        has_d=has_d,
        has_seed=has_seed,
        matmul_dtype=mm,
        precision=dtype_policy.precision,
        save_residuals=save_residuals,
    )

    y, ybar, states, final_ssm, final_kv = pl.pallas_call(
        kernel,
        grid=(batch, nc, nheads),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shapes,
        scratch_shapes=(pltpu.VMEM((nheads, headdim, state_dim), jnp.float32),),
        compiler_params=cfg.compiler_params(),
        interpret=interpret,
    )(
        q, k, v, z_in, adt_b, gamma_b, scale_b, phi.astype(jnp.float32),
        L.head_rows(q_bias), L.head_rows(k_bias), d_in,
        seed_ssm_in, seed_kv_in, seed_coef_in,
    )

    final_k = final_kv[:, :, 0, :state_dim]
    final_v = final_kv[:, :, 0, state_dim:]
    if save_residuals:
        return y, final_ssm, final_k, final_v, ForwardResiduals(
            ybar=ybar, chunk_states=states
        )
    return y, final_ssm, final_k, final_v
