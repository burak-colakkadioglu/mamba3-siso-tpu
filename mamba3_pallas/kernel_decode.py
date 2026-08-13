"""Single-token decode kernel for Mamba-3 SISO.

No sequential grid axis, because one token is one step. The three-term trapezoidal
update appears here in its original form rather than the collapsed ``scale``-weighted one
the chunked kernel uses: collapsing needs the *next* token's ``dt``/``lambda``, which at
decode time has not been generated yet. So the carried ``(k_prev, v_prev)`` pair pays its
``beta`` term now:

    S' = alpha * S + beta * (v_prev (x) k_prev) + gamma * (v (x) k~)
    y  = S' q~ + D_h v,   then y *= silu(z)

with ``alpha = exp(dt A)``, ``beta = alpha * dt * (1 - lambda)``, ``gamma = dt * lambda``.

``S`` is ``(P, N)`` = 32 KiB in f32 per head and ``S' q~`` is a matrix-vector product, so
the step is bandwidth-bound. The whole point is to touch ``S`` exactly once. The state
buffers are donated via ``input_output_aliases`` so a generation loop updates them in
place instead of reallocating every step.

Two grids, both correct, differing in VMEM footprint per step:

* `siso_decode_folded` -- grid ``(batch,)``, all heads in one block. One grid step per
  batch element instead of ``H`` of them, and the head-shared ``q``/``k``/``angles`` are
  passed as single rows and broadcast inside the kernel rather than duplicated ``H`` times
  in HBM. Needs ``H * P * N * 4`` bytes of state resident (1 MiB at ``H=32``, 4 MiB at
  ``H=64``) plus double buffering. This is the default.
* `siso_decode` -- grid ``(batch, nheads)``, one head per step, 32 KiB of state
  regardless of ``H``. Use this when the folded block does not fit, which happens at large
  head counts or on v3 (16 MiB VMEM against v5e's 128).

`decode_step` picks between them. Both are checked against `reference.siso_step`.

At small batch the wall time is dominated by the host, not the kernel. A blocking
per-token call at ``B=1`` is ~270 us against ~19 us of device time: ~180 us of host
round-trip (avoidable by not syncing every token) and ~72 us of per-call dispatch
(12 arguments to flatten, plus donation bookkeeping). Both shrink as batch grows and are
gone by ``B=64``. `decode_scan` runs many steps under one ``lax.scan`` if you want the
device-only floor.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from . import layout as L
from . import reference as R

TWO_PI = 2.0 * math.pi


def _decode_kernel(
    # inputs (post-squeeze)
    q_ref,          # (1, N)
    k_ref,          # (1, N)
    v_ref,          # (1, P)
    z_ref,          # (1, P)
    scal_ref,       # (1, 4)     adt | dt | lam | unused, packed
    angles_ref,     # (1, Nr)    raw, pre-tanh
    qb_ref,         # (1, N)
    kb_ref,         # (1, N)
    d_ref,          # (1, 1)
    ssm_ref,        # (P, N)     carried state, donated
    kv_ref,         # (1, N+P)   carried (k_prev | v_prev), donated
    phi_ref,        # (1, Nr)    carried angle, donated
    # outputs
    y_ref,          # (1, P)
    ssm_out_ref,    # (P, N)
    kv_out_ref,     # (1, N+P)
    phi_out_ref,    # (1, Nr)
    *,
    state_dim: int,
    headdim: int,
    has_z: bool,
    has_d: bool,
    matmul_dtype: Any,
    precision: Any,
):
    adt = scal_ref[0, 0]
    dt = scal_ref[0, 1]
    lam = scal_ref[0, 2]

    # ---- advance the rotary angle -----------------------------------------------
    theta = jnp.tanh(angles_ref[...].astype(jnp.float32)) * jnp.float32(math.pi) * dt
    phi = phi_ref[...].astype(jnp.float32) + theta
    # Wrap to keep f32 precision usable over long generations.
    phi = phi - jnp.float32(TWO_PI) * jnp.floor(phi / jnp.float32(TWO_PI))
    phi_out_ref[...] = phi

    u = q_ref[...] + qb_ref[...]
    w = k_ref[...] + kb_ref[...]
    cos_full, sin_full = L.rope_cos_sin(phi, state_dim)
    q_rot = L.rope_apply(u, cos_full.astype(u.dtype), sin_full.astype(u.dtype))
    k_rot = L.rope_apply(w, cos_full.astype(w.dtype), sin_full.astype(w.dtype))

    v = v_ref[...].astype(jnp.float32)

    # ---- three-term state update ------------------------------------------------
    alpha = jnp.exp(adt)
    beta = alpha * dt * (1.0 - lam)
    gamma = dt * lam

    kv = kv_ref[0].astype(jnp.float32)
    k_prev = kv[:state_dim]
    v_prev = kv[state_dim:]

    ssm = (
        alpha * ssm_ref[...]
        + beta * (v_prev[:, None] * k_prev[None, :])
        + gamma * (v[0][:, None] * k_rot[0].astype(jnp.float32)[None, :])
    )
    ssm_out_ref[...] = ssm

    # ---- output ------------------------------------------------------------------
    # (1, N) @ (P, N).T -> (1, P): a matrix-vector product against the state.
    out = jax.lax.dot_general(
        q_rot.astype(matmul_dtype),
        ssm.astype(matmul_dtype),
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
        precision=precision,
    )
    if has_d:
        out = out + d_ref[0, 0] * v
    if has_z:
        z = z_ref[...].astype(jnp.float32)
        out = out * (z * jax.nn.sigmoid(z))
    y_ref[...] = out.astype(y_ref.dtype)

    # k is stored rotated but *unscaled*: its scale depends on the next token.
    kv_out_ref[...] = jnp.concatenate(
        [k_rot.astype(jnp.float32), v.astype(jnp.float32)], axis=-1
    )


def _folded_kernel(
    # inputs
    q_ref,          # (1, N)      head-shared, one row
    k_ref,          # (1, N)      head-shared
    v_ref,          # (H, P)
    z_ref,          # (H, P)
    scal_ref,       # (H, 4)      adt | dt | lam | unused
    angles_ref,     # (1, Nr)     head-shared, raw
    qb_ref,         # (H, N)
    kb_ref,         # (H, N)
    d_ref,          # (H, 1)
    ssm_ref,        # (H, P, N)   carried state, donated
    kv_ref,         # (H, N+P)    carried (k_prev | v_prev), donated
    phi_ref,        # (H, Nr)     carried angle, donated
    # outputs
    y_ref,          # (H, P)
    ssm_out_ref,    # (H, P, N)
    kv_out_ref,     # (H, N+P)
    phi_out_ref,    # (H, Nr)
    *,
    state_dim: int,
    headdim: int,
    has_z: bool,
    has_d: bool,
    matmul_dtype: Any,
    precision: Any,
):
    """One decode step for every head of one batch element.

    ``q``/``k``/``angles`` arrive as single rows and broadcast against the ``(H, ...)``
    per-head tensors inside the kernel, so the 32x HBM duplication the per-head grid
    needs never happens. ``phi`` is still per-head: it accumulates ``tanh(angle) * pi *
    dt`` and ``dt`` is per-head, so the angles diverge across heads after step one even
    though their rates do not.
    """
    adt = scal_ref[:, 0:1]                     # (H, 1)
    dt = scal_ref[:, 1:2]
    lam = scal_ref[:, 2:3]

    # ---- advance the rotary angle (per head: dt differs) ------------------------
    rates = jnp.tanh(angles_ref[...].astype(jnp.float32)) * jnp.float32(math.pi)
    phi = phi_ref[...].astype(jnp.float32) + rates * dt          # (1,Nr)*(H,1) -> (H,Nr)
    phi = phi - jnp.float32(TWO_PI) * jnp.floor(phi / jnp.float32(TWO_PI))
    phi_out_ref[...] = phi

    u = q_ref[...] + qb_ref[...]               # (1,N) + (H,N) -> (H,N)
    w = k_ref[...] + kb_ref[...]
    cos_full, sin_full = L.rope_cos_sin(phi, state_dim)
    q_rot = L.rope_apply(u, cos_full.astype(u.dtype), sin_full.astype(u.dtype))
    k_rot = L.rope_apply(w, cos_full.astype(w.dtype), sin_full.astype(w.dtype))

    v = v_ref[...].astype(jnp.float32)         # (H, P)

    # ---- three-term state update ------------------------------------------------
    alpha = jnp.exp(adt)                       # (H, 1)
    beta = alpha * dt * (1.0 - lam)
    gamma = dt * lam

    kv = kv_ref[...].astype(jnp.float32)       # (H, N+P)
    k_prev = kv[:, :state_dim]
    v_prev = kv[:, state_dim:]

    # Outer products per head: (H,P,1) * (H,1,N). alpha/beta/gamma are (H,1) and index
    # as (H,1,1) against the (H,P,N) state.
    ssm = (
        alpha[:, :, None] * ssm_ref[...]
        + beta[:, :, None] * (v_prev[:, :, None] * k_prev[:, None, :])
        + gamma[:, :, None] * (v[:, :, None] * k_rot.astype(jnp.float32)[:, None, :])
    )
    ssm_out_ref[...] = ssm

    # ---- output ------------------------------------------------------------------
    # Batched over heads: contract N, keep H as the batch dim. Per head this uses one row
    # of a 128x128 MXU, which looks wasteful, but the contraction runs along the lane axis
    # and that is what the MXU does natively. Doing it on the VPU instead means
    # `sum(..., -1)`, a tree of lane-crossing shuffles.
    out = jax.lax.dot_general(
        ssm.astype(matmul_dtype),
        q_rot.astype(matmul_dtype),
        (((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
        precision=precision,
    )

    if has_d:
        out = out + d_ref[...] * v             # (H,1) * (H,P)
    if has_z:
        z = z_ref[...].astype(jnp.float32)
        out = out * (z * jax.nn.sigmoid(z))
    y_ref[...] = out.astype(y_ref.dtype)

    kv_out_ref[...] = jnp.concatenate(
        [k_rot.astype(jnp.float32), v.astype(jnp.float32)], axis=-1
    )


def _pack_state(state: R.SISOState) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """`reference.SISOState` -> the three buffers the kernel donates."""
    kv = jnp.concatenate(
        [state.k.astype(jnp.float32), state.v.astype(jnp.float32)], axis=-1
    )
    return state.ssm.astype(jnp.float32), kv[:, :, None, :], state.phi.astype(jnp.float32)[:, :, None, :]


def _unpack_state(
    ssm: jnp.ndarray, kv: jnp.ndarray, phi: jnp.ndarray, state_dim: int
) -> R.SISOState:
    return R.SISOState(
        ssm=ssm, k=kv[:, :, 0, :state_dim], v=kv[:, :, 0, state_dim:], phi=phi[:, :, 0, :]
    )


def siso_decode(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    adt: jnp.ndarray,
    dt: jnp.ndarray,
    lam: jnp.ndarray,
    angles: jnp.ndarray,
    q_bias: jnp.ndarray,
    k_bias: jnp.ndarray,
    state: R.SISOState,
    d_skip: jnp.ndarray | None = None,
    z: jnp.ndarray | None = None,
    dtype_policy: L.DtypePolicy = L.BF16,
    interpret: Any = False,
) -> tuple[jnp.ndarray, R.SISOState]:
    """One decode step.

    Args:
      q, k: ``(B, N)`` post-RMSNorm C and B, permuted layout.
      v: ``(B, H, P)``.
      adt, dt, lam: ``(B, H)``.
      angles: ``(B, Nr)`` **raw** (pre-tanh) rotary rates -- the kernel applies
        ``tanh`` and the ``dt`` scaling itself, since it also has to accumulate.
      q_bias, k_bias: ``(H, N)``.
      state: Carry-in.
      d_skip: ``(H,)`` or None. z: ``(B, H, P)`` or None.

    Returns:
      ``(y, next_state)`` with ``y`` shaped ``(B, H, P)``.
    """
    batch, state_dim = q.shape
    nheads, headdim = v.shape[1], v.shape[2]
    n_angles = angles.shape[-1]
    has_z = z is not None
    has_d = d_skip is not None
    mm = dtype_policy.matmul

    # One row per (b, h) for q/k/v/z: rank-2 blocks with a singleton leading axis,
    # the same shape rule the chunked kernel's scalar streams obey.
    q_in = jnp.broadcast_to(q[:, None, None, :], (batch, nheads, 1, state_dim))
    k_in = jnp.broadcast_to(k[:, None, None, :], (batch, nheads, 1, state_dim))
    v_in = v[:, :, None, :]
    z_in = (z if has_z else jnp.zeros_like(v))[:, :, None, :]
    # adt/dt/lam packed into one 4-wide row: three separate 1-wide arrays would each
    # need their own block and buy nothing.
    scal = jnp.stack(
        [
            adt.astype(jnp.float32),
            dt.astype(jnp.float32),
            lam.astype(jnp.float32),
            jnp.zeros_like(adt, jnp.float32),
        ],
        axis=-1,
    )[:, :, None, :]
    angles_in = jnp.broadcast_to(
        angles[:, None, None, :], (batch, nheads, 1, n_angles)
    )
    d_in = L.head_rows(
        d_skip.astype(jnp.float32) if has_d else jnp.zeros((nheads,), jnp.float32)
    )
    ssm_in, kv_in, phi_in = _pack_state(state)

    row = lambda width: pl.BlockSpec((None, None, 1, width), lambda b, h: (b, h, 0, 0))
    head = lambda width: pl.BlockSpec((None, 1, width), lambda b, h: (h, 0, 0))
    ssm_spec = pl.BlockSpec((None, None, headdim, state_dim), lambda b, h: (b, h, 0, 0))

    in_specs = [
        row(state_dim), row(state_dim), row(headdim), row(headdim),
        row(4), row(n_angles),
        head(state_dim), head(state_dim), head(1),
        ssm_spec, row(state_dim + headdim), row(n_angles),
    ]
    out_specs = [row(headdim), ssm_spec, row(state_dim + headdim), row(n_angles)]
    out_shapes = [
        jax.ShapeDtypeStruct((batch, nheads, 1, headdim), v.dtype),
        jax.ShapeDtypeStruct((batch, nheads, headdim, state_dim), jnp.float32),
        jax.ShapeDtypeStruct((batch, nheads, 1, state_dim + headdim), jnp.float32),
        jax.ShapeDtypeStruct((batch, nheads, 1, n_angles), jnp.float32),
    ]

    kernel = functools.partial(
        _decode_kernel,
        state_dim=state_dim,
        headdim=headdim,
        has_z=has_z,
        has_d=has_d,
        matmul_dtype=mm,
        precision=dtype_policy.precision,
    )

    y, ssm_out, kv_out, phi_out = pl.pallas_call(
        kernel,
        grid=(batch, nheads),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shapes,
        compiler_params=L.CompilerParams(dimension_semantics=("parallel", "parallel")),
        # Donate the three state buffers onto their outputs so a generation loop
        # updates in place. Indices are into the flat input list above.
        input_output_aliases={9: 1, 10: 2, 11: 3},
        interpret=interpret,
    )(
        q_in, k_in, v_in, z_in, scal, angles_in,
        L.head_rows(q_bias), L.head_rows(k_bias), d_in,
        ssm_in, kv_in, phi_in,
    )

    return y[:, :, 0, :], _unpack_state(ssm_out, kv_out, phi_out, state_dim)


def siso_decode_folded(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    adt: jnp.ndarray,
    dt: jnp.ndarray,
    lam: jnp.ndarray,
    angles: jnp.ndarray,
    q_bias: jnp.ndarray,
    k_bias: jnp.ndarray,
    state: R.SISOState,
    d_skip: jnp.ndarray | None = None,
    z: jnp.ndarray | None = None,
    dtype_policy: L.DtypePolicy = L.BF16,
    interpret: Any = False,
) -> tuple[jnp.ndarray, R.SISOState]:
    """One decode step, all heads per grid step. Signature matches `siso_decode`.

    The head axis lives in the block, so ``q``/``k``/``angles`` are passed as one row each
    and broadcast inside the kernel rather than being duplicated ``H`` times in HBM. VMEM
    per grid step is ``H * P * N * 4`` bytes of state (1 MiB at ``H=32``) plus double
    buffering, which is why `siso_decode` still exists: at head counts where this block
    does not fit, the per-head grid does.
    """
    batch, state_dim = q.shape
    nheads, headdim = v.shape[1], v.shape[2]
    n_angles = angles.shape[-1]
    has_z = z is not None
    has_d = d_skip is not None

    row = lambda width: pl.BlockSpec((None, 1, width), lambda b: (b, 0, 0))
    per_head = lambda width: pl.BlockSpec((None, nheads, width), lambda b: (b, 0, 0))
    head_only = lambda width: pl.BlockSpec((nheads, width), lambda b: (0, 0))
    ssm_spec = pl.BlockSpec((None, nheads, headdim, state_dim), lambda b: (b, 0, 0, 0))

    q_in = q[:, None, :]                       # (B, 1, N) -- no head duplication
    k_in = k[:, None, :]
    angles_in = angles[:, None, :]
    v_in = v
    z_in = z if has_z else jnp.zeros_like(v)
    scal = jnp.stack(
        [
            adt.astype(jnp.float32),
            dt.astype(jnp.float32),
            lam.astype(jnp.float32),
            jnp.zeros_like(adt, jnp.float32),
        ],
        axis=-1,
    )                                          # (B, H, 4)
    d_in = (
        d_skip.astype(jnp.float32) if has_d else jnp.zeros((nheads,), jnp.float32)
    ).reshape(nheads, 1)
    kv_in = jnp.concatenate(
        [state.k.astype(jnp.float32), state.v.astype(jnp.float32)], axis=-1
    )                                          # (B, H, N+P)
    ssm_in = state.ssm.astype(jnp.float32)
    phi_in = state.phi.astype(jnp.float32)

    in_specs = [
        row(state_dim), row(state_dim), per_head(headdim), per_head(headdim),
        per_head(4), row(n_angles),
        head_only(state_dim), head_only(state_dim), head_only(1),
        ssm_spec, per_head(state_dim + headdim), per_head(n_angles),
    ]
    out_specs = [
        per_head(headdim), ssm_spec, per_head(state_dim + headdim), per_head(n_angles)
    ]
    out_shapes = [
        jax.ShapeDtypeStruct((batch, nheads, headdim), v.dtype),
        jax.ShapeDtypeStruct((batch, nheads, headdim, state_dim), jnp.float32),
        jax.ShapeDtypeStruct((batch, nheads, state_dim + headdim), jnp.float32),
        jax.ShapeDtypeStruct((batch, nheads, n_angles), jnp.float32),
    ]

    kernel = functools.partial(
        _folded_kernel,
        state_dim=state_dim,
        headdim=headdim,
        has_z=has_z,
        has_d=has_d,
        matmul_dtype=dtype_policy.matmul,
        precision=dtype_policy.precision,
    )

    y, ssm_out, kv_out, phi_out = pl.pallas_call(
        kernel,
        grid=(batch,),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shapes,
        compiler_params=L.CompilerParams(dimension_semantics=("parallel",)),
        input_output_aliases={9: 1, 10: 2, 11: 3},
        interpret=interpret,
    )(
        q_in, k_in, v_in, z_in, scal, angles_in,
        q_bias, k_bias, d_in,
        ssm_in, kv_in, phi_in,
    )

    return y, R.SISOState(
        ssm=ssm_out, k=kv_out[..., :state_dim], v=kv_out[..., state_dim:], phi=phi_out
    )


def decode_step(*args, folded: bool = True, **kw) -> tuple[jnp.ndarray, R.SISOState]:
    """`siso_decode_folded` or `siso_decode`. The default is the folded kernel."""
    if folded:
        return siso_decode_folded(*args, **kw)
    return siso_decode(*args, **kw)


def decode_scan(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    adt: jnp.ndarray,
    dt: jnp.ndarray,
    lam: jnp.ndarray,
    angles: jnp.ndarray,
    q_bias: jnp.ndarray,
    k_bias: jnp.ndarray,
    state: R.SISOState,
    d_skip: jnp.ndarray | None = None,
    z: jnp.ndarray | None = None,
    dtype_policy: L.DtypePolicy = L.BF16,
    interpret: Any = False,
    folded: bool = True,
) -> tuple[jnp.ndarray, R.SISOState]:
    """``L`` decode steps under one ``lax.scan``, state as carry.

    Same shapes and result as `decode_loop`, but one XLA program instead of ``L`` of them.
    Real generation cannot use this, since the next token is not known until the current
    one is sampled. It is here for two things: measuring what a token costs on the device
    with the host out of the way, and running a known-length continuation (teacher forcing,
    a fixed prompt suffix) without paying dispatch per token.

    At ``B=1`` this is ~19 us/token against ~270 us for a blocking per-token call. The
    difference is host cost, not kernel. At ``B=64`` and up the two are within 0.1% of each
    other, so at large batch just call `decode_step` per token.
    """
    seqlen = q.shape[1]
    step = functools.partial(
        decode_step,
        dtype_policy=dtype_policy,
        interpret=interpret,
        folded=folded,
    )

    def body(carry, xs):
        qt, kt, vt, at, dtt, lt, angt, zt = xs
        y, carry = step(
            qt, kt, vt, at, dtt, lt, angt, q_bias, k_bias, carry,
            d_skip=d_skip, z=zt if z is not None else None,
        )
        return carry, y

    # Move the time axis to the front so scan slices it, then put it back. The
    # transposes are outside the loop, so they cost one pass over the inputs rather
    # than one dispatch per token.
    xs = (
        q.swapaxes(0, 1),                      # (L, B, N)
        k.swapaxes(0, 1),
        v.transpose(2, 0, 1, 3),               # (L, B, H, P)
        adt.transpose(2, 0, 1),                # (L, B, H)
        dt.transpose(2, 0, 1),
        lam.transpose(2, 0, 1),
        angles.swapaxes(0, 1),                 # (L, B, Nr)
        (z if z is not None else jnp.zeros_like(v)).transpose(2, 0, 1, 3),
    )
    state, ys = jax.lax.scan(body, state, xs, length=seqlen)
    return ys.transpose(1, 2, 0, 3), state


def decode_loop(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    adt: jnp.ndarray,
    dt: jnp.ndarray,
    lam: jnp.ndarray,
    angles: jnp.ndarray,
    q_bias: jnp.ndarray,
    k_bias: jnp.ndarray,
    state: R.SISOState,
    d_skip: jnp.ndarray | None = None,
    z: jnp.ndarray | None = None,
    dtype_policy: L.DtypePolicy = L.BF16,
    interpret: Any = False,
    folded: bool = False,
) -> tuple[jnp.ndarray, R.SISOState]:
    """Run one decode step over ``L`` tokens in a Python loop. For tests.

    Exists to check a decode chain against one chunked prefill call: shapes match the
    prefill entry point (``q``/``k`` are ``(B, L, N)`` etc.) so the two are directly
    comparable. Defaults to the per-head kernel so that grid keeps getting exercised.

    For anything timed use `decode_scan`, which is the same computation under one
    ``lax.scan`` instead of ``L`` separate dispatches.
    """
    seqlen = q.shape[1]
    outs = []
    for t in range(seqlen):
        y, state = decode_step(
            q[:, t], k[:, t], v[:, :, t], adt[:, :, t], dt[:, :, t], lam[:, :, t],
            angles[:, t], q_bias, k_bias, state,
            d_skip=d_skip, z=None if z is None else z[:, :, t],
            dtype_policy=dtype_policy, interpret=interpret,
            folded=folded,
        )
        outs.append(y)
    return jnp.stack(outs, axis=2), state
