"""``jax.custom_vjp`` glue tying the forward and backward kernels together.

The differentiable boundary sits at ``(q, k, v, adt, gamma, scale, phi, ...)`` rather
than at the raw projections. That is deliberate: ``gamma``/``scale`` and ``phi`` are
built by `reference.preprocess` in plain JAX, so ordinary autodiff carries their
cotangents back to ``dt``/``lambda``/``angles`` -- no hand-written kernel needed for
either, and the ``t+1`` coupling inside ``scale`` stays correct by construction.

`siso` is the differentiable core. `siso_segment` wraps it with the preprocessing and
state plumbing that a caller actually wants.
"""

from __future__ import annotations

import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from . import kernel_bwd as KB
from . import kernel_fwd as KF
from . import layout as L
from . import reference as R


class SISOOutput(NamedTuple):
    """Result of a segment.

    Attributes:
      y: ``(B, H, L, P)`` gated output.
      state: Carry for the next segment or for decoding.
    """

    y: jnp.ndarray
    state: R.SISOState


@functools.partial(
    jax.custom_vjp,
    nondiff_argnums=(14, 15, 16, 17),  # chunk, policy_name, interpret, device_kind
)
def siso(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    adt: jnp.ndarray,
    gamma: jnp.ndarray,
    scale: jnp.ndarray,
    phi: jnp.ndarray,
    q_bias: jnp.ndarray,
    k_bias: jnp.ndarray,
    d_skip: jnp.ndarray | None,
    z: jnp.ndarray | None,
    seed_ssm: jnp.ndarray | None,
    seed_kv: jnp.ndarray | None,
    seed_coef: jnp.ndarray | None,
    chunk: int,
    policy_name: str,
    interpret: Any,
    device_kind: str | None,
):
    """Differentiable chunked SISO scan.

    Args:
      q, k: ``(B, L, N)`` post-RMSNorm C and B, permuted layout, shared across heads.
      v: ``(B, H, L, P)``.
      adt, gamma, scale: ``(B, H, L)`` f32 from `reference.preprocess`.
      phi: ``(B, H, L, Nr)`` cumulative rotary angles.
      q_bias, k_bias: ``(H, N)``.
      d_skip: ``(H,)`` or None. z: ``(B, H, L, P)`` or None.
      seed_ssm: ``(B, H, P, N)`` or None. seed_kv: ``(B, H, N + P)`` -- the previous
        segment's rotated-unscaled k concatenated with its v, packed so the two
        travel as one differentiable leaf. seed_coef: ``(B, H)`` = ``dt_0 (1-lam_0)``.
      chunk, policy_name, interpret, device_kind: static configuration.

    Returns:
      ``(y, final_ssm, final_k, final_v)``. ``final_k`` is rotated but unscaled.
    """
    pol = L.policy(policy_name)
    state_dim = q.shape[-1]
    seed_k = seed_kv[..., :state_dim] if seed_kv is not None else None
    seed_v = seed_kv[..., state_dim:] if seed_kv is not None else None
    y, fs, fk, fv = KF.siso_forward(
        q, k, v, adt, gamma, scale, phi, q_bias, k_bias,
        d_skip=d_skip, z=z,
        seed_ssm=seed_ssm, seed_k=seed_k, seed_v=seed_v, seed_coef=seed_coef,
        chunk=chunk, dtype_policy=pol, save_residuals=False,
        interpret=interpret, device_kind=device_kind,
    )
    return y, fs, fk, fv


def _siso_fwd(
    q, k, v, adt, gamma, scale, phi, q_bias, k_bias, d_skip, z,
    seed_ssm, seed_kv, seed_coef, chunk, policy_name, interpret, device_kind,
):
    pol = L.policy(policy_name)
    state_dim = q.shape[-1]
    seed_k = seed_kv[..., :state_dim] if seed_kv is not None else None
    seed_v = seed_kv[..., state_dim:] if seed_kv is not None else None
    y, fs, fk, fv, res = KF.siso_forward(
        q, k, v, adt, gamma, scale, phi, q_bias, k_bias,
        d_skip=d_skip, z=z,
        seed_ssm=seed_ssm, seed_k=seed_k, seed_v=seed_v, seed_coef=seed_coef,
        chunk=chunk, dtype_policy=pol, save_residuals=True,
        interpret=interpret, device_kind=device_kind,
    )
    residuals = (
        q, k, v, adt, gamma, scale, phi, q_bias, k_bias, d_skip, z,
        seed_ssm, seed_kv, seed_coef, res.ybar, res.chunk_states,
    )
    return (y, fs, fk, fv), residuals


def _siso_bwd(chunk, policy_name, interpret, device_kind, residuals, cotangents):
    (
        q, k, v, adt, gamma, scale, phi, q_bias, k_bias, d_skip, z,
        seed_ssm, seed_kv, seed_coef, ybar, chunk_states,
    ) = residuals
    dy, dfinal_ssm, dfinal_k, dfinal_v = cotangents
    pol = L.policy(policy_name)
    state_dim = q.shape[-1]
    seed_k = seed_kv[..., :state_dim] if seed_kv is not None else None
    seed_v = seed_kv[..., state_dim:] if seed_kv is not None else None

    g = KB.siso_backward(
        dy, q, k, v, adt, gamma, scale, phi, q_bias, k_bias, ybar, chunk_states,
        d_skip=d_skip, z=z,
        seed_ssm=seed_ssm, seed_k=seed_k, seed_v=seed_v, seed_coef=seed_coef,
        dfinal_ssm=dfinal_ssm, dfinal_k=dfinal_k, dfinal_v=dfinal_v,
        chunk=chunk, dtype_policy=pol, interpret=interpret, device_kind=device_kind,
    )

    dseed_kv = (
        jnp.concatenate([g.dseed_k, g.dseed_v], axis=-1)
        if seed_kv is not None
        else None
    )
    # Cast back to each primal's dtype -- custom_vjp requires an exact match.
    like = lambda x, p: None if p is None else x.astype(p.dtype)
    return (
        like(g.dq, q), like(g.dk, k), like(g.dv, v),
        like(g.dadt, adt), like(g.dgamma, gamma), like(g.dscale, scale),
        like(g.dphi, phi), like(g.dq_bias, q_bias), like(g.dk_bias, k_bias),
        like(g.dd_skip, d_skip), like(g.dz, z),
        like(g.dseed_ssm, seed_ssm), like(dseed_kv, seed_kv), like(g.dseed_coef, seed_coef),
    )


siso.defvjp(_siso_fwd, _siso_bwd)


def siso_segment(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    dt_raw: jnp.ndarray,
    a_raw: jnp.ndarray,
    trap_raw: jnp.ndarray,
    angles: jnp.ndarray,
    dt_bias: jnp.ndarray,
    q_bias: jnp.ndarray,
    k_bias: jnp.ndarray,
    d_skip: jnp.ndarray | None = None,
    z: jnp.ndarray | None = None,
    state: R.SISOState | None = None,
    a_floor: float = 1e-4,
    chunk: int = 128,
    policy_name: str = "bf16",
    interpret: Any = False,
    device_kind: str | None = None,
) -> SISOOutput:
    """Preprocess raw projections, run the scan, and return the carry.

    This is the entry point a layer should call. It owns the one subtlety of
    segmented execution: ``gamma``/``scale``/``phi`` are derived from *this
    segment's* raw inputs, never sliced from a whole-sequence version. ``scale``
    reaches one position forward, so a sliced copy would hand the last token a
    successor term that belongs to the next segment -- and then double-count it when
    the carried ``(k, v)`` pair supplies the same term on the next call.

    Args:
      q, k: ``(B, L, N)`` post-RMSNorm, permuted layout.
      v: ``(B, H, L, P)``.
      dt_raw, a_raw, trap_raw: ``(B, L, H)`` raw ``in_proj`` slices.
      angles: ``(B, L, Nr)`` raw rotary rates.
      dt_bias: ``(H,)``. q_bias, k_bias: ``(H, N)``.
      d_skip: ``(H,)`` or None. z: ``(B, H, L, P)`` or None.
      state: Carry-in from a previous segment, or None.

    Returns:
      `SISOOutput`.
    """
    pre, _ = R.preprocess(
        dt_raw, a_raw, trap_raw, angles, dt_bias,
        a_floor=a_floor, phi_init=None if state is None else state.phi,
    )

    if state is None:
        seed_ssm = seed_kv = seed_coef = None
    else:
        seed_ssm = state.ssm
        seed_kv = jnp.concatenate(
            [state.k.astype(jnp.float32), state.v.astype(jnp.float32)], axis=-1
        )
        seed_coef = pre.dt[:, :, 0] * (1.0 - pre.lam[:, :, 0])

    y, fs, fk, fv = siso(
        q, k, v, pre.adt, pre.gamma, pre.scale, pre.phi, q_bias, k_bias,
        d_skip, z, seed_ssm, seed_kv, seed_coef,
        chunk, policy_name, interpret, device_kind,
    )
    return SISOOutput(
        y=y, state=R.SISOState(ssm=fs, k=fk, v=fv, phi=pre.phi[:, :, -1, :])
    )
