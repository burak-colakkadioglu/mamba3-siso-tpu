"""Convert a PyTorch Mamba-3 SISO checkpoint into `layer.SISOParams`.

Three things have to change on the way across:

1. **Transpose.** ``torch.nn.Linear`` stores ``(out_features, in_features)``; the JAX
   side applies ``u @ W`` and wants ``(in, out)``.
2. **The rotary permutation.** Upstream rotates adjacent channel pairs; this
   implementation rotates a contiguous half-split, because on TPU the ``d_state``
   axis is the 128-lane dimension and a stride-2 split costs a lane shuffle every
   chunk. Applying `layout.deinterleave_perm` once, offline, makes the two layouts
   numerically identical -- see `layout.deinterleave_perm` for the proof sketch.
   It touches every tensor that lives on the state axis: the ``B``/``C`` slices of
   ``in_proj``, the two biases, and the two norm gains.
3. **Squeeze the MIMO rank.** Upstream keeps ``B_bias``/``C_bias`` as
   ``(H, R, N)``; SISO has ``R == 1``.

Everything else -- ``dt_bias``, ``D``, ``out_proj`` -- passes through unchanged.
``A``, ``dt`` and ``lambda`` are per-head scalars, so they are permutation-invariant,
and ``q~ . k~ == u . w`` because rotations are orthogonal.
"""

from __future__ import annotations

from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np

from . import layer as LY
from . import layout as L


def _to_numpy(x: Any) -> np.ndarray:
    """Accept a torch tensor, numpy array, or anything with ``__array__``."""
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return np.asarray(x, dtype=np.float32)


def _perm(n: int) -> np.ndarray:
    return np.asarray(L.deinterleave_perm(n))


def config_from_state_dict(
    sd: Mapping[str, Any], headdim: int = 64, **overrides
) -> LY.SISOConfig:
    """Infer a `layer.SISOConfig` from checkpoint shapes.

    ``d_model`` and ``d_in_proj`` come from ``in_proj.weight``, ``d_state`` and
    ``nheads`` from ``B_bias``. ``headdim`` cannot be inferred (only the product
    ``nheads * headdim == d_inner`` is visible), so it is a parameter; 64 is
    upstream's default.
    """
    in_proj = _to_numpy(sd["in_proj.weight"])
    d_in_proj, d_model = in_proj.shape
    b_bias = _to_numpy(sd["B_bias"])
    nheads, d_state = b_bias.shape[0], b_bias.shape[-1]
    d_inner = nheads * headdim
    if d_inner % d_model != 0:
        raise ValueError(
            f"nheads*headdim = {d_inner} is not a multiple of d_model = {d_model}; "
            f"headdim={headdim} is probably wrong for this checkpoint"
        )
    expand = d_inner // d_model

    cfg = LY.SISOConfig(
        d_model=d_model, d_state=d_state, headdim=headdim, expand=expand, **overrides
    )
    if cfg.d_in_proj != d_in_proj:
        raise ValueError(
            f"in_proj width {d_in_proj} does not match the config's {cfg.d_in_proj}. "
            "This checkpoint may be MIMO (mimo_rank > 1), which is out of scope, or "
            f"headdim is not {headdim}."
        )
    return cfg


def torch_to_jax(
    sd: Mapping[str, Any], cfg: LY.SISOConfig, dtype=jnp.float32
) -> LY.SISOParams:
    """Build `layer.SISOParams` from a torch ``state_dict``.

    Args:
      sd: Keys ``in_proj.weight``, ``out_proj.weight``, ``dt_bias``, ``B_bias``,
        ``C_bias``, ``D``, and the two norm gains. Norm gains are accepted under
        either ``B_norm.weight`` (upstream ``RMSNormGated``) or ``B_norm_weight``
        (the vendored `torch_ref` module).
      cfg: From `config_from_state_dict`, or built by hand.

    Returns:
      `layer.SISOParams` in the permuted layout the kernels expect.
    """
    n = cfg.d_state
    perm = _perm(n)

    # (out, in) -> (in, out).
    in_proj = _to_numpy(sd["in_proj.weight"]).T.copy()
    out_proj = _to_numpy(sd["out_proj.weight"]).T.copy()

    # Permute the two state-axis blocks of in_proj in place. Order of the fused
    # projection is [z, x, B, C, dd_dt, dd_A, trap, angles], so B and C start after
    # 2 * d_inner columns.
    b_start = 2 * cfg.d_inner
    c_start = b_start + n
    for start in (b_start, c_start):
        block = in_proj[:, start : start + n]
        in_proj[:, start : start + n] = block[:, perm]

    def norm_gain(prefix: str) -> np.ndarray:
        for key in (f"{prefix}_norm.weight", f"{prefix}_norm_weight"):
            if key in sd:
                return _to_numpy(sd[key])
        raise KeyError(f"missing {prefix} norm weight ({prefix}_norm.weight)")

    # (H, R, N) -> (H, N); SISO always has R == 1.
    def bias(prefix: str) -> np.ndarray:
        arr = _to_numpy(sd[f"{prefix}_bias"])
        if arr.ndim == 3:
            if arr.shape[1] != 1:
                raise ValueError(
                    f"{prefix}_bias has mimo_rank {arr.shape[1]}; MIMO is out of scope"
                )
            arr = arr[:, 0]
        return arr

    return LY.SISOParams(
        in_proj=jnp.asarray(in_proj, dtype),
        out_proj=jnp.asarray(out_proj, dtype),
        dt_bias=jnp.asarray(_to_numpy(sd["dt_bias"]), dtype),
        b_norm_weight=jnp.asarray(norm_gain("B")[perm], dtype),
        c_norm_weight=jnp.asarray(norm_gain("C")[perm], dtype),
        b_bias=jnp.asarray(bias("B")[:, perm], dtype),
        c_bias=jnp.asarray(bias("C")[:, perm], dtype),
        d_skip=jnp.asarray(_to_numpy(sd["D"]), dtype),
    )


def jax_to_torch(params: LY.SISOParams, cfg: LY.SISOConfig) -> dict[str, np.ndarray]:
    """Inverse of `torch_to_jax`, as numpy arrays in torch's shapes and order.

    Useful for exporting a TPU-trained layer back to the CUDA kernels, and as a
    round-trip check on the permutation.
    """
    n = cfg.d_state
    inv = np.asarray(L.interleave_perm(n))

    in_proj = np.asarray(params.in_proj, np.float32).copy()
    b_start = 2 * cfg.d_inner
    for start in (b_start, b_start + n):
        block = in_proj[:, start : start + n]
        in_proj[:, start : start + n] = block[:, inv]

    return {
        "in_proj.weight": in_proj.T.copy(),
        "out_proj.weight": np.asarray(params.out_proj, np.float32).T.copy(),
        "dt_bias": np.asarray(params.dt_bias, np.float32),
        "B_norm.weight": np.asarray(params.b_norm_weight, np.float32)[inv],
        "C_norm.weight": np.asarray(params.c_norm_weight, np.float32)[inv],
        "B_bias": np.asarray(params.b_bias, np.float32)[:, inv][:, None, :],
        "C_bias": np.asarray(params.c_bias, np.float32)[:, inv][:, None, :],
        "D": np.asarray(params.d_skip, np.float32),
    }
