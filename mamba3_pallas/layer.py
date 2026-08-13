"""The Mamba-3 SISO layer: projections around the kernel.

Two APIs over one implementation. `mamba3_siso_layer` is a plain function taking a
`SISOParams` pytree -- no framework dependency, so it works with Optax, a raw
training loop, or whatever else. `Mamba3SISO` is a thin ``flax.nnx.Module`` that
holds the same arrays as ``nnx.Param``s and calls straight through. Flax is imported
lazily, so this module is usable without it installed.

Layer structure, matching ``mamba/mamba_ssm/modules/mamba3.py``:

    in_proj -> [z, x, B, C, dd_dt, dd_A, trap, angles]
    B, C   -> RMSNorm (no bias; "BCNorm", the paper's analogue of QKNorm)
    kernel -> gated by silu(z)
    out_proj

Mamba-3 has no short causal convolution: the state-input convolution implied by
trapezoidal discretization plus the learned B/C biases replace it. The paper's Table 5(a)
ablation at 440M has adding the conv back at 15.85 ppl against 15.72 without it.

Note the roles: ``C`` is the query (read out of the state) and ``B`` is the key
(written into it), so ``q = C`` and ``k = B`` at the kernel boundary.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from . import layout as L
from . import reference as R
from . import siso as SISO


@dataclasses.dataclass(frozen=True)
class SISOConfig:
    """Shapes and hyperparameters. Mirrors ``Mamba3.__init__``.

    Attributes:
      d_model: Model width.
      d_state: ``N``, the SSM state dimension per head.
      headdim: ``P``, channels per head.
      expand: ``d_inner = expand * d_model``.
      rope_fraction: Share of ``d_state`` that rotates; 0.5 or 1.0. With 0.5 and
        ``d_state=128`` there are 32 angles, and half the channels pass through.
      chunk: Kernel chunk length. Multiple of 128.
      a_floor: ``A`` is clamped to ``<= -a_floor`` so the state always decays.
      norm_eps: Epsilon for the B/C RMSNorms.
      policy_name: ``"bf16"`` or ``"f32"``.
    """

    d_model: int
    d_state: int = 128
    headdim: int = 64
    expand: int = 2
    rope_fraction: float = 0.5
    chunk: int = 128
    a_floor: float = 1e-4
    norm_eps: float = 1e-5
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    policy_name: str = "bf16"

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    @property
    def nheads(self) -> int:
        if self.d_inner % self.headdim != 0:
            raise ValueError(f"d_inner {self.d_inner} not divisible by headdim {self.headdim}")
        return self.d_inner // self.headdim

    @property
    def n_angles(self) -> int:
        """Rotary pairs. Even, and at most ``d_state // 2``."""
        if self.rope_fraction not in (0.5, 1.0):
            raise ValueError(f"rope_fraction must be 0.5 or 1.0, got {self.rope_fraction}")
        width = int(self.d_state * self.rope_fraction)
        if width % 2:
            width -= 1
        n = width // 2
        if n <= 0:
            raise ValueError("rope_fraction too small: no rotary angles")
        return n

    @property
    def d_in_proj(self) -> int:
        """Width of the fused input projection. Order is fixed by `split_in_proj`."""
        return 2 * self.d_inner + 2 * self.d_state + 3 * self.nheads + self.n_angles

    def __post_init__(self):
        L.validate_chunk(self.chunk)
        _ = self.nheads, self.n_angles


class SISOParams(NamedTuple):
    """Layer weights.

    Attributes:
      in_proj: ``(d_model, d_in_proj)``. Column-major relative to torch's
        ``nn.Linear`` (which stores ``(out, in)``); `convert` transposes.
      out_proj: ``(d_inner, d_model)``.
      dt_bias: ``(H,)`` initialized so ``softplus(dt_bias)`` is log-uniform in
        ``[dt_min, dt_max]``.
      b_norm_weight, c_norm_weight: ``(N,)`` RMSNorm gains, in permuted layout.
      b_bias, c_bias: ``(H, N)`` per-head channel biases, permuted layout. Note
        these are added *after* the norm and default to 1, not 0.
      d_skip: ``(H,)`` skip-connection weight.
    """

    in_proj: jnp.ndarray
    out_proj: jnp.ndarray
    dt_bias: jnp.ndarray
    b_norm_weight: jnp.ndarray
    c_norm_weight: jnp.ndarray
    b_bias: jnp.ndarray
    c_bias: jnp.ndarray
    d_skip: jnp.ndarray


def init_params(cfg: SISOConfig, key: jax.Array, dtype=jnp.float32) -> SISOParams:
    """Fresh weights, matching upstream's initialization.

    ``dt_bias`` gets the inverse-softplus of a log-uniform ``dt`` draw, so the
    initial timescales span ``[dt_min, dt_max]``. The B/C biases start at 1 (not 0),
    as in ``Mamba3.__init__``.
    """
    k_in, k_out, k_dt = jax.random.split(key, 3)
    scale_in = 1.0 / math.sqrt(cfg.d_model)
    scale_out = 1.0 / math.sqrt(cfg.d_inner)

    dt = jnp.exp(
        jax.random.uniform(k_dt, (cfg.nheads,), dtype=jnp.float32)
        * (math.log(cfg.dt_max) - math.log(cfg.dt_min))
        + math.log(cfg.dt_min)
    )
    dt = jnp.clip(dt, cfg.dt_init_floor, None)
    # inverse softplus: log(expm1(dt)) == dt + log(-expm1(-dt))
    dt_bias = dt + jnp.log(-jnp.expm1(-dt))

    return SISOParams(
        in_proj=jax.random.uniform(
            k_in, (cfg.d_model, cfg.d_in_proj), dtype, -scale_in, scale_in
        ),
        out_proj=jax.random.uniform(
            k_out, (cfg.d_inner, cfg.d_model), dtype, -scale_out, scale_out
        ),
        dt_bias=dt_bias,
        b_norm_weight=jnp.ones((cfg.d_state,), dtype),
        c_norm_weight=jnp.ones((cfg.d_state,), dtype),
        b_bias=jnp.ones((cfg.nheads, cfg.d_state), dtype),
        c_bias=jnp.ones((cfg.nheads, cfg.d_state), dtype),
        d_skip=jnp.ones((cfg.nheads,), dtype),
    )


def split_in_proj(
    proj: jnp.ndarray, cfg: SISOConfig
) -> tuple[jnp.ndarray, ...]:
    """Split the fused projection.

    Order is ``[z, x, B, C, dd_dt, dd_A, trap, angles]`` -- fixed by upstream's
    ``in_proj`` layout, so checkpoints depend on it.

    Returns:
      ``(z, x, B, C, dt_raw, a_raw, trap_raw, angles)``. ``z``/``x`` come back as
      ``(B, L, H, P)``; ``B``/``C`` as ``(B, L, N)``; the three scalar streams as
      ``(B, L, H)``; ``angles`` as ``(B, L, Nr)``.
    """
    sizes = (
        cfg.d_inner, cfg.d_inner, cfg.d_state, cfg.d_state,
        cfg.nheads, cfg.nheads, cfg.nheads, cfg.n_angles,
    )
    # Split points must be Python ints: jnp.cumsum here would be a traced value and
    # jnp.split would reject it under jit.
    bounds = list(itertools.accumulate(sizes))[:-1]
    z, x, b, c, dt_raw, a_raw, trap_raw, angles = jnp.split(proj, bounds, axis=-1)
    batch, seqlen = proj.shape[0], proj.shape[1]
    z = z.reshape(batch, seqlen, cfg.nheads, cfg.headdim)
    x = x.reshape(batch, seqlen, cfg.nheads, cfg.headdim)
    return z, x, b, c, dt_raw, a_raw, trap_raw, angles


def rms_norm(x: jnp.ndarray, weight: jnp.ndarray, eps: float) -> jnp.ndarray:
    """RMSNorm over the last axis, computed in f32.

    Matches ``rms_norm_ref`` in ``mamba/mamba_ssm/ops/triton/layernorm_gated.py``:
    no mean subtraction, no bias, gain applied after scaling.
    """
    xf = x.astype(jnp.float32)
    rstd = jax.lax.rsqrt(jnp.mean(jnp.square(xf), axis=-1, keepdims=True) + eps)
    return (xf * rstd * weight.astype(jnp.float32)).astype(x.dtype)


def mamba3_siso_layer(
    params: SISOParams,
    u: jnp.ndarray,
    cfg: SISOConfig,
    state: R.SISOState | None = None,
    interpret: Any = False,
    device_kind: str | None = None,
) -> tuple[jnp.ndarray, R.SISOState]:
    """Forward pass of one Mamba-3 SISO layer.

    Args:
      params: `SISOParams`. B/C-related entries must already be in the permuted
        (half-split rotary) layout -- `convert.torch_to_jax` handles that.
      u: ``(B, L, d_model)``.
      cfg: `SISOConfig`.
      state: Carry-in for segmented prefill, or None.
      interpret: Pass `layout.interpret_mode()` to run on CPU.

    Returns:
      ``(out, state)`` with ``out`` shaped ``(B, L, d_model)``.
    """
    batch, seqlen, _ = u.shape
    # HIGHEST when the layer is running f32: the TPU MXU has no f32 multiplier, so at
    # the default precision an f32 matmul truncates its operands to bf16. Under the
    # bf16 policy the operands are already bf16 and DEFAULT is both correct and faster.
    prec = L.policy(cfg.policy_name).precision
    proj = jnp.matmul(u, params.in_proj.astype(u.dtype), precision=prec)
    z, x, b, c, dt_raw, a_raw, trap_raw, angles = split_in_proj(proj, cfg)

    # BCNorm, then the per-head channel biases. The biases are added inside the
    # kernel (it needs the pre-rotation vectors anyway for the diagonal term), so
    # only the norm happens here.
    b = rms_norm(b, params.b_norm_weight, cfg.norm_eps)
    c = rms_norm(c, params.c_norm_weight, cfg.norm_eps)

    # C reads out of the state (query), B writes into it (key).
    out = SISO.siso_segment(
        q=c,
        k=b,
        v=jnp.transpose(x, (0, 2, 1, 3)),          # (B, H, L, P)
        dt_raw=dt_raw,
        a_raw=a_raw,
        trap_raw=trap_raw,
        angles=angles,
        dt_bias=params.dt_bias,
        q_bias=params.c_bias,
        k_bias=params.b_bias,
        d_skip=params.d_skip,
        z=jnp.transpose(z, (0, 2, 1, 3)),
        state=state,
        a_floor=cfg.a_floor,
        chunk=cfg.chunk,
        policy_name=cfg.policy_name,
        interpret=interpret,
        device_kind=device_kind,
    )

    y = jnp.transpose(out.y, (0, 2, 1, 3)).reshape(batch, seqlen, cfg.d_inner)
    out_proj = jnp.matmul(
        y.astype(u.dtype), params.out_proj.astype(u.dtype), precision=prec
    )
    return out_proj, out.state


def _make_nnx_module():
    """Build the ``nnx.Module`` subclass on first use, so flax stays optional."""
    from flax import nnx

    class Mamba3SISO(nnx.Module):
        """Mamba-3 SISO layer as a Flax NNX module.

        Holds `SISOParams` as ``nnx.Param``s and delegates to
        `mamba3_siso_layer`. All the maths lives in the functional path; this is
        only parameter management.
        """

        def __init__(self, cfg: SISOConfig, *, rngs: nnx.Rngs, dtype=jnp.float32):
            self.cfg = cfg
            p = init_params(cfg, rngs.params(), dtype)
            for name, value in zip(p._fields, p):
                setattr(self, name, nnx.Param(value))

        def params(self) -> SISOParams:
            return SISOParams(*(getattr(self, n).value for n in SISOParams._fields))

        def __call__(
            self,
            u: jnp.ndarray,
            state: R.SISOState | None = None,
            interpret: Any = False,
        ) -> tuple[jnp.ndarray, R.SISOState]:
            return mamba3_siso_layer(
                self.params(), u, self.cfg, state=state, interpret=interpret
            )

    return Mamba3SISO


def __getattr__(name: str):
    """Lazily expose ``Mamba3SISO`` without importing flax at module load."""
    if name == "Mamba3SISO":
        return _make_nnx_module()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
