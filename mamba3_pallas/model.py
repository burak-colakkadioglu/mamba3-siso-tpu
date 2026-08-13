"""A stacked Mamba-3 SISO language model.

`layer.mamba3_siso_layer` is one layer. This wraps ``n_layers`` of them in a pre-norm
residual stack with an embedding and an output head, which is what ``MambaLMHeadModel``
does upstream with ``d_intermediate=0``: no interleaved MLP, since the SISO block's own
``z`` gate is its nonlinearity.

    x = embed[tokens]
    for each block:  x = x + siso(rms_norm(x))
    logits = rms_norm(x) @ head

Deliberately plain. No dropout, no fused anything. If training misbehaves the cause should
be the kernel or the data, not this file.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from . import layer as LY
from . import layout as L
from . import reference as R


@dataclasses.dataclass(frozen=True)
class LMConfig:
    """Model shape. `siso` is the per-layer config, shared by every block.

    Attributes:
      vocab_size: Token count. Byte-level data uses 256.
      n_layers: Number of SISO blocks.
      siso: `layer.SISOConfig` for each block.
      tie_head: Reuse the embedding matrix as the output head. Saves
        ``vocab * d_model`` parameters and usually helps at small scale.
    """

    vocab_size: int
    n_layers: int
    siso: LY.SISOConfig
    tie_head: bool = True

    @property
    def d_model(self) -> int:
        return self.siso.d_model

    def param_count(self) -> int:
        """Analytic parameter count, for reporting.

        Checked against ``sum(x.size for x in jax.tree.leaves(init_lm(...)))``; the two
        agree to within the ``nheads`` fudge below, which is close enough for a log line.
        """
        c = self.siso
        per_layer = (
            c.d_model * c.d_in_proj          # in_proj
            + c.d_inner * c.d_model          # out_proj
            + c.nheads * 2                   # dt_bias, d_skip
            + c.d_state * 2                  # b/c norm gains
            + c.nheads * c.d_state * 2       # b/c biases
            + c.d_model                      # block norm gain
        )
        head = 0 if self.tie_head else self.d_model * self.vocab_size
        return (
            self.vocab_size * self.d_model   # embedding
            + self.n_layers * per_layer
            + self.d_model                   # final norm
            + head
        )


class LMParams(NamedTuple):
    """Weights. ``blocks`` is a list of ``(norm_gain, SISOParams)`` per layer."""

    embed: jnp.ndarray
    blocks: list[tuple[jnp.ndarray, LY.SISOParams]]
    final_norm: jnp.ndarray
    head: jnp.ndarray | None


def init_lm(cfg: LMConfig, key: jax.Array, dtype=jnp.float32) -> LMParams:
    """Fresh weights.

    ``out_proj`` is scaled by ``1/sqrt(2*n_layers)`` -- the standard residual-stack
    correction (GPT-2's, and what upstream's ``_init_weights`` applies). Without it the
    residual stream's variance grows with depth and deep stacks start out badly
    conditioned.
    """
    keys = jax.random.split(key, cfg.n_layers + 2)
    embed = jax.random.normal(keys[0], (cfg.vocab_size, cfg.d_model), dtype) * 0.02

    blocks = []
    resid_scale = 1.0 / math.sqrt(2.0 * cfg.n_layers)
    for i in range(cfg.n_layers):
        p = LY.init_params(cfg.siso, keys[i + 1], dtype)
        p = p._replace(out_proj=p.out_proj * resid_scale)
        blocks.append((jnp.ones((cfg.d_model,), dtype), p))

    head = (
        None
        if cfg.tie_head
        else jax.random.normal(keys[-1], (cfg.d_model, cfg.vocab_size), dtype) * 0.02
    )
    return LMParams(embed=embed, blocks=blocks, final_norm=jnp.ones((cfg.d_model,), dtype),
                    head=head)


def lm_forward(
    params: LMParams,
    tokens: jnp.ndarray,
    cfg: LMConfig,
    interpret: Any = False,
    device_kind: str | None = None,
    states: list[R.SISOState] | None = None,
) -> jnp.ndarray | tuple[jnp.ndarray, list[R.SISOState]]:
    """``(B, L)`` int tokens -> ``(B, L, vocab)`` logits.

    Pre-norm residual: each block reads a normalized copy and adds its output back to an
    unnormalized stream, so gradients reach layer 0 through a clean identity path.

    Args:
      states: Per-layer carry-in for segmented prefill. When given, the per-layer
        carry-out is returned alongside the logits, which is what `prefill` and the
        decode path need.
    """
    x = params.embed[tokens]
    new_states: list[R.SISOState] = []
    for i, (norm_gain, p) in enumerate(params.blocks):
        h = LY.rms_norm(x, norm_gain, cfg.siso.norm_eps)
        y, st = LY.mamba3_siso_layer(
            p, h, cfg.siso, state=None if states is None else states[i],
            interpret=interpret, device_kind=device_kind,
        )
        new_states.append(st)
        x = x + y
    x = LY.rms_norm(x, params.final_norm, cfg.siso.norm_eps)
    head = params.embed.T if cfg.tie_head else params.head
    logits = jnp.matmul(
        x.astype(head.dtype), head, precision=L.policy(cfg.siso.policy_name).precision
    )
    return (logits, new_states) if states is not None else logits


def zero_states(cfg: LMConfig, batch: int, dtype=jnp.float32) -> list[R.SISOState]:
    """A fresh per-layer carry, for generation from an empty context."""
    c = cfg.siso
    return [
        R.zero_state(batch, c.nheads, c.headdim, c.d_state, c.n_angles, dtype)
        for _ in range(cfg.n_layers)
    ]


def lm_decode_step(
    params: LMParams,
    token: jnp.ndarray,
    states: list[R.SISOState],
    cfg: LMConfig,
    interpret: Any = False,
) -> tuple[jnp.ndarray, list[R.SISOState]]:
    """One token through the stack using `kernel_decode`. ``(B,)`` -> ``(B, vocab)``.

    This is the O(1)-per-token generation path. It mirrors `lm_forward`'s structure but
    calls the decode kernel, so it needs the same projection and normalization steps that
    `layer.mamba3_siso_layer` does -- inlined here rather than factored out, because the
    prefill layer works on ``(B, L, ...)`` and this works on ``(B, ...)``, and threading a
    length axis through both made the shared version harder to read than two copies.
    """
    from . import kernel_decode as KD

    prec = L.policy(cfg.siso.policy_name).precision
    c = cfg.siso
    x = params.embed[token]                                   # (B, d_model)
    out_states: list[R.SISOState] = []
    for i, (norm_gain, p) in enumerate(params.blocks):
        h = LY.rms_norm(x, norm_gain, c.norm_eps)
        proj = jnp.matmul(h, p.in_proj.astype(h.dtype), precision=prec)
        z, xx, b, cc, dt_raw, a_raw, trap_raw, angles = LY.split_in_proj(
            proj[:, None, :], c
        )
        b = LY.rms_norm(b, p.b_norm_weight, c.norm_eps)[:, 0]         # (B, N)
        cc = LY.rms_norm(cc, p.c_norm_weight, c.norm_eps)[:, 0]
        # `preprocess` on a length-1 sequence: softplus/heavy-tail/sigmoid, no cumsum.
        dt = jax.nn.softplus(dt_raw[:, 0] + p.dt_bias)                # (B, H)
        a = jnp.minimum(-R.heavy_tail(a_raw[:, 0].astype(jnp.float32)), -c.a_floor)
        lam = jax.nn.sigmoid(trap_raw[:, 0].astype(jnp.float32))
        y, st = KD.decode_step(
            cc, b, xx[:, 0],
            a * dt, dt, lam, angles[:, 0],
            p.c_bias, p.b_bias, states[i],
            d_skip=p.d_skip, z=z[:, 0],
            dtype_policy=L.policy(c.policy_name), interpret=interpret,
        )
        out_states.append(st)
        x = x + jnp.matmul(
            y.reshape(y.shape[0], c.d_inner).astype(x.dtype),
            p.out_proj.astype(x.dtype), precision=prec,
        )
    x = LY.rms_norm(x, params.final_norm, c.norm_eps)
    head = params.embed.T if cfg.tie_head else params.head
    return jnp.matmul(x.astype(head.dtype), head, precision=prec), out_states
