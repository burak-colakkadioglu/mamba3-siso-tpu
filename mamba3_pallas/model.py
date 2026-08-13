"""A stacked Mamba-3 SISO language model.

`layer.mamba3_siso_layer` is one layer. This wraps ``n_layers`` of them in a pre-norm
residual stack with an embedding and an output head, which is what ``MambaLMHeadModel``
does upstream.

With ``d_intermediate=0`` a block is just the mixer, since the SISO block's own ``z`` gate
is its nonlinearity:

    x = embed[tokens]
    for each block:  x = x + siso(rms_norm(x))
    logits = rms_norm(x) @ head

With ``d_intermediate > 0`` each block gets a gated MLP after the mixer, on its own
residual and its own norm. That is what the released `state-spaces` Mamba-3 checkpoints
use (the 1.5b has ``d_model=2048, d_intermediate=4096``):

    for each block:  x = x + siso(rms_norm(x))
                     x = x + mlp(rms_norm2(x))

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
      vocab_size: Token count. Byte-level data uses 256. Released checkpoints pad this
        to a multiple of ``pad_vocab_size_multiple``, so a converted config carries the
        padded number, not the tokenizer's.
      n_layers: Number of SISO blocks.
      siso: `layer.SISOConfig` for each block.
      tie_head: Reuse the embedding matrix as the output head. Saves
        ``vocab * d_model`` parameters and usually helps at small scale.
      d_intermediate: Hidden width of the per-block gated MLP. 0 means no MLP, which is
        the byte-level default here; the released Mamba-3 checkpoints use ``2 * d_model``.
    """

    vocab_size: int
    n_layers: int
    siso: LY.SISOConfig
    tie_head: bool = True
    d_intermediate: int = 0

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
        if self.d_intermediate:
            per_layer += (
                c.d_model * 2 * self.d_intermediate   # fc1, gate and value fused
                + self.d_intermediate * c.d_model     # fc2
                + c.d_model                           # mlp norm gain
            )
        head = 0 if self.tie_head else self.d_model * self.vocab_size
        return (
            self.vocab_size * self.d_model   # embedding
            + self.n_layers * per_layer
            + self.d_model                   # final norm
            + head
        )


class MLPParams(NamedTuple):
    """Gated MLP weights, matching upstream's ``GatedMLP``.

    ``fc1`` is one matrix producing ``2 * d_intermediate`` columns which then split into
    value and gate halves, rather than two separate matrices. That is upstream's layout,
    so a checkpoint's ``mlp.fc1.weight`` transposes straight in.
    """

    fc1: jnp.ndarray        # (d_model, 2 * d_intermediate)
    fc2: jnp.ndarray        # (d_intermediate, d_model)


class BlockParams(NamedTuple):
    """One residual block: the mixer, plus optionally an MLP on its own norm."""

    norm_gain: jnp.ndarray
    siso: LY.SISOParams
    mlp_norm_gain: jnp.ndarray | None = None
    mlp: MLPParams | None = None


class LMParams(NamedTuple):
    """Weights. ``blocks`` is a list of `BlockParams`, one per layer."""

    embed: jnp.ndarray
    blocks: list[BlockParams]
    final_norm: jnp.ndarray
    head: jnp.ndarray | None


def gated_mlp(p: MLPParams, x: jnp.ndarray, precision=None) -> jnp.ndarray:
    """``fc2(value * silu(gate))``, where ``fc1`` produces both halves at once.

    Upstream splits ``fc1``'s output with ``chunk(2, dim=-1)``, so the value half is the
    first ``d_intermediate`` columns and the gate half is the second. Getting that order
    backwards still runs and still trains, it just silently mismatches a checkpoint, so
    the order is load-bearing rather than cosmetic.
    """
    h = jnp.matmul(x, p.fc1.astype(x.dtype), precision=precision)
    value, gate = jnp.split(h, 2, axis=-1)
    return jnp.matmul(
        (value * jax.nn.silu(gate)).astype(p.fc2.dtype), p.fc2, precision=precision
    )



def init_lm(cfg: LMConfig, key: jax.Array, dtype=jnp.float32) -> LMParams:
    """Fresh weights.

    ``out_proj`` is scaled by ``1/sqrt(2*n_layers)`` -- the standard residual-stack
    correction (GPT-2's, and what upstream's ``_init_weights`` applies). Without it the
    residual stream's variance grows with depth and deep stacks start out badly
    conditioned. ``fc2`` gets the same treatment for the same reason, since with an MLP
    each layer writes to the residual stream twice.
    """
    n_keys = 2 * cfg.n_layers + 2
    keys = jax.random.split(key, n_keys)
    embed = jax.random.normal(keys[0], (cfg.vocab_size, cfg.d_model), dtype) * 0.02

    blocks = []
    resid_scale = 1.0 / math.sqrt(2.0 * cfg.n_layers)
    di = cfg.d_intermediate
    for i in range(cfg.n_layers):
        p = LY.init_params(cfg.siso, keys[i + 1], dtype)
        p = p._replace(out_proj=p.out_proj * resid_scale)
        mlp = None
        mlp_norm = None
        if di:
            k1, k2 = jax.random.split(keys[cfg.n_layers + 1 + i])
            # Standard fan-in init. fc1 is (d_model, 2*di) because value and gate are
            # fused into one matmul, matching upstream's GatedMLP.
            fc1 = jax.random.normal(k1, (cfg.d_model, 2 * di), dtype) / math.sqrt(
                cfg.d_model
            )
            fc2 = (
                jax.random.normal(k2, (di, cfg.d_model), dtype)
                / math.sqrt(di)
                * resid_scale
            )
            mlp = MLPParams(fc1=fc1, fc2=fc2)
            mlp_norm = jnp.ones((cfg.d_model,), dtype)
        blocks.append(
            BlockParams(
                norm_gain=jnp.ones((cfg.d_model,), dtype),
                siso=p,
                mlp_norm_gain=mlp_norm,
                mlp=mlp,
            )
        )

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
    prec = L.policy(cfg.siso.policy_name).precision
    new_states: list[R.SISOState] = []
    for i, b in enumerate(params.blocks):
        h = LY.rms_norm(x, b.norm_gain, cfg.siso.norm_eps)
        y, st = LY.mamba3_siso_layer(
            b.siso, h, cfg.siso, state=None if states is None else states[i],
            interpret=interpret, device_kind=device_kind,
        )
        new_states.append(st)
        x = x + y
        if b.mlp is not None:
            # Second residual, own norm. Upstream's Block does exactly this when
            # d_intermediate > 0.
            x = x + gated_mlp(
                b.mlp, LY.rms_norm(x, b.mlp_norm_gain, cfg.siso.norm_eps), prec
            )
    x = LY.rms_norm(x, params.final_norm, cfg.siso.norm_eps)
    head = params.embed.T if cfg.tie_head else params.head
    logits = jnp.matmul(x.astype(head.dtype), head, precision=prec)
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
    for i, blk in enumerate(params.blocks):
        p = blk.siso
        h = LY.rms_norm(x, blk.norm_gain, c.norm_eps)
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
        if blk.mlp is not None:
            x = x + gated_mlp(
                blk.mlp, LY.rms_norm(x, blk.mlp_norm_gain, c.norm_eps), prec
            )
    x = LY.rms_norm(x, params.final_norm, c.norm_eps)
    head = params.embed.T if cfg.tie_head else params.head
    return jnp.matmul(x.astype(head.dtype), head, precision=prec), out_states
