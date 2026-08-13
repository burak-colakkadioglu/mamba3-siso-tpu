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

`torch_to_jax` handles one mixer. `load_lm_state_dict` handles a whole released
checkpoint: ``state-spaces/mamba3-siso-*`` ships a ``MambaLMHeadModel`` state dict with
``backbone.layers.N.mixer.*`` keys plus an embedding, per-block norms, an optional gated
MLP and a final norm, so it needs the key mapping and the `model.LMConfig` too.
"""

from __future__ import annotations

from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np

from . import layer as LY
from . import layout as L


def _to_numpy(x: Any) -> np.ndarray:
    """Accept a torch tensor, numpy array, or anything with ``__array__``.

    Torch tensors go through ``.float()`` before numpy sees them. numpy has no native
    bfloat16, so ``np.asarray(bf16_tensor, dtype=np.float32)`` raises
    ``TypeError: Got unsupported ScalarType BFloat16`` rather than converting, and the
    released Mamba-3 checkpoints are stored bf16. Widening on the torch side first works
    for bf16, fp16 and f32 alike.
    """
    if hasattr(x, "detach"):
        x = x.detach().cpu()
        if hasattr(x, "float"):
            x = x.float()
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


# --------------------------------------------------------------------------------------
# Whole released checkpoints
# --------------------------------------------------------------------------------------
#
# A `MambaLMHeadModel` state dict looks like this (n_layer blocks):
#
#     backbone.embedding.weight                     (vocab, d_model)
#     backbone.layers.0.norm.weight                 (d_model,)
#     backbone.layers.0.mixer.in_proj.weight        (d_in_proj, d_model)
#     backbone.layers.0.mixer.out_proj.weight       (d_model, d_inner)
#     backbone.layers.0.mixer.dt_bias               (nheads,)
#     backbone.layers.0.mixer.B_bias                (nheads, mimo_rank, d_state)
#     backbone.layers.0.mixer.C_bias                (nheads, mimo_rank, d_state)
#     backbone.layers.0.mixer.B_norm.weight         (d_state,)
#     backbone.layers.0.mixer.C_norm.weight         (d_state,)
#     backbone.layers.0.mixer.D                     (nheads,)
#     backbone.layers.0.norm2.weight                (d_model,)      if d_intermediate
#     backbone.layers.0.mlp.fc1.weight              (2*d_int, d_model)
#     backbone.layers.0.mlp.fc2.weight              (d_model, d_int)
#     backbone.norm_f.weight                        (d_model,)
#     lm_head.weight                                (vocab, d_model)
#
# `lm_head.weight` is the same tensor object as the embedding when tie_embeddings is set,
# so it is present but redundant.

MIXER_PREFIX = "backbone.layers.{i}.mixer."


def config_from_hf(cfg_json: Mapping[str, Any], **overrides) -> "Any":
    """A released ``config.json`` -> `model.LMConfig`.

    Reads the fields the kernels need and ignores the rest (``fused_add_norm``,
    ``residual_in_fp32`` and friends describe upstream's CUDA plumbing, not the math).

    Two things are easy to get wrong:

    * **vocab padding.** Upstream pads ``vocab_size`` up to
      ``pad_vocab_size_multiple`` *inside* ``MambaLMHeadModel.__init__``, so the
      embedding is wider than the config's ``vocab_size`` and the checkpoint's actual
      width is the padded one. The padded number is what goes in the config here.
    * **chunk.** ``config.json`` carries upstream's chunk size (64 for the released
      models), which Mosaic cannot use. It is a tiling parameter, not a weight shape, so
      it is overridden to 128 unless the caller says otherwise; the chunked algorithm is
      identical at any chunk, which is what the ``segments`` test stage checks.

    Raises:
      ValueError: If the checkpoint is MIMO or has attention layers, neither of which
        these kernels implement.
    """
    from . import model as M

    ssm = dict(cfg_json.get("ssm_cfg", {}))
    layer_name = ssm.pop("layer", "Mamba3")
    if layer_name != "Mamba3":
        raise ValueError(f"ssm_cfg.layer is {layer_name!r}, expected 'Mamba3'")
    if ssm.get("mimo", False) or ssm.get("mimo_rank", 1) not in (1, None):
        raise ValueError("this checkpoint is MIMO; only SISO is implemented")
    if cfg_json.get("attn_layer_idx"):
        raise ValueError(
            f"checkpoint has attention layers at {cfg_json['attn_layer_idx']}; "
            "hybrid attention/SSM stacks are not implemented"
        )
    if ssm.get("ngroups", 1) != 1:
        raise ValueError(f"ngroups={ssm['ngroups']}, only ngroups=1 (SISO) is implemented")

    vocab = int(cfg_json["vocab_size"])
    mult = int(cfg_json.get("pad_vocab_size_multiple", 1) or 1)
    if vocab % mult:
        vocab += mult - (vocab % mult)

    siso_kw: dict[str, Any] = {
        "d_model": int(cfg_json["d_model"]),
        "d_state": int(ssm.get("d_state", 128)),
        "headdim": int(ssm.get("headdim", 64)),
        "expand": int(ssm.get("expand", 2)),
        "rope_fraction": float(ssm.get("rope_fraction", 0.5)),
        "chunk": 128,
    }
    for src, dst in (("A_floor", "a_floor"), ("dt_min", "dt_min"), ("dt_max", "dt_max"),
                     ("dt_init_floor", "dt_init_floor")):
        if src in ssm:
            siso_kw[dst] = float(ssm[src])
    siso_kw.update({k: v for k, v in overrides.items() if k not in ("d_intermediate",)})

    return M.LMConfig(
        vocab_size=vocab,
        n_layers=int(cfg_json["n_layer"]),
        siso=LY.SISOConfig(**siso_kw),
        tie_head=bool(cfg_json.get("tie_embeddings", True)),
        d_intermediate=int(cfg_json.get("d_intermediate", 0)),
    )


def load_lm_state_dict(sd: Mapping[str, Any], cfg: "Any", dtype=jnp.float32) -> "Any":
    """A released ``MambaLMHeadModel`` state dict -> `model.LMParams`.

    Args:
      sd: The full checkpoint, with ``backbone.``-prefixed keys.
      cfg: From `config_from_hf`.

    Raises:
      KeyError: If a key the config implies is absent, with the key name. Loading a
        partially-mapped model would produce plausible-looking garbage, so this is strict.
    """
    from . import model as M

    def get(key: str) -> np.ndarray:
        if key not in sd:
            raise KeyError(f"checkpoint is missing {key!r}")
        return _to_numpy(sd[key])

    embed = get("backbone.embedding.weight")
    if embed.shape != (cfg.vocab_size, cfg.d_model):
        raise ValueError(
            f"embedding is {embed.shape}, config implies "
            f"{(cfg.vocab_size, cfg.d_model)}; check pad_vocab_size_multiple"
        )

    blocks = []
    for i in range(cfg.n_layers):
        pre = MIXER_PREFIX.format(i=i)
        mixer = {k[len(pre):]: sd[k] for k in sd if k.startswith(pre)}
        if not mixer:
            raise KeyError(f"no keys under {pre!r}; is this a Mamba-3 checkpoint?")
        siso = torch_to_jax(mixer, cfg.siso, dtype)

        mlp = None
        mlp_norm = None
        if cfg.d_intermediate:
            mlp = M.MLPParams(
                fc1=jnp.asarray(get(f"backbone.layers.{i}.mlp.fc1.weight").T, dtype),
                fc2=jnp.asarray(get(f"backbone.layers.{i}.mlp.fc2.weight").T, dtype),
            )
            mlp_norm = jnp.asarray(get(f"backbone.layers.{i}.norm2.weight"), dtype)
        blocks.append(
            M.BlockParams(
                norm_gain=jnp.asarray(get(f"backbone.layers.{i}.norm.weight"), dtype),
                siso=siso,
                mlp_norm_gain=mlp_norm,
                mlp=mlp,
            )
        )

    # Tied heads share storage upstream, so `lm_head.weight` is present either way; only
    # read it when the config says it is a separate matrix.
    head = None if cfg.tie_head else jnp.asarray(get("lm_head.weight").T, dtype)
    return M.LMParams(
        embed=jnp.asarray(embed, dtype),
        blocks=blocks,
        final_norm=jnp.asarray(get("backbone.norm_f.weight"), dtype),
        head=head,
    )


def load_pretrained(
    repo_id: str, dtype=jnp.float32, revision: str | None = None, **cfg_overrides
) -> tuple[Any, Any]:
    """Download a released checkpoint from the Hugging Face Hub. ``(params, cfg)``.

        params, cfg = convert.load_pretrained("state-spaces/mamba3-siso-1.5b")

    Needs ``huggingface_hub`` and ``torch`` (the weights are a ``pytorch_model.bin``,
    which is a pickle, so torch has to unpickle it). Both are optional dependencies here.
    """
    import json

    from huggingface_hub import hf_hub_download

    cfg_path = hf_hub_download(repo_id, "config.json", revision=revision)
    with open(cfg_path) as fh:
        cfg = config_from_hf(json.load(fh), **cfg_overrides)

    import torch

    bin_path = hf_hub_download(repo_id, "pytorch_model.bin", revision=revision)
    sd = torch.load(bin_path, map_location="cpu", weights_only=True)
    return load_lm_state_dict(sd, cfg, dtype), cfg

