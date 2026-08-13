"""Save and load trained weights.

`train.py` keeps the best-validation params in memory and then the process exits, so
without this the weights are gone when the run ends. `save` writes a single ``.npz``
holding every array plus the `model.LMConfig` that produced them, and `load` gives both
back, so a checkpoint is self describing: you do not have to remember which
``--d-model`` and ``--n-layers`` you trained with.

    from mamba3_pallas import checkpoint as CP
    CP.save("model.npz", params, cfg)
    params, cfg = CP.load("model.npz")

Format is plain ``numpy.savez``: one entry per array under a readable name, plus a
``__config__`` entry holding JSON. No pickle, so loading a file from someone else does
not execute their code, and anything that reads npz can inspect it without this package.

Weights are stored in whatever dtype they are in memory, which for `model.init_lm` is
float32. Pass ``dtype=jnp.bfloat16`` to `save` to halve the file at the cost of some
precision; the kernels run in bf16 anyway under the default policy. bfloat16 is not a
native numpy dtype, so the npz records it as an opaque 2 byte type and the JSON carries the
real dtype name per array; `load` restores it with a ``view``. That is transparent here but
means a bf16 checkpoint read by bare numpy shows up as ``|V2``.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import jax
import numpy as np

from . import layer as LY
from . import model as M

#: Bumped if the on-disk layout ever changes in a way that old readers cannot handle.
FORMAT_VERSION = 1

#: The npz entry holding the JSON config. Not a valid leaf name, so it cannot collide.
CONFIG_KEY = "__config__"


def _flat_key(keypath: tuple[Any, ...]) -> str:
    """A readable, stable name for one leaf of an `model.LMParams` tree.

    ``jax.tree_util.keystr`` gives ``.blocks[0][1].in_proj``, which is accurate but ugly
    in a file listing and mixes three separators. This maps the two positional indices
    that `LMParams.blocks` uses onto names:

        .blocks[0][0]           -> blocks.0.norm_gain
        .blocks[0][1].in_proj   -> blocks.0.in_proj
        .embed                  -> embed
    """
    parts: list[str] = []
    for k in keypath:
        if isinstance(k, jax.tree_util.GetAttrKey):
            parts.append(k.name)
        elif isinstance(k, jax.tree_util.SequenceKey):
            parts.append(str(k.idx))
        else:                                   # pragma: no cover - defensive
            parts.append(str(k).strip(".[]'\""))
    # blocks is a list of (norm_gain, SISOParams) pairs: name the tuple slots.
    if len(parts) >= 3 and parts[0] == "blocks":
        slot = parts[2]
        rest = parts[3:]
        if slot == "0":
            return f"blocks.{parts[1]}.norm_gain"
        if slot == "1" and rest:
            return f"blocks.{parts[1]}." + ".".join(rest)
    return ".".join(parts)


def config_to_dict(cfg: M.LMConfig) -> dict[str, Any]:
    """`model.LMConfig` -> a JSON-safe dict. Inverse of `config_from_dict`."""
    c = cfg.siso
    return {
        "format_version": FORMAT_VERSION,
        "vocab_size": cfg.vocab_size,
        "n_layers": cfg.n_layers,
        "tie_head": cfg.tie_head,
        "siso": {
            "d_model": c.d_model,
            "d_state": c.d_state,
            "headdim": c.headdim,
            "expand": c.expand,
            "rope_fraction": c.rope_fraction,
            "chunk": c.chunk,
            "a_floor": c.a_floor,
            "norm_eps": c.norm_eps,
            "dt_min": c.dt_min,
            "dt_max": c.dt_max,
            "dt_init_floor": c.dt_init_floor,
            "policy_name": c.policy_name,
        },
    }


def config_from_dict(d: dict[str, Any]) -> M.LMConfig:
    """A dict from `config_to_dict` -> `model.LMConfig`.

    Unknown keys in ``siso`` are dropped rather than raising, so a checkpoint written by
    a newer version that added a config field still loads here. Missing keys fall back to
    `layer.SISOConfig`'s defaults.
    """
    got = d.get("format_version", 0)
    if got > FORMAT_VERSION:
        raise ValueError(
            f"checkpoint format version {got} is newer than this package supports "
            f"({FORMAT_VERSION}); upgrade mamba3-siso-tpu"
        )
    fields = {f.name for f in dataclasses.fields(LY.SISOConfig)}
    siso_kw = {k: v for k, v in d["siso"].items() if k in fields}
    return M.LMConfig(
        vocab_size=int(d["vocab_size"]),
        n_layers=int(d["n_layers"]),
        siso=LY.SISOConfig(**siso_kw),
        tie_head=bool(d["tie_head"]),
    )


def _np_dtype(dtype: Any) -> Any:
    """Normalize a dtype argument for `numpy.ndarray.astype`.

    ``jnp.float32`` is not ``np.float32``, it is a jax scalar-type object, and passing it
    straight to ``astype`` on an ml_dtypes-backed array (a bf16 checkpoint) raises
    ``ValueError: setting an array element with a sequence``. ``np.dtype()`` resolves both
    spellings, and ml_dtypes registers ``bfloat16`` with numpy so it resolves too.
    """
    return None if dtype is None else np.dtype(dtype)


def save(path: str, params: M.LMParams, cfg: M.LMConfig, dtype: Any = None) -> str:
    """Write ``params`` and ``cfg`` to ``path`` as an npz. Returns the path written.

    Params may be sharded across devices (they are, after a `train.make_step` run under
    ``shard_map``); ``np.asarray`` gathers each one to the host, so no unsharding step is
    needed on the caller's side.

    Args:
      dtype: Cast every array before writing. ``jnp.bfloat16`` halves the file. Note numpy
        has no native bfloat16, so a bf16 checkpoint stores ml_dtypes-backed arrays and
        needs jax (or ml_dtypes) installed to read; float16 is portable but has a much
        smaller exponent range. Default keeps whatever dtype the params are in.
    """
    flat = jax.tree_util.tree_flatten_with_path(params)[0]
    dt = _np_dtype(dtype)
    arrays: dict[str, np.ndarray] = {}
    for keypath, leaf in flat:
        name = _flat_key(keypath)
        if name in arrays:                      # pragma: no cover - defensive
            raise ValueError(f"duplicate leaf name {name!r}; _flat_key is not injective")
        a = np.asarray(leaf)
        arrays[name] = a.astype(dt) if dt is not None else a
    arrays[CONFIG_KEY] = np.frombuffer(
        json.dumps(
            {**config_to_dict(cfg), "dtypes": {k: str(v.dtype) for k, v in arrays.items()}}
        ).encode("utf-8"),
        dtype=np.uint8,
    )
    np.savez(path, **arrays)
    # np.savez appends .npz when the name lacks it, so report what actually exists.
    return path if path.endswith(".npz") else path + ".npz"


def load(path: str, dtype: Any = None) -> tuple[M.LMParams, M.LMConfig]:
    """Read a checkpoint written by `save`. Returns ``(params, cfg)``.

    The config in the file rebuilds the tree structure, so this does not need to be told
    the model shape. Every array the structure expects has to be present, and a shape
    mismatch raises rather than loading a silently wrong model.

    Args:
      dtype: Cast arrays on load, e.g. ``jnp.float32`` to widen a bf16 checkpoint.
    """
    with np.load(path, allow_pickle=False) as z:
        if CONFIG_KEY not in z.files:
            raise ValueError(
                f"{path} has no {CONFIG_KEY} entry; not a mamba3-siso-tpu checkpoint"
            )
        dt = _np_dtype(dtype)
        meta = json.loads(bytes(z[CONFIG_KEY]).decode("utf-8"))
        cfg = config_from_dict(meta)
        stored = meta.get("dtypes", {})
        # Build the empty tree from cfg, then fill it by name. Using the same
        # `init_lm` structure means a renamed or missing leaf is caught here.
        template = M.init_lm(cfg, jax.random.key(0))
        flat = jax.tree_util.tree_flatten_with_path(template)[0]
        leaves = []
        for keypath, ref in flat:
            name = _flat_key(keypath)
            if name not in z.files:
                raise ValueError(f"{path} is missing array {name!r}")
            a = z[name]
            if a.shape != ref.shape:
                raise ValueError(
                    f"{name}: checkpoint has shape {a.shape}, config implies {ref.shape}"
                )
            # npz has no bfloat16, so such arrays come back as an opaque void type of the
            # right width. Reinterpret them using the dtype name the writer recorded.
            if a.dtype.kind == "V" and name in stored:
                a = a.view(np.dtype(stored[name]))
            leaves.append(a.astype(dt) if dt is not None else a)
        extra = set(z.files) - {CONFIG_KEY} - {_flat_key(k) for k, _ in flat}
        if extra:
            raise ValueError(f"{path} has unexpected arrays: {sorted(extra)}")
    treedef = jax.tree.structure(template)
    return jax.tree.unflatten(treedef, leaves), cfg


