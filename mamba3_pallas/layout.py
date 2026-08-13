"""Layouts, device config, and Mosaic-lowering checks for the Mamba-3 SISO TPU kernels.

Nothing here touches a TPU. The device table is read out of Pallas' own hardware
database (``pltpu.get_tpu_info_for_chip``) so it stays correct without hardcoding,
and ``assert_lowers`` runs the real Mosaic lowering under an abstract mesh, which
means block-shape bugs are caught on a CPU laptop instead of on Kaggle.

Two facts about Pallas TPU that shape every kernel in this package:

1. ``cumsum`` has no lowering. Within-chunk prefix sums must go through the MXU as
   a matmul against a lower-triangular ones matrix -- see `tril_ones` / `prefix_sum`.
2. A per-token scalar stream cannot be passed as ``(B, H, L)``. Mosaic wants the
   last two block dims divisible by (8, 128) or equal to the array dims; the legal
   shape is ``(B, H, nchunks, 1, chunk)`` -- see `scalars_to_blocks`.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# --------------------------------------------------------------------------------------
# JAX version guards.
#
# Developed against jax 0.11.0. Kaggle's TPU runtime pins its own version, and two
# names have moved historically, so resolve them once here rather than at each use.
# --------------------------------------------------------------------------------------

JAX_VERSION = jax.__version__

#: ``pltpu.CompilerParams`` in jax >= 0.6, ``TPUCompilerParams`` before that.
CompilerParams = getattr(pltpu, "CompilerParams", None) or pltpu.TPUCompilerParams

#: ``pltpu.InterpretParams`` gates the TPU-semantics CPU simulator. Older jax only
#: has plain ``interpret=True``; `interpret_mode` papers over the difference.
InterpretParams = getattr(pltpu, "InterpretParams", None)

#: ``jax.experimental.layout`` arrived in 0.4.x and was renamed once. Used to pin operand
#: layouts to what Mosaic requires -- see `descending_format`. ``None`` on older jax, in
#: which case the layout helpers degrade to no-ops.
try:  # pragma: no cover - version dependent
    from jax.experimental.layout import Format, Layout
except ImportError:  # pragma: no cover
    try:
        from jax.experimental.layout import DeviceLocalLayout as Layout  # type: ignore
        from jax.experimental.layout import Layout as Format  # type: ignore
    except ImportError:
        Format = Layout = None  # type: ignore


def interpret_mode(detect_races: bool = False, nan_uninitialized: bool = True) -> Any:
    """Value to pass as ``pallas_call(interpret=...)`` for CPU simulation.

    Returns ``InterpretParams`` when available (simulates VMEM/HBM/DMA and can flag
    races and reads of uninitialized memory), else falls back to ``True``.
    """
    if InterpretParams is None:
        return True
    return InterpretParams(
        detect_races=detect_races,
        uninitialized_memory="nan" if nan_uninitialized else "zero",
    )


# --------------------------------------------------------------------------------------
# Device configuration
# --------------------------------------------------------------------------------------

#: Kaggle offers "TPU VM v3-8" and "TPU v5e-8"; v5e is the default target.
DEFAULT_DEVICE_KIND = "TPU v5e"

_DEVICE_KIND_ALIASES = {
    "v3": "TPU v3",
    "tpu v3": "TPU v3",
    "v4": "TPU v4",
    "tpu v4": "TPU v4",
    "v5e": "TPU v5e",
    "tpu v5e": "TPU v5e",
    "tpu v5 lite": "TPU v5e",
    "v5p": "TPU v5p",
    "tpu v5p": "TPU v5p",
    "v6e": "TPU v6e",
    "tpu v6e": "TPU v6e",
}


def canonical_device_kind(kind: str) -> str:
    """Map a loose name (``"v5e"``) onto a JAX ``device_kind`` (``"TPU v5e"``)."""
    return _DEVICE_KIND_ALIASES.get(kind.strip().lower(), kind)


@dataclasses.dataclass(frozen=True)
class BlockConfig:
    """Tunables for one (device, problem) pair.

    Attributes:
      chunk: Sequence rows per grid step. Must be a multiple of 128 -- the scalar
        streams put ``chunk`` on the minormost block axis, where Mosaic requires it.
      vmem_limit_bytes: Passed through to the Mosaic compiler. ``None`` leaves the
        default. v3's 16 MiB VMEM is small enough to matter.
      dimension_semantics: One entry per grid axis. The chunk axis carries the
        recurrence through VMEM scratch, so it must stay ``"arbitrary"``.
    """

    chunk: int = 128
    vmem_limit_bytes: int | None = None
    #: One entry per grid axis, in ``(batch, nchunks, nheads)`` order. ``nchunks``
    #: carries the recurrence through VMEM scratch so it must stay ``"arbitrary"``.
    #: ``nheads`` is *also* arbitrary rather than parallel: it is innermost, and
    #: Megacore may only split a prefix of parallel axes, so marking it parallel would
    #: let the two halves of a v4/v5p chip disagree about which head's scratch slot
    #: they own. Only ``batch`` is splittable, which is what we want anyway.
    dimension_semantics: tuple[str, ...] = ("parallel", "arbitrary", "arbitrary")

    def compiler_params(self) -> Any:
        kwargs: dict[str, Any] = {"dimension_semantics": self.dimension_semantics}
        if self.vmem_limit_bytes is not None:
            kwargs["vmem_limit_bytes"] = self.vmem_limit_bytes
        return CompilerParams(**kwargs)


#: ``device_kind`` -> ``pltpu.ChipVersion`` value string. Mirrors JAX's own private
#: ``chip_version_from_device_kind``, which is not part of the public pltpu surface.
_DEVICE_KIND_TO_CHIP = {
    "TPU v2": "v2",
    "TPU v3": "v3",
    "TPU v4": "v4",
    "TPU v4 lite": "v4i",
    "TPU v5e": "v5e",
    "TPU v5 lite": "v5e",
    "TPU v5": "v5p",
    "TPU v5p": "v5p",
    "TPU v6e": "v6e",
    "TPU v6 lite": "v6e",
    "TPU7": "7",
    "TPU7x": "7x",
    "TPU8i": "8i",
}


def tpu_info(kind: str = DEFAULT_DEVICE_KIND, num_cores: int = 1):
    """Hardware facts for a chip kind, with no TPU attached.

    Wraps ``pltpu.get_tpu_info_for_chip``. Note every field is *per TensorCore*.
    """
    chip_str = _DEVICE_KIND_TO_CHIP.get(canonical_device_kind(kind))
    if chip_str is None:
        raise ValueError(f"Unknown TPU device kind: {kind!r}")
    return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion(chip_str), num_cores)


def block_config(kind: str | None = None, chunk: int | None = None) -> BlockConfig:
    """Pick a `BlockConfig` for a device kind, or for the live device if ``None``.

    v3 has 16 MiB of VMEM against v5e's 128 MiB, so it gets a smaller chunk and an
    explicit VMEM ceiling. Everything else is shared.
    """
    if kind is None:
        kind = current_device_kind()
    kind = canonical_device_kind(kind)
    try:
        info = tpu_info(kind)
        vmem = info.vmem_capacity_bytes
    except ValueError:
        vmem = None  # CPU / unknown: leave the compiler at its defaults.

    # Mosaic's *scoped* VMEM budget defaults to 16 MiB regardless of the chip's total,
    # and the kernels exceed it once the chunk grows: at chunk=512 the backward needs
    # ~17.4 MiB of live tiles (chunk^2 for the two attention matrices dominates, at
    # 2 MiB each, plus ~1.8 MiB of chunk*N), and Mosaic refuses with
    # "CompileTimeScopedVmemOom ... exceeded scoped vmem limit". Raise it explicitly to
    # half the chip's physical VMEM -- generous for the tiles above, and still well
    # inside what the double-buffered block windows leave free.
    if vmem is None:
        cfg = BlockConfig(chunk=128, vmem_limit_bytes=None)
    elif vmem <= 32 * 1024 * 1024:
        # v3 has only 16 MiB total, so half of it is *below* the 16 MiB default and
        # would make things worse. Ask for the whole chip and let Mosaic complain if
        # the block windows do not fit.
        cfg = BlockConfig(chunk=128, vmem_limit_bytes=vmem)
    else:
        cfg = BlockConfig(chunk=128, vmem_limit_bytes=vmem // 2)

    if chunk is not None:
        cfg = dataclasses.replace(cfg, chunk=chunk)
    validate_chunk(cfg.chunk)
    return cfg


def current_device_kind() -> str:
    """``device_kind`` of the default device, e.g. ``"TPU v5e"`` or ``"cpu"``."""
    try:
        return jax.devices()[0].device_kind
    except Exception:  # pragma: no cover - no backend at all
        return "cpu"


def on_tpu() -> bool:
    try:
        return jax.devices()[0].platform == "tpu"
    except Exception:  # pragma: no cover
        return False


def no_tpu_message() -> str:
    """Why ``--tpu`` failed, when the accelerator is in fact selected.

    A TPU admits one client process at a time, so a ``!python -m ...`` cell launched
    from a notebook whose kernel already imported JAX finds the device taken.
    """
    return (
        f"No TPU in this process (backend {jax.default_backend()!r}). If the notebook "
        "kernel already imported JAX it holds the device, and a `!python -m ...` cell "
        "is a separate process.\nEither restart the kernel and rerun via the CLI, or "
        "call it in-process: tests.main(['--tpu'])"
    )


def validate_chunk(chunk: int) -> None:
    """Chunk must be a positive multiple of 128.

    The per-token scalar streams (``adt``, ``gamma``, ``scale``) are shaped
    ``(B, H, nc, 1, chunk)``, which puts ``chunk`` on the minormost axis where
    Mosaic demands a multiple of 128 (the lane count). 64 does not lower.
    """
    if chunk <= 0 or chunk % 128 != 0:
        raise ValueError(
            f"chunk must be a positive multiple of 128 (Mosaic lane constraint), got {chunk}"
        )


# --------------------------------------------------------------------------------------
# Dtype policy
# --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DtypePolicy:
    """Which dtype each role uses, and the matmul precision that goes with it.

    Matmul operands go to bf16 with f32 accumulation, matching the upstream Triton
    kernel (which also stores its rotated/scaled K in bf16). State, decay, and
    angles stay f32 because they accumulate over the whole sequence.
    """

    matmul: jnp.dtype = jnp.bfloat16
    accum: jnp.dtype = jnp.float32
    state: jnp.dtype = jnp.float32
    saved_state: jnp.dtype = jnp.bfloat16

    @property
    def highest(self) -> bool:
        return self.matmul == jnp.float32

    @property
    def precision(self):
        """``precision=`` for every ``dot_general``/``einsum`` under this policy.

        This is not cosmetic. On TPU the MXU has no f32 multiplier: an f32
        ``dot_general`` at ``Precision.DEFAULT`` silently truncates both operands to
        bf16, so an "f32" path is really bf16 with f32 accumulation and lands around
        1e-2 relative error. ``HIGHEST`` requests the multi-pass decomposition
        (bf16x6) that recovers true f32.

        Under the bf16 policy the operands are already bf16, so there is nothing to
        truncate and DEFAULT is both correct and fastest.
        """
        return jax.lax.Precision.HIGHEST if self.highest else jax.lax.Precision.DEFAULT


#: Production setting: bf16 MXU operands, f32 accumulate.
BF16 = DtypePolicy()

#: Validation setting: everything f32, so kernel-vs-reference diffs are algorithmic
#: rather than rounding.
F32 = DtypePolicy(matmul=jnp.float32, saved_state=jnp.float32)


def policy(name: str) -> DtypePolicy:
    key = name.strip().lower()
    if key in ("bf16", "bfloat16", "default"):
        return BF16
    if key in ("f32", "fp32", "float32", "highest"):
        return F32
    raise ValueError(f"Unknown dtype policy {name!r}; want 'bf16' or 'f32'")


# --------------------------------------------------------------------------------------
# The de-interleave permutation
# --------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def deinterleave_perm(n: int) -> jnp.ndarray:
    """Index array pi mapping interleaved rotary pairs to a half-split layout.

    The PyTorch reference rotates *adjacent* channels: it views the state axis as
    ``(n/2, 2)`` and mixes ``[..., 0]`` with ``[..., 1]``. On TPU that axis is the
    128-lane dimension, so a stride-2 split is a lane shuffle repeated every chunk.

    Applying ``pi`` once, offline, to every tensor that lives on the state axis
    turns the rotation into a contiguous half-split (NeoX style): pair ``j`` becomes
    channels ``(j, n/2 + j)``, so the rotation is two 64-lane slices and one
    concatenate.

        new[j]       = old[2j]
        new[n/2 + j] = old[2j + 1]

    Rotation and permutation commute in the sense that matters here:
    ``pi(ref_rope(u)) == neox_rope(pi(u))`` exactly, and ``q~ . k~ == u . w``
    because rotations are orthogonal.
    """
    if n % 2 != 0:
        raise ValueError(f"state dim must be even for rotary pairing, got {n}")
    return jnp.concatenate([jnp.arange(0, n, 2), jnp.arange(1, n, 2)])


@functools.lru_cache(maxsize=None)
def interleave_perm(n: int) -> jnp.ndarray:
    """Inverse of `deinterleave_perm` -- for converting weights back to torch order."""
    return jnp.argsort(deinterleave_perm(n))


def apply_perm(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """Apply pi along ``axis`` (which must have even length)."""
    return jnp.take(x, deinterleave_perm(x.shape[axis]), axis=axis)


def unapply_perm(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """Undo `apply_perm`."""
    return jnp.take(x, interleave_perm(x.shape[axis]), axis=axis)


# --------------------------------------------------------------------------------------
# Rotary helpers (half-split layout)
# --------------------------------------------------------------------------------------


def rope_cos_sin(phi: jnp.ndarray, state_dim: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build full-width cos/sin factors from per-pair angles.

    Args:
      phi: ``(..., n_angles)`` cumulative angles, f32.
      state_dim: Full state width ``N``; ``n_angles <= N // 2``.

    Returns:
      ``(cos_full, sin_full)``, each ``(..., N)``, laid out for the half-split rotation.
      Pairs beyond ``n_angles`` pass through unrotated (``cos = 1, sin = 0``), which is
      the ``rope_fraction < 1`` case and mirrors the reference's ``F.pad(cos, value=1)``.

    The angle is padded and doubled *before* the transcendentals rather than padding
    cos/sin after them. ``cos(0) == 1`` and ``sin(0) == 0`` are exactly the pad values
    the half-width form needs, so this way the padding is free: two concatenates instead
    of four. Concatenates cross the 128-lane register boundary, so halving them matters
    more than the extra cos/sin work.
    """
    half = state_dim // 2
    n_angles = phi.shape[-1]
    if n_angles > half:
        raise ValueError(f"n_angles ({n_angles}) exceeds state_dim // 2 ({half})")

    if n_angles < half:
        pad = jnp.zeros((*phi.shape[:-1], half - n_angles), phi.dtype)
        phi = jnp.concatenate([phi, pad], axis=-1)
    phi_full = jnp.concatenate([phi, phi], axis=-1)
    return jnp.cos(phi_full), jnp.sin(phi_full)


def rope_apply(x: jnp.ndarray, cos_full: jnp.ndarray, sin_full: jnp.ndarray) -> jnp.ndarray:
    """Rotate ``x`` (half-split layout) by the angles baked into cos/sin.

    ``rot_half(x) = concat(-x_hi, x_lo)``, which crosses the lane axis. `pltpu.roll` plus
    a sign mask is the same permutation and the hardware serves it as a lane-rotate, so
    that is the path used inside a kernel.

    Outside a kernel it cannot be: ``pltpu.roll`` is a Pallas primitive with no eval rule
    there, and this function is also called from `reference` and the backward epilogue as
    ordinary JAX. Hence the ``_in_kernel()`` check. Both spellings are bit-identical.
    Getting the check wrong surfaces immediately as
    ``NotImplementedError: Evaluation rule for 'roll'``.
    """
    half = x.shape[-1] // 2
    if _in_kernel():
        rolled = pltpu.roll(x, half, x.ndim - 1)
        sign = jnp.where(
            jax.lax.broadcasted_iota(jnp.int32, x.shape, x.ndim - 1) < half, -1.0, 1.0
        )
        rot = rolled * sign.astype(x.dtype)
    else:
        lo, hi = x[..., :half], x[..., half:]
        rot = jnp.concatenate([-hi, lo], axis=-1)
    return x * cos_full + rot * sin_full


def _in_kernel() -> bool:
    """True when tracing inside a ``pallas_call`` body.

    ``pltpu`` primitives only have lowering rules there. The check is a capability
    probe rather than a type test: ask the current trace whether the primitive can be
    bound at all, which stays correct if JAX's internal trace classes move.
    """
    try:
        return not jax.core.trace_state_clean()
    except Exception:  # pragma: no cover - private API moved
        return False


def rope_dphi(x_rot: jnp.ndarray, dx_rot: jnp.ndarray, n_angles: int) -> jnp.ndarray:
    """Gradient of a rotation w.r.t. its angle, given the rotated value and its cotangent.

    ``d/dphi R(phi) u = R(phi + pi/2) u``, and in half-split layout applying
    ``R(pi/2)`` to an already-rotated vector is just ``rot_half``. So

        dphi_j = dx_lo[j] * (-x_hi[j]) + dx_hi[j] * x_lo[j]

    Only the first ``n_angles`` pairs carry an angle; the rest are unrotated.
    """
    half = x_rot.shape[-1] // 2
    lo, hi = x_rot[..., :half], x_rot[..., half:]
    dlo, dhi = dx_rot[..., :half], dx_rot[..., half:]
    dphi = dhi * lo - dlo * hi
    return dphi[..., :n_angles]


# --------------------------------------------------------------------------------------
# Prefix sums without ``cumsum``
# --------------------------------------------------------------------------------------


def tril_ones(n: int, dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
    """``(n, n)`` lower-triangular ones, inclusive of the diagonal.

    Built from iota rather than ``jnp.tril`` so it lowers inside a Pallas kernel.
    """
    i = jax.lax.broadcasted_iota(jnp.int32, (n, n), 0)
    j = jax.lax.broadcasted_iota(jnp.int32, (n, n), 1)
    return (i >= j).astype(dtype)


def row_iota(n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """The ``(i, j)`` iota pair used for both `tril_ones` and the causal mask."""
    i = jax.lax.broadcasted_iota(jnp.int32, (n, n), 0)
    j = jax.lax.broadcasted_iota(jnp.int32, (n, n), 1)
    return i, j


def prefix_sum(row: jnp.ndarray, tril: jnp.ndarray) -> jnp.ndarray:
    """Inclusive prefix sum of a ``(1, n)`` row, as an MXU matmul.

    Pallas TPU has no ``cumsum`` primitive (nor ``associative_scan``), so the
    within-chunk decay accumulation goes through the systolic array instead:
    ``A = row @ tril.T`` gives ``A[t] = sum_{s <= t} row[s]``.

    Always ``HIGHEST``, regardless of dtype policy. ``tril`` is exactly 0/1 so it
    survives any truncation, but ``row`` holds the log-domain decay ``adt * log2(e)``
    -- truncating that to bf16 (which is what DEFAULT does on TPU, since the MXU has
    no f32 multiplier) would put ~3 decimal digits of error into every ``exp2``
    downstream. The cost is a few extra passes over one ``(1, n)`` row.

    Args:
      row: ``(1, n)`` f32.
      tril: ``(n, n)`` from `tril_ones`, hoisted so it is shared with the mask.

    Returns:
      ``(n,)`` f32.
    """
    return jax.lax.dot_general(
        row, tril, (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )[0]


def suffix_sum(row: jnp.ndarray, tril: jnp.ndarray) -> jnp.ndarray:
    """Inclusive *reverse* prefix sum: ``out[t] = sum_{s >= t} row[s]``.

    Same trick as `prefix_sum` but contracting against ``tril``'s first axis
    instead of its second, which is ``triu`` without materializing one. Used by the
    backward pass to turn per-position decay adjoints into per-position ``adt``
    adjoints. ``HIGHEST`` for the same reason.
    """
    return jax.lax.dot_general(
        row, tril, (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )[0]


# --------------------------------------------------------------------------------------
# Scalar-stream reshaping
# --------------------------------------------------------------------------------------


def num_chunks(seqlen: int, chunk: int) -> int:
    if seqlen % chunk != 0:
        raise ValueError(
            f"seqlen ({seqlen}) must be a multiple of chunk ({chunk}); "
            "pad the sequence before calling."
        )
    return seqlen // chunk


def scalars_to_blocks(x: jnp.ndarray, chunk: int) -> jnp.ndarray:
    """``(B, H, L) -> (B, H, L // chunk, 1, chunk)``, a free reshape.

    Mosaic will not accept ``(B, H, L)`` with block ``(None, None, chunk)``: for a
    rank-3 array the last two block dims must be divisible by (8, 128) or match the
    array dims, and ``chunk`` alone satisfies neither. Adding a length-1 sublane
    axis fixes it. Inside the kernel the ref is read as ``ref[0] -> (chunk,)``.
    """
    b, h, l = x.shape
    return x.reshape(b, h, num_chunks(l, chunk), 1, chunk)


def blocks_to_scalars(x: jnp.ndarray) -> jnp.ndarray:
    """Inverse of `scalars_to_blocks`."""
    b, h, nc, one, chunk = x.shape
    assert one == 1, f"expected a singleton sublane axis, got {one}"
    return x.reshape(b, h, nc * chunk)


# --------------------------------------------------------------------------------------
# BlockSpec factories
#
# Named so the kernels read declaratively and so every layout decision lives in one
# file. Each has been checked against Mosaic via `assert_lowers`.
#
# GRID ORDER IS ``(batch, nchunks, nheads)`` -- head innermost. That ordering is the
# whole reason `q`/`k` are cheap. They are ``(B, L, N)``, shared across heads
# (``ngroups=1``), and their index maps ignore ``h``; Pallas elides an HBM copy when
# *lexicographically adjacent* grid steps map to the same window, so with ``h``
# innermost a q/k block is fetched once and reused across all ``H`` heads. With ``h``
# outer (the original ordering) every head re-walked the same ``nchunks`` blocks, and
# q/k moved ``H`` times -- 1024 MiB instead of 32 MiB at ``B=8, H=32, L=8192``, nearly
# doubling total traffic.
#
# The price is that the SSM state scratch must hold every head at once, ``(H, P, N)``
# f32 rather than ``(P, N)``: 1 MiB at ``H=32`` against v5e's 128 MiB of VMEM, so it is
# affordable. Correctness is unaffected -- for a fixed ``(b, h)`` the chunk index still
# increases monotonically, which is the only ordering the recurrence needs.
# --------------------------------------------------------------------------------------


def spec_shared_bl_n(chunk: int, n: int) -> pl.BlockSpec:
    """``(B, L, N)`` shared across heads -- the B and C projections.

    The index map drops ``h``, and ``h`` is the innermost grid axis, so consecutive
    grid steps reuse this window and Pallas elides the copy. This is the payoff of the
    grid ordering described above.
    """
    return pl.BlockSpec((None, chunk, n), lambda b, c, h: (b, c, 0))


def spec_bhl_p(chunk: int, p: int) -> pl.BlockSpec:
    """``(B, H, L, P)`` per-head -- v, z, y, ybar."""
    return pl.BlockSpec((None, None, chunk, p), lambda b, c, h: (b, h, c, 0))


def spec_bhl_nr(chunk: int, nr: int) -> pl.BlockSpec:
    """``(B, H, L, Nr)`` per-head cumulative rotary angles."""
    return pl.BlockSpec((None, None, chunk, nr), lambda b, c, h: (b, h, c, 0))


def spec_scalar(chunk: int) -> pl.BlockSpec:
    """``(B, H, nc, 1, chunk)`` per-token scalar stream. See `scalars_to_blocks`."""
    return pl.BlockSpec((None, None, None, 1, chunk), lambda b, c, h: (b, h, c, 0, 0))


def spec_chunk_state(p: int, n: int) -> pl.BlockSpec:
    """``(B, H, nc, P, N)`` per-chunk saved state for the backward pass."""
    return pl.BlockSpec((None, None, None, p, n), lambda b, c, h: (b, h, c, 0, 0))


def spec_head_row(width: int) -> pl.BlockSpec:
    """``(H, 1, width)`` per-head parameter row (q_bias, k_bias, D).

    The singleton middle axis is load-bearing, exactly as in `spec_scalar`: a rank-2
    ``(H, width)`` array with block ``(None, width)`` leaves a rank-1 block, which
    Mosaic rejects. Use `head_rows` to produce the array. Kernel sees ``(1, width)``.
    """
    return pl.BlockSpec((None, 1, width), lambda b, c, h: (h, 0, 0))


def head_rows(x: jnp.ndarray) -> jnp.ndarray:
    """``(H, ...) -> (H, 1, ...)``, the shape `spec_head_row` expects."""
    return x.reshape(x.shape[0], 1, *x.shape[1:]) if x.ndim > 1 else x.reshape(-1, 1, 1)


def spec_final_state(nheads: int, p: int, n: int) -> pl.BlockSpec:
    """``(B, H, P, N)`` final SSM state -- one block spanning *all* heads.

    Why the whole head axis rather than ``(b, h)``. Pallas requires that every
    invocation writing a given output slice be *consecutive*. With ``h`` innermost, a
    per-``(b, h)`` window is revisited at every chunk with other heads interleaved
    between the visits, which Pallas rejects outright ("Revisited block ... in
    iteration (0, 1, 0)"). Making the block span ``H`` keeps the window fixed for the
    whole ``b`` iteration, so all writes to it are trivially consecutive. The kernel
    indexes the ref by ``h``.
    """
    return pl.BlockSpec((None, nheads, p, n), lambda b, c, h: (b, 0, 0, 0))


def spec_final_row(nheads: int, width: int) -> pl.BlockSpec:
    """``(B, H, 1, width)`` final k/v row -- all heads in one block, as above."""
    return pl.BlockSpec((None, nheads, 1, width), lambda b, c, h: (b, 0, 0, 0))


def spec_seed_state(p: int, n: int) -> pl.BlockSpec:
    """``(B, H, P, N)`` carry-in state, read per ``(b, h)``.

    Inputs have no consecutive-write rule, so this one can stay per-head.
    """
    return pl.BlockSpec((None, None, p, n), lambda b, c, h: (b, h, 0, 0))


def spec_seed_row(width: int) -> pl.BlockSpec:
    """``(B, H, 1, width)`` carry-in k/v row, read per ``(b, h)``."""
    return pl.BlockSpec((None, None, 1, width), lambda b, c, h: (b, h, 0, 0))


def spec_headsum_bl_n(chunk: int, n: int) -> pl.BlockSpec:
    """``(B, L, N)`` accumulated across heads -- ``dq``/``dk``.

    The index map ignores ``h``, so every head in a ``(b, c)`` step writes the same
    window. That is exactly what makes an in-kernel head reduction legal: those writes
    are consecutive under the ``(B, nc, H)`` ordering, which is Pallas' requirement.
    The kernel initializes at ``h == 0`` and accumulates after.
    """
    return pl.BlockSpec((None, chunk, n), lambda b, c, h: (b, c, 0))


def spec_per_step_head_row(width: int) -> pl.BlockSpec:
    """``(B, nc, H, 1, width)`` -- one private slot per grid step.

    For per-head quantities that need summing over ``batch`` and ``chunk``
    (``dq_bias``, ``dk_bias``). Accumulating in-kernel is not an option: with ``h``
    innermost, the writes for a fixed ``h`` are separated by every other head, and
    Pallas requires writes to one slice to be consecutive. Giving each step its own
    slot sidesteps that, and the buffer is tiny -- ``B * nc * H * N`` f32 is 8 MiB at
    ``B=8, nc=64, H=32, N=128`` -- so the trailing ``sum`` over two axes is cheap.
    """
    return pl.BlockSpec(
        (None, None, None, 1, width), lambda b, c, h: (b, c, h, 0, 0)
    )


def spec_scratch_bl_p(chunk: int, p: int) -> pl.BlockSpec:
    """``(B, nc, chunk, P)`` write-only scratch shared by every head.

    Used for residual outputs the caller did not ask for. Every head writes the same
    window, which is legal because they are consecutive under this grid ordering, and
    the buffer is ``1/H`` the size of a real residual. The kernel body is unchanged --
    the ref has the same ``(chunk, P)`` shape either way.
    """
    return pl.BlockSpec((None, None, chunk, p), lambda b, c, h: (b, c, 0, 0))


def spec_scratch_state(p: int, n: int) -> pl.BlockSpec:
    """``(B, nc, P, N)`` write-only scratch shared by every head. See above."""
    return pl.BlockSpec((None, None, p, n), lambda b, c, h: (b, c, 0, 0))


def reversed_index_map(fn: Callable[..., tuple], nc: int) -> Callable[..., tuple]:
    """Wrap an index map so the chunk axis is walked backwards.

    The Pallas grid always counts up; the backward pass wants chunk ``nc - 1 - c``.
    Rewriting the index maps rather than the grid keeps the sequential-axis guarantees
    intact. ``c`` is the *middle* grid axis under the ordering above.
    """

    @functools.wraps(fn)
    def wrapped(b, c, h):
        return fn(b, nc - 1 - c, h)

    return wrapped


def reverse_spec(spec: pl.BlockSpec, nc: int) -> pl.BlockSpec:
    """`reversed_index_map` applied to an existing `pl.BlockSpec`.

    ``spec.index_map`` is wrapped by Pallas, so unwrap it before composing.
    """
    inner = getattr(spec.index_map, "index_map", spec.index_map)
    return pl.BlockSpec(spec.block_shape, reversed_index_map(inner, nc))


# --------------------------------------------------------------------------------------
# Operand layouts
# --------------------------------------------------------------------------------------
#
# Mosaic custom calls do not negotiate layouts: every operand must arrive in the default
# descending (row-major) layout. XLA, left to itself, picks a *different* preferred layout
# for the big activation tensors, and the mismatch is paid as a materializing copy outside
# the kernel. Measured on v5e at B=8 H=32 L=8192 chunk=128, from the real trace:
#
#     %siso_forward.1  (the kernel)                        20.547 ms   77.4%
#     %copy.2.phi_ref  f32[8,32,8192,32]                    2.145 ms    8.1%
#     %copy.1.z_ref    bf16[8,32,8192,64]                   1.266 ms    4.8%
#     %copy.v_ref      bf16[8,32,8192,64]                   1.265 ms    4.8%
#     %copy.4          bf16[8,32,8192,64]{2,3,1,0}          1.200 ms    4.5%
#     ------------------------------------------------------------------
#     not the kernel                                        6.00 ms   22.6%
#
# Note `copy.4`'s layout: ``{2,3,1,0}``, dims 2 and 3 swapped, where every other tensor is
# ``{3,2,1,0}``. That is XLA's preference, and the reason is tiling. With ``{3,2,1,0}`` the
# minor two dims of ``(B, H, L, P)`` are ``(8192, 64)``, and TPU tiling ``T(8,128)`` pads a
# 64-wide minor dim to 128 -- so the buffer occupies 2x its logical size. Swapped, they are
# ``(64, 8192)``, which tiles exactly. ``phi`` is worse: ``Nr=32`` minor pads 4x.
#
# So ``P=64`` against a 128-lane register is the root cause, and the copies are XLA
# converting between the layout it wants for the buffers and the one Mosaic demands. Three
# copies in (v, z, phi) and one out (y).
#
# The cost is flat in chunk size (6.00 ms at chunk 128, 6.01 at 512) because it is
# per-tensor rather than per-grid-step, which is also why raising the chunk stopped buying
# anything past 512.
#
# Pinning the layouts at the jit boundary removes them, measured bit-exact on v5e:
#
#     chunk 128   27.49 -> 21.66 ms   (5.84 ms, 21.2%)
#     chunk 512   23.52 -> 17.56 ms   (5.96 ms, 25.4%)
#
# which is the 6.00/6.01 ms the trace predicted, to within noise. Forward throughput goes
# from 2.79 M to 3.77 M tokens/s at chunk 512 bf16, and f32 gains more than bf16 (up to
# 43% at chunk 128) because its tensors are twice the bytes to convert.
#
# The catch is *where* the pin has to go. See `descending_format`: an in-trace
# ``device_put`` is silently dropped, so this can only be fixed by whoever owns the jit
# that produces v/z/phi. The kernel cannot fix it for its callers.


def descending_format(x: Any) -> Any:
    """`Format` requesting the default descending layout, preserving sharding.

    Passing this as ``jax.jit(in_shardings=...)`` / ``out_shardings=...`` pins the
    operand to the layout Mosaic requires, so XLA produces it that way instead of
    converting at the call boundary. Measured on v5e at ``B=8 H=32 L=8192``: **21.2%
    faster at chunk 128 (27.49 -> 21.66 ms) and 25.4% at chunk 512 (23.52 -> 17.56)**,
    bit-exact, which matches the 6.00/6.01 ms of copies the trace attributed to layout
    conversion almost exactly.

    Returns ``None`` when the running jax has no layout API, so callers can fall back
    silently.

    **This only works at a jit boundary.** ``jax.device_put(x, fmt)`` called *inside* a
    traced function is silently dropped -- verified by asking for a non-default layout and
    getting the default back -- and there is no ``lax.with_layout_constraint``. So a
    library cannot pin the layout of its own intermediates: ``siso_forward`` has no way to
    fix this for its callers from the inside. Whoever owns the ``jit`` that produces
    ``v``/``z``/``phi`` has to pin them. See `pin_descending` for the argument case and the
    README for the model case.
    """
    if Format is None or Layout is None:
        return None
    ndim = len(x.shape)
    return Format(Layout(major_to_minor=tuple(range(ndim))), getattr(x, "sharding", None))


def pin_descending(tree: Any) -> Any:
    """`jax.device_put` every array in ``tree`` into the default descending layout.

    Use on arrays that are *arguments* to a jit -- a benchmark's inputs, a loaded
    checkpoint. This is an eager ``device_put``, so it does real work once and then the
    arrays stay in that layout.

    It does **not** work on intermediates inside a jit (see `descending_format`). For a
    model, pin the layout on the ``jit`` that produces the tensors instead:

        fn = jax.jit(step, in_shardings=fmts, out_shardings=out_fmt)

    Pinning only the kernel's own boundary would move the conversion rather than remove
    it, which is the one thing the benchmark's 21-25% figure could hide.
    """
    if Format is None:
        return tree

    def put(x):
        if not hasattr(x, "shape") or not hasattr(x, "dtype"):
            return x
        fmt = descending_format(x)
        return x if fmt is None else jax.device_put(x, fmt)

    return jax.tree.map(put, tree)


def layout_of(x: Any) -> str:
    """``major_to_minor`` of an array as a short string, or ``"?"``. For diagnostics."""
    try:
        return str(tuple(x.format.layout.major_to_minor))
    except Exception:  # pragma: no cover - no layout API, or a tracer
        return "?"


# --------------------------------------------------------------------------------------
# Ahead-of-time Mosaic validation
# --------------------------------------------------------------------------------------


def abstract_tpu_mesh(kind: str = DEFAULT_DEVICE_KIND, num_cores: int = 1):
    """Context manager that makes JAX believe the default device is a given TPU.

    This is what lets the real Mosaic lowering run on a CPU-only machine: the
    lowering path asks for ``device_kind`` and ``num_cores``, and an abstract mesh
    supplies both without any hardware.
    """
    import jax._src.mesh as mesh_lib  # private, but the only way to set this
    from jax.sharding import AbstractMesh

    device = mesh_lib.AbstractDevice(
        device_kind=canonical_device_kind(kind), num_cores=num_cores, platform="tpu"
    )
    mesh = AbstractMesh((1,), ("_aot",), abstract_device=device)
    return mesh_lib.use_abstract_mesh(mesh)


def lower_for_tpu(
    fn: Callable[..., Any],
    *args: Any,
    kind: str = DEFAULT_DEVICE_KIND,
    num_cores: int = 1,
) -> str:
    """Lower ``fn`` through Mosaic for a TPU kind and return the MLIR.

    Args are ``jax.ShapeDtypeStruct``s. Raises whatever Mosaic raises -- typically
    ``ValueError`` for an illegal block shape or ``NotImplementedError`` for a
    primitive with no TPU lowering (``cumsum`` being the one that bit us).
    """
    with abstract_tpu_mesh(kind, num_cores):
        return pl.lower_as_mlir(fn, *args, platforms=["tpu"])


def assert_lowers(
    fn: Callable[..., Any],
    *args: Any,
    kinds: Sequence[str] = ("TPU v5e", "TPU v3"),
    label: str = "",
) -> dict[str, int]:
    """Assert ``fn`` lowers for every listed chip; return MLIR size per kind.

    Cheap enough (~1 s) to run as the first test stage, and it catches the two
    failure classes that interpret mode cannot see: illegal block shapes and
    unlowerable primitives.
    """
    sizes: dict[str, int] = {}
    for kind in kinds:
        try:
            sizes[canonical_device_kind(kind)] = len(lower_for_tpu(fn, *args, kind=kind))
        except Exception as exc:  # surface which target failed
            name = f"{label} " if label else ""
            raise AssertionError(
                f"{name}failed to lower for {canonical_device_kind(kind)}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    return sizes


def describe_environment() -> str:
    """One-line summary for the top of a test or benchmark run."""
    kind = current_device_kind()
    parts = [f"jax {JAX_VERSION}", f"device {kind}", f"n_devices {len(jax.devices())}"]
    try:
        info = tpu_info(kind)
        parts.append(f"vmem {info.vmem_capacity_bytes / 2**20:.0f}MiB")
        parts.append(f"hbm_bw {info.mem_bw_bytes_per_second / 1e9:.0f}GB/s")
    except ValueError:
        parts.append("(not a TPU: interpret mode only)")
    return "  ".join(parts)
