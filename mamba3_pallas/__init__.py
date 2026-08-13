"""Mamba-3 SISO for TPU, as JAX/Pallas kernels.

A standalone reimplementation of the SISO layer from
`Mamba-3: Improved Sequence Modeling using State Space Principles
<https://arxiv.org/abs/2603.15569>`_. The upstream ``state-spaces/mamba`` repo ships
Triton (prefill/training) and CuTe DSL (decode) kernels, both CUDA-only; this package
targets TPU via Pallas and stays weight-compatible so PyTorch checkpoints port over.

Typical use::

    from mamba3_pallas import layer as LY

    cfg = LY.SISOConfig(d_model=1024, d_state=128, headdim=64, chunk=128)
    params = LY.init_params(cfg, jax.random.key(0))
    y, state = LY.mamba3_siso_layer(params, u, cfg)   # u: (B, L, d_model)

Modules:
  layout         Device config, the rotary de-interleave permutation, BlockSpecs, and
                 an ahead-of-time Mosaic lowering check that runs without a TPU.
  reference      Pure-JAX recurrent and chunked references; the gradient oracle.
  kernel_fwd     Chunked forward pallas_call.
  kernel_bwd     Reverse-chunk-scan backward, plus its elementwise JAX epilogue.
  kernel_decode  Single-token step with donated state buffers. Two grid shapes:
                 ``(batch, nheads)`` and the head-folded ``(batch,)``; ``decode_step``
                 picks between them and defaults to folded.
  siso           jax.custom_vjp glue; ``siso_segment`` is the entry point.
  layer          SISOConfig/SISOParams, the functional layer, optional Flax NNX module.
  model          A stacked pre-norm LM around ``n_layers`` SISO blocks.
  convert        torch state_dict -> SISOParams (applies the permutation).
  checkpoint     Save/load trained weights as a self-describing npz.
  torch_ref      Vendored PyTorch reference, CPU-only.
  tests          Staged correctness suite.
  train          Byte-level LM training.

``tests`` and ``train`` are omitted from the default import: they pull in optional
dependencies (torch, optax) and are meant to be run as ``python -m``.
"""

from . import (
    checkpoint,
    convert,
    kernel_bwd,
    kernel_decode,
    kernel_fwd,
    layer,
    layout,
    model,
    reference,
    siso,
)

__all__ = [
    "layout",
    "reference",
    "kernel_fwd",
    "kernel_bwd",
    "kernel_decode",
    "siso",
    "layer",
    "model",
    "convert",
    "checkpoint",
]
