# mamba3-siso-tpu

Mamba-3 SISO as JAX/Pallas TPU kernels. Forward, backward, single token decode.

SISO only, the `ngroups=1` variant where every head gets its own B/C. MIMO is not
implemented, see [constraints](#constraints).

Upstream [state-spaces/mamba](https://github.com/state-spaces/mamba) ships Triton for
prefill/training and CuTe DSL for decode, both CUDA only. This is the same layer for TPU.
Weight compatible with the PyTorch `Mamba3` module so you can load a real checkpoint.

Built on v5e-8, falls back to v3-8 automatically.

## Results

8 layers, 30.8M params, enwik8 with the standard protocol split (train on first 90M bytes,
test on last 5M):

| | bits per char | params |
|---|---|---|
| [gzip -9](https://www.mattmahoney.net/dc/text.html) | 2.92 | |
| [LN HM-LSTM](https://arxiv.org/abs/1609.01704) | 1.32 | 35M |
| **this** | **1.289** | **30.8M** |
| [large mLSTM](https://arxiv.org/abs/1609.07959) | 1.24 | 46M |
| [12L Transformer-XL](https://arxiv.org/abs/1901.02860) | 1.06 | 41M |
| [24L Transformer-XL](https://arxiv.org/abs/1901.02860) | 0.99 | 277M |
| [+ dynamic eval](https://arxiv.org/abs/1904.08378) | 0.94 | 277M |

So it lands between the LSTM baselines, and the 46M mLSTM from 2016 is still ahead of it.
This is one layer type trained once with no recipe tuning, not a leaderboard attempt. What
the number is for is showing the kernels train a real model to a sane loss on real data.
Baseline rows are the published values from those papers, not reruns.

Sample, one byte at a time through the decode kernel:

```
</text>
    </revision>
  </page>
  <page>
    <title>Bryant County, Illinois</title>
    <id>96315</id>
    <revision>
      <id>28180870</id>
      <timestamp>2005-11-14T16:13:47Z</timestamp>
      <contributor>
        <username>Leslie Mateus</username>
```

Closes 3 nested tags in the right order, opens a new page, writes the full revision schema
with a valid timestamp. The state is holding structure over hundreds of bytes, which is the
point of the layer.

## Speed

One v5e chip, B=8 H=32 L=8192 N=128 P=64, bf16:

| | |
|---|---|
| forward | 17.4 ms, **3.77M tok/s** |
| forward + backward | 80.1 ms |
| vs `lax.associative_scan` | **60x** faster at L=8192 |
| training on 8 chips | **~495K tok/s** at 30.8M params |
| decode, B=256 | 1.4 ms/token |

Every number here was measured by hand on a v5e-8. Nothing measures throughput
automatically, so if you change the kernels, re-measure. The forward is compute bound at
these shapes and the decode is bandwidth bound, which is why the two are reported
differently.

## Install

```bash
git clone https://github.com/burak-colakkadioglu/mamba3-siso-tpu
cd mamba3-siso-tpu
pip install -e .
```

On Kaggle or Colab TPU jax is already there, don't reinstall it. Extras:

```bash
pip install -e ".[torch,train]"   # torch parity tests + training
```

## Use

```python
import jax
from mamba3_pallas import layer as LY

cfg = LY.SISOConfig(d_model=1024, d_state=128, headdim=64, chunk=128)
params = LY.init_params(cfg, jax.random.key(0))
y, state = LY.mamba3_siso_layer(params, u, cfg)      # u: (B, L, d_model)
```

**Pin your operand layouts, it's worth 21-26% of forward time.** The library can't do this
for you because an in-trace `device_put` gets dropped. You have to pin the jit that calls
the layer:

```python
from mamba3_pallas import layout as L

fmts = jax.tree.map(L.descending_format, args)
step = jax.jit(fn, in_shardings=fmts, out_shardings=out_fmt)
```

Reason: Mosaic wants the default descending layout on every operand, XLA prefers a
different one when `P=64`, and the mismatch gets paid as a transposing copy on `v`, `z`,
`phi` and `y`. Details in
[internals](docs/internals.md#operand-layouts-worth-21-26-of-forward).

## Released checkpoints

The `state-spaces/mamba3-siso-*` models load and run:

```python
from mamba3_pallas import convert as CV, model as M

params, cfg = CV.load_pretrained("state-spaces/mamba3-siso-1.5b")
logits = M.lm_forward(params, tokens, cfg)
```

That handles the `backbone.layers.N.mixer.*` key mapping, the transposes, the rotary
permutation, the gated MLP those checkpoints carry (`d_intermediate = 2 * d_model`) and the
vocab padding. Needs `torch` to unpickle the `.bin` and `huggingface_hub` to fetch it.
Tokenizer is not in those repos, they point at `meta-llama/Llama-3.1-8B`, which is gated.

MIMO checkpoints and hybrid attention stacks are refused with a clear error rather than
half loaded.

Generating from one, through the decode kernel:

```python
from transformers import AutoTokenizer
from mamba3_pallas import convert as CV, train as T
import jax

tk = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
params, cfg = CV.load_pretrained("state-spaces/mamba3-siso-1.5b")

ids = T.generate(
    params, cfg, jax.random.key(0),
    prompt_ids=tk.encode("The capital of France is"),
    n_tokens=100, temperature=0.8, rep_penalty=1.0,
    stop_ids={tk.eos_token_id},
)
print(tk.decode(ids))
```

`generate` works on token ids so any tokenizer drives it. `train.sample` is the byte level
wrapper over it, which is what the byte models here use.

Single layer, from a raw `state_dict`:

```python
cfg = CV.config_from_state_dict(sd, headdim=64)
params = CV.torch_to_jax(sd, cfg)
```

## Tests

```bash
python -m mamba3_pallas.tests          # CPU, interpret mode
python -m mamba3_pallas.tests --tpu    # real shapes, needs a TPU
```

All pass on a v5e-8. Almost all of it also passes with no TPU at all, 118 checks: `lower` 40,
`shapes` 8, `refs` 4, `rotation` 4, `forward` 13, `backward` 13, `segments` 6, `torch` 3,
`checkpoint` 14, `pretrained` 13. That is every stage except `decode` and `train`, which are
too slow in interpret mode to sit in a quick run. Run them yourself:

```bash
python -m mamba3_pallas.tests --stage lower --stage shapes --stage refs --stage rotation \
    --stage forward --stage backward --stage segments --stage torch --stage checkpoint \
    --stage pretrained
```

Two things make that possible on a machine with no TPU:

- **40 Mosaic lowering configs.** `--stage lower` runs the real Mosaic lowering under an
  abstract TPU mesh. chunk 128/256/512/1024, bf16/f32, v5e and v3. Block shape bugs and
  missing primitive lowerings get caught without a TPU.
- **All the numerical stages** in interpret mode. Slow but correct, so parity vs the
  references and vs PyTorch is checked too.

No hosted CI here, so no badge, run the command instead. Nothing checks throughput either
way, only correctness.

## Train

```bash
python -m mamba3_pallas.train --steps 18000 --d-model 768 --n-layers 8 --batch 8 \
    --corpus enwik8 --protocol-split --save enwik8-31m.npz
```

Downloads and caches the corpus, shards the batch over every chip, keeps the best
validation checkpoint, and tells you if your config is actually comparable to published
numbers. `--corpus` also takes `enwik9`, `text8`, `tiny_shakespeare`, `synthetic`, or a
path (Kaggle `/kaggle/input/...` mounts work).

`--save` writes the best-validation weights, without it they are gone when the run ends.
`--save-dtype bfloat16` halves the file. `--resume path.npz` picks a run back up, and the
model shape comes from the file so you don't have to repeat the `--d-model`/`--n-layers`
flags.

Two flags for larger models on a v5e-8:

- `--shard-optimizer` shards Adam's two moments across chips (ZeRO-1) instead of
  replicating them. Saves `2 * params * 4 / n_dev` bytes per chip, 5.1 GiB at 787M on 8
  chips, which is what lets a big model fit at a useful batch. Costs one all-gather of the
  updates per step. The loss curve is identical, verified step for step.
- `--pin-layouts` pins operand layouts at the step's jit boundary, worth ~20% of forward
  time. Multi chip only, since there is no sharding to rebuild on a single device.

```bash
python -m mamba3_pallas.train --steps 2000 --d-model 2816 --n-layers 16 --batch 4 \
    --seqlen 512 --chunk 512 --corpus enwik8 --shard-optimizer --pin-layouts
```

Training on your own text, pass `--prompt` too. The generation prefix defaults to something
the named corpora contain (`[[The ` for enwik8, which is wiki link syntax) and a prompt your
corpus never contains is out of distribution, so the first bytes come out as noise before
the model recovers. `--rep-penalty 1.0` is also worth trying on prose; the 1.15 default suits
enwik8 entity runs and over-corrects on naturally repetitive text.

Weights are a plain npz, no pickle, with the config stored alongside them:

```python
from mamba3_pallas import checkpoint as CP, model as M

params, cfg = CP.load("enwik8-31m.npz")
logits = M.lm_forward(params, tokens, cfg)      # tokens: (B, L) int32, raw bytes
```

## Files

```
mamba3_pallas/
  layout.py        device config, rotary permutation, BlockSpecs, AOT Mosaic check
  reference.py     pure JAX references, gradient oracle
  kernel_fwd.py    chunked forward pallas_call
  kernel_bwd.py    reverse chunk scan backward
  kernel_decode.py single token step, per-head and head-folded grids
  siso.py          custom_vjp glue
  layer.py         config, params, the layer, optional Flax NNX module
  model.py         stacked LM
  convert.py       torch state_dict -> params, released checkpoint loader
  checkpoint.py    save/load trained weights as npz
  torch_ref.py     vendored PyTorch reference, CPU only
  tests.py         12 test stages
  train.py         byte level LM training
docs/internals.md  design decisions and constraints
```

## Constraints

- SISO only. MIMO (`nheads > ngroups`, heads sharing one B/C) is not implemented. The
  kernels assume `ngroups=1` in their index maps, so it's not a flag you can flip.
- `chunk` has to be a multiple of 128 (Mosaic lane constraint).
- Sequence length has to be a multiple of `chunk`. Pad it.
- Multi chip is data parallel via `shard_map`. Mosaic kernels can't be auto partitioned so
  `jit` with input shardings just fails, this isn't a style choice.
- Variable length / packed sequences are not implemented either. Segmented prefill works
  (chain the carry), packing many short sequences into one row does not.

## A note on AI use

I want to be upfront about this, since it's a fair thing to wonder about. I used AI to help
write the documentation and the inline comments in this repo. The code itself, the kernels,
the layout machinery, the tests and the training loop, is written by me.

The honest reason is that I'm not very good at explaining things to people who haven't been
looking at the same code I have. I understand what I write, but when I try to describe it, it
comes out either too terse or too tangled to be useful to anyone else. It's something I'm
working on, and I'd like to be writing all of this myself eventually.

I read over everything before it goes in, but I'm sure some of it still slips past me. If a
comment or a section here doesn't match the behaviour you see, I'd genuinely appreciate you
telling me, and I'm sorry for the time it costs you to work it out.

## Citation

The layer is from the Mamba-3 paper:

```bibtex
@misc{lahoti2026mamba3improvedsequencemodeling,
      title={Mamba-3: Improved Sequence Modeling using State Space Principles},
      author={Aakash Lahoti and Kevin Y. Li and Berlin Chen and Caitlin Wang and Aviv Bick and J. Zico Kolter and Tri Dao and Albert Gu},
      year={2026},
      eprint={2603.15569},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.15569},
}
```

## License

Apache 2.0. Parts derived from
[state-spaces/mamba](https://github.com/state-spaces/mamba), also Apache 2.0. The
Mamba-3 files it derives from are Copyright (c) 2025-2026 Dao AI Lab, Goombalab.
[NOTICE](NOTICE) says which files, which headers, and what changed.
