# Internals

Why the kernels are shaped the way they are. Short version is in the
[README](../README.md).

The layer is the SISO block from [Mamba-3: Improved Sequence Modeling using State Space
Principles](https://arxiv.org/abs/2603.15569). Upstream `state-spaces/mamba` has Triton
kernels for prefill/training and CuTe DSL for decode, both CUDA only. This is forward,
backward and single token decode as Pallas TPU kernels, weight compatible with the PyTorch
`Mamba3` module.

SISO means `ngroups=1`: every head carries its own B/C. The MIMO variant, where a group of
heads shares one B/C, is a different data flow through the same recurrence and is not
implemented here. See [not implemented](#not-implemented).

## What it computes

Per `(batch, head)`, with `N = d_state = 128`, `P = headdim = 64`:

```
lambda_t = sigmoid(trap_t)              theta_t = tanh(angle_t) * pi
gamma_t  = dt_t * lambda_t              phi_t   = sum_{i<=t} theta_i dt_i  (mod 2pi)
scale_t  = gamma_t + dt_{t+1} (1 - lambda_{t+1})
u_t = q_t + q_bias[h]                   w_t = k_t + k_bias[h]
q~_t = R(phi_t) u_t                     khat_t = scale_t * R(phi_t) w_t
h_t  = exp(dt_t A_t) h_{t-1} + khat_t (x) v_t
y_t  = (q~_t h_t + gamma_t (u_t . w_t) v_t + D_h v_t) * silu(z_t)
```

`C` is the query (reads out of the state) and `B` is the key (writes into it), so
`q = C` and `k = B` at the kernel boundary.

### The three term recurrence collapses to one term

The paper writes `h_t = alpha_t h_{t-1} + beta_t B_{t-1}x_{t-1} + gamma_t B_t x_t`, but
position `s`'s total contribution to `h_t` telescopes to `scale_s * alpha_{s+1..t}`. So the
kernel runs an ordinary single term scan on `khat = scale * k~` and the trapezoid costs one
shifted slice instead of a second scan. This is the paper's
`L = (1-semiseparable decay) x (2-band)` factorization. Checked numerically against an
explicit `lax.scan` of the three term form.

The last position has no successor so its shifted term is zero, which is why `SISOState`
has to carry an unscaled `k`.

### Rotary uses a permuted layout

Upstream rotates adjacent channel pairs. On TPU `d_state` is the 128 lane axis, so a
stride-2 split is a lane shuffle every chunk. A fixed de-interleave permutation applied once
at conversion time turns it into a contiguous half split: two 64 lane slices and one
concatenate. `convert.py` does the permutation, and `perm(ref_rope(u))` equals
`neox_rope(perm(u))` bit exactly.

## TPU constraints that shape the code

### cumsum has no Pallas TPU lowering

Neither does `associative_scan`. Within chunk prefix sums go through the MXU as a matmul
against a lower triangular ones matrix (`layout.prefix_sum`). The cumulative rotary angle
stays in XLA, which is also what upstream does with a dedicated kernel pair.

### The MXU has no f32 multiplier

An f32 `dot_general` at default precision silently truncates both operands to bf16 and
accumulates in f32. That is around 1e-2 relative error, so an "f32" path that is really
bf16. The f32 policy passes `Precision.HIGHEST` at every matmul, which requests the
multi pass (bf16x6) decomposition that recovers true f32. `layout.prefix_sum` and
`suffix_sum` use HIGHEST unconditionally since they carry log domain decay into every
downstream `exp2`. Under bf16 the operands are already bf16 so `DEFAULT` is correct and
faster.

CPU does have an f32 multiplier, so this is invisible in interpret mode. It only shows up
on hardware.

### Block shape rules

Mosaic wants the last two block dims divisible by (8, 128), or equal to the array dims. Two
things follow:

- **Per token scalar streams can't be `(B, H, L)`.** With block `(None, None, chunk)` the
  last two dims are `(1, chunk)` after the squeeze, which satisfies neither rule. The legal
  shape is `(B, H, nchunks, 1, chunk)`, see `layout.scalars_to_blocks`.
- **Per head params can't be `(H, N)`.** A `(None, N)` block is rank 1, which Mosaic
  rejects. They go as `(H, 1, N)` with `layout.head_rows`.

`chunk` has to be a multiple of 128 because it ends up on the minormost axis of those
scalar streams.

### Scoped VMEM defaults to 16 MiB

Regardless of what the chip actually has. The backward exceeds it once the chunk grows: at
chunk 512 the live tiles are ~17.4 MiB, mostly the two `chunk x chunk` attention matrices at
2 MiB each, and it fails with `CompileTimeScopedVmemOom`. `layout.block_config` sets
`vmem_limit_bytes` from the chip's real capacity: half of it on v5e, all of it on v3 where
half would be *below* the default and make things worse.

## Grid ordering: head is innermost

The grid is `(batch, nchunks, nheads)`. `q`/`k` are `(B, L, N)` and shared across heads
(`ngroups=1`), and their index maps ignore `h`. Pallas elides an HBM copy when
lexicographically adjacent grid steps map to the same window, so with `h` innermost a q/k
block is fetched once and reused across all `H` heads. With `(B, H, nchunks)` instead, every
head re-walks the same blocks and q/k move `H` times: 1024 MiB instead of 32 MiB at
`B=8 H=32 L=8192`.

Two things had to change for that ordering:

- **The state scratch spans all heads**, `(H, P, N)` f32 rather than `(P, N)`. 1 MiB at
  `H=32` against 128 MiB of VMEM. The kernel indexes it by `pl.program_id(2)`.
- **Final state outputs are one block spanning `H`.** Pallas requires all invocations
  writing a given output slice to be consecutive. With `h` innermost a per-`(b,h)` window
  gets revisited every chunk with other heads in between, and Pallas rejects that outright
  (`Revisited block ... in iteration (0, 1, 0)`). A block spanning `H` keeps the window
  fixed for the whole `b` iteration. Inputs have no such rule, hence the `spec_seed_*` vs
  `spec_final_*` split in `layout.py`.

Correctness is unaffected: for a fixed `(b, h)` the chunk index still increases
monotonically, which is the only ordering the recurrence needs.

`nchunks` is `arbitrary` because it carries the recurrence. `nheads` is arbitrary too, since
Megacore only splits a prefix of parallel axes and splitting the scratch would let two
halves of a v4/v5p chip disagree about which head's slot they own. Only `batch` is parallel.

## Operand layouts, worth 21-26% of forward

Mosaic custom calls don't negotiate layouts. Every operand has to arrive in the default
descending layout. XLA, left alone, picks a different preferred layout for the big
activation tensors, and the mismatch gets paid as a materializing copy outside the kernel.

Root cause is `P=64` against 128 lanes. With `{3,2,1,0}` the minor two dims of `(B,H,L,P)`
are `(8192, 64)`, and TPU tiling `T(8,128)` pads a 64 wide minor dim to 128, so the buffer
takes 2x its logical size. Swapped to `{2,3,1,0}` they're `(64, 8192)` which tiles exactly.
`phi` is worse: `Nr=32` minor pads 4x. So XLA converts on the way in (`v`, `z`, `phi`) and
on the way back out (`y`).

Fix is to pin the layouts at the jit boundary, which is worth 21% at chunk 128 and 25% at
chunk 512, bit exact:

```python
fmts = tuple(L.descending_format(a) for a in call_args)
out_fmt = jax.tree.map(
    lambda o: L.Format(L.Layout(major_to_minor=tuple(range(len(o.shape)))), shd),
    jax.eval_shape(fwd, *call_args),
)
fn = jax.jit(fwd, in_shardings=fmts, out_shardings=out_fmt)
```

Pin the outputs too, not just the inputs, or you fix 3 copies out of 4.

**This can't be done from inside the library.** `jax.device_put(x, fmt)` inside a traced
function is silently dropped (ask for a non default layout, get the default back), and there
is no `lax.with_layout_constraint`. At a jit boundary the same request is honored. So
whoever owns the jit that produces `v`/`z`/`phi` has to pin them.

## The backward

Reverse chunk scan. The grid walks chunks backwards via `layout.reversed_index_map` rather
than reversing the grid itself, which keeps the sequential axis guarantees intact.

Four things are fused into the kernel rather than done as a JAX epilogue: the rotation
pullback, the head reduction for `dq`/`dk`, the `u . w` diagonal, and the SiLU gate. Fusing
the head reduction is legal because `h` is the innermost grid axis, so all writes to a given
`(b, c)` window are consecutive, which is exactly Pallas' requirement.

One subtlety in `dadt` that took two tries to get right: the `A_last` term has to be added
*after* the suffix sum, not before.

```python
dA = jnp.sum(q_rot * dq_rot, -1) - jnp.sum(k_hat * dk_hat, -1)
a_last_term = jnp.exp2(A_last) * jnp.sum(G * S) + jnp.sum(k_hat * dk_inter)
dadt = L.suffix_sum(dA[None, :], tril) + a_last_term
```

Adding it before the suffix sum makes it accumulate `chunk` times over.

## Decode

Two grids, both correct, differing in VMEM per step:

- `siso_decode_folded`, grid `(batch,)`, all heads in one block. Head shared
  `q`/`k`/`angles` are passed as single rows and broadcast inside the kernel instead of
  being duplicated `H` times in HBM. Needs `H * P * N * 4` bytes resident, 1 MiB at `H=32`,
  4 MiB at `H=64`, plus double buffering. Default.
- `siso_decode`, grid `(batch, nheads)`, 32 KiB regardless of `H`. For when the folded block
  doesn't fit: large head counts, or v3 with its 16 MiB VMEM.

The update is in three term form here, not the collapsed one the chunked kernel uses.
Collapsing needs the *next* token's `dt`/`lambda` and at decode time that hasn't been
generated yet, so the carried `(k_prev, v_prev)` pair pays its `beta` term on arrival.

State buffers are donated via `input_output_aliases`. Donation deletes the buffer you passed
in, so a generation loop has to thread the state through rather than reusing one object.
Pass `float32` for `state.k`/`state.v` while you're at it: the kernel returns f32 there, and
a bf16 carry in makes the loop compile twice.

At small batch, wall time is host dominated. A blocking per token call at `B=1` is ~270 us
against ~19 us of device time. ~180 us is round trip you can avoid by not syncing every
token, ~72 us is per call dispatch you can't while sampling is per token. Both are gone by
`B=64`. `decode_scan` runs many steps under one `lax.scan` if you want the device only
number.

## Multi chip

Data parallel over `batch`, wrapped in `jax.shard_map`. Two reasons it has to be that way:

- v5e has one TensorCore per chip, so the Pallas grid can't scale past a single chip.
- **Mosaic kernels can't be automatically partitioned.** A `pallas_call` under plain `jit`
  with input shardings is rejected outright: `NotImplementedError: ... Please wrap the call
  in a shard_map`. `shard_map` hands each device a full size local shard and calls the kernel
  per device, which is the right execution model anyway.

`in_specs` shards every batched argument on axis 0 and replicates the three per head
parameters (`q_bias`, `k_bias`, `d_skip`) which have no batch axis. `check_vma=False`
because `pallas_call` builds its outputs from `ShapeDtypeStruct`s that carry no
`manual_axis_type`, which the varying manual axes check wants.

`shard_map` won't reshard for you either. Arguments have to already carry the sharding
`in_specs` declares, so params and optimizer state get placed replicated once and each batch
gets placed sharded on the way in.

## Training memory

Data parallel replicates params and optimizer state on every chip, so per-chip memory is
set by the model, not the batch. The training step donates params and optimizer state
(`donate_argnums=(0, 1)`), and at scale that is the difference between fitting and not.

Undonated, a step holds six param-sized buffers at once: params, grads, Adam's two moments,
the updates, and the new params. XLA can't write the new params over the old ones unless you
tell it the old ones are dead. At 787M f32 that is 2.93 GiB each:

```
6 x 2.93 = 17.6 GiB   against a v5e's 15.75, OOM
4 x 2.93 = 11.7 GiB   donated, ~4 GiB left for activations
```

Two things follow. Donation deletes the buffers you pass in, so the loop has to thread
params and `opt_state` through the return value, and a "best so far" snapshot has to be a
real copy: `jax.tree.map(lambda t: t, params)` rebuilds the tree around the *same* buffers
and dangles on the next step. `train.py` copies that snapshot to host numpy, which also
keeps it out of HBM.

Also worth knowing: this floor is independent of `seqlen`. The model is recurrent, activation
memory per step doesn't grow with sequence length the way attention's does, so shortening
the sequence does not help an OOM that is really about parameter copies. What does help is
`--shard-optimizer`: Adam's two moments are the only replicated state that has no reason to
be replicated, since each element's update depends only on that element. Raveling the params
into one vector, sharding it over the mesh and keeping moments for the local slice turns
`2 * P * 4` bytes per chip into `2 * P * 4 / n_dev`, 5.87 GiB down to 0.73 GiB at 787M on 8
chips. The cost is one all-gather of the updates per step; params stay replicated so the
forward is untouched.

Two API details that path ran into. `jax.make_mesh` builds `Explicit` axes, and
`with_sharding_constraint` only accepts `Auto` ones, so the sharded/replicated moves use
`jax.sharding.reshard`. And `jax.eval_shape` reports shardings carrying an `AbstractMesh`,
which `out_shardings` rejects (`_device_assignment is not implemented`), so the specs get
rebuilt against the concrete mesh.

## Numerics

Tolerances follow upstream's own metric: 95th percentile relative error over entries with
`|ref| >= 1e-2`. Angles use a circular metric since they wrap at `2pi`.

One tolerance is looser on TPU than CPU, on purpose. The recurrent reference does `L`
sequential `exp()` multiplies while the chunked form does one `exp2(cumsum)` per chunk.
Algebraically identical, numerically not: 2.3e-06 on CPU, 1.8e-04 on TPU at L=2048. In f64
they agree to 1.2e-08, which confirms it's the two formulations and not a kernel bug. So
`refs` and the decode chain get 1e-3 on TPU.

## Test stages

```
0  lower     Mosaic lowering, both chips, chunk 128/256/512/1024, bf16/f32  (no TPU)
1  shapes    tiny run with race detection, output shapes, finiteness
2  refs      three term lax.scan vs the chunked form
2b rotation  compute parity exactly from hand built weights
3  forward   Pallas forward vs chunked reference, f32 and bf16
4  backward  Pallas VJP vs jax.grad of the reference, plus finite differences
5  segments  split the sequence, chain the carry, compare to one call
6  decode    both grids vs the step reference, scan vs loop, chain vs prefill, handoff
7  torch     PyTorch reference on CPU vs JAX through convert.py
7b checkpoint save/load: same config, same tree, same logits, bf16 storage
7c pretrained released-checkpoint key mapping and gated MLP vs a torch stack
8  train     memorize a token sequence (asserted), parity by SGD (reported)
```

**Stage 2b is the interesting one.** Rotational dynamics are what the complex/RoPE part of
Mamba-3 is for, and the paper's Table 5(b) has Mamba-2 at 0.90% on parity (chance) against
Mamba-3 at 100%. Instead of training for
it, this stage builds weights that compute parity in closed form: only token 0 writes into
the state, decay is ~1, and `tanh(angle)*pi*dt = pi*bit`, so `y_t = cos(phi_t) =
(-1)^parity`. Sign accuracy has to be exactly 1.0. A mispaired rotation, a wrong
permutation, or a dropped `phi` accumulation all break it, and it's deterministic and fast
where a training run is neither.

Stage 8 also trains parity by Adam and reports without asserting. It stays at chance, but so
does the pure JAX reference with no Pallas involved, at default `dt_bias` and with `dt_bias`
retuned. Default init gives `dt` in [0.006, 0.020], so `tanh(angle)*pi*dt` reaches ~0.06 rad
per token against the `pi` it needs, and the paper trains this inside a full model with a
tuned recipe rather than one layer plus a linear head. Asserting on it would be asserting on
the optimizer.

## Running without a TPU

`--stage lower` runs the real Mosaic lowering under an abstract TPU mesh
(`layout.abstract_tpu_mesh` + `pl.lower_as_mlir`). No hardware, but block shape bugs and
missing primitive lowerings are caught for real. That's how the `cumsum` problem and the
rank-1 block problem were both found.

Everything else runs in Pallas interpret mode, which simulates VMEM/DMA in Python. Correct
but ~1000x slower, so local shapes are small. Interpret mode also defaults
`uninitialized_memory="nan"`, which is why reading the state scratch before
`pl.when(c == 0)` writes it would poison the output instead of passing quietly.

## Constraints

- `chunk` must be a multiple of 128.
- Sequence length must be a multiple of `chunk`. Pad if not.
- `ngroups=1` only. SISO, not MIMO.
- `rope_fraction` is 0.5 or 1.0.
- Multi chip is data parallel only, see above.
- `bf16` is the throughput policy. `f32` is for parity checking and runs slower, since
  HIGHEST is a 6 pass MXU decomposition.

## Not implemented

- **MIMO**, the `nheads > ngroups` variant where a group of heads shares one B/C. The
  grid ordering section above leans on `ngroups=1`: `q`/`k` are `(B, L, N)` with no head
  axis, so their index maps ignore `h` and Pallas reuses one fetched block across all `H`
  heads. MIMO makes them `(B, L, G, N)`, which changes those index maps, the state scratch
  layout and the `dq`/`dk` head reduction in the backward. It's a real port, not a flag.
- Variable length / packed sequences. Segmented prefill works, chaining the carry across
  calls, but packing several short sequences into one row does not.
- Anything below bf16.
