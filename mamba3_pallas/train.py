"""Byte level LM training on top of the stacked SISO model.

Run:
    python -m mamba3_pallas.train --steps 18000 --corpus enwik8 --protocol-split

Byte level vocab on purpose, no tokenizer between the kernels and the loss. Data is enwik8
or enwik9 (downloaded and cached), or a synthetic structured task if you want to skip the
download.

**Comparing to published numbers.** enwik8 has a specific protocol: train on the first 90M
bytes, validate on the next 5M, test on the last 5M, report bits per char on the test split.
Since the data is a byte stream, bpc and bits/byte are the same number. Published character
level results on that protocol, with their parameter counts:

    gzip -9                                   2.92 bpc          (Mahoney's benchmark)
    LN HM-LSTM                                1.32 bpc    35M   (Chung et al. 2016)
    large mLSTM                               1.24 bpc    46M   (Krause et al. 2016)
    12L Transformer-XL                        1.06 bpc    41M   (Dai et al. 2019, Table 2)
    24L Transformer-XL                        0.99 bpc   277M   (Dai et al. 2019, Table 2)
    Transformer-XL + dynamic eval             0.94 bpc   277M   (Krause et al. 2019)

Pass `--protocol-split` to match it. Without that flag the run trains on ~95M bytes and
holds out 5 MiB, which folds the benchmark's test split into training, so the number is close
but not the real thing. `--corpus enwik9` trains on 949M bytes, 10x what the benchmark
allows, so those numbers are not comparable to the enwik8 leaderboard at all. The run header
prints which case you are in.

The Mamba papers report Pile perplexity and zero shot accuracy rather than enwik8 bpc, so
there is no Mamba number to line up against here.
"""

from __future__ import annotations

import argparse
import functools
import math
import os
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from . import checkpoint as CP
from . import layer as LY
from . import layout as L
from . import model as M


def synthetic_corpus(n_bytes: int, seed: int = 0) -> np.ndarray:
    """A byte stream with long-range structure and a known entropy floor.

    Three interleaved regularities, chosen so a model that only learns local statistics
    cannot do well:

    1. A repeating vocabulary of 64 "words" drawn from a Zipf distribution, so unigram
       and bigram statistics carry real information (an n-gram model does okay).
    2. Bracket nesting: an opening byte is matched by its closing byte after a variable
       gap, which needs state that survives the gap.
    3. A copy task: every ~200 bytes a marker introduces an 8-byte key, and the same key
       must be reproduced after the next marker. This is the part that needs genuine
       recall rather than smoothing.

    The floor is not analytic, but the copy and bracket structure means a model with
    working state should reach clearly below what a bigram model gets, and the gap
    between them is the signal.
    """
    rng = np.random.default_rng(seed)
    words = [
        bytes(rng.integers(97, 123, size=rng.integers(3, 9))) for _ in range(64)
    ]
    zipf = 1.0 / np.arange(1, len(words) + 1)
    zipf /= zipf.sum()

    out = bytearray()
    open_stack: list[int] = []
    pending_key: bytes | None = None
    while len(out) < n_bytes:
        r = rng.random()
        if r < 0.04 and len(open_stack) < 4:
            b = int(rng.integers(0, 3))
            out += bytes([40 + 2 * b])                  # '(', '*', ','... openers
            open_stack.append(b)
        elif r < 0.08 and open_stack:
            b = open_stack.pop()
            out += bytes([41 + 2 * b])                  # matching closer
        elif r < 0.10:
            # copy task: emit a marker plus a key, or reproduce the pending key
            if pending_key is None:
                pending_key = bytes(rng.integers(65, 91, size=8))
                out += b"\x01" + pending_key
            else:
                out += b"\x02" + pending_key
                pending_key = None
        else:
            out += words[int(rng.choice(len(words), p=zipf))]
            out += b" "
    return np.frombuffer(bytes(out[:n_bytes]), dtype=np.uint8)


#: Byte corpora that can be fetched, each with several candidate URLs tried in order so a
#: dead mirror is a logged line and not a traceback. First URL is the canonical home: Matt
#: Mahoney's compression-benchmark directory, which is where every published bits/char
#: number for these files is measured. `tiny_shakespeare` is 1 MB, for checking the pipeline
#: end to end without a 36 MB download.
#:
#: Use `enwik9` at 30M parameters and up. enwik8 is the first 100 MB of enwik9, so the
#: distribution is the same, but 1 GB is 10x the text and a model that big will otherwise
#: memorize enwik8 in a handful of epochs. Note that a number trained on enwik9 is not
#: comparable to the enwik8 leaderboard, since the benchmark fixes training data at 90 MB.
#:
#: Each entry is ``(candidate URLs, member name inside the archive or None if raw)``.
CORPORA: dict[str, tuple[tuple[str, ...], str | None]] = {
    "enwik8": (
        (
            "https://mattmahoney.net/dc/enwik8.zip",
            "http://mattmahoney.net/dc/enwik8.zip",
            "https://data.deepai.org/enwik8.zip",
        ),
        "enwik8",
    ),
    "enwik9": (
        (
            "https://mattmahoney.net/dc/enwik9.zip",
            "http://mattmahoney.net/dc/enwik9.zip",
            "https://data.deepai.org/enwik9.zip",
        ),
        "enwik9",
    ),
    "text8": (
        (
            "https://mattmahoney.net/dc/text8.zip",
            "http://mattmahoney.net/dc/text8.zip",
        ),
        "text8",
    ),
    "tiny_shakespeare": (
        (
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
            "tinyshakespeare/input.txt",
        ),
        None,
    ),
}


def fetch_corpus(name: str, cache_dir: str = "~/.cache/mamba3_pallas") -> str:
    """Download and unpack a named corpus, returning the local path. Cached.

    Tries each candidate URL in turn and reports which one worked, so a dead mirror is a
    logged line rather than a traceback. Raises with actionable alternatives if all fail --
    on Kaggle the usual fix is to attach the dataset in the sidebar and pass its
    ``/kaggle/input/...`` path directly to ``--corpus``.
    """
    import urllib.error
    import urllib.request
    import zipfile

    urls, member = CORPORA[name]
    cache = os.path.expanduser(cache_dir)
    os.makedirs(cache, exist_ok=True)
    dest = os.path.join(cache, member or name)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest

    errors: list[str] = []
    for url in urls:
        archive = url.endswith(".zip")
        tmp = dest + (".zip" if archive else ".part")
        try:
            print(f"fetching {name}: {url}", flush=True)
            req = urllib.request.Request(
                url, headers={"User-Agent": "mamba3-pallas/1.0"}
            )
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                f.write(r.read())
        except Exception as exc:  # noqa: BLE001 - any failure means try the next mirror
            errors.append(f"  {url}\n    {type(exc).__name__}: {exc}")
            continue
        if archive:
            with zipfile.ZipFile(tmp) as zf:
                names = zf.namelist()
                inner = member if member in names else names[0]
                with zf.open(inner) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
            os.remove(tmp)
        else:
            os.replace(tmp, dest)
        print(f"cached {dest} ({os.path.getsize(dest) / 2**20:.1f} MiB)", flush=True)
        return dest

    raise RuntimeError(
        f"could not fetch {name!r}; every mirror failed:\n"
        + "\n".join(errors)
        + "\n\nAlternatives:\n"
        f"  * attach the dataset in Kaggle's sidebar and pass its path:\n"
        f"      --corpus /kaggle/input/<dataset>/{member or name}\n"
        f"  * download it yourself and pass the path\n"
        f"  * --corpus tiny_shakespeare  (1 MB, different URL)\n"
        f"  * --corpus synthetic         (no download; see the note it prints)"
    )


def load_corpus(path: str | None, n_bytes: int = 0) -> tuple[np.ndarray, str]:
    """Bytes to train on, from a path, a known corpus name, or the synthetic fallback.

    ``path`` may be a filename, a directory (the largest file in it is used -- Kaggle
    mounts datasets as directories), one of `CORPORA`, or None for synthetic.
    ``n_bytes`` of 0 means read the whole file.
    """
    if path in CORPORA:
        path = fetch_corpus(path)
    if path and os.path.isdir(path):
        files = [
            os.path.join(path, f) for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ]
        if not files:
            raise RuntimeError(f"--corpus {path!r} is an empty directory")
        path = max(files, key=os.path.getsize)
        print(f"using largest file in that directory: {path}")
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            raw = f.read() if n_bytes <= 0 else f.read(n_bytes)
        return np.frombuffer(raw, dtype=np.uint8), f"{os.path.basename(path)} bytes"
    if path:
        raise RuntimeError(
            f"--corpus {path!r} not found. Pass a file, a directory, one of "
            f"{sorted(CORPORA)}, or 'synthetic'."
        )
    return synthetic_corpus(n_bytes if n_bytes > 0 else 8 << 20), (
        "synthetic structured bytes"
    )


def split_corpus(
    data: np.ndarray, val_frac: float = 0.05, val_cap: int = 5 << 20,
    protocol: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """``(train, val)`` by a contiguous tail split.

    Contiguous, not interleaved: with a recurrent model and 512-token windows, random
    interleaving would put validation bytes inside training windows and the held-out loss
    would be meaningless.

    Args:
      protocol: Use enwik8's exact benchmark split -- first 90M bytes to train, *last* 5M
        as the test slice, and the 5M validation slice in between left unused. This is what
        published bpc numbers are measured on, so it is the only setting whose result is
        directly comparable to them.
      val_frac, val_cap: Used when ``protocol`` is False. 5% of the corpus, capped at 5 MiB
        so enwik9 does not hold out 50 MB of text the model could train on.

    The default (non-protocol) split trains on ~95M bytes of enwik8 rather than 90M,
    folding the benchmark's test split into training and evaluating at the test position.
    That is fine for tracking progress and wrong for claiming a leaderboard number.
    """
    if protocol:
        if len(data) < 100_000_000:
            raise ValueError(
                f"--protocol-split needs the full 100M-byte enwik8; got {len(data)} bytes. "
                f"Use --corpus enwik8 --corpus-bytes 0."
            )
        return data[:90_000_000], data[95_000_000:100_000_000]
    n_val = min(max(1, int(len(data) * val_frac)), val_cap)
    return data[:-n_val], data[-n_val:]


def batches(data: np.ndarray, batch: int, seqlen: int, key: jax.Array):
    """Infinite stream of ``(inputs, targets)``, both ``(batch, seqlen)`` int32."""
    n = len(data) - seqlen - 1
    while True:
        key, sub = jax.random.split(key)
        idx = np.asarray(jax.random.randint(sub, (batch,), 0, n))
        xs = np.stack([data[i : i + seqlen] for i in idx])
        ys = np.stack([data[i + 1 : i + seqlen + 1] for i in idx])
        yield jnp.asarray(xs, jnp.int32), jnp.asarray(ys, jnp.int32)


def bigram_baseline(data: np.ndarray) -> float:
    """Cross-entropy of a smoothed bigram model, in nats.

    The number a model must beat to be doing anything beyond local statistics. Computed
    on the same bytes the model trains on, so it is if anything generous to the baseline.
    """
    counts = np.ones((256, 256), np.float64)          # Laplace
    np.add.at(counts, (data[:-1], data[1:]), 1.0)
    probs = counts / counts.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(probs[data[:-1], data[1:]])))


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


def make_step(cfg: M.LMConfig, opt, interpret: Any, mesh=None):
    """One jitted AdamW step. Returns ``(loss, params, opt_state)``.

    With ``mesh``, the batch is split across every chip and the layer runs under
    ``jax.shard_map``. Two things force that shape:

    * v5e has one TensorCore per chip, so the Pallas grid cannot scale past a single chip.
      Data parallelism over chips is the only axis available.
    * **Mosaic kernels cannot be automatically partitioned.** A ``pallas_call`` under plain
      ``jit`` with input shardings is rejected outright, GSPMD has no way to reason about
      the kernel body. ``shard_map`` hands each device a full size local shard and calls
      the kernel per device, which is the right execution model anyway.

    Gradients are averaged across the mesh with ``jax.lax.pmean`` inside the shard_map, so
    every replica applies the same update and the params stay in step without an explicit
    all-gather.
    """
    import optax

    def loss_fn(params, xs, ys):
        logits = M.lm_forward(params, xs, cfg, interpret=interpret)
        return optax.softmax_cross_entropy_with_integer_labels(logits, ys).mean()

    if mesh is None:
        @jax.jit
        def step(params, opt_state, xs, ys):
            loss, grads = jax.value_and_grad(loss_fn)(params, xs, ys)
            updates, opt_state = opt.update(grads, opt_state, params)
            return loss, optax.apply_updates(params, updates), opt_state

        return step

    from jax.sharding import PartitionSpec as P

    def sharded_grad(params, xs, ys):
        """Per-device loss and gradient, averaged over the data axis."""
        loss, grads = jax.value_and_grad(loss_fn)(params, xs, ys)
        return jax.lax.pmean(loss, "data"), jax.lax.pmean(grads, "data")

    # Params replicated, data sharded on axis 0. `check_vma=False` because pallas_call
    # builds its outputs from ShapeDtypeStructs that carry no manual_axis_type, which the
    # varying-manual-axes check wants.
    grad_fn = jax.shard_map(
        sharded_grad,
        mesh=mesh,
        in_specs=(P(), P("data"), P("data")),
        out_specs=(P(), P()),
        check_vma=False,
    )

    @jax.jit
    def step(params, opt_state, xs, ys):
        loss, grads = grad_fn(params, xs, ys)
        updates, opt_state = opt.update(grads, opt_state, params)
        return loss, optax.apply_updates(params, updates), opt_state

    return step


@functools.partial(jax.jit, static_argnums=(4,))
def _pick_token(
    logits: jnp.ndarray,
    seen: jnp.ndarray,
    key: jax.Array,
    params: jnp.ndarray,
    vocab: int,
) -> jnp.ndarray:
    """Temperature + nucleus (top-p) + repetition penalty, then sample. One jitted call.

    Fused on purpose. Done eagerly, each of argsort / cumsum / scatter / softmax is its own
    XLA dispatch, about 10 per token, and each one costs roughly what the decode kernel
    costs. `seen` is a fixed size 256 wide mask rather than a variable length index array
    so the shape never changes and nothing re-traces.

    Args:
      logits: ``(vocab,)`` raw model output.
      seen: ``(vocab,)`` bool, bytes appearing in the recent window.
      params: ``(3,)`` f32 of ``(temperature, top_p, rep_penalty)``. Traced, not static,
        so changing them does not recompile.

    The filters are here because plain temperature sampling degenerates on enwik8: the
    corpus has long runs of HTML numeric entities, those runs are the most locally
    predictable text in the file, and a per-byte-likelihood model will emit
    ``&amp;#963;&amp;#951;...`` forever once it lands there. Sampling time only, the
    reported loss is untouched.
    """
    temperature, top_p, rep_penalty = params[0], params[1], params[2]
    lg = logits.astype(jnp.float32)

    # Repetition penalty on recently-seen bytes, before top-p so a penalized byte can
    # actually fall out of the nucleus. Sign-aware: dividing a negative logit would make
    # it *more* likely.
    penalized = jnp.where(lg > 0, lg / rep_penalty, lg * rep_penalty)
    lg = jnp.where(seen, penalized, lg)
    lg = lg / temperature

    # Nucleus: keep the smallest prefix of the sorted distribution whose mass reaches
    # top_p. `cumsum - probs` is the mass *strictly before* each entry, so the top-1 is
    # always kept even when its own mass already exceeds top_p.
    order = jnp.argsort(lg)[::-1]
    probs = jax.nn.softmax(lg[order])
    keep = (jnp.cumsum(probs) - probs) < top_p
    masked = jnp.where(keep, lg[order], -jnp.inf)
    lg = jnp.full_like(lg, -jnp.inf).at[order].set(masked)

    return jax.random.categorical(key, lg) % vocab


def sample(
    params: M.LMParams,
    cfg: M.LMConfig,
    key: jax.Array,
    prompt: bytes = b"the ",
    n_tokens: int = 192,
    temperature: float = 0.9,
    top_p: float = 0.95,
    rep_penalty: float = 1.15,
    rep_window: int = 64,
    interpret: Any = False,
    stream: bool = True,
) -> bytes:
    """Generate bytes with the decode kernel: one prefill, then O(1) per token.

    Three things that matter here:

    1. **Prefill once** to build the per layer state, then step. O(n), not O(n^2).
    2. **jit the step.** Un-jitted, every call re-traces and recompiles all the Pallas
       kernels in the stack, so it never finishes.
    3. **Donate the state**, like a generation loop should.

    Sampling is temperature + top-p + a repetition penalty over the last `rep_window`
    bytes. See `_pick_token`.

    `stream` prints bytes as they arrive. At ~270 us per token that is legible in real
    time, and a silent generator looks hung.
    """
    pad_to = cfg.siso.chunk
    ctx = np.frombuffer(prompt, dtype=np.uint8)
    if len(ctx) < pad_to:
        ctx = np.concatenate([np.zeros(pad_to - len(ctx), np.uint8), ctx])
    ctx = ctx[-pad_to:]

    prefill = jax.jit(
        functools.partial(M.lm_forward, cfg=cfg, interpret=interpret)
    )
    step = jax.jit(
        functools.partial(M.lm_decode_step, cfg=cfg, interpret=interpret)
    )

    logits, states = prefill(
        params, jnp.asarray(ctx[None], jnp.int32), states=M.zero_states(cfg, 1)
    )
    last = logits[0, -1]

    out = bytearray(prompt)
    if stream:
        print(prompt.decode("utf-8", errors="replace"), end="", flush=True)
    filt = jnp.asarray([temperature, top_p, rep_penalty], jnp.float32)
    # Fixed-size mask, updated on the host as a ring buffer. A variable-length index
    # array would re-trace `_pick_token` on every distinct window size.
    seen = np.zeros(cfg.vocab_size, bool)
    window: list[int] = list(prompt[-rep_window:])
    for b in window:
        seen[b] = True
    for _ in range(n_tokens):
        key, sub = jax.random.split(key)
        nxt = int(_pick_token(last, jnp.asarray(seen), sub, filt, cfg.vocab_size))
        out.append(nxt)
        window.append(nxt)
        if len(window) > rep_window:
            dropped = window.pop(0)
            if dropped not in window:
                seen[dropped] = False
        seen[nxt] = True
        if stream:
            print(
                bytes([nxt]).decode("utf-8", errors="replace"), end="", flush=True
            )
        logits, states = step(params, jnp.asarray([nxt], jnp.int32), states)
        last = logits[0]
    if stream:
        print(flush=True)
    return bytes(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--headdim", type=int, default=64)
    ap.add_argument("--d-state", type=int, default=128)
    ap.add_argument(
        "--d-intermediate", type=int, default=0,
        help="hidden width of a per-block gated MLP; 0 means no MLP. The released "
             "Mamba-3 checkpoints use 2*d_model, byte level models here do fine without.",
    )
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--policy", default="bf16")
    ap.add_argument(
        "--corpus", default="enwik8",
        help="'enwik8', 'text8', 'tiny_shakespeare', 'synthetic', or a file/directory "
             "path (e.g. a Kaggle /kaggle/input/... mount). Named corpora are downloaded "
             "and cached; several mirrors are tried.",
    )
    # 0 means "all of it". The 7M-parameter run overfit 30 MB (train-val gap widened from
    # -0.119 to -0.195 bpb going 2M -> 7M params), and enwik8 is 95 MB on disk, so capping
    # at 32 MB was throwing away two thirds of the fix.
    ap.add_argument("--corpus-bytes", type=int, default=0,
                    help="bytes of corpus to use; 0 = all of it")
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--devices", type=int, default=0,
        help="chips to use; 0 = all of them. --batch is per-chip, so the global batch "
             "is --batch x devices. Use 1 to force the single-chip path.",
    )
    ap.add_argument(
        "--protocol-split", action="store_true",
        help="use enwik8's exact benchmark split (train on the first 90M bytes, test on "
             "the last 5M). The only setting whose bpc is comparable to published numbers.",
    )
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--rep-penalty", type=float, default=1.15,
                    help="1.0 disables. enwik8 entity runs need ~1.1-1.2; naturally "
                         "repetitive text like children's stories reads better at 1.0")
    ap.add_argument("--sample-bytes", type=int, default=384)
    ap.add_argument(
        "--prompt", default=None, metavar="TEXT",
        help="prefix to generate from. Default is picked from the corpus, so pass this "
             "when training on your own text: a prompt the corpus never contains is out "
             "of distribution and the first tokens come out as noise.",
    )
    ap.add_argument(
        "--save", default=None, metavar="PATH",
        help="write the best-validation weights to PATH as an npz (see "
             "checkpoint.py). Without this the weights are discarded when the run ends.",
    )
    ap.add_argument(
        "--save-dtype", default="float32", choices=("float32", "bfloat16"),
        help="dtype to store weights in; bfloat16 halves the file",
    )
    ap.add_argument(
        "--resume", default=None, metavar="PATH",
        help="start from the weights in PATH instead of a fresh init. The model shape "
             "comes from the checkpoint, so the --d-model/--n-layers flags are ignored.",
    )
    args = ap.parse_args(argv)

    try:
        import optax
    except ImportError:
        print("optax is required: pip install optax")
        return 2

    interpret = False if L.on_tpu() else L.interpret_mode()
    if not L.on_tpu():
        print(L.no_tpu_message())
        print("\nRunning in interpret mode: correct but ~1000x slower. Use a tiny")
        print("--steps/--d-model to check the plumbing; the loss curve needs a TPU.\n")

    raw, label = load_corpus(
        None if args.corpus == "synthetic" else args.corpus, args.corpus_bytes
    )
    data, val_data = split_corpus(raw, protocol=args.protocol_split)
    resumed = None
    if args.resume:
        # The checkpoint owns the model shape: rebuilding it from the flags would silently
        # produce a different tree if they disagree, and the load would fail on shapes.
        resumed, cfg = CP.load(args.resume, dtype=jnp.float32)
        siso = cfg.siso
        print(f"resuming from {args.resume}: {cfg.n_layers} layers,"
              f" d_model={siso.d_model}, {cfg.param_count() / 1e6:.2f}M params")
    else:
        siso = LY.SISOConfig(
            d_model=args.d_model, d_state=args.d_state, headdim=args.headdim,
            chunk=args.chunk, policy_name=args.policy,
        )
        cfg = M.LMConfig(
            vocab_size=256, n_layers=args.n_layers, siso=siso,
            d_intermediate=args.d_intermediate,
        )

    # Data parallelism over chips. `--batch` is per device so the global batch grows with
    # the mesh, which keeps a given --batch comparable across device counts and makes the
    # speedup show up as tokens/s. `--devices 1` forces the single chip path.
    #
    # Not gated on on_tpu(): the shard_map path also runs on CPU with
    # XLA_FLAGS=--xla_force_host_platform_device_count=N, which is handy for testing.
    n_dev = len(jax.devices())
    if args.devices > 0:
        n_dev = min(n_dev, args.devices)
    mesh = jax.make_mesh((n_dev,), ("data",)) if n_dev > 1 else None
    global_batch = args.batch * (n_dev if mesh else 1)

    # Is this run's number comparable to published enwik8 bpc? Only if it followed the
    # benchmark's split on the benchmark's corpus.
    is_enwik8 = "enwik8" in label
    comparable = is_enwik8 and args.protocol_split
    split_desc = (
        "enwik8 protocol split: train first 90M, test last 5M"
        if args.protocol_split else "contiguous tail"
    )

    print("=" * 78)
    print(L.describe_environment())
    print(f"data: {label}, {len(data) / 1e6:.1f}M bytes train"
          f" + {len(val_data) / 1e6:.1f}M held-out ({split_desc})")
    print(f"model: {cfg.n_layers} x SISO(d_model={siso.d_model}, d_state={siso.d_state},"
          f" P={siso.headdim}, H={siso.nheads})"
          f"{f' + MLP({cfg.d_intermediate})' if cfg.d_intermediate else ''},"
          f" {cfg.param_count() / 1e6:.2f}M params")
    if mesh:
        print(f"shard: {n_dev} chips via shard_map, batch {args.batch}/chip"
              f" = {global_batch} global (Mosaic cannot be auto-partitioned)")
    else:
        print(f"shard: single chip"
              f"{' (--devices 1)' if args.devices == 1 else ''}")
    print(f"train: {args.steps} steps, batch {global_batch} x {args.seqlen},"
          f" {global_batch * args.seqlen * args.steps / 1e6:.1f}M tokens seen,"
          f" {global_batch * args.seqlen * args.steps / max(cfg.param_count(), 1):.1f}"
          f" tok/param, {args.policy}")
    if comparable:
        print("bpc:   COMPARABLE to published enwik8 numbers (protocol split, enwik8)")
    else:
        why = []
        if not is_enwik8:
            why.append(f"corpus is {label.split()[0]}, not enwik8")
        if not args.protocol_split:
            why.append("split is not the 90M/5M protocol")
        print(f"bpc:   NOT comparable to published enwik8 numbers ({'; '.join(why)})")
    print("=" * 78)

    bigram = bigram_baseline(data[: min(len(data), 4 << 20)])
    print(f"\nbaselines (nats/byte):  uniform {math.log(256):.3f}"
          f"   smoothed bigram {bigram:.3f}")
    print("Published enwik8 character-level results (bits/char == bits/byte here),")
    print("all on the 90M/5M/5M protocol split:")
    print("  gzip -9 2.92 | LN HM-LSTM 35M 1.32 | large mLSTM 46M 1.24")
    print("  Transformer-XL 41M 1.06 | 277M 0.99 | +dyn.eval 277M 0.94")
    print("Held-out loss is what counts; train loss falls by memorizing.\n")

    params = M.init_lm(cfg, jax.random.key(args.seed)) if resumed is None else resumed
    sched = optax.warmup_cosine_decay_schedule(
        init_value=args.lr * 0.1, peak_value=args.lr, warmup_steps=args.warmup,
        decay_steps=max(args.steps, args.warmup + 1), end_value=args.lr * 0.1,
    )
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.1))
    opt_state = opt.init(params)
    step = make_step(cfg, opt, interpret, mesh=mesh)
    stream = batches(data, global_batch, args.seqlen, jax.random.key(args.seed + 1))

    # `shard_map` rejects arguments whose sharding does not already match `in_specs` --
    # it will not reshard for you. So params and optimizer state are placed replicated
    # once here, and each batch is placed sharded on the way in.
    if mesh is None:
        shard_batch = lambda t: t
    else:
        from jax.sharding import NamedSharding, PartitionSpec as P

        repl = NamedSharding(mesh, P())
        sh = NamedSharding(mesh, P("data"))
        place = lambda t: jax.device_put(t, repl) if hasattr(t, "shape") else t
        params = jax.tree.map(place, params)
        opt_state = jax.tree.map(place, opt_state)
        shard_batch = lambda t: jax.device_put(t, sh)

    import optax as _optax  # local alias, already imported above

    if mesh is None:
        @jax.jit
        def val_loss(params, xs, ys):
            logits = M.lm_forward(params, xs, cfg, interpret=interpret)
            return _optax.softmax_cross_entropy_with_integer_labels(logits, ys).mean()
    else:
        from jax.sharding import PartitionSpec as P

        def _vl(params, xs, ys):
            logits = M.lm_forward(params, xs, cfg, interpret=interpret)
            return jax.lax.pmean(
                _optax.softmax_cross_entropy_with_integer_labels(logits, ys).mean(),
                "data",
            )

        val_loss = jax.jit(
            jax.shard_map(
                _vl, mesh=mesh, in_specs=(P(), P("data"), P("data")), out_specs=P(),
                check_vma=False,
            )
        )

    def evaluate(params) -> float:
        """Mean loss over held-out bytes. Fixed key, so runs are comparable."""
        vs = batches(val_data, global_batch, args.seqlen, jax.random.key(1234))
        tot = 0.0
        for _ in range(args.val_batches):
            xs, ys = next(vs)
            tot += float(val_loss(params, shard_batch(xs), shard_batch(ys)))
        return tot / args.val_batches

    print(f"{'step':>6s} {'loss':>8s} {'bits/byte':>10s} {'val bpb':>9s}"
          f" {'tok/s':>12s} {'elapsed':>9s}")
    t0 = time.perf_counter()
    losses: list[float] = []
    val_hist: list[tuple[int, float]] = []
    # Keep the best-validation params. Held-out loss on a small corpus turns around well
    # before the schedule ends, so reporting the final params understates the model by
    # however far past its optimum the run went. Costs one params-sized copy.
    best: tuple[float, int, Any] | None = None
    for i in range(1, args.steps + 1):
        xs, ys = next(stream)
        loss, params, opt_state = step(
            params, opt_state, shard_batch(xs), shard_batch(ys)
        )
        losses.append(float(loss))
        if i % args.log_every == 0 or i == 1:
            jax.block_until_ready(loss)
            recent = float(np.mean(losses[-args.log_every :]))
            vcol = ""
            if args.val_every and (i % args.val_every == 0 or i == args.steps):
                v = evaluate(params)
                val_hist.append((i, v))
                mark = " "
                if best is None or v < best[0]:
                    best = (v, i, jax.tree.map(lambda t: t, params))
                    mark = "*"
                vcol = f"{v / math.log(2):8.3f}{mark}"
            el = time.perf_counter() - t0
            print(f"{i:6d} {recent:8.4f} {recent / math.log(2):10.3f} {vcol:>9s}"
                  f" {global_batch * args.seqlen * i / el:12,.0f} {el:8.1f}s")

    # Average the last 5% of steps, floored at 10 and capped at 200. A fixed window like
    # `losses[-50:]` would cover most of a short run and report a loss the model passed
    # long ago.
    tail = min(200, max(10, args.steps // 20))
    final = float(np.mean(losses[-tail:]))
    val_final = val_hist[-1][1] if val_hist else evaluate(params)
    tokens_seen = global_batch * args.seqlen * args.steps
    if best is not None and best[1] < args.steps:
        params = best[2]                      # generate from the best checkpoint
    val = best[0] if best is not None else val_final

    print(f"\ntrain loss {final:.4f} nats ({final / math.log(2):.3f} bits/byte)"
          f"   [mean of last {tail} steps]")
    print(f"HELD-OUT   {val:.4f} nats ({val / math.log(2):.3f} bits/byte)"
          f"   <- the number that counts")
    if best is not None and best[1] < args.steps:
        print(f"  best at step {best[1]} of {args.steps};"
              f" final was {val_final / math.log(2):.3f} bpb"
              f" ({(val_final - val) / math.log(2):+.3f} worse -- trained past the"
              f" optimum, sampling uses the best checkpoint)")
    gap = final - val
    print(f"  train-val gap {gap:+.4f} nats"
          f"  {'(memorizing)' if gap < -0.15 else '(generalizing)'}")
    print(f"  vs uniform {math.log(256):.3f}: {math.log(256) - val:+.3f}")
    print(f"  vs bigram  {bigram:.3f}: {bigram - val:+.3f}"
          f"  {'BEATS bigram' if val < bigram else 'does NOT beat bigram'}")
    if args.corpus == "synthetic":
        print("  NOTE: synthetic data. The corpus is 64 repeated words, so a local model")
        print("  does unusually well (bigram ~0.8 nats vs ~2.5-3 on real text) and a")
        print("  sequence model can partly memorize it. Use --corpus enwik8 for a number")
        print("  comparable to published results.")
    else:
        epochs = tokens_seen / max(len(data), 1)
        print(f"  {cfg.param_count() / 1e6:.1f}M params, {tokens_seen / 1e6:.0f}M tokens"
              f" ({tokens_seen / cfg.param_count():.0f} tok/param, {epochs:.2f} epochs"
              f" over {len(data) / 1e6:.0f}M bytes)")
        if comparable:
            print(f"  COMPARABLE to published enwik8 bpc: gzip 2.92 | HM-LSTM 35M 1.32")
            print(f"  | mLSTM 46M 1.24 | Transformer-XL 41M 1.06, 277M 0.99")
            print(f"  {cfg.param_count() / 1e6:.0f}M params at {val / math.log(2):.3f} bpc,"
                  f" measured on the same split, so it reads against those directly.")
        else:
            print(f"  NOT comparable to published enwik8 bpc"
                  f"{' (enwik9 trains on 10x the data the benchmark allows)' if 'enwik9' in label else ''}"
                  f"{'' if args.protocol_split else ' (non-protocol split)'}.")
            print(f"  For a comparable number: --corpus enwik8 --protocol-split")
        if epochs > 4 and gap < -0.15:
            print(f"  -> {epochs:.1f} epochs is enough to memorize this corpus at this size;"
                  f" more data (enwik9)")
            print(f"     will help more than more steps.")
        elif epochs < 1.5 and gap > -0.1 and best is not None and best[1] >= args.steps:
            print(f"  -> still improving at the final step and only {epochs:.2f} epochs in;"
                  f" more steps will help.")

    # Save before sampling. Sampling is the part most likely to raise (an unlucky prompt,
    # a decode-path bug), and losing an hour of TPU time to that would be avoidable.
    if args.save:
        written = CP.save(args.save, params, cfg, dtype=getattr(jnp, args.save_dtype))
        mb = os.path.getsize(written) / (1 << 20)
        print(f"\nsaved {written} ({mb:.1f} MiB, {args.save_dtype},"
              f" val {val / math.log(2):.3f} bpb)")
        print(f"  reload it with: mamba3_pallas.checkpoint.load({written!r})")

    key = jax.random.key(args.seed + 2)
    prompt = default_prompt(args.corpus, label) if args.prompt is None else (
        args.prompt.encode("utf-8")
    )
    print(f"\n--- sample (T={args.temperature} top_p={args.top_p}"
          f" rep={args.rep_penalty},"
          f" {'decode kernel' if L.on_tpu() else 'interpret'}) ---")
    t1 = time.perf_counter()
    if mesh is not None:
        # Generation is batch-1 and single-device. The trained params carry an Explicit
        # sharding from the mesh, and a plain `jit` inside `sample` then refuses them
        # ("Length of device assignment 1 is not equal to the size of the mesh N"), so
        # copy them to one device *and* clear the sharding by round-tripping through
        # numpy. Cheap: this happens once, after training.
        params = jax.tree.map(
            lambda t: jnp.asarray(np.asarray(t)) if hasattr(t, "shape") else t,
            params,
        )
    text = sample(
        params, cfg, key, prompt=prompt, n_tokens=args.sample_bytes,
        temperature=args.temperature, top_p=args.top_p,
        rep_penalty=args.rep_penalty, interpret=interpret,
    )
    el = time.perf_counter() - t1
    n = len(text) - len(prompt)
    print(f"--- {n} bytes in {el:.1f}s ({n / el:.1f} B/s) ---")
    body = text[len(prompt):]
    print(f"    {uniq_bytes(body)} distinct bytes, {degeneracy(body)}")
    return 0


def default_prompt(corpus: str, label: str) -> bytes:
    """A generation prefix the trained corpus actually contains.

    This matters more than it looks. The prompt is the model's whole context at token 0,
    and a prefix that never appears in training is out of distribution: the first several
    bytes come out as noise until the model hits something it recognizes (often an
    end-of-document marker) and restarts cleanly. It reads like a broken model and is not.

    ``[[The `` is wiki link syntax, which is right for enwik8/enwik9 and wrong for
    everything else, so only those get it. A path corpus gets a bare capital letter, since
    nothing can be assumed about its contents. Pass ``--prompt`` to override.
    """
    if corpus == "synthetic":
        return b"the "
    if "enwik" in label:
        return b"[[The "
    if "text8" in label:
        return b" the "
    if "shakespeare" in label:
        return b"KING RICHARD"
    return b"The "


def uniq_bytes(body: bytes) -> int:
    return len(set(body))


def degeneracy(
    body: bytes, max_coverage: float = 0.35, max_period: int = 32
) -> str:
    """Describe repetition in generated text, or say there is none.

    Measures **coverage**: what fraction of the sample its longest repeating run occupies.
    Raw repeat count does not work as a threshold. Six repeated bytes is 1.6% of a 384 byte
    sample and completely normal, while a 10 byte HTML entity repeating 180 times covers the
    whole thing. Coverage separates those by two orders of magnitude, a count cannot.

    Whitespace only periods are skipped. Indentation and blank line runs are real structure
    in enwik8 markup, so a model reproducing them is working, not looping.

    Returns a one line verdict with the period, its repeat count and its coverage, so the
    number behind the call is always visible.
    """
    if not body:
        return "empty"
    best = (0, 0, 0.0)                       # (period, reps, coverage)
    for k in range(1, min(max_period, len(body) // 2) + 1):
        i = 0
        while i + 2 * k <= len(body):
            if body[i : i + k] != body[i + k : i + 2 * k]:
                i += 1
                continue
            reps = 2
            j = i + 2 * k
            while j + k <= len(body) and body[j : j + k] == body[i : i + k]:
                reps += 1
                j += k
            period = body[i : i + k]
            if period.strip():               # skip whitespace-only runs
                cov = (reps * k) / len(body)
                if cov > best[2]:
                    best = (k, reps, cov)
            i = j
    k, reps, cov = best
    if k == 0:
        return "no degeneration (no non-whitespace repeats)"
    detail = f"{k}-byte period x{reps}, {cov:.0%} of output"
    return (
        f"DEGENERATE: {detail}" if cov > max_coverage
        else f"no degeneration (longest: {detail})"
    )


if __name__ == "__main__":
    raise SystemExit(main())
