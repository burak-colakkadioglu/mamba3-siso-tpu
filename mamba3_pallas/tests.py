"""Staged test suite for the Mamba-3 SISO TPU kernels.

Runs the same checks locally (CPU, ``interpret`` mode) and on a TPU. Stages are
ordered cheapest-first, so a break shows up in the fastest thing that can see it:

  0 lower      real Mosaic lowering per chip kind. No TPU needed, no execution.
  1 shapes     tiny kernel run, race detection on.
  2 refs       recurrent vs chunked JAX reference. No kernel involved.
  2b rotation  compute parity exactly from hand-built weights -- the capability the
               rotary path exists for, checked without training.
  3 forward    Pallas forward vs chunked reference.
  4 backward   Pallas VJP vs jax.grad of the chunked reference.
  5 segments   split the sequence, chain the state, compare to one call.
  6 decode     both decode grids vs the step reference, chain vs prefill, and the
               prefill-then-decode handoff.
  7 torch      PyTorch reference on CPU vs JAX through convert.py.
  7b checkpoint save/load round-trip: same config, same tree, same logits.
  8 train      overfit a tiny LM task; also the paper's parity task.

Usage::

    python -m mamba3_pallas.tests                 # everything the platform allows
    python -m mamba3_pallas.tests --stage lower   # one stage
    python -m mamba3_pallas.tests --tpu           # full shapes, interpret off

Accuracy metric follows upstream's own: the 95th-percentile relative error over
entries with ``|ref| >= 1e-2`` (``relative_error`` in
``mamba/tests/ops/triton/test_mamba3_siso.py``). Angles use a circular metric, since
they wrap.
"""

from __future__ import annotations

import argparse
import functools
import math
import sys
import time
import traceback
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from . import convert as CV
from . import kernel_decode as KD
from . import kernel_fwd as KF
from . import layer as LY
from . import layout as L
from . import reference as R
from . import siso as SISO

# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def rel_error(
    got: Any,
    ref: Any,
    *,
    mag_floor: float = 1e-2,
    pct: float = 95.0,
    angle: bool = False,
) -> float:
    """Upstream's accuracy metric.

    Elementwise relative error, restricted to entries whose reference magnitude is at
    least ``mag_floor`` (below that, relative error is meaningless), reduced by the
    95th percentile rather than the max so one unlucky cancellation does not dominate.

    Args:
      angle: Use circular *absolute* error in radians instead. Cumulative angles wrap
        at 2*pi, so 0.01 and 2*pi - 0.01 are close, not maximally distant.
    """
    g = np.asarray(got, np.float64)
    r = np.asarray(ref, np.float64)
    if g.shape != r.shape:
        raise AssertionError(f"shape mismatch: {g.shape} vs {r.shape}")
    if angle:
        d = np.abs((g - r + math.pi) % (2 * math.pi) - math.pi)
        return float(np.percentile(d, pct))
    mask = np.abs(r) >= mag_floor
    if not mask.any():
        return 0.0
    return float(np.percentile(np.abs(g - r)[mask] / np.abs(r)[mask], pct))


def max_rel(got: Any, ref: Any) -> float:
    """Max relative error, normalized by the reference's largest magnitude."""
    g = np.asarray(got, np.float64)
    r = np.asarray(ref, np.float64)
    denom = max(float(np.max(np.abs(r))), 1e-30)
    return float(np.max(np.abs(g - r)) / denom)


# --------------------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------------------


class Report:
    """Collects check outcomes and prints one table at the end."""

    def __init__(self):
        self.rows: list[tuple[str, str, float, float, bool]] = []
        self.errors: list[tuple[str, str]] = []

    def check(self, stage: str, name: str, value: float, tol: float) -> bool:
        ok = bool(value <= tol) and math.isfinite(value)
        self.rows.append((stage, name, value, tol, ok))
        flag = "ok  " if ok else "FAIL"
        print(f"    {flag} {name:34s} {value:.3e}  (tol {tol:.0e})")
        return ok

    def note(self, stage: str, name: str) -> None:
        self.rows.append((stage, name, 0.0, math.inf, True))
        print(f"    ok   {name}")

    def fail(self, stage: str, name: str, exc: BaseException) -> None:
        self.errors.append((f"{stage}:{name}", "".join(traceback.format_exception(exc))))
        self.rows.append((stage, name, math.nan, 0.0, False))
        print(f"    FAIL {name}: {type(exc).__name__}: {exc}")

    def summary(self) -> int:
        n_fail = sum(1 for *_, ok in self.rows if not ok)
        print("\n" + "=" * 74)
        print(f"{len(self.rows) - n_fail}/{len(self.rows)} checks passed")
        if self.errors:
            print("\nexceptions:")
            for name, tb in self.errors:
                print(f"\n--- {name} ---\n{tb}")
        by_stage: dict[str, list[bool]] = {}
        for stage, _, _, _, ok in self.rows:
            by_stage.setdefault(stage, []).append(ok)
        for stage, oks in by_stage.items():
            mark = "PASS" if all(oks) else "FAIL"
            print(f"  {mark}  {stage:10s} {sum(oks)}/{len(oks)}")
        return 1 if n_fail else 0


class Env:
    """Problem sizes and execution mode for one run."""

    def __init__(self, tpu: bool, chunk: int = 128, policy: str = "f32"):
        self.tpu = tpu
        self.chunk = chunk
        self.policy = policy
        # Interpret mode simulates VMEM/DMA in Python and is ~1000x slower, so the
        # local shapes stay small. The kernels are shape-agnostic; only the chunk
        # count changes.
        if tpu:
            self.batch, self.nheads, self.seqlen = 4, 8, 1024
        else:
            self.batch, self.nheads, self.seqlen = 2, 3, 256
        self.state_dim = 128
        self.headdim = 64
        self.n_angles = 32

    @property
    def interpret(self) -> Any:
        return False if self.tpu else L.interpret_mode()

    @property
    def interpret_strict(self) -> Any:
        """With race detection -- for the small shape-check stage only."""
        return False if self.tpu else L.interpret_mode(detect_races=True)

    def __str__(self) -> str:
        mode = "tpu" if self.tpu else "cpu/interpret"
        return (
            f"{mode}  B={self.batch} H={self.nheads} L={self.seqlen} "
            f"N={self.state_dim} P={self.headdim} chunk={self.chunk} {self.policy}"
        )


def make_inputs(env: Env, seed: int = 0) -> dict[str, jnp.ndarray]:
    """Random inputs at the kernel boundary, with realistic magnitudes.

    ``q``/``k`` are RMS-normalized because the real layer feeds them through
    BCNorm, and ``dt`` is log-uniform in ``[1e-3, 1e-1]`` as at initialization --
    both matter, since the accuracy of the decay path depends on ``adt`` being small.
    """
    b, h, l = env.batch, env.nheads, env.seqlen
    n, p, nr = env.state_dim, env.headdim, env.n_angles
    keys = jax.random.split(jax.random.key(seed), 12)

    def norm_rows(x):
        return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-6)

    dt = jnp.exp(
        jax.random.uniform(keys[4], (b, l, h)) * (math.log(0.1) - math.log(0.001))
        + math.log(0.001)
    )
    return {
        "q": norm_rows(jax.random.normal(keys[0], (b, l, n))),
        "k": norm_rows(jax.random.normal(keys[1], (b, l, n))),
        "v": jax.random.normal(keys[2], (b, h, l, p)),
        "z": jax.random.normal(keys[3], (b, h, l, p)),
        # dt_raw is inverse-softplus of dt, so preprocess() recovers dt exactly.
        "dt_raw": jnp.log(jnp.expm1(dt)),
        "a_raw": jax.random.normal(keys[5], (b, l, h)),
        "trap_raw": jax.random.normal(keys[6], (b, l, h)),
        "angles": jax.random.normal(keys[7], (b, l, nr)) * 0.5,
        "dt_bias": jnp.zeros((h,)),
        "q_bias": jax.random.normal(keys[8], (h, n)) * 0.1,
        "k_bias": jax.random.normal(keys[9], (h, n)) * 0.1,
        "d_skip": jax.random.normal(keys[10], (h,)),
        "dy": jax.random.normal(keys[11], (b, h, l, p)),
    }


# --------------------------------------------------------------------------------------
# Stage 0: Mosaic lowering
# --------------------------------------------------------------------------------------


def stage_lower(env: Env, rep: Report) -> None:
    """Lower all three kernels for both target chips, at both chunk sizes.

    This is the only check that sees Mosaic's block-shape and primitive rules, and it
    needs no hardware -- so it runs first and it runs everywhere.
    """
    b, h, l = 2, 3, 1024
    n, p, nr = 128, 64, 32
    f = lambda shape, dt=jnp.float32: jax.ShapeDtypeStruct(shape, dt)
    kinds = ("TPU v5e", "TPU v3")

    # 1024 is covered because it is the fastest forward on v5e (0.358 us/token against
    # 0.415 at chunk 128). The backward is what gates using it: its live
    # tiles go as chunk^2, 8 MiB per attention matrix at 1024 against 2 MiB at 512.
    for chunk in (128, 256, 512, 1024):
        nc = l // chunk
        for pol_name in ("bf16", "f32"):
            pol = L.policy(pol_name)
            tag = f"chunk={chunk} {pol_name}"

            def fwd(q, k, v, adt, gam, scl, phi, qb, kb, d, z, kind="TPU v5e"):
                return KF.siso_forward(
                    q, k, v, adt, gam, scl, phi, qb, kb, d, z,
                    chunk=chunk, dtype_policy=pol, save_residuals=True,
                    device_kind=kind,
                )

            fwd_sig = [
                f((b, l, n)), f((b, l, n)), f((b, h, l, p)),
                f((b, h, l)), f((b, h, l)), f((b, h, l)), f((b, h, l, nr)),
                f((h, n)), f((h, n)), f((h,)), f((b, h, l, p)),
            ]

            def bwd(dy, ybar, z, q, k, v, adt, gam, scl, phi, qb, kb, cs, d, dfs,
                    kind="TPU v5e"):
                from . import kernel_bwd as KB

                return KB.siso_backward_kernel(
                    dy, ybar, z, q, k, v, adt, gam, scl, phi, qb, kb, cs,
                    d_skip=d, dfinal_ssm=dfs, chunk=chunk, dtype_policy=pol,
                    device_kind=kind,
                )

            bwd_sig = [
                f((b, h, l, p)), f((b, h, l, p)), f((b, h, l, p)),
                f((b, l, n)), f((b, l, n)), f((b, h, l, p)),
                f((b, h, l)), f((b, h, l)), f((b, h, l)), f((b, h, l, nr)),
                f((h, n)), f((h, n)), f((b, h, nc, p, n), pol.saved_state),
                f((h,)), f((b, h, p, n)),
            ]

            for label, fn, sig in (("fwd", fwd, fwd_sig), ("bwd", bwd, bwd_sig)):
                for kind in kinds:
                    name = f"lower {label} {tag} {kind}"
                    try:
                        L.lower_for_tpu(
                            functools.partial(fn, kind=kind), *sig, kind=kind
                        )
                        rep.note("lower", name)
                    except Exception as exc:  # noqa: BLE001
                        rep.fail("lower", name, exc)

    # Decode: no chunking, so one shape per policy. Both grid shapes, and both
    # readout units for the folded one -- the folded kernel's (H, P, N) block and its
    # lane reduction are exactly the parts a shape rule could reject, and interpret
    # mode cannot see that.
    for pol_name in ("bf16", "f32"):
        pol = L.policy(pol_name)

        def dec(q, k, v, adt, dt, lam, ang, qb, kb, ssm, sk, sv, sphi, d, z, *, fn):
            state = R.SISOState(ssm=ssm, k=sk, v=sv, phi=sphi)
            return fn(q, k, v, adt, dt, lam, ang, qb, kb, state, d, z, dtype_policy=pol)

        dec_sig = [
            f((8, n)), f((8, n)), f((8, h, p)), f((8, h)), f((8, h)), f((8, h)),
            f((8, nr)), f((h, n)), f((h, n)), f((8, h, p, n)), f((8, h, n)),
            f((8, h, p)), f((8, h, nr)), f((h,)), f((8, h, p)),
        ]
        variants = [
            ("per-head", KD.siso_decode),
            ("folded", KD.siso_decode_folded),
        ]
        for vname, vfn in variants:
            for kind in kinds:
                name = f"lower decode {vname} {pol_name} {kind}"
                try:
                    L.lower_for_tpu(
                        functools.partial(dec, fn=vfn), *dec_sig, kind=kind
                    )
                    rep.note("lower", name)
                except Exception as exc:  # noqa: BLE001
                    rep.fail("lower", name, exc)


# --------------------------------------------------------------------------------------
# Stage 1: shape / index sanity
# --------------------------------------------------------------------------------------


def stage_shapes(env: Env, rep: Report) -> None:
    """Smallest legal problem, race detection on, output shapes asserted.

    Catches index-map and scratch-init mistakes in a second. ``uninitialized_memory``
    defaults to NaN in interpret mode, so reading the state scratch before
    ``pl.when(c == 0)`` writes it would poison the output rather than pass silently.
    """
    small = Env(tpu=False, chunk=128, policy="f32")
    small.batch, small.nheads, small.seqlen = 1, 1, 128
    x = make_inputs(small, seed=7)
    pre, _ = R.preprocess(
        x["dt_raw"], x["a_raw"], x["trap_raw"], x["angles"], x["dt_bias"]
    )
    try:
        y, fs, fk, fv, res = KF.siso_forward(
            x["q"], x["k"], x["v"], pre.adt, pre.gamma, pre.scale, pre.phi,
            x["q_bias"], x["k_bias"], x["d_skip"], x["z"],
            chunk=128, dtype_policy=L.F32, save_residuals=True,
            interpret=env.interpret_strict, device_kind="TPU v5e",
        )
    except Exception as exc:  # noqa: BLE001
        rep.fail("shapes", "forward runs", exc)
        return

    b, h, l = small.batch, small.nheads, small.seqlen
    n, p = small.state_dim, small.headdim
    expected = {
        "y": (y.shape, (b, h, l, p)),
        "final_ssm": (fs.shape, (b, h, p, n)),
        "final_k": (fk.shape, (b, h, n)),
        "final_v": (fv.shape, (b, h, p)),
        "ybar": (res.ybar.shape, (b, h, l, p)),
        "chunk_states": (res.chunk_states.shape, (b, h, 1, p, n)),
    }
    for name, (got, want) in expected.items():
        if got == want:
            rep.note("shapes", f"{name} shape {got}")
        else:
            rep.fail("shapes", f"{name} shape", AssertionError(f"{got} != {want}"))

    finite = all(
        bool(jnp.all(jnp.isfinite(t))) for t in (y, fs, fk, fv, res.ybar, res.chunk_states)
    )
    if finite:
        rep.note("shapes", "all outputs finite (no uninitialized reads)")
    else:
        rep.fail("shapes", "outputs finite", AssertionError("NaN or inf in output"))

    # Layout pinning must not change the result. The trace found 22.6% of forward runtime
    # in XLA copies converting between the layout it prefers for (B,H,L,P) -- {2,3,1,0},
    # because P=64 minor pads to 128 lanes -- and the descending layout Mosaic demands.
    # `profile --only layout` and `bench`'s `fwd pinned` row ask jit for descending on
    # both sides to remove them; this pins that the answer is identical either way.
    if L.Format is None:
        rep.note("shapes", "layout pinning unavailable in this jax (skipped)")
        return
    call_args = (
        x["q"], x["k"], x["v"], pre.adt, pre.gamma, pre.scale, pre.phi,
        x["q_bias"], x["k_bias"], x["d_skip"], x["z"],
    )
    fwd_fn = functools.partial(
        KF.siso_forward, chunk=128, dtype_policy=L.F32, save_residuals=False,
        interpret=env.interpret, device_kind="TPU v5e",
    )
    try:
        fmts = tuple(L.descending_format(a) for a in call_args)
        shd = call_args[0].sharding
        out_fmt = jax.tree.map(
            lambda o: L.Format(
                L.Layout(major_to_minor=tuple(range(len(o.shape)))), shd
            ),
            jax.eval_shape(fwd_fn, *call_args),
        )
        pinned_args = tuple(jax.device_put(a, f) for a, f in zip(call_args, fmts))
        y_pin = jax.jit(fwd_fn, in_shardings=fmts, out_shardings=out_fmt)(*pinned_args)[0]
        y_ref = jax.jit(fwd_fn)(*call_args)[0]
        rep.check("shapes", "pinned layout == default", rel_error(y_pin, y_ref), 0.0)
    except Exception as exc:  # noqa: BLE001
        rep.fail("shapes", "pinned layout == default", exc)


# --------------------------------------------------------------------------------------
# Stage 2: reference cross-check
# --------------------------------------------------------------------------------------


def stage_refs(env: Env, rep: Report) -> None:
    """``lax.scan`` three-term recurrence vs the chunked linear-attention form.

    No kernel here. This validates the algebra that lets the kernel run a *single*-term
    scan on ``scale * k``: if the trapezoid collapse were wrong, everything downstream
    would be consistently wrong together.

    On the tolerance. In f64 the two formulations agree to **1.2e-08** -- the algebra is
    exact. In f32 they diverge because they accumulate decay differently: the recurrent
    form does ``L`` sequential ``exp(adt_t)`` multiplies, the chunked form one
    ``exp2(cumsum)`` per chunk. That gap is 2.3e-06 on CPU but **1.8e-04 on TPU at
    L=1024**, because the TPU transcendental unit is looser than CPU libm and the error
    compounds along the sequential path only. Injecting a 3e-6 relative error into
    ``exp`` on CPU reproduces ~7e-05, which brackets the observed value.

    So this is a numerics check between two formulations, not a correctness check --
    `stage_rotation` is the one that would catch a wrong recurrence -- and the threshold
    is set from that measurement. For comparison, upstream's own
    reference-vs-reference test uses ``rtol=1e-4`` at L=2048.
    """
    x = make_inputs(env)
    pre, _ = R.preprocess(
        x["dt_raw"], x["a_raw"], x["trap_raw"], x["angles"], x["dt_bias"]
    )
    y_rec, st_rec = R.siso_recurrent(
        x["q"], x["k"], x["v"], pre.adt, pre.dt, pre.lam, pre.phi,
        x["q_bias"], x["k_bias"], x["d_skip"], x["z"],
    )
    y_chk, st_chk = R.siso_chunked(
        x["q"], x["k"], x["v"], pre.adt, pre.gamma, pre.scale, pre.phi,
        x["q_bias"], x["k_bias"], x["d_skip"], x["z"], chunk=env.chunk,
    )
    # Scales with sequence length; 1e-5 is right at L=256 (CPU), 1e-3 leaves headroom
    # to L=8192 on TPU.
    tol = 1e-5 if not env.tpu else 1e-3
    rep.check("refs", "3-term scan vs chunked: y", rel_error(y_chk, y_rec), tol)
    rep.check("refs", "3-term scan vs chunked: ssm", rel_error(st_chk.ssm, st_rec.ssm), tol)
    rep.check("refs", "3-term scan vs chunked: k", rel_error(st_chk.k, st_rec.k), 1e-5)
    rep.check("refs", "3-term scan vs chunked: v", max_rel(st_chk.v, st_rec.v), 1e-6)


# --------------------------------------------------------------------------------------
# Stage 2b: rotational state tracking, analytically
# --------------------------------------------------------------------------------------


def stage_rotation(env: Env, rep: Report) -> None:
    """Compute parity exactly, with hand-built weights and no training.

    This is the check that the rotary path can express rotational dynamics -- the
    capability the complex/RoPE part of Mamba-3 exists for, and the thing the paper
    reports Mamba-2 cannot do at all. Doing it analytically instead of by gradient
    descent makes it deterministic, fast, and free of any optimization confound.

    Construction: let only token 0 write into the state (``k_t = 0`` for ``t > 0``) and
    set the decay to ~1, so ``h_t = khat_0 (x) v_0`` for every ``t``. With
    ``q = k = e_0`` the readout collapses to a single cosine,

        y_t = q~_t . h_t = cos(phi_t - phi_0) * v_0

    and choosing ``tanh(angle_t) * pi * dt_t = pi * bit_t`` makes ``phi_t`` advance by
    ``pi`` on each set bit, so ``y_t = (-1)^(running XOR)``. Sign accuracy must be
    exactly 1.0; a mis-paired rotation, a wrong permutation, or a dropped ``phi``
    accumulation all break it.
    """
    batch, nheads = 2, 1
    seqlen = 2 * env.chunk
    n, p, nr = 128, 64, 32

    bits = jax.random.bernoulli(jax.random.key(0), 0.5, (batch, seqlen)).astype(jnp.float32)
    bits = bits.at[:, 0].set(0.0)  # phi_0 = 0 anchors the reference frame
    parity = jnp.cumsum(bits, axis=1) % 2

    # tanh(angle) * pi * dt = pi * bit, with dt = 1.
    atanh_one = math.atanh(1 - 1e-6)
    angles = jnp.where(bits[:, :, None] > 0.5, atanh_one, 0.0) * jnp.ones((1, 1, nr))
    dt = jnp.ones((batch, nheads, seqlen))
    lam = jnp.ones((batch, nheads, seqlen))       # gamma = dt * lam = 1
    adt = jnp.full((batch, nheads, seqlen), -1e-6)  # alpha ~ 1: no decay
    gamma, scale = R.trapezoid_coeffs(dt, lam)
    phi, _ = R.cumulative_angles(
        jnp.repeat(angles[:, None, :, :], nheads, axis=1), dt
    )

    e0 = jnp.zeros((n,)).at[0].set(1.0)
    q = jnp.broadcast_to(e0, (batch, seqlen, n))
    k = jnp.zeros((batch, seqlen, n)).at[:, 0, :].set(e0)
    v = jnp.ones((batch, nheads, seqlen, p))
    qb = jnp.zeros((nheads, n))
    kb = jnp.zeros((nheads, n))
    want_sign = jnp.where(parity[:, None, :, None] > 0.5, -1.0, 1.0)

    y_ref, _ = R.siso_recurrent(q, k, v, adt, dt, lam, phi, qb, kb, None, None)
    y_ker, *_ = KF.siso_forward(
        q, k, v, adt, gamma, scale, phi, qb, kb, None, None,
        chunk=env.chunk, dtype_policy=L.F32, interpret=env.interpret,
    )

    for name, y in (("recurrent reference", y_ref), ("pallas kernel", y_ker)):
        acc = float(jnp.mean((jnp.sign(y) == jnp.sign(want_sign)).astype(jnp.float32)))
        rep.check("rotation", f"{name}: parity sign error", 1.0 - acc, 0.0)
        # The residual is bounded by the two epsilons above (atanh cutoff and the
        # 1e-6 decay), not by kernel error.
        rep.check(
            "rotation",
            f"{name}: |y - (-1)^parity|",
            float(jnp.max(jnp.abs(y - want_sign))),
            1e-3,
        )


# --------------------------------------------------------------------------------------
# Stage 3: forward parity
# --------------------------------------------------------------------------------------


def stage_forward(env: Env, rep: Report) -> None:
    """Pallas forward vs the chunked reference, in both dtype policies."""
    x = make_inputs(env)
    pre, _ = R.preprocess(
        x["dt_raw"], x["a_raw"], x["trap_raw"], x["angles"], x["dt_bias"]
    )
    y_ref, st_ref, dbg = R.siso_chunked(
        x["q"], x["k"], x["v"], pre.adt, pre.gamma, pre.scale, pre.phi,
        x["q_bias"], x["k_bias"], x["d_skip"], x["z"], chunk=env.chunk,
        return_debug=True,
    )

    for pol_name, tol in (("f32", 2e-5), ("bf16", 5e-2)):
        pol = L.policy(pol_name)
        cast = (lambda t: t.astype(jnp.bfloat16)) if pol_name == "bf16" else (lambda t: t)
        y, fs, fk, fv, res = KF.siso_forward(
            cast(x["q"]), cast(x["k"]), cast(x["v"]),
            pre.adt, pre.gamma, pre.scale, pre.phi,
            x["q_bias"], x["k_bias"], x["d_skip"], cast(x["z"]),
            chunk=env.chunk, dtype_policy=pol, save_residuals=True,
            interpret=env.interpret,
        )
        rep.check("forward", f"{pol_name} y", rel_error(y, y_ref), tol)
        rep.check("forward", f"{pol_name} final ssm", rel_error(fs, st_ref.ssm), tol)
        rep.check("forward", f"{pol_name} final k", rel_error(fk, st_ref.k), tol)
        rep.check("forward", f"{pol_name} final v", max_rel(fv, st_ref.v), tol)
        rep.check("forward", f"{pol_name} ybar residual", rel_error(res.ybar, dbg["ybar"]), tol)
        rep.check(
            "forward",
            f"{pol_name} chunk_states residual",
            rel_error(res.chunk_states, dbg["chunk_states"]),
            tol,
        )

    # The no-gate / no-skip path takes different branches in the kernel body.
    y_plain, *_ = KF.siso_forward(
        x["q"], x["k"], x["v"], pre.adt, pre.gamma, pre.scale, pre.phi,
        x["q_bias"], x["k_bias"], None, None,
        chunk=env.chunk, dtype_policy=L.F32, interpret=env.interpret,
    )
    y_plain_ref, _ = R.siso_chunked(
        x["q"], x["k"], x["v"], pre.adt, pre.gamma, pre.scale, pre.phi,
        x["q_bias"], x["k_bias"], None, None, chunk=env.chunk,
    )
    rep.check("forward", "no-gate no-skip y", rel_error(y_plain, y_plain_ref), 2e-5)


# --------------------------------------------------------------------------------------
# Stage 4: backward parity
# --------------------------------------------------------------------------------------


def stage_backward(env: Env, rep: Report) -> None:
    """Pallas VJP vs ``jax.grad`` of the chunked reference.

    Differentiating through ``preprocess`` too, so this covers the whole chain:
    ``dgamma``/``dscale`` back to ``dt``/``trap`` (including ``scale``'s ``t+1``
    coupling) and ``dphi`` back to ``angles``.
    """
    x = make_inputs(env)
    names = [
        "q", "k", "v", "dt_raw", "a_raw", "trap_raw", "angles",
        "dt_bias", "q_bias", "k_bias", "d_skip", "z",
    ]
    primals = tuple(x[n] for n in names)
    dy = x["dy"]

    def kernel_loss(*args):
        out = SISO.siso_segment(
            *args[:10], d_skip=args[10], z=args[11],
            chunk=env.chunk, policy_name="f32", interpret=env.interpret,
        )
        return jnp.sum(out.y * dy)

    def ref_loss(*args):
        pre, _ = R.preprocess(args[3], args[4], args[5], args[6], args[7])
        y, _ = R.siso_chunked(
            args[0], args[1], args[2], pre.adt, pre.gamma, pre.scale, pre.phi,
            args[8], args[9], args[10], args[11], chunk=env.chunk,
        )
        return jnp.sum(y * dy)

    argnums = tuple(range(len(names)))
    try:
        got = jax.grad(kernel_loss, argnums=argnums)(*primals)
        want = jax.grad(ref_loss, argnums=argnums)(*primals)
    except Exception as exc:  # noqa: BLE001
        rep.fail("backward", "vjp runs", exc)
        return

    for name, g, w in zip(names, got, want):
        rep.check("backward", f"d{name}", rel_error(g, w), 2e-4)

    # Finite differences against the hand-written VJP. The reference comparison above
    # could in principle be fooled if both sides shared a mistake; this cannot, since
    # it only evaluates the forward. Small and f32, because central differences on a
    # bf16 kernel would drown in rounding.
    if not env.tpu:
        from jax import test_util as jtu  # not re-exported as jax.test_util in 0.11

        tiny = Env(tpu=False, chunk=128, policy="f32")
        tiny.batch, tiny.nheads, tiny.seqlen = 1, 1, 128
        xs = make_inputs(tiny, seed=3)

        def f(q, v, dt_raw):
            out = SISO.siso_segment(
                q, xs["k"], v, dt_raw, xs["a_raw"], xs["trap_raw"], xs["angles"],
                xs["dt_bias"], xs["q_bias"], xs["k_bias"],
                d_skip=xs["d_skip"], z=None,
                chunk=128, policy_name="f32", interpret=env.interpret,
            )
            return jnp.sum(out.y**2)

        try:
            # eps=1e-3, not the 1e-4 default: at 1e-4 the central difference of an
            # f32 loss of order 1e4 is dominated by rounding, and the *pure-JAX
            # reference* fails this check by the same 8.3% as the kernel does. The
            # step size is the thing being tested there, not the gradient.
            jtu.check_grads(
                f, (xs["q"], xs["v"], xs["dt_raw"]), order=1, modes=("rev",),
                atol=2e-2, rtol=2e-2, eps=1e-3,
            )
            rep.note("backward", "check_grads(order=1) vs finite differences")
        except Exception as exc:  # noqa: BLE001
            rep.fail("backward", "check_grads(order=1)", exc)


# --------------------------------------------------------------------------------------
# Stage 5: state passing
# --------------------------------------------------------------------------------------


def stage_segments(env: Env, rep: Report) -> None:
    """Split the sequence, chain the carry, compare against one call.

    The check that catches beta-seeding errors. Note each segment preprocesses its
    *own* raw inputs: ``scale`` reaches one position forward, so slicing a
    whole-sequence ``scale`` would give a segment's last token a successor term that
    belongs to the next segment, and then double-count it via the carried ``(k, v)``.
    """
    x = make_inputs(env)
    half = env.seqlen // 2
    if half % env.chunk:
        rep.note("segments", f"skipped: seqlen/2={half} not a multiple of {env.chunk}")
        return

    def run(sl, state=None, policy="f32"):
        return SISO.siso_segment(
            x["q"][:, sl], x["k"][:, sl], x["v"][:, :, sl],
            x["dt_raw"][:, sl], x["a_raw"][:, sl], x["trap_raw"][:, sl],
            x["angles"][:, sl], x["dt_bias"], x["q_bias"], x["k_bias"],
            d_skip=x["d_skip"], z=x["z"][:, :, sl], state=state,
            chunk=env.chunk, policy_name=policy, interpret=env.interpret,
        )

    full = run(slice(None))
    o1 = run(slice(0, half))
    o2 = run(slice(half, env.seqlen), state=o1.state)
    y_split = jnp.concatenate([o1.y, o2.y], axis=2)

    rep.check("segments", "2 segments vs 1 call: y", rel_error(y_split, full.y), 2e-5)
    rep.check(
        "segments", "2 segments vs 1 call: ssm", rel_error(o2.state.ssm, full.state.ssm), 2e-5
    )
    rep.check(
        "segments", "2 segments vs 1 call: k", rel_error(o2.state.k, full.state.k), 2e-5
    )
    rep.check(
        "segments",
        "2 segments vs 1 call: phi",
        rel_error(o2.state.phi, full.state.phi, angle=True),
        1e-3,
    )

    # Gradients must also flow correctly through the chain.
    def chained_loss(q, v):
        a = SISO.siso_segment(
            q[:, :half], x["k"][:, :half], v[:, :, :half],
            x["dt_raw"][:, :half], x["a_raw"][:, :half], x["trap_raw"][:, :half],
            x["angles"][:, :half], x["dt_bias"], x["q_bias"], x["k_bias"],
            d_skip=x["d_skip"], z=x["z"][:, :, :half],
            chunk=env.chunk, policy_name="f32", interpret=env.interpret,
        )
        b = SISO.siso_segment(
            q[:, half:], x["k"][:, half:], v[:, :, half:],
            x["dt_raw"][:, half:], x["a_raw"][:, half:], x["trap_raw"][:, half:],
            x["angles"][:, half:], x["dt_bias"], x["q_bias"], x["k_bias"],
            d_skip=x["d_skip"], z=x["z"][:, :, half:], state=a.state,
            chunk=env.chunk, policy_name="f32", interpret=env.interpret,
        )
        return jnp.sum(jnp.concatenate([a.y, b.y], axis=2) * x["dy"])

    def single_loss(q, v):
        pre, _ = R.preprocess(
            x["dt_raw"], x["a_raw"], x["trap_raw"], x["angles"], x["dt_bias"]
        )
        y, _ = R.siso_chunked(
            q, x["k"], v, pre.adt, pre.gamma, pre.scale, pre.phi,
            x["q_bias"], x["k_bias"], x["d_skip"], x["z"], chunk=env.chunk,
        )
        return jnp.sum(y * x["dy"])

    gq, gv = jax.grad(chained_loss, argnums=(0, 1))(x["q"], x["v"])
    rq, rv = jax.grad(single_loss, argnums=(0, 1))(x["q"], x["v"])
    rep.check("segments", "chained grad dq", rel_error(gq, rq), 2e-4)
    rep.check("segments", "chained grad dv", rel_error(gv, rv), 2e-4)


# --------------------------------------------------------------------------------------
# Stage 6: decode
# --------------------------------------------------------------------------------------


def stage_decode(env: Env, rep: Report) -> None:
    """Decode chain vs prefill, plus the prefill-then-decode handoff.

    Interpret mode dispatches one simulated kernel per token, so the local run uses a
    short chain; on TPU it uses the full sequence.
    """
    dec_env = Env(tpu=env.tpu, chunk=env.chunk, policy=env.policy)
    if not env.tpu:
        # One simulated kernel launch per token, and interpret mode is ~1000x slower
        # than hardware, so the local chain is deliberately tiny: 2 * chunk tokens at
        # B=1, H=2 is ~5 min. Anything larger is not worth the wall clock -- the same
        # stage on TPU runs the full sequence in seconds.
        dec_env.batch, dec_env.nheads, dec_env.seqlen = 1, 2, 2 * env.chunk
    x = make_inputs(dec_env, seed=11)
    pre, _ = R.preprocess(
        x["dt_raw"], x["a_raw"], x["trap_raw"], x["angles"], x["dt_bias"]
    )
    b, h = dec_env.batch, dec_env.nheads
    p, n, nr = dec_env.headdim, dec_env.state_dim, dec_env.n_angles
    zero = R.zero_state(b, h, p, n, nr)

    # One step against the pure-JAX step reference.
    y1, s1 = KD.siso_decode(
        x["q"][:, 0], x["k"][:, 0], x["v"][:, :, 0],
        pre.adt[:, :, 0], pre.dt[:, :, 0], pre.lam[:, :, 0], x["angles"][:, 0],
        x["q_bias"], x["k_bias"], zero, x["d_skip"], x["z"][:, :, 0],
        dtype_policy=L.F32, interpret=env.interpret,
    )
    yr, sr = R.siso_step(
        x["q"][:, 0], x["k"][:, 0], x["v"][:, :, 0],
        pre.adt[:, :, 0], pre.dt[:, :, 0], pre.lam[:, :, 0], x["angles"][:, 0],
        x["q_bias"], x["k_bias"], zero, x["d_skip"], x["z"][:, :, 0],
    )
    rep.check("decode", "1 step vs step ref: y", rel_error(y1, yr), 2e-5)
    rep.check("decode", "1 step vs step ref: ssm", rel_error(s1.ssm, sr.ssm), 2e-5)
    rep.check("decode", "1 step vs step ref: phi", rel_error(s1.phi, sr.phi, angle=True), 1e-4)

    # The folded kernel, both readout units, from a *non-zero* state: the head axis
    # lives in the block there, so a mis-broadcast of the head-shared q/k/angles
    # against the per-head phi/dt would be invisible from a zero state.
    seed_state = R.SISOState(
        ssm=jax.random.normal(jax.random.key(31), zero.ssm.shape, jnp.float32),
        k=jax.random.normal(jax.random.key(32), zero.k.shape, jnp.float32),
        v=jax.random.normal(jax.random.key(33), zero.v.shape, jnp.float32),
        phi=jax.random.uniform(jax.random.key(34), zero.phi.shape, jnp.float32) * 6.0,
    )
    step_args = (
        x["q"][:, 0], x["k"][:, 0], x["v"][:, :, 0],
        pre.adt[:, :, 0], pre.dt[:, :, 0], pre.lam[:, :, 0], x["angles"][:, 0],
        x["q_bias"], x["k_bias"], seed_state, x["d_skip"], x["z"][:, :, 0],
    )
    y_ph, s_ph = KD.siso_decode(
        *step_args, dtype_policy=L.F32, interpret=env.interpret
    )
    y_sr, s_sr = R.siso_step(*step_args)
    rep.check("decode", "per-head seeded vs step ref", rel_error(y_ph, y_sr), 2e-5)
    y_f, s_f = KD.siso_decode_folded(
        *step_args, dtype_policy=L.F32, interpret=env.interpret
    )
    rep.check("decode", "folded vs step ref: y", rel_error(y_f, y_sr), 2e-5)
    rep.check("decode", "folded vs step ref: ssm", rel_error(s_f.ssm, s_sr.ssm), 2e-5)
    rep.check(
        "decode", "folded vs step ref: phi",
        rel_error(s_f.phi, s_sr.phi, angle=True), 1e-4,
    )
    rep.check("decode", "folded vs per-head: y", rel_error(y_f, y_ph), 2e-5)

    # `decode_scan` is one lax.scan over the same steps, so it has to agree with the
    # Python loop it replaces. Short chain locally: interpret mode dispatches a simulated
    # kernel per token and this is a second chain on top of the full one below.
    scan_n = dec_env.seqlen if env.tpu else 4
    scan_args = (
        x["q"][:, :scan_n], x["k"][:, :scan_n], x["v"][:, :, :scan_n],
        pre.adt[:, :, :scan_n], pre.dt[:, :, :scan_n], pre.lam[:, :, :scan_n],
        x["angles"][:, :scan_n], x["q_bias"], x["k_bias"], zero,
        x["d_skip"], x["z"][:, :, :scan_n],
    )
    y_ref_loop, s_ref_loop = KD.decode_loop(
        *scan_args, dtype_policy=L.F32, interpret=env.interpret
    )
    for folded in (False, True):
        y_sc, s_sc = KD.decode_scan(
            *scan_args, dtype_policy=L.F32, interpret=env.interpret, folded=folded
        )
        tag = "folded" if folded else "per-head"
        rep.check(
            "decode", f"scan/{tag} vs loop: y", rel_error(y_sc, y_ref_loop), 2e-5
        )
        rep.check(
            "decode", f"scan/{tag} vs loop: ssm",
            rel_error(s_sc.ssm, s_ref_loop.ssm), 2e-5,
        )

    # Full chain from a zero state vs one prefill call.
    y_dec, s_dec = KD.decode_loop(
        x["q"], x["k"], x["v"], pre.adt, pre.dt, pre.lam, x["angles"],
        x["q_bias"], x["k_bias"], zero, x["d_skip"], x["z"],
        dtype_policy=L.F32, interpret=env.interpret,
    )
    y_pre, fs, fk, fv = KF.siso_forward(
        x["q"], x["k"], x["v"], pre.adt, pre.gamma, pre.scale, pre.phi,
        x["q_bias"], x["k_bias"], x["d_skip"], x["z"],
        chunk=env.chunk, dtype_policy=L.F32, interpret=env.interpret,
    )
    # Same two-formulation gap as `stage_refs`: the decode chain does L sequential
    # exp() multiplies while prefill does one exp2(cumsum) per chunk, and the TPU's
    # transcendental unit makes that visible. 1.8e-04 measured at L=2048 on v5e.
    chain_tol = 2e-5 if not env.tpu else 1e-3
    rep.check("decode", "chain vs prefill: y", rel_error(y_dec, y_pre), chain_tol)
    rep.check("decode", "chain vs prefill: ssm", rel_error(s_dec.ssm, fs), chain_tol)
    rep.check("decode", "chain vs prefill: k", rel_error(s_dec.k, fk), 2e-5)

    # Prefill half, decode the rest: the handoff that matters in production.
    half = dec_env.seqlen // 2
    if half % env.chunk == 0:
        pre1, _ = R.preprocess(
            x["dt_raw"][:, :half], x["a_raw"][:, :half], x["trap_raw"][:, :half],
            x["angles"][:, :half], x["dt_bias"],
        )
        y_a, ssm_a, k_a, v_a = KF.siso_forward(
            x["q"][:, :half], x["k"][:, :half], x["v"][:, :, :half],
            pre1.adt, pre1.gamma, pre1.scale, pre1.phi,
            x["q_bias"], x["k_bias"], x["d_skip"], x["z"][:, :, :half],
            chunk=env.chunk, dtype_policy=L.F32, interpret=env.interpret,
        )
        carry = R.SISOState(ssm=ssm_a, k=k_a, v=v_a, phi=pre1.phi[:, :, -1, :])
        pre2, _ = R.preprocess(
            x["dt_raw"][:, half:], x["a_raw"][:, half:], x["trap_raw"][:, half:],
            x["angles"][:, half:], x["dt_bias"], phi_init=carry.phi,
        )
        y_b, _ = KD.decode_loop(
            x["q"][:, half:], x["k"][:, half:], x["v"][:, :, half:],
            pre2.adt, pre2.dt, pre2.lam, x["angles"][:, half:],
            x["q_bias"], x["k_bias"], carry, x["d_skip"], x["z"][:, :, half:],
            dtype_policy=L.F32, interpret=env.interpret,
        )
        rep.check(
            "decode",
            "prefill->decode handoff",
            rel_error(jnp.concatenate([y_a, y_b], axis=2), y_pre),
            chain_tol,
        )
    else:
        rep.note("decode", f"handoff skipped: half={half} not a multiple of {env.chunk}")


# --------------------------------------------------------------------------------------
# Stage 7: cross-framework parity
# --------------------------------------------------------------------------------------


def stage_torch(env: Env, rep: Report) -> None:
    """PyTorch reference on CPU vs the JAX/Pallas layer, through ``convert.py``.

    This is what validates the parts no self-consistency check can: the ``in_proj``
    split order, the rotary permutation, bias/norm placement, and the C-is-query /
    B-is-key assignment. Skipped if torch is unavailable.
    """
    try:
        import torch

        from . import torch_ref as TR
    except ImportError as exc:
        rep.note("torch", f"skipped: {exc}")
        return

    torch.manual_seed(0)
    d_model = 256
    module = TR.Mamba3SISOTorch(d_model=d_model, d_state=128, headdim=64)
    sd = module.state_dict()

    cfg = CV.config_from_state_dict(sd, headdim=64, chunk=env.chunk, policy_name="f32")
    params = CV.torch_to_jax(sd, cfg)

    # The permutation must be exactly invertible, or checkpoints drift on export.
    back = CV.jax_to_torch(params, cfg)
    worst = max(
        float(np.max(np.abs(back[key] - np.asarray(sd[key].detach(), np.float32))))
        for key in back
        if key in sd
    )
    rep.check("torch", "state_dict round-trip", worst, 1e-12)

    u = np.asarray(
        np.random.default_rng(0).normal(size=(2, 256, d_model)) * 0.5, np.float32
    )
    with torch.no_grad():
        y_torch = module(torch.from_numpy(u)).numpy()
    y_jax, _ = LY.mamba3_siso_layer(
        params, jnp.asarray(u), cfg, interpret=env.interpret
    )
    rep.check("torch", "torch vs jax layer forward", rel_error(y_jax, y_torch), 5e-2)
    rep.check("torch", "torch vs jax layer (max rel)", max_rel(y_jax, y_torch), 1e-3)


# --------------------------------------------------------------------------------------
# Stage 7b: checkpoint round-trip
# --------------------------------------------------------------------------------------


def stage_checkpoint(env: Env, rep: Report) -> None:
    """`checkpoint.save` then `checkpoint.load` must return the same model.

    Cheap and pure host code, no kernel involved, but worth asserting: a checkpoint that
    silently drops a leaf or reassociates the ``blocks`` list would produce a model that
    loads fine and generates garbage. Checks the tie_head and untied trees separately
    since they differ by one array, and checks that bf16 storage halves the file and
    survives a widening load.
    """
    import os
    import tempfile

    from . import checkpoint as CP
    from . import model as M

    siso = LY.SISOConfig(
        d_model=128, d_state=128, headdim=64, chunk=env.chunk, policy_name="f32"
    )
    with tempfile.TemporaryDirectory() as d:
        for tie in (True, False):
            cfg = M.LMConfig(vocab_size=256, n_layers=2, siso=siso, tie_head=tie)
            params = M.init_lm(cfg, jax.random.key(3))
            path = CP.save(os.path.join(d, f"t{tie}.npz"), params, cfg)
            got, got_cfg = CP.load(path)
            tag = "tied" if tie else "untied"
            rep.check(
                "checkpoint", f"{tag}: config round-trip",
                0.0 if got_cfg == cfg else 1.0, 1e-12,
            )
            rep.check(
                "checkpoint", f"{tag}: tree structure",
                0.0 if jax.tree.structure(got) == jax.tree.structure(params) else 1.0,
                1e-12,
            )
            worst = max(
                float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
                for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(got))
            )
            rep.check("checkpoint", f"{tag}: weights bit-exact", worst, 0.0)

        # Same logits out of the loaded model as the saved one. This is the property that
        # actually matters and it catches a permuted-but-complete tree, which the
        # per-leaf check above would pass.
        cfg = M.LMConfig(vocab_size=256, n_layers=2, siso=siso, tie_head=True)
        params = M.init_lm(cfg, jax.random.key(4))
        path = CP.save(os.path.join(d, "logits.npz"), params, cfg)
        got, got_cfg = CP.load(path)
        tokens = jnp.asarray(
            np.random.default_rng(0).integers(0, 256, (1, env.chunk)), jnp.int32
        )
        a = M.lm_forward(params, tokens, cfg, interpret=env.interpret)
        b = M.lm_forward(got, tokens, got_cfg, interpret=env.interpret)
        rep.check("checkpoint", "identical logits after reload", max_rel(b, a), 0.0)

        # bfloat16 storage: half the bytes, and a widening load must give back a usable
        # float32 tree rather than the opaque void dtype npz stores bf16 as.
        f32 = CP.save(os.path.join(d, "f32.npz"), params, cfg)
        bf16 = CP.save(os.path.join(d, "bf16.npz"), params, cfg, dtype=jnp.bfloat16)
        ratio = os.path.getsize(bf16) / os.path.getsize(f32)
        rep.check("checkpoint", "bf16 file is half of f32", abs(ratio - 0.5), 0.02)
        wide, _ = CP.load(bf16, dtype=jnp.float32)
        rep.check(
            "checkpoint", "bf16 load widens to f32",
            0.0 if np.asarray(wide.embed).dtype == np.float32 else 1.0, 1e-12,
        )
        rep.check(
            "checkpoint", "bf16 weights within bf16 epsilon",
            max_rel(wide.embed, params.embed), 1e-2,
        )

        # A corrupt file must raise, not load a wrong model.
        with np.load(f32, allow_pickle=False) as z:
            entries = {k: z[k] for k in z.files}
        del entries["blocks.1.in_proj"]
        broken = os.path.join(d, "broken.npz")
        np.savez(broken, **entries)
        raised = 0.0
        try:
            CP.load(broken)
        except ValueError:
            raised = 1.0
        rep.check("checkpoint", "missing array raises", 1.0 - raised, 1e-12)



# --------------------------------------------------------------------------------------
# Stage 8: training behaviour
# --------------------------------------------------------------------------------------


def stage_train(env: Env, rep: Report) -> None:
    """Two learning checks: can it memorize, and can it do parity.

    Memorizing a fixed token sequence proves the gradients are usable end to end and
    runs anywhere. Parity is the sharper test -- ``h_t = R(pi x_t) h_{t-1}`` needs
    genuine rotational dynamics, and the paper's Table 5(b) puts Mamba-2 at 0.90% (chance)
    against Mamba-3 at 100% -- but it needs hundreds of optimizer steps, which interpret
    mode cannot deliver (~1 s/step on CPU, and 60 steps only reaches 0.54
    accuracy, i.e. still indistinguishable from chance). So parity is TPU-only.
    """
    try:
        import optax
    except ImportError as exc:
        rep.note("train", f"skipped: {exc}")
        return

    cfg = LY.SISOConfig(d_model=128, d_state=128, headdim=64, chunk=env.chunk,
                        policy_name="f32")

    # ---- memorization -----------------------------------------------------------
    vocab, seqlen = 17, env.chunk
    keys = jax.random.split(jax.random.key(3), 4)
    tokens = jax.random.randint(keys[0], (4, seqlen + 1), 0, vocab)
    embed = jax.random.normal(keys[1], (vocab, cfg.d_model)) * 0.3
    head = jax.random.normal(keys[2], (cfg.d_model, vocab)) * 0.1
    params = LY.init_params(cfg, keys[3])

    def lm_loss(tree):
        p, h = tree
        out, _ = LY.mamba3_siso_layer(p, embed[tokens[:, :-1]], cfg,
                                      interpret=env.interpret)
        return optax.softmax_cross_entropy_with_integer_labels(
            out @ h, tokens[:, 1:]
        ).mean()

    tree = (params, head)
    opt = optax.adam(3e-3)
    opt_state = opt.init(tree)
    step = jax.jit(jax.value_and_grad(lm_loss))
    first = last = None
    for i in range(40):
        loss, grads = step(tree)
        updates, opt_state = opt.update(grads, opt_state, tree)
        tree = optax.apply_updates(tree, updates)
        if i == 0:
            first = float(loss)
        last = float(loss)
    print(
        f"    ..   memorization loss {first:.3f} -> {last:.3f} "
        f"(ln {vocab} = {math.log(vocab):.3f})"
    )
    rep.check("train", "memorization loss ratio", last / first, 0.25)

    if not env.tpu:
        rep.note("train", "parity task skipped (needs TPU; too slow to train in interpret mode)")
        return

    # ---- parity, by gradient descent --------------------------------------------
    # Reported, not asserted. `stage_rotation` already proves the kernel *can*
    # represent parity -- it builds the weights analytically and gets exactly 1.000
    # sign accuracy. Whether Adam *finds* those weights from this initialization is a
    # separate, optimization-shaped question, and measured here it does not: the
    # pure-JAX reference (no Pallas at all) also sits at 0.502 after 600 steps, both
    # with dt_bias at its default and with dt_bias retuned so softplus(dt_bias) ~ 1.
    # Default init gives dt in [0.006, 0.020], so tanh(angle) * pi * dt reaches ~0.06
    # rad per token -- two orders short of the pi it needs -- and the paper trains
    # this task inside a full model with a tuned recipe, not one layer plus a linear
    # head. Asserting on it would be asserting on the optimizer.
    pkeys = jax.random.split(jax.random.key(5), 3)
    bits = jax.random.bernoulli(pkeys[0], 0.5, (64, seqlen)).astype(jnp.int32)
    labels = jnp.cumsum(bits, axis=1) % 2
    p_embed = jax.random.normal(pkeys[1], (2, cfg.d_model)) * 0.5
    p_head = jax.random.normal(pkeys[2], (cfg.d_model, 2)) * 0.1
    p_params = LY.init_params(cfg, jax.random.key(6))

    def parity_out(tree):
        p, h = tree
        out, _ = LY.mamba3_siso_layer(p, p_embed[bits], cfg, interpret=env.interpret)
        return out @ h

    def parity_loss(tree):
        return optax.softmax_cross_entropy_with_integer_labels(
            parity_out(tree), labels
        ).mean()

    tree = (p_params, p_head)
    opt = optax.adam(3e-3)
    opt_state = opt.init(tree)
    step = jax.jit(jax.value_and_grad(parity_loss))
    acc_fn = jax.jit(
        lambda t: jnp.mean((jnp.argmax(parity_out(t), axis=-1) == labels).astype(jnp.float32))
    )

    n_steps = 1500
    best = 0.0
    for i in range(n_steps):
        loss, grads = step(tree)
        updates, opt_state = opt.update(grads, opt_state, tree)
        tree = optax.apply_updates(tree, updates)
        if (i + 1) % 250 == 0:
            acc = float(acc_fn(tree))
            best = max(best, acc)
            print(f"    ..   parity step {i + 1:4d}  loss {float(loss):.4f}  acc {acc:.3f}")
    print(f"    ..   best parity accuracy {best:.3f} (chance 0.5; informational only)")
    rep.note("train", f"parity by SGD reached {best:.3f} (not asserted; see stage rotation)")


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

STAGES: dict[str, Callable[[Env, Report], None]] = {
    "lower": stage_lower,
    "shapes": stage_shapes,
    "refs": stage_refs,
    "rotation": stage_rotation,
    "forward": stage_forward,
    "backward": stage_backward,
    "segments": stage_segments,
    "decode": stage_decode,
    "torch": stage_torch,
    "checkpoint": stage_checkpoint,
    "train": stage_train,
}


def main(argv: list[str] | None = None) -> int:
    """Run the suite. Also callable in-process: ``tests.main(["--tpu"])``."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--stage", action="append", choices=sorted(STAGES), default=None,
        help="run only these stages (repeatable); default is all",
    )
    ap.add_argument("--tpu", action="store_true", help="full shapes, interpret off")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=None)
    args = ap.parse_args(argv)

    if args.tpu and not L.on_tpu():
        print(L.no_tpu_message())
        return 2

    env = Env(tpu=args.tpu, chunk=args.chunk)
    if args.seqlen is not None:
        env.seqlen = args.seqlen

    print("=" * 74)
    print(L.describe_environment())
    print(env)
    print("=" * 74)

    rep = Report()
    for name in args.stage or list(STAGES):
        print(f"\n[{name}]")
        started = time.time()
        try:
            STAGES[name](env, rep)
        except Exception as exc:  # noqa: BLE001
            rep.fail(name, "stage raised", exc)
        print(f"    ({time.time() - started:.1f}s)")

    return rep.summary()


if __name__ == "__main__":
    sys.exit(main())
