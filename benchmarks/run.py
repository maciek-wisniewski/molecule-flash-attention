"""Time, peak memory and accuracy of molflash against the baselines.

    python benchmarks/run.py speed --scenario complex --n 1024 4096 16384 --out results.jsonl
    python benchmarks/run.py accuracy --out results.jsonl
    python benchmarks/run.py primitives --n 8192 --out results.jsonl

Every record is one JSON line; benchmarks/report.py turns a file of them into the tables and
figures in docs/benchmarks.md.
"""

import argparse
import json
import statistics
import time

import torch
from baselines import METHODS
from scenarios import SCENARIOS

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# Every method is timed by repeating backward through one graph (retain_graph=True), which
# compiled backwards with donated buffers refuse.
import torch._functorch.config  # noqa: E402

torch._functorch.config.donated_buffer = False


def _gpu():
    return torch.cuda.get_device_name()


def _setup(scenario, b, n, heads, head_dim, dtype, seed=0):
    torch.manual_seed(seed)
    sc = SCENARIOS[scenario](b, n, heads, "cuda")
    q, k, v = (
        torch.randn(b, heads, n, head_dim, device="cuda", dtype=dtype, requires_grad=True)
        for _ in range(3)
    )
    return sc, q, k, v


def _median_ms(fn, reps):
    times = []
    for _ in range(reps):
        a, z = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        z.record()
        times.append((a, z))
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(z) for a, z in times)


def measure(method, scenario, b, n, heads, head_dim, dtype, warmup=2, reps=10):
    """``{"fwd_ms", "fwdbwd_ms", "peak_gb", "first_s"}`` or ``{"error": ...}``."""
    fn = METHODS[method]
    rec = {}
    try:
        sc, q, k, v = _setup(scenario, b, n, heads, head_dim, dtype)
        kw = {} if method == "sdpa (no bias)" else sc.kwargs()
        inputs = [q, k, v, *sc.params.values()]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        o = fn(q, k, v, kw)
        go = torch.randn_like(o)
        o.backward(go, retain_graph=True)
        torch.cuda.synchronize()
        rec["first_s"] = time.perf_counter() - t0

        def fwd():
            with torch.no_grad():
                fn(q, k, v, kw)

        def fwdbwd():
            for x in inputs:
                x.grad = None
            fn(q, k, v, kw).backward(go, retain_graph=True)

        for _ in range(warmup):
            fwdbwd()
        rec["fwd_ms"] = _median_ms(fwd, reps)
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        rec["fwdbwd_ms"] = _median_ms(fwdbwd, reps)
        rec["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
        rec["extra_gb"] = (torch.cuda.max_memory_allocated() - base) / 2**30
    except torch.OutOfMemoryError:
        rec = {"error": "OOM"}
    except Exception as e:  # noqa: BLE001 - a baseline that cannot run is a result, not a crash
        oom = "out of memory" in str(e).lower()
        rec = {"error": "OOM" if oom else f"unsupported: {type(e).__name__}: {str(e)[:300]}"}
    finally:
        torch.cuda.empty_cache()
    return rec


def _emit(out, rec):
    line = json.dumps(rec)
    print(line, flush=True)
    if out:
        with open(out, "a") as f:
            f.write(line + "\n")


def cmd_speed(a):
    skip = set()
    for scenario in a.scenario:
        for n in a.n:
            for method in a.methods:
                base = dict(
                    kind="speed",
                    gpu=_gpu(),
                    scenario=scenario,
                    method=method,
                    n=n,
                    batch=a.batch,
                    heads=a.heads,
                    head_dim=a.head_dim,
                    dtype=a.dtype,
                )
                if (scenario, method) in skip:
                    _emit(a.out, {**base, "error": "skipped after OOM or timeout at smaller n"})
                    continue
                rec = measure(
                    method, scenario, a.batch, n, a.heads, a.head_dim, DTYPES[a.dtype], reps=a.reps
                )
                if "error" in rec or rec.get("fwdbwd_ms", 0) > a.max_ms:
                    skip.add((scenario, method))
                _emit(a.out, {**base, **rec})


def cmd_accuracy(a):
    """Error of every method against fp32 eager on the same rounded inputs."""
    for scenario in a.scenario:
        for dtype in ("bf16", "fp16", "fp32"):
            sc, q, k, v = _setup(scenario, 1, a.n[0], a.heads, a.head_dim, DTYPES[dtype])
            kw = sc.kwargs()
            qf, kf, vf = (x.detach().float().requires_grad_(True) for x in (q, k, v))
            ref = METHODS["eager"](qf, kf, vf, kw)
            go = torch.randn_like(ref)
            want = torch.autograd.grad(
                ref, [qf, kf, vf, *sc.params.values()], go, retain_graph=True
            )
            for method in a.methods:
                if method in ("eager", "sdpa (no bias)"):
                    continue
                try:
                    o = METHODS[method](q, k, v, kw)
                    got = torch.autograd.grad(
                        o,
                        [q, k, v, *sc.params.values()],
                        go.to(o.dtype),
                        retain_graph=True,
                        allow_unused=True,
                    )
                except Exception as e:  # noqa: BLE001
                    _emit(
                        a.out,
                        dict(
                            kind="accuracy",
                            scenario=scenario,
                            method=method,
                            dtype=dtype,
                            error=str(e)[:200],
                        ),
                    )
                    continue
                names = ["out", "dq", "dk", "dv", *sc.params]
                errs = {}
                for name, x, y in zip(names, [o, *got], [ref, *want], strict=True):
                    if x is None:  # e.g. cuEquivariance derives q, k, v from one input
                        continue
                    errs[name] = (
                        (x.float() - y).abs().max() / y.abs().max().clamp(min=1e-6)
                    ).item()
                _emit(
                    a.out,
                    dict(
                        kind="accuracy",
                        gpu=_gpu(),
                        scenario=scenario,
                        method=method,
                        dtype=dtype,
                        n=a.n[0],
                        rel_err=errs,
                    ),
                )


def cmd_primitives(a):
    """Cost of each primitive alone, relative to the same kernel with no terms."""
    import molflash as mf
    from molflash import terms

    n, h, d = a.n[0], a.heads, a.head_dim
    torch.manual_seed(0)
    from scenarios import _chain_bonds, coordinates

    pos = coordinates(1, n, "cuda")
    index, types = _chain_bonds(1, n, "cuda")
    cases = {
        "none": lambda p: {},
        "EdgeBias (4 slots)": lambda p: dict(bias=terms.bonds(p["bond"], index, types)),
        "NodeBias (rank 8)": lambda p: dict(bias=mf.NodeBias(p["u"], p["w"])),
        "PairBias code": lambda p: dict(
            bias=terms.distance_buckets(p["code"], pos, max_distance=16.0)
        ),
        "PairBias interpolated": lambda p: dict(
            bias=terms.distance_function(p["curve"], pos, grid=(0.0, 16.0))
        ),
        "PairMask (15 A)": lambda p: dict(mask=terms.distance_cutoff(pos, cutoff=15.0)),
        "PairScale": lambda p: dict(
            multiplier=mf.PairScale(
                terms._distance_fn(), pos, p["curve"].detach().abs(), grid=(0.0, 16.0)
            )
        ),
        "PairGate": lambda p: dict(gate=terms.cosine_cutoff(pos, cutoff=15.0)),
    }
    for name, make in cases.items():
        params = {
            "bond": torch.randn(5, h, device="cuda", requires_grad=True),
            "u": torch.randn(1, h, n, 8, device="cuda", requires_grad=True),
            "w": torch.randn(1, h, n, 8, device="cuda", requires_grad=True),
            "code": torch.randn(h, 32, device="cuda", requires_grad=True),
            "curve": torch.randn(h, 64, device="cuda", requires_grad=True),
        }
        q, k, v = (
            torch.randn(1, h, n, d, device="cuda", dtype=DTYPES[a.dtype], requires_grad=True)
            for _ in range(3)
        )
        kw = make(params)
        o = mf.attention(q, k, v, **kw)
        go = torch.randn_like(o)

        def fwd(q=q, k=k, v=v, kw=kw):
            with torch.no_grad():
                mf.attention(q, k, v, **kw)

        def fwdbwd(q=q, k=k, v=v, kw=kw, go=go):
            mf.attention(q, k, v, **kw).backward(go, retain_graph=True)

        fwdbwd()
        _emit(
            a.out,
            dict(
                kind="primitive",
                gpu=_gpu(),
                primitive=name,
                n=n,
                heads=h,
                head_dim=d,
                dtype=a.dtype,
                fwd_ms=_median_ms(fwd, a.reps),
                fwdbwd_ms=_median_ms(fwdbwd, a.reps),
            ),
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["speed", "accuracy", "primitives"])
    p.add_argument("--scenario", nargs="+", default=list(SCENARIOS))
    p.add_argument("--methods", nargs="+", default=list(METHODS))
    p.add_argument("--n", nargs="+", type=int, default=[1024, 2048, 4096, 8192, 16384, 32768])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    p.add_argument("--reps", type=int, default=10)
    p.add_argument(
        "--max-ms",
        type=float,
        default=30_000.0,
        help="stop growing n for a method once one step takes longer than this",
    )
    p.add_argument("--out")
    a = p.parse_args()
    {
        "speed": cmd_speed,
                "accuracy": cmd_accuracy,
        "primitives": cmd_primitives,
    }[a.command](a)


if __name__ == "__main__":
    main()
