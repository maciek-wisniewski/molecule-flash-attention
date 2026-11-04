"""Forward and forward+backward time of one attention call with a single term, per kind of term.

    python benchmarks/ablate_terms.py > ablate_terms.jsonl

N = 8192, 8 heads of 64, bfloat16, batch 1. "frozen" tables do not require grad, "learned" ones
do. Prints one JSON line per case, times in ms (median of 10 CUDA-event-timed calls).
"""

import json
import statistics

import molflash as mf
import torch
from molflash import terms

torch.manual_seed(0)
dev = "cuda"
N, H, D = 8192, 8, 64
gpu = torch.cuda.get_device_name()
q, k, v = (
    torch.randn(1, H, N, D, device=dev, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)
)
pos = torch.randn(1, N, 3, device=dev) * 10
lab = torch.zeros(1, N, dtype=torch.int32, device=dev)
go = torch.randn(1, H, N, D, device=dev, dtype=torch.bfloat16)


@mf.pair_function
def zero(a, b):
    return a * 0


def t_ms(fn, reps=10):
    for _ in range(3):
        fn()
    ev = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        fn()
        b.record()
        ev.append((a, b))
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) for a, b in ev)


def run(name, kw):
    def f():
        return mf.attention(q, k, v, **kw)

    with torch.no_grad():
        fw = t_ms(f)
    fb = t_ms(lambda: f().backward(go))
    print(json.dumps(dict(gpu=gpu, case=name, fwd=round(fw, 3), fwdbwd=round(fb, 3))), flush=True)


def g(*s):
    return torch.randn(*s, device=dev, requires_grad=True)


def n(*s):
    return torch.randn(*s, device=dev)


run("none", {})
run("const code, frozen", dict(bias=mf.PairBias(zero, lab, n(H, 1))))
run("const code, learned", dict(bias=mf.PairBias(zero, lab, g(H, 1))))
run("buckets, frozen", dict(bias=terms.distance_buckets(n(H, 32), pos, max_distance=16.0)))
run("buckets, learned", dict(bias=terms.distance_buckets(g(H, 32), pos, max_distance=16.0)))
run("interp, frozen", dict(bias=terms.distance_function(n(H, 64), pos, grid=(0.0, 20.0))))
run("interp, learned", dict(bias=terms.distance_function(g(H, 64), pos, grid=(0.0, 20.0))))
run(
    "interp shared heads, learned",
    dict(bias=terms.distance_function(g(1, 64), pos, grid=(0.0, 20.0))),
)
run("edge 4", dict(bias=mf.EdgeBias(torch.randint(-1, N, (1, N, 4), device=dev), g(1, N, 4, H))))
run("node r8", dict(bias=mf.NodeBias(g(1, H, N, 8), g(1, H, N, 8))))
