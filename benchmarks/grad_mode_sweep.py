"""Forward+backward time of a learned distance table under each table-gradient mode, by table size.

    python benchmarks/grad_mode_sweep.py > grad_mode_sweep.jsonl

"reduce" accumulates the table gradient in registers, "private" in a buffer per program; the
module limits that normally choose between them are overridden for the sweep. N = 8192, 8 heads
of 64, bfloat16, batch 1. Prints one JSON line per (size, mode, kind), time in ms.
"""

import json
import statistics

import molflash as mf
import torch
from molflash import kernels, terms

torch.manual_seed(0)
dev = "cuda"
N, H, D = 8192, 8, 64
gpu = torch.cuda.get_device_name()
q, k, v = (
    torch.randn(1, H, N, D, device=dev, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)
)
pos = torch.randn(1, N, 3, device=dev) * 10
go = torch.randn(1, H, N, D, device=dev, dtype=torch.bfloat16)


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


for size in (16, 32, 64, 128):
    for mode in ("reduce", "private"):
        kernels._REDUCE_MAX_CODES = kernels._REDUCE_MAX_GRID = 10**6 if mode == "reduce" else 0
        for kind in ("buckets", "interp"):
            t = torch.randn(H, size, device=dev, requires_grad=True)
            term = (
                terms.distance_buckets(t, pos, max_distance=16.0)
                if kind == "buckets"
                else terms.distance_function(t, pos, grid=(0.0, 20.0))
            )
            ms = t_ms(lambda term=term: mf.attention(q, k, v, bias=term).backward(go))
            rec = dict(gpu=gpu, size=size, mode=mode, kind=kind, fwdbwd=round(ms, 2))
            print(json.dumps(rec), flush=True)
