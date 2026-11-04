"""Forward and forward+backward time of single terms under each kernel block configuration.

    python benchmarks/tune_blocks.py > tune_blocks.jsonl

For each block shape (BLOCK_M, BLOCK_N, warps, stages) it times attention with no terms, a
learned 64-point distance table and a 32-entry distance-bucket table, with the tables learned and
frozen. The learned interpolated table runs once with private slices per program and once with
atomic additions straight into the shared table. N = 8192, 8 heads of 64, bfloat16, batch 1.
Prints one JSON line per case, time in ms.
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


def case(make, grad_table=True):
    tab_i = torch.randn(H, 64, device=dev, requires_grad=grad_table)
    tab_c = torch.randn(H, 32, device=dev, requires_grad=grad_table)
    kw = make(tab_i, tab_c)

    def f():
        return mf.attention(q, k, v, **kw)

    def fb():
        f().backward(go)

    with torch.no_grad():
        fw = t_ms(f)
    return dict(fwd=fw, fwdbwd=t_ms(fb))


cases = {
    "none": lambda ti, tc: {},
    "interp": lambda ti, tc: dict(bias=terms.distance_function(ti, pos, grid=(0.0, 20.0))),
    "code": lambda ti, tc: dict(bias=terms.distance_buckets(tc, pos, max_distance=16.0)),
}
configs = [
    (32, 32, 4, 1),
    (64, 32, 4, 1),
    (32, 64, 4, 1),
    (64, 64, 4, 1),
    (64, 64, 8, 2),
    (128, 64, 8, 2),
]
for cfg in configs:
    kernels._blocks = lambda st, device, backward=False, cfg=cfg: cfg
    for name, make in cases.items():
        for grad in [True, False] if name != "none" else [True]:
            for mode in ["private", "shared"] if name == "interp" and grad else ["default"]:
                kernels._PRIVATE_BUDGET = 0 if mode == "shared" else 256 * 2**20
                try:
                    r = case(make, grad)
                except Exception as e:  # a block shape may not fit in shared memory
                    r = {"error": str(e)[:200]}
                rec = dict(gpu=gpu, blocks=cfg, case=name, table_grad=grad, mode=mode, **r)
                print(json.dumps(rec), flush=True)
