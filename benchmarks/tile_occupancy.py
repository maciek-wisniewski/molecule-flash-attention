"""How many tiles a distance cutoff keeps on the benchmark coordinates, and the fit of
T(N) = alpha N^2 + beta K(N) to measured times.

    python tile_occupancy.py [--cutoff 15] [--results a100.jsonl --scenario protein]
    python tile_occupancy.py --xyz 1AON.xyz ...

Runs on the CPU. K(N) counts the forward tiles (32 queries x 64 keys) with at least one pair
closer than the cutoff; the coordinates are those of scenarios.coordinates, or, with --xyz,
real structures (one "x y z" line per heavy atom, in file order) in their own order, in Morton
order and in random order. To write such a file from an mmCIF or PDB file with gemmi:

    st = gemmi.read_structure(path); st.remove_waters(); st.remove_hydrogens()
    lines = [f"{a.pos.x} {a.pos.y} {a.pos.z}" for ch in st[0] for r in ch for a in r]
"""

import argparse
import json
import math

import torch
from scenarios import coordinates

NS = [1024, 2048, 4096, 8192, 16384, 32768]


def morton(x, cell=4.0):
    """The order of scenarios.coordinates: Morton code of 4 A cells."""
    c = ((x - x.min(0).values) / cell).long()
    code = torch.zeros(len(x), dtype=torch.long)
    for bit in range(10):
        for axis in range(3):
            code |= ((c[:, axis] >> bit) & 1) << (3 * bit + axis)
    return x[code.argsort()]


def occupancy(n, cutoff, bm=32, bn=64, chunk=1024, x=None):
    if x is None:
        x = coordinates(1, n, "cpu")[0]
    n = len(x) // bn * bn
    x = x[:n]
    pairs = 0
    kept = torch.zeros(n // bm, n // bn, dtype=torch.bool)
    for s in range(0, n, chunk):
        near = torch.cdist(x[s : s + chunk], x) < cutoff
        pairs += near.sum().item()
        rows = near.view(-1, bm, n).any(1)
        kept[s // bm : (s + chunk) // bm] = rows.view(-1, n // bn, bn).any(-1)
    return pairs / n, kept.float().mean().item(), kept.sum(1).float().mean().item()


def fit(results, scenario, frac):
    times = {}
    with open(results) as f:
        for line in f:
            if "NaN" in line or not line.strip():
                continue
            r = json.loads(line)
            ours = r.get("kind") == "speed" and r["method"] == "molflash"
            if ours and r["scenario"] == scenario and "fwdbwd_ms" in r:
                times[r["n"]] = r["fwdbwd_ms"]
    ns = [n for n in NS if n >= 4096 and n in times]
    a = torch.tensor([[n * n / 1e9, frac[n] * n * n / 1e9] for n in ns], dtype=torch.float64)
    y = torch.tensor([times[n] for n in ns], dtype=torch.float64)
    coef = torch.linalg.lstsq(a, y[:, None]).solution[:, 0]
    pred = a @ coef
    print(f"fit on {scenario}: alpha {coef[0]:.1f}, beta {coef[1]:.1f} ms per 1e9 pairs")
    for n, p, t in zip(ns, pred.tolist(), y.tolist(), strict=True):
        share = coef[0].item() * n * n / 1e9 / p
        print(f"  N={n}: measured {t:.1f} ms, fitted {p:.1f} ms, mask test {share:.0%}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cutoff", type=float, default=15.0)
    p.add_argument("--results", help="run.py speed output to fit")
    p.add_argument("--scenario", nargs="+", default=["protein", "complex"])
    p.add_argument("--xyz", nargs="+", help="real structures, one 'x y z' line per atom")
    a = p.parse_args()
    torch.manual_seed(0)
    if a.xyz:
        for path in a.xyz:
            with open(path) as f:
                x = torch.tensor([[float(v) for v in line.split()] for line in f])
            orders = {"file": x, "morton": morton(x), "random": x[torch.randperm(len(x))]}
            for name, y in orders.items():
                nbrs, kept, _ = occupancy(0, a.cutoff, x=y)
                print(
                    f"{path} ({len(x)} atoms), {name} order: {nbrs:.0f} neighbours, "
                    f"{kept:.1%} of tiles kept"
                )
        return
    print(f"bulk neighbours: {4 / 3 * math.pi * a.cutoff**3 / 20.0:.0f}")
    frac = {}
    for n in NS:
        nbrs, frac[n], per_row = occupancy(n, a.cutoff)
        print(f"N={n}: {nbrs:.0f} neighbours, {frac[n]:.1%} of tiles kept, {per_row:.1f} per row")
    if a.results:
        for s in a.scenario:
            fit(a.results, s, frac)


if __name__ == "__main__":
    main()
