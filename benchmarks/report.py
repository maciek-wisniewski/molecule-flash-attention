"""Markdown tables from run.py output.

python benchmarks/report.py results.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

ORDER = [
    "molflash",
    "flex",
    "flashbias",
    "cuequivariance",
    "xformers",
    "ds4sci",
    "turbopairformer",
    "sdpa+bias",
    "eager",
    "sdpa (no bias)",
]


def _load(paths):
    recs = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    # A size skipped after a failure at a smaller size inherits the reason: OOM stays OOM,
    # anything else (too slow, unsupported) is shown as n/a.
    last = {}
    for r in recs:
        if r.get("kind") != "speed":
            continue
        key = (r["gpu"], r["scenario"], r["method"])
        if r.get("error", "").startswith("skipped"):
            if last.get(key) in ("OOM", "slow"):
                r["error"] = last[key]
        elif "error" in r:
            last[key] = r["error"]
        elif r.get("fwdbwd_ms", 0) > 30_000:
            last[key] = "slow"
    return recs


def _cell(r, key, fmt):
    if r is None:
        return ""
    if "error" in r:
        return {"OOM": "OOM", "slow": "> 30 s"}.get(r["error"], "n/a")
    return fmt.format(r[key])


def speed_tables(recs) -> str:
    out = []
    groups = defaultdict(dict)
    for r in recs:
        if r["kind"] == "speed":
            groups[(r["gpu"], r["scenario"], r["dtype"], r["heads"], r["head_dim"])][
                (r["method"], r["n"])
            ] = r
    for (gpu, scenario, dtype, h, d), rows in sorted(groups.items()):
        ns = sorted({n for _, n in rows})
        methods = [m for m in ORDER if any(mm == m for mm, _ in rows)]
        for title, key, fmt in (
            ("forward + backward, ms", "fwdbwd_ms", "{:.1f}"),
            ("forward, ms", "fwd_ms", "{:.1f}"),
            ("peak memory, GB", "peak_gb", "{:.2f}"),
        ):
            out.append(f"**{scenario}**, {gpu}, {dtype}, {h} heads x {d}: {title}\n")
            out.append("| method | " + " | ".join(f"N={n}" for n in ns) + " |")
            out.append("|---" * (len(ns) + 1) + "|")
            for m in methods:
                cells = [_cell(rows.get((m, n)), key, fmt) for n in ns]
                out.append(f"| {m} | " + " | ".join(cells) + " |")
            out.append("")
    return "\n".join(out)


def summary_table(recs, gpu: str, ns=(4096, 16384, 32768)) -> str:
    """One row per scenario and method: fwd+bwd time and peak memory at a few sizes."""
    rows = {}
    for r in recs:
        if r["kind"] == "speed" and r["gpu"] == gpu and r["n"] in ns:
            rows[(r["scenario"], r["method"], r["n"])] = r
    scenarios = sorted({k[0] for k in rows})
    head = " | ".join(f"N={n}" for n in ns)
    out = [f"| scenario | method | {head} |", "|---|---|" + "---|" * len(ns)]
    for sc in scenarios:
        for m in ORDER:
            cells = []
            for n in ns:
                r = rows.get((sc, m, n))
                if r is None:
                    cells.append("")
                elif "error" in r:
                    cells.append({"OOM": "OOM", "slow": "> 30 s"}.get(r["error"], "n/a"))
                else:
                    cells.append(f"{r['fwdbwd_ms']:.1f} ms, {r['peak_gb']:.2f} GB")
            if any(cells):
                out.append(f"| {sc} | {m} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def accuracy_table(recs) -> str:
    # cuEquivariance projects its own q, k, v and gates the output: a different function.
    latest = {}
    for r in recs:
        if r["kind"] == "accuracy" and "rel_err" in r and r["method"] != "cuequivariance":
            latest[(r.get("gpu"), r["scenario"], r["method"], r["dtype"])] = r
    rows = list(latest.values())
    if not rows:
        return ""
    out = [
        "| scenario | method | dtype | output | worst q/k/v grad | worst parameter grad |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        e = r["rel_err"]
        qkv = max([e[x] for x in ("dq", "dk", "dv") if x in e] or [float("nan")])
        params = max([v for k, v in e.items() if k not in ("out", "dq", "dk", "dv")] or [0.0])
        out.append(
            f"| {r['scenario']} | {r['method']} | {r['dtype']} | {e['out']:.1e} | "
            f"{qkv:.1e} | {params:.1e} |"
        )
    return "\n".join(out)


def primitive_table(recs) -> str:
    latest = {}
    for r in recs:
        if r["kind"] == "primitive":
            latest[(r["gpu"], r["n"], r["primitive"])] = r
    rows = list(latest.values())
    if not rows:
        return ""
    base = {(r["gpu"], r["n"]): r for r in rows if r["primitive"] == "none"}
    out = [
        "| GPU | N | primitive | forward ms | x no terms | fwd+bwd ms | x no terms |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        b = base[(r["gpu"], r["n"])]
        out.append(
            f"| {r['gpu']} | {r['n']} | {r['primitive']} | {r['fwd_ms']:.2f} | "
            f"{r['fwd_ms'] / b['fwd_ms']:.2f} | {r['fwdbwd_ms']:.2f} | "
            f"{r['fwdbwd_ms'] / b['fwdbwd_ms']:.2f} |"
        )
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="+")
    p.add_argument("--summary", help="print only the summary table for this GPU name")
    a = p.parse_args()
    recs = _load(a.results)
    if a.summary:
        print(summary_table(recs, a.summary))
        return
    for part in (
        speed_tables(recs),
        accuracy_table(recs),
        primitive_table(recs),
    ):
        if part:
            print(part, end="\n\n")


if __name__ == "__main__":
    main()
