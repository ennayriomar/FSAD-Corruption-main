"""Paper-style tables from the unit files written by run_protocols.py.

Aggregation (as in the paper): per seed, average over all corruption conditions; then report the mean and
sample SD over seeds. Signed differences are in percentage points.

  python summarize.py --results results/dinov2
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RANK = ("image_auroc", "pixel_auroc", "aupro")
OPER = ("image_test_fpr", "image_test_fnr", "mean_iou_anomalous")


def load(results: Path):
    units = defaultdict(dict)   # (category, shots, protocol) -> {(condition, seed): unit}
    for p in sorted((results / "units").glob("*.json")):
        u = json.loads(p.read_text())
        units[(u["category"], u["shots"], u["protocol"])][(f"{u['corruption']}{u['severity']}", u["seed"])] = u
    return units


def per_seed(cell: dict, get) -> dict:
    """seed -> mean over conditions of get(unit)."""
    by_seed = defaultdict(list)
    for (_, seed), u in cell.items():
        by_seed[seed].append(get(u))
    return {s: float(np.mean(v)) * 100 for s, v in by_seed.items()}


def ms(values: dict) -> str:
    v = np.array(list(values.values()))
    return f"{v.mean():.2f} ± {v.std(ddof=1):.2f}" if len(v) > 1 else f"{v.mean():.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/dinov2")
    args = ap.parse_args()
    results = Path(args.results)
    units = load(results)
    cats = sorted({c for c, _, _ in units})
    shots = sorted({k for _, k, _ in units})
    lines, rows = [f"# Results: {results.name}\n"], []

    lines += ["## Ranking metrics (%, mean ± seed SD)\n",
              "| k | Category | Protocol | Image AUROC | Pixel AUROC | AUPRO@0.3 |", "|---|---|---|---|---|---|"]
    for k in shots:
        for c in cats:
            for proto in ("clean", "P1", "P2"):
                cell = units.get((c, k, proto))
                if not cell:
                    continue
                vals = [per_seed(cell, lambda u, m=m: u["metrics"][m]) for m in RANK]
                lines.append(f"| {k} | {c} | {proto} | " + " | ".join(ms(v) for v in vals) + " |")
                rows.append({"shots": k, "category": c, "protocol": proto, "variant": "fixed",
                             **{m: float(np.mean(list(v.values()))) for m, v in zip(RANK, vals)}})

    lines += ["\n## Operating point at the clean-calibrated thresholds (%)\n",
              "| k | Category | Protocol | Thresholds | FPR | FNR | mIoU (anomalous) |", "|---|---|---|---|---|---|---|"]
    for k in shots:
        for c in cats:
            for proto in ("clean", "P1", "P2"):
                cell = units.get((c, k, proto))
                if not cell:
                    continue
                variants = [("fixed", lambda u, m: u["metrics"][m])]
                if proto != "clean" and "recalibration" in next(iter(cell.values())):
                    variants += [(v, lambda u, m, v=v: u["recalibration"][v][m]) for v in ("recal50", "recal10")]
                for name, get in variants:
                    vals = [per_seed(cell, lambda u, m=m, get=get: get(u, m)) for m in OPER]
                    lines.append(f"| {k} | {c} | {proto} | {name} | " + " | ".join(ms(v) for v in vals) + " |")
                    rows.append({"shots": k, "category": c, "protocol": proto, "variant": name,
                                 **{m: float(np.mean(list(v.values()))) for m, v in zip(OPER, vals)}})

    (results / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(results / "summary.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["shots", "category", "protocol", "variant", *RANK, *OPER]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("\n".join(lines))
    print(f"\nwritten: {results / 'summary.md'} and {results / 'summary.csv'}")


if __name__ == "__main__":
    main()
