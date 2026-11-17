"""Compare your run with reference values from the paper's verified runs.

First run the two small commands listed in the README ("Check your installation"), then:
  python check_reproduction.py --results results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--tol", type=float, default=1e-6, help="allowed absolute difference (0 on the paper's setup)")
    args = ap.parse_args()
    reference = json.loads((HERE / "reference/reference_units.json").read_text())
    worst, missing, checked = 0.0, [], 0
    for name, ref in reference["units"].items():
        path = Path(args.results) / ref["backbone"] / "units" / f"{name}.json"
        if not path.exists():
            missing.append(f"{ref['backbone']}/{name}")
            continue
        u = json.loads(path.read_text())
        diffs = [abs(u["metrics"][k] - v) for k, v in ref["metrics"].items()]
        diffs += [abs(u["thresholds"][k] - v) for k, v in ref["thresholds"].items()]
        for variant, vals in ref.get("recalibration", {}).items():
            diffs += [abs(u["recalibration"][variant][k] - v) for k, v in vals.items()]
        worst = max([worst, *diffs])
        checked += 1
    print(f"checked {checked} units, missing {len(missing)}; max abs difference = {worst:.3g}")
    if missing:
        print("missing (run the README commands first):", *missing, sep="\n  ")
    print("PASS" if checked and not missing and worst <= args.tol else "CHECK" if checked else "NOTHING TO CHECK")


if __name__ == "__main__":
    main()
