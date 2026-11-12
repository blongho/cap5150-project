"""Compare a fresh pipeline run with the reference values in configs/expected_results.json.

Checks the cleaned data (app counts per year), the listed permissions, and the mean AUT(F1)
and AUT(recall) over seeds of every experiment cell in the reference. Exits with status 1 and
lists every mismatch if the run does not reproduce the reference.
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from drift.config import resolve


def compare(name: str, actual, expected, tol: float, problems: list[str]) -> None:
    if isinstance(expected, float):
        if abs(float(actual) - expected) > tol:
            problems.append(f"{name}: got {actual:.4f}, expected {expected:.4f} (tol {tol})")
    elif actual != expected:
        problems.append(f"{name}: got {actual!r}, expected {expected!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expected", default="configs/expected_results.json")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    exp = json.loads(resolve(args.expected).read_text())
    res = resolve(args.results_dir)
    cells = exp["cells"]
    needed = ["year_counts.csv", *(f"{c}_aut.csv" for c in cells)]
    missing = [f for f in needed if not (res / f).exists()]
    records = sorted(res.glob(f"{next(iter(cells))}_seed*.json"))
    if not records:
        missing.append(f"{next(iter(cells))}_seed*.json")
    if missing:
        print(f"FAIL: missing result files {missing}; run `make all` first", file=sys.stderr)
        return 1

    tol = exp["tolerance"]
    counts = pd.read_csv(res / "year_counts.csv", index_col=0)
    record = json.loads(records[0].read_text())

    problems: list[str] = []
    compare("n_apps", int(counts["total"].sum()), exp["n_apps"], 0, problems)
    compare("n_malware", int(counts["malware"].sum()), exp["n_malware"], 0, problems)
    compare("year_totals", {str(k): int(v) for k, v in counts["total"].items()},
            exp["year_totals"], 0, problems)
    compare("listed_permissions", record["listed_permissions"], exp["listed_permissions"], 0,
            problems)
    n_checks = 4
    for cell, values in cells.items():
        auts = pd.read_csv(res / f"{cell}_aut.csv")
        for metric in ("aut_f1", "aut_recall"):
            compare(f"{cell}.{metric}", auts[metric].mean(), values[f"{metric}_mean"],
                    float(tol[metric]), problems)
            n_checks += 1

    if problems:
        print("FAIL: results do not reproduce the reference:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print(f"OK: all {n_checks} checks reproduce the reference "
          f"(tolerance AUT(F1) {tol['aut_f1']}, AUT(recall) {tol['aut_recall']}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
