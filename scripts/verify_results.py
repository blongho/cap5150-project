"""Compare a fresh pipeline run with the reference values in configs/expected_results.json.

Exits with status 1 and lists every mismatch if the run does not reproduce the reference.
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
    needed = ["year_counts.csv", "milestone_aut.csv", "milestone_run.json"]
    missing = [f for f in needed if not (res / f).exists()]
    if missing:
        print(f"FAIL: missing result files {missing}; run `make all` first", file=sys.stderr)
        return 1

    tol = float(exp["tolerance"])
    counts = pd.read_csv(res / "year_counts.csv", index_col=0)
    aut = pd.read_csv(res / "milestone_aut.csv", header=[0, 1], index_col=0)
    run = json.loads((res / "milestone_run.json").read_text())

    problems: list[str] = []
    compare("n_apps", int(counts["total"].sum()), exp["n_apps"], tol, problems)
    compare("n_malware", int(counts["malware"].sum()), exp["n_malware"], tol, problems)
    compare("year_totals", {str(k): int(v) for k, v in counts["total"].items()},
            exp["year_totals"], tol, problems)
    compare("listed_permissions", run["listed_permissions"], exp["listed_permissions"], tol,
            problems)
    compare("replication_accuracy", run["replication"]["accuracy"],
            exp["replication_accuracy"], tol, problems)
    for metric in ("aut_f1", "aut_recall"):
        for cond, value in exp[f"{metric}_mean"].items():
            compare(f"{metric}[{cond}]", aut.loc[cond, (metric, "mean")], value, tol, problems)

    if problems:
        print("FAIL: results do not reproduce the reference:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print(f"OK: all {len(exp['aut_f1_mean']) * 2 + 5} checks reproduce the reference "
          f"(tolerance {tol}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
