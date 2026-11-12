"""KS drift tests (per permission, BH-corrected, and on RF scores) for each test year.

Writes results/ks_tests.csv, results/ks_scores.csv and results/ks_run.json.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from drift.config import load_config, resolve
from drift.data import LABEL, permission_columns, time_split
from drift.ks_tests import run_ks_analysis
from drift.models import build_model
from drift.runinfo import run_info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(resolve(args.config))
    results = resolve(cfg["experiment"]["results_dir"])
    results.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(resolve(cfg["data"]["processed_path"]))
    perms = permission_columns(df.columns)
    split = cfg["split"]
    train, tests = time_split(df, split["train_end_year"], split["test_years"])
    seed = cfg["ks"]["seed"]
    model = build_model(cfg["model"], seed).fit(train[perms], train[LABEL])
    table = run_ks_analysis(train, tests, perms, model, results, cfg["ks"]["alpha"],
                            {**run_info(), "model": cfg["model"]["name"], "seed": seed})
    print("Permissions flagged (BH-adjusted p <= alpha) per test year:")
    print(table.groupby("year")["flagged"].sum().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
