"""Replication check and masking check.

Writes results/replication.json and results/masking_check.csv.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

from drift.analysis import masking_check, replication
from drift.config import load_config, resolve
from drift.data import permission_columns
from drift.permissions import change_years, select
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
    p = cfg["permissions"]
    table = pd.read_csv(resolve(p["list_path"]))
    listed = select(table, p["categories"], p["min_api"], p["max_api"])

    rep = replication(df, perms, cfg["replication"], cfg["model"].get("n_jobs", -1))
    (results / "replication.json").write_text(
        json.dumps({**rep, "config": cfg["replication"], **run_info()}, indent=2, default=float))
    check = masking_check(df, change_years(table, listed, p["categories"], p["min_api"],
                                           p["max_api"]))
    check.to_csv(results / "masking_check.csv", index=False)
    print(f"Replication accuracy {rep['accuracy']:.4f} "
          f"(reference {rep['reference_accuracy']}, difference {rep['difference']:+.4f})")
    print(check.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
