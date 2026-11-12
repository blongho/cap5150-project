"""Run the H1-H3 Wilcoxon tests on per-seed AUT and write results/hypothesis_tests.csv."""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from drift.config import load_config, resolve
from drift.stats import run_hypotheses


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(resolve(args.config))
    s = cfg["stats"]
    table = run_hypotheses(resolve(cfg["experiment"]["results_dir"]), metric=s["metric"],
                           alpha=s["alpha"])
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.drop(columns=["claim"]).to_string(index=False, float_format="%.4f"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
