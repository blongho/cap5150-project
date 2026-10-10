"""Clean the raw KronoDroid real-device files and write the processed dataset."""

from __future__ import annotations

import argparse
import logging
import sys

from drift.config import load_config, resolve
from drift.data import clean, load_raw, year_counts

log = logging.getLogger("prepare_data")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(resolve(args.config))
    d = cfg["data"]

    raw_dir = resolve(d["raw_dir"])
    df = clean(
        load_raw(raw_dir / d["malware_csv"], raw_dir / d["benign_csv"]),
        d["date_column"],
        d["min_year"],
        d["max_year"],
    )
    out = resolve(d["processed_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    results = resolve(cfg["experiment"]["results_dir"])
    results.mkdir(parents=True, exist_ok=True)
    counts = year_counts(df)
    counts.to_csv(results / "year_counts.csv")
    log.info("wrote %s (%d rows) and results/year_counts.csv", out, len(df))
    print(counts.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
