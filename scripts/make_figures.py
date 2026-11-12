"""Build all report figures and LaTeX tables from results/ CSVs (no experiments are run)."""

from __future__ import annotations

import argparse
import sys

from drift.config import load_config, resolve
from drift.figures import make_all


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(resolve(args.config))
    exp = cfg["experiment"]
    for path in make_all(resolve(exp["results_dir"]), resolve(exp["figures_dir"])):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
