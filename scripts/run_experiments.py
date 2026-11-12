"""Run the full experiment grid (model x defense x condition x seed) into results/.

Existing Run_Records are reused, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import logging
import sys

from drift.config import load_config, resolve
from drift.experiments import CONDITIONS, DEFENSES, MODELS, run_experiments


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--defenses", nargs="+", default=list(DEFENSES), choices=DEFENSES)
    ap.add_argument("--conditions", nargs="+", choices=CONDITIONS,
                    help="default: unmasked masked, plus the optional conditions enabled in "
                         "the config")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(resolve(args.config))
    written = run_experiments(cfg, args.models, args.defenses, args.conditions)
    print(f"{len(written)} new run records written to {resolve(cfg['experiment']['results_dir'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
