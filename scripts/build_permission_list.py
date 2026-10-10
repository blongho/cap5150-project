"""Build data/deprecated_permissions.csv from the downloaded AOSP manifests."""

from __future__ import annotations

import argparse
import logging
import sys

from drift.config import load_config, resolve
from drift.data import load_raw, permission_columns
from drift.permissions import build_list
from drift.sources import API_LEVELS

log = logging.getLogger("build_permission_list")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(resolve(args.config))
    d, p = cfg["data"], cfg["permissions"]

    raw_dir = resolve(d["raw_dir"])
    raw = load_raw(raw_dir / d["malware_csv"], raw_dir / d["benign_csv"])
    perms = permission_columns(raw.columns)
    aosp_dir = resolve(p["aosp_dir"])
    manifests = {
        api: aosp_dir / f"api{api}_{tag}.xml" for api, (tag, _) in API_LEVELS.items()
    }
    missing = [str(m) for m in manifests.values() if not m.exists()]
    if missing:
        raise FileNotFoundError(f"Missing AOSP manifests (run make data): {missing}")
    years = {api: year for api, (_, year) in API_LEVELS.items()}

    table = build_list(manifests, years, perms)
    out = resolve(p["list_path"])
    table.to_csv(out, index=False)
    log.info("wrote %d entries to %s", len(table), out)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
