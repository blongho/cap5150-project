"""Download KronoDroid (real device) and AOSP permission manifests, verifying SHA-256.

KronoDroid checksums are pinned in src/drift/sources.py. AOSP checksums are pinned in
data/aosp_manifest_sha256.txt; pass --record-aosp once to (re)create that file.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import urllib.request
import zipfile
from pathlib import Path

from drift.config import load_config, resolve
from drift.sources import AOSP_CHECKSUMS, AOSP_URL, API_LEVELS, KRONODROID_BASE, KRONODROID_FILES

log = logging.getLogger("download_data")
TIMEOUT_S = 60


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, expected: str | None) -> None:
    if dest.exists() and expected and sha256(dest) == expected:
        log.info("ok (cached) %s", dest.name)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading %s", url)
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as resp, tmp.open("wb") as out:
        out.write(resp.read())
    actual = sha256(tmp)
    if expected and actual != expected:
        raise RuntimeError(f"Checksum mismatch for {dest.name}: expected {expected}, got {actual}")
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--record-aosp", action="store_true", help="write AOSP checksum file")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(resolve(args.config))

    raw_dir = resolve(cfg["data"]["raw_dir"])
    for name, digest in KRONODROID_FILES.items():
        zpath = raw_dir / name
        fetch(f"{KRONODROID_BASE}/{name}", zpath, digest)
        csv_name = name.replace(".zip", ".csv")
        if not (raw_dir / csv_name).exists():
            with zipfile.ZipFile(zpath) as zf:
                if zf.namelist() != [csv_name]:
                    raise RuntimeError(f"Unexpected archive contents in {name}: {zf.namelist()}")
                zf.extract(csv_name, raw_dir)

    aosp_dir = resolve(cfg["permissions"]["aosp_dir"])
    checksum_file = resolve(AOSP_CHECKSUMS)
    pinned: dict[str, str] = {}
    if checksum_file.exists() and not args.record_aosp:
        for line in checksum_file.read_text().splitlines():
            digest, name = line.split()
            pinned[name] = digest
    elif not args.record_aosp:
        raise RuntimeError(f"{checksum_file} missing; run with --record-aosp to create it")

    recorded = []
    for api, (tag, _year) in sorted(API_LEVELS.items()):
        dest = aosp_dir / f"api{api}_{tag}.xml"
        fetch(AOSP_URL.format(tag=tag), dest, pinned.get(dest.name))
        recorded.append(f"{sha256(dest)}  {dest.name}")
    if args.record_aosp:
        checksum_file.write_text("\n".join(recorded) + "\n")
        log.info("recorded AOSP checksums in %s", checksum_file)
    log.info("all downloads verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
