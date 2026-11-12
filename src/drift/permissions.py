"""Derive deprecated / restricted permissions from AOSP framework manifests.

A permission is *deprecated* at the first API level whose manifest documents it with
``@deprecated``. It is *restricted* at the first API level whose manifest gives it the
``hardRestricted`` or ``softRestricted`` permission flag (introduced in Android 10).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ANDROID_NS = "{http://schemas.android.com/apk/res/android}"
PREFIX = "android.permission."
COLUMNS = ["permission", "category", "api_level", "year", "flag", "source"]


def parse_manifest(path: Path) -> dict[str, dict[str, bool | str]]:
    """Return {short_name: {deprecated, restricted, flag}} for android.permission.* entries."""
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.parse(path, parser=parser).getroot()
    out: dict[str, dict[str, bool | str]] = {}
    last_comment = ""
    for node in root:
        if node.tag is ET.Comment:
            last_comment = node.text or ""
            continue
        if node.tag == "permission":
            name = node.get(f"{ANDROID_NS}name", "")
            if name.startswith(PREFIX):
                flags = node.get(f"{ANDROID_NS}permissionFlags", "")
                restricted = [f for f in ("hardRestricted", "softRestricted") if f in flags]
                out[name[len(PREFIX) :]] = {
                    "deprecated": "@deprecated" in last_comment,
                    "restricted": bool(restricted),
                    "flag": restricted[0] if restricted else "",
                }
        last_comment = ""  # a comment documents only the element right after it
    return out


def build_list(
    manifests: dict[int, Path], years: dict[int, int], dataset_permissions: list[str]
) -> pd.DataFrame:
    """First API level at which each dataset permission became deprecated or restricted."""
    parsed = {api: parse_manifest(p) for api, p in sorted(manifests.items())}
    rows = []
    for perm in dataset_permissions:
        for category in ("deprecated", "restricted"):
            for api, entries in parsed.items():
                info = entries.get(perm)
                if info and info[category]:
                    rows.append(
                        {
                            "permission": perm,
                            "category": category,
                            "api_level": api,
                            "year": years[api],
                            "flag": info["flag"] if category == "restricted" else "@deprecated",
                            "source": manifests[api].name,
                        }
                    )
                    break
    return pd.DataFrame(rows, columns=COLUMNS).sort_values(["category", "api_level", "permission"])


def select(
    table: pd.DataFrame, categories: list[str], min_api: int, max_api: int
) -> list[str]:
    """Permissions in the given categories whose change falls in [min_api, max_api]."""
    keep = table[
        table["category"].isin(categories) & table["api_level"].between(min_api, max_api)
    ]
    return sorted(keep["permission"].unique())


def change_years(
    table: pd.DataFrame, permissions: list[str], categories: list[str], min_api: int, max_api: int
) -> dict[str, int]:
    """First year each permission changed (earliest across the given categories and APIs)."""
    keep = table[
        table["permission"].isin(permissions)
        & table["category"].isin(categories)
        & table["api_level"].between(min_api, max_api)
    ]
    return {p: int(y) for p, y in keep.groupby("permission")["year"].min().items()}
