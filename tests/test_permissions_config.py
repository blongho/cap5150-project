import pandas as pd
import pytest
import yaml

from drift.config import ConfigError, load_config, resolve
from drift.permissions import build_list, parse_manifest, select

XML = """<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android" package="android">
  <!-- @deprecated No longer used. -->
  <permission android:name="android.permission.OLD" android:protectionLevel="normal" />
  <permission android:name="android.permission.SMS" {flags} android:protectionLevel="dangerous" />
  <!-- ordinary documentation -->
  <permission android:name="android.permission.PLAIN" android:protectionLevel="normal" />
  <permission android:name="com.other.permission.OLD" android:protectionLevel="normal" />
</manifest>
"""


def write(tmp_path, name, flags=""):
    p = tmp_path / name
    p.write_text(XML.format(flags=flags))
    return p


def test_parse_manifest(tmp_path):
    info = parse_manifest(write(tmp_path, "m.xml", 'android:permissionFlags="hardRestricted"'))
    assert info["OLD"]["deprecated"] and not info["OLD"]["restricted"]
    assert info["SMS"]["restricted"] and info["SMS"]["flag"] == "hardRestricted"
    assert not info["PLAIN"]["deprecated"] and not info["PLAIN"]["restricted"]
    assert "com.other.permission.OLD" not in info


def test_build_list_uses_first_api(tmp_path):
    manifests = {28: write(tmp_path, "a.xml"),
                 29: write(tmp_path, "b.xml", 'android:permissionFlags="softRestricted"')}
    table = build_list(manifests, {28: 2018, 29: 2019}, ["OLD", "SMS", "PLAIN"])
    rows = {(r.permission, r.category): r.api_level for r in table.itertuples()}
    assert rows == {("OLD", "deprecated"): 28, ("SMS", "restricted"): 29}


def test_select_filters_category_and_api():
    t = pd.DataFrame({"permission": ["A", "B", "C"], "category": ["deprecated", "restricted",
                      "deprecated"], "api_level": [21, 29, 31]})
    assert select(t, ["deprecated"], 21, 30) == ["A"]
    assert select(t, ["deprecated", "restricted"], 23, 34) == ["B", "C"]


def test_repo_permission_list_matches_dataset():
    """Every listed permission must be a KronoDroid permission column."""
    path = resolve("data/deprecated_permissions.csv")
    if not path.exists():
        pytest.skip("run `make permissions` first")
    table = pd.read_csv(path)
    from drift.data import EXPECTED_PERMISSIONS  # noqa: F401  (documents the 166-column contract)
    raw = resolve("data/raw/real_malware_v1.csv")
    if not raw.exists():
        pytest.skip("run `make data` first")
    cols = pd.read_csv(raw, nrows=0).columns
    assert set(table["permission"]) <= set(cols)


def test_base_config_is_valid():
    load_config(resolve("configs/base.yaml"))


def test_config_rejects_test_years_before_training(tmp_path):
    cfg = yaml.safe_load(resolve("configs/base.yaml").read_text())
    cfg["split"]["test_years"] = [2012, 2014]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ConfigError):
        load_config(path)
