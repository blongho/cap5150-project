"""Load and validate the YAML experiment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_SECTIONS = ("data", "permissions", "split", "model", "experiment", "replication")


class ConfigError(ValueError):
    """Raised when the configuration is missing keys or has invalid values."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config and fail fast on missing sections or inconsistent splits."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ConfigError(f"Config {path} is missing sections: {missing}")

    data, split = cfg["data"], cfg["split"]
    if not data["min_year"] <= split["train_end_year"] < data["max_year"]:
        raise ConfigError("split.train_end_year must lie inside [data.min_year, data.max_year)")
    test_years = split["test_years"]
    if not test_years or min(test_years) <= split["train_end_year"]:
        # TESSERACT C1: every training sample must precede every test sample.
        raise ConfigError("All split.test_years must come after split.train_end_year")
    if max(test_years) > data["max_year"]:
        raise ConfigError("split.test_years exceed data.max_year")

    perms = cfg["permissions"]
    unknown = set(perms["categories"]) - {"deprecated", "restricted"}
    if unknown:
        raise ConfigError(f"Unknown permission categories: {sorted(unknown)}")
    if perms["min_api"] > perms["max_api"]:
        raise ConfigError("permissions.min_api must be <= permissions.max_api")
    return cfg


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    p = Path(path)
    return p if p.is_absolute() else repo_root() / p
