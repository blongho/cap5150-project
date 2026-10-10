"""Simulated permission loss (masking) and the pruning defense."""

from __future__ import annotations

import pandas as pd


def _check(columns: list[str], permissions: list[str]) -> None:
    missing = sorted(set(permissions) - set(columns))
    if missing:
        raise KeyError(f"Permissions not in the feature set: {missing}")


def mask(X: pd.DataFrame, permissions: list[str]) -> pd.DataFrame:
    """Return a copy of X with the listed permission columns set to 0.

    Models a deployed detector whose input no longer contains these permissions.
    """
    _check(list(X.columns), permissions)
    out = X.copy()
    out[permissions] = 0
    return out


def prune(X: pd.DataFrame, permissions: list[str]) -> pd.DataFrame:
    """Return a copy of X without the listed permission columns (the pruning defense)."""
    _check(list(X.columns), permissions)
    return X.drop(columns=permissions)
