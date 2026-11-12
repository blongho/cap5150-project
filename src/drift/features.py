"""Simulated permission loss (masking) and the pruning defense."""

from __future__ import annotations

import pandas as pd

from drift.data import YEAR


def _check(columns: list[str], permissions: list[str]) -> None:
    missing = sorted(set(permissions) - set(columns))
    if missing:
        raise KeyError(f"Permissions not in the feature set: {missing}")


def mask(
    X: pd.DataFrame, permissions: list[str], change_years: dict[str, int] | None = None
) -> pd.DataFrame:
    """Return a copy of X with the listed permission columns set to 0.

    Models a deployed detector whose input no longer contains these permissions. With
    `change_years`, each permission in it is zeroed only in rows whose `year` column is on or
    after its change year (timed masking); X must then have a `year` column, and every
    permission in `change_years` must be a column of X.
    """
    _check(list(X.columns), permissions)
    out = X.copy()
    if change_years is None:
        out[permissions] = 0
        return out
    _check(list(X.columns), list(change_years))
    if YEAR not in X.columns:
        raise KeyError(f"Timed masking needs a '{YEAR}' column")
    for perm in permissions:
        if perm in change_years:
            out.loc[out[YEAR] >= change_years[perm], perm] = 0
    return out


def prune(X: pd.DataFrame, permissions: list[str]) -> pd.DataFrame:
    """Return a copy of X without the listed permission columns (the pruning defense)."""
    _check(list(X.columns), permissions)
    return X.drop(columns=permissions)
