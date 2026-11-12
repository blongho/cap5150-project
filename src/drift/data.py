"""Load, clean and split the KronoDroid real-device data."""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

LABEL = "Malware"
YEAR = "year"
HASH = "sha256"
# In KronoDroid the 166 standard permission flags sit between these two columns.
FIRST_PERMISSION = "ACCEPT_HANDOVER"
LAST_PERMISSION = "WRITE_VOICEMAIL"
EXPECTED_PERMISSIONS = 166


def permission_columns(columns: pd.Index | list[str]) -> list[str]:
    cols = list(columns)
    try:
        start, end = cols.index(FIRST_PERMISSION), cols.index(LAST_PERMISSION)
    except ValueError as exc:
        raise ValueError("Permission columns not found; is this a KronoDroid file?") from exc
    perms = cols[start : end + 1]
    if len(perms) != EXPECTED_PERMISSIONS:
        raise ValueError(f"Expected {EXPECTED_PERMISSIONS} permission columns, got {len(perms)}")
    return perms


def load_raw(malware_csv, benign_csv) -> pd.DataFrame:
    malware = pd.read_csv(malware_csv, low_memory=False)
    benign = pd.read_csv(benign_csv, low_memory=False)
    if list(malware.columns) != list(benign.columns):
        raise ValueError("Malware and benign files have different columns")
    if not (malware[LABEL] == 1).all() or not (benign[LABEL] == 0).all():
        raise ValueError("Unexpected labels: malware file must be 1 and benign file 0")
    return pd.concat([malware, benign], ignore_index=True)


def clean(raw: pd.DataFrame, date_column: str, min_year: int, max_year: int) -> pd.DataFrame:
    """Keep permission flags, label and year; drop bad dates and conflicting duplicates.

    Returns one row per app with binary permission features, the label and the year.
    """
    perms = permission_columns(raw.columns)
    df = raw[[HASH, date_column, LABEL, *perms]].copy()
    n0 = len(df)

    # Apps whose hash appears with both labels cannot be labelled reliably: drop all copies.
    labels_per_hash = df.groupby(HASH)[LABEL].nunique()
    conflicting = labels_per_hash[labels_per_hash > 1].index
    df = df[~df[HASH].isin(conflicting)]
    df = df.drop_duplicates(subset=HASH, keep="first")

    dates = pd.to_datetime(df[date_column], format="%m/%d/%Y", errors="coerce")
    df[YEAR] = dates.dt.year
    df = df[df[YEAR].between(min_year, max_year)].copy()
    df[YEAR] = df[YEAR].astype(int)

    df[perms] = (df[perms] > 0).astype("int8")  # flags are binary; guard against counts
    df = df.drop(columns=[date_column]).reset_index(drop=True)
    log.info(
        "clean: %d -> %d rows (%d conflicting hashes dropped)", n0, len(df), len(conflicting)
    )
    return df


def year_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Samples, malware count and malware share per year."""
    t = df.groupby(YEAR)[LABEL].agg(total="size", malware="sum")
    t["benign"] = t["total"] - t["malware"]
    t["malware_share"] = (t["malware"] / t["total"]).round(3)
    return t[["benign", "malware", "total", "malware_share"]]


def time_split(df: pd.DataFrame, train_end_year: int, test_years: list[int]):
    """Training set (years <= train_end_year) and one test set per later year."""
    if min(test_years) <= train_end_year:
        raise ValueError("Test years must come after the training window")
    train = df[df[YEAR] <= train_end_year]
    tests = {y: df[df[YEAR] == y] for y in test_years}
    empty = [y for y, t in tests.items() if t.empty]
    if empty:
        raise ValueError(f"No samples for test years {empty}")
    return train, tests


def subsample_malware(df_year: pd.DataFrame, target_share: float, seed: int) -> pd.DataFrame:
    """Drop random malware rows so malware makes up `target_share` of the year.

    Keeps every benign row and round(target_share * n_benign / (1 - target_share)) malware
    rows. Raises ValueError unless 0 < target_share < the current malware share.
    """
    current = float(df_year[LABEL].mean())
    if not 0 < target_share < current:
        raise ValueError(
            f"target_share {target_share} must lie in (0, current share {current:.3f})"
        )
    benign = df_year[df_year[LABEL] == 0]
    n_malware = round(target_share * len(benign) / (1 - target_share))
    malware = df_year[df_year[LABEL] == 1].sample(n=n_malware, random_state=seed)
    return df_year.loc[df_year.index.isin(benign.index.union(malware.index))]
