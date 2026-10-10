import pandas as pd
import pytest

from drift.data import YEAR, clean, permission_columns, time_split, year_counts


def test_permission_columns_finds_166(raw_frame, perms):
    assert permission_columns(raw_frame.columns) == perms


def test_permission_columns_rejects_other_files():
    with pytest.raises(ValueError):
        permission_columns(["a", "b"])


def test_clean_drops_bad_rows_and_conflicts(raw_frame):
    df = clean(raw_frame, "HighestModDate", 2008, 2020)
    assert sorted(df["sha256"]) == ["a", "b", "c"]
    assert df["sha256"].is_unique
    assert set(df[YEAR]) == {2012, 2014, 2015}
    assert df.iloc[:, 2:-1].isin([0, 1]).all().all()


def test_time_split_has_no_overlap(raw_frame):
    df = clean(raw_frame, "HighestModDate", 2008, 2020)
    train, tests = time_split(df, 2013, [2014, 2015])
    assert train[YEAR].max() <= 2013
    assert all(t[YEAR].min() > train[YEAR].max() for t in tests.values())
    assert set(train["sha256"]).isdisjoint(set(pd.concat(tests.values())["sha256"]))


def test_time_split_rejects_overlapping_years(raw_frame):
    df = clean(raw_frame, "HighestModDate", 2008, 2020)
    with pytest.raises(ValueError):
        time_split(df, 2014, [2014])


def test_time_split_rejects_empty_year(raw_frame):
    df = clean(raw_frame, "HighestModDate", 2008, 2020)
    with pytest.raises(ValueError):
        time_split(df, 2013, [2014, 2019])


def test_year_counts(raw_frame):
    counts = year_counts(clean(raw_frame, "HighestModDate", 2008, 2020))
    assert counts.loc[2012, "malware"] == 1 and counts.loc[2014, "benign"] == 1
