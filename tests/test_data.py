import pandas as pd
import pytest

from drift.data import (
    YEAR,
    clean,
    permission_columns,
    subsample_malware,
    time_split,
    year_counts,
)


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


def _year_frame(n_malware: int, n_benign: int) -> pd.DataFrame:
    return pd.DataFrame({"sha256": [f"h{i}" for i in range(n_malware + n_benign)],
                         "Malware": [1] * n_malware + [0] * n_benign})


def test_subsample_malware_hits_target_share_and_keeps_benign():
    df = _year_frame(n_malware=600, n_benign=900)
    out = subsample_malware(df, target_share=0.10, seed=0)
    assert (out["Malware"] == 0).sum() == 900
    assert out["Malware"].mean() == pytest.approx(0.10, abs=0.001)
    assert list(out.index) == sorted(out.index)  # original order kept
    pd.testing.assert_frame_equal(out, subsample_malware(df, 0.10, seed=0))


@pytest.mark.parametrize("target", [0.4, 0.5, 0.0])
def test_subsample_malware_rejects_targets_not_below_current_share(target):
    with pytest.raises(ValueError):
        subsample_malware(_year_frame(n_malware=400, n_benign=600), target, seed=0)
