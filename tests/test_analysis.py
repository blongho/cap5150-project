"""Tests for the replication and masking checks."""

import numpy as np
import pandas as pd
import pytest

from drift.analysis import masking_check, replication


def test_masking_check_rates_before_and_after_change_year():
    df = pd.DataFrame({
        "year": [2013, 2013, 2015, 2015],
        "Malware": [1, 0, 1, 0],
        "GET_TASKS": [0, 0, 1, 0],
    })
    t = masking_check(df, {"GET_TASKS": 2014}).set_index("class")
    assert (t.loc["malware", "rate_before"], t.loc["malware", "rate_after"]) == (0.0, 1.0)
    assert (t.loc["benign", "rate_before"], t.loc["benign", "rate_after"]) == (0.0, 0.0)
    assert t.loc["malware", "years_after_in_data"] == 1


def test_masking_check_change_after_data_has_no_after_rate():
    df = pd.DataFrame({"year": [2013, 2013], "Malware": [1, 0], "P": [1, 0]})
    t = masking_check(df, {"P": 2021})
    assert t["rate_after"].isna().all()


def test_replication_reports_difference_to_reference():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.integers(0, 2, size=(200, 4)), columns=list("ABCD"))
    df = X.assign(Malware=X["A"])
    rep = replication(df, list("ABCD"), {"test_size": 0.25, "seed": 1,
                                        "reference_accuracy": 0.9}, n_jobs=1)
    assert rep["accuracy"] == 1.0
    assert rep["difference"] == pytest.approx(0.1)
