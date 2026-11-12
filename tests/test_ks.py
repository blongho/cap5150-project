"""Tests for the KS drift helpers."""

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from drift.ks_tests import bh_correct, ks_on_scores, ks_per_feature, run_ks_analysis


def test_bh_flags_nothing_at_high_p():
    assert not bh_correct(np.ones(50) * 0.9, 0.05).any()


def test_bh_flags_everything_at_zero_p():
    assert bh_correct(np.zeros(50), 0.05).all()


def test_bh_is_idempotent():
    p = np.random.default_rng(0).uniform(0, 0.1, size=50)
    np.testing.assert_array_equal(bh_correct(p, 0.05), bh_correct(p, 0.05))


def test_bh_is_less_strict_than_bonferroni():
    p = np.array([0.001, 0.01, 0.02, 0.03, 0.5])
    # Bonferroni at 0.05 keeps only p <= 0.01; BH keeps the first four.
    assert list(bh_correct(p, 0.05)) == [True, True, True, True, False]


def _frames():
    rng = np.random.default_rng(0)
    n = 400
    train = pd.DataFrame({"A": rng.binomial(1, 0.2, n), "B": rng.binomial(1, 0.5, n),
                          "Malware": rng.integers(0, 2, n)})
    test = pd.DataFrame({"A": rng.binomial(1, 0.8, n), "B": rng.binomial(1, 0.5, n),
                         "Malware": rng.integers(0, 2, n)})
    return train, test


def test_ks_per_feature_detects_the_shifted_feature():
    train, test = _frames()
    t = ks_per_feature(train, test, ["A", "B"]).set_index("permission")
    assert t.loc["A", "p_value"] < 1e-10
    assert t.loc["B", "p_value"] > 0.01


def test_ks_on_scores():
    res = ks_on_scores(np.zeros(100), np.ones(100))
    assert res["statistic"] == 1.0 and res["p_value"] < 1e-10


def test_run_ks_analysis_writes_outputs(tmp_path):
    train, test = _frames()
    model = RandomForestClassifier(n_estimators=5, random_state=0)
    model.fit(train[["A", "B"]], train["Malware"])
    table = run_ks_analysis(train, {2015: test, 2014: train}, ["A", "B"], model, tmp_path, 0.05)
    assert list(table.columns) == ["year", "permission", "statistic", "p_raw", "p_adj",
                                   "flagged"]
    assert list(table["year"]) == [2014, 2014, 2015, 2015]
    flagged = table.set_index(["year", "permission"])["flagged"]
    assert not flagged[(2014, "A")] and flagged[(2015, "A")]
    assert (tmp_path / "ks_scores.csv").exists()
    meta = json.loads((tmp_path / "ks_run.json").read_text())
    assert "Benjamini-Hochberg" in meta["correction"]
    assert meta["flagged_per_year"] == {"2014": 0, "2015": 1}
