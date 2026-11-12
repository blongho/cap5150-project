"""Tests for the retraining defense helpers."""

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from drift.data import HASH, LABEL
from drift.defenses import cumulative_budget, finetune_mlp, retrain_rf, sample_budget

PERMS = ["P0", "P1", "P2"]


def _year(n_malware: int, n_benign: int, prefix: str = "y", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_malware + n_benign
    df = pd.DataFrame(rng.integers(0, 2, size=(n, len(PERMS))), columns=PERMS)
    df.insert(0, HASH, [f"{prefix}{i}" for i in range(n)])
    df[LABEL] = [1] * n_malware + [0] * n_benign
    return df


def test_sample_budget_preserves_malware_share():
    year = _year(n_malware=300, n_benign=700)
    budget, _ = sample_budget(year, frac=0.05, seed=0)
    assert len(budget) == 50
    assert abs(budget[LABEL].mean() - year[LABEL].mean()) <= 0.05


def test_sample_budget_and_eval_are_disjoint_and_cover_the_year():
    year = _year(n_malware=120, n_benign=80)
    budget, eval_df = sample_budget(year, frac=0.05, seed=3)
    assert set(budget[HASH]).isdisjoint(eval_df[HASH])
    assert len(budget) + len(eval_df) == len(year)


def test_sample_budget_is_reproducible():
    year = _year(n_malware=120, n_benign=80)
    a, _ = sample_budget(year, frac=0.05, seed=7)
    b, _ = sample_budget(year, frac=0.05, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_sample_budget_with_too_few_malware_warns_and_labels_nothing(caplog):
    year = _year(n_malware=1, n_benign=50)
    with caplog.at_level(logging.WARNING):
        budget, eval_df = sample_budget(year, frac=0.05, seed=0)
    assert budget.empty
    pd.testing.assert_frame_equal(eval_df, year)
    assert "fewer than" in caplog.text


def test_cumulative_budget():
    a, b, c = (_year(3, 3, prefix=p) for p in "abc")
    pd.testing.assert_frame_equal(cumulative_budget([a, b, c]), pd.concat([a, b, c]))
    assert cumulative_budget([]).empty


class RecordingRF(RandomForestClassifier):
    """Random forest that records every training set it is fitted on."""

    seen: list[pd.DataFrame] = []

    def fit(self, X, y, sample_weight=None):
        RecordingRF.seen.append(X.copy())
        return super().fit(X, y, sample_weight)


def test_retrain_rf_never_trains_on_eval_rows():
    RecordingRF.seen = []
    train = _year(40, 40, prefix="t")
    budget, eval_df = sample_budget(_year(100, 100, prefix="e", seed=1), frac=0.05, seed=0)
    base = RecordingRF(n_estimators=3, random_state=0)
    model = retrain_rf(base, train[PERMS], train[LABEL], budget, PERMS, seed=5)

    assert model is not base
    assert model.random_state == 5
    (X_fit,) = RecordingRF.seen
    assert len(X_fit) == len(train) + len(budget)
    # Rows are identified by their features plus position; eval rows must not appear.
    budget_rows = budget[PERMS].reset_index(drop=True)
    pd.testing.assert_frame_equal(X_fit.iloc[len(train):].reset_index(drop=True), budget_rows)
    assert set(budget[HASH]).isdisjoint(eval_df[HASH])


def test_retrain_rf_with_small_budget_returns_model_unchanged(caplog):
    base = RandomForestClassifier(n_estimators=3, random_state=0)
    train = _year(10, 10)
    with caplog.at_level(logging.WARNING):
        assert retrain_rf(base, train[PERMS], train[LABEL], _year(1, 5), PERMS, 0) is base
        assert retrain_rf(base, train[PERMS], train[LABEL], pd.DataFrame(), PERMS, 0) is base
    assert "not retrained" in caplog.text


def test_finetune_mlp_with_small_budget_returns_model_unchanged(caplog):
    sentinel = object()
    with caplog.at_level(logging.WARNING):
        assert finetune_mlp(sentinel, _year(5, 1), PERMS, epochs=1, lr=1e-3) is sentinel
    assert "not fine-tuned" in caplog.text


def test_finetune_mlp_updates_in_place():
    pytest.importorskip("torch")
    from drift.models import build_model

    train = _year(60, 60, prefix="t")
    mlp = build_model({"name": "mlp", "hidden_size": 8, "epochs": 2}, seed=0)
    mlp.fit(train[PERMS], train[LABEL])
    before = mlp.predict_proba(train[PERMS])
    out = finetune_mlp(mlp, _year(20, 20, prefix="b", seed=2), PERMS, epochs=3, lr=0.05)
    assert out is mlp
    assert not np.array_equal(before, mlp.predict_proba(train[PERMS]))
