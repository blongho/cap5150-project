"""Tests for the MLP model and the build_model factory's mlp branch."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from drift.models import build_model


def _toy(n: int = 100, d: int = 10, seed: int = 0):
    """Binary permission-like features with a learnable label."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, d)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] >= 1).astype(int)
    return X, y


def _small_mlp(seed: int = 0):
    return build_model({"name": "mlp", "hidden_size": 16, "epochs": 5}, seed=seed)


def test_unknown_mlp_key_raises():
    with pytest.raises(ValueError, match="bad_key"):
        build_model({"name": "mlp", "bad_key": 1}, seed=0)


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model({"name": "svm"}, seed=0)


def test_random_forest_branch_unchanged():
    rf = build_model({"name": "random_forest", "n_estimators": 7, "n_jobs": 1}, seed=3)
    assert (rf.n_estimators, rf.random_state, rf.n_jobs) == (7, 3, 1)


# Tests below need torch, which has no wheels for Intel macOS with Python 3.13.
requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch is not installed"
)


@requires_torch
def test_build_model_defaults_have_interface():
    model = build_model({"name": "mlp"}, seed=0)
    for method in ("fit", "predict", "predict_proba"):
        assert callable(getattr(model, method))
    assert (model.hidden_size, model.dropout, model.epochs, model.lr, model.patience) == (
        256, 0.3, 50, 1e-3, 5,
    )
    assert model.random_state == 0


@requires_torch
def test_build_model_passes_config():
    cfg = {"name": "mlp", "hidden_size": 8, "dropout": 0.1, "epochs": 3, "lr": 0.01,
           "patience": 2, "batch_size": 16, "val_fraction": 0.2}
    model = build_model(cfg, seed=7)
    assert (model.hidden_size, model.dropout, model.epochs, model.lr, model.patience,
            model.batch_size, model.val_fraction, model.random_state) == (
        8, 0.1, 3, 0.01, 2, 16, 0.2, 7,
    )


@requires_torch
def test_fit_predict_binary_outputs():
    X, y = _toy()
    pred = _small_mlp().fit(X, y).predict(X)
    assert pred.shape == (100,)
    assert set(np.unique(pred)) <= {0, 1}


@requires_torch
def test_predict_proba_is_a_distribution():
    X, y = _toy()
    proba = _small_mlp().fit(X, y).predict_proba(X)
    assert proba.shape == (100, 2)
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


@requires_torch
def test_same_seed_same_predictions():
    X, y = _toy()
    a = _small_mlp(seed=11).fit(X, y)
    b = _small_mlp(seed=11).fit(X, y)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))


@requires_torch
def test_refit_with_same_seed_is_identical():
    X, y = _toy()
    model = _small_mlp(seed=5)
    first = model.fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(first, model.fit(X, y).predict_proba(X))


@requires_torch
def test_accepts_dataframe_and_learns():
    X, y = _toy(n=400)
    frame = pd.DataFrame(X, columns=[f"P{i}" for i in range(X.shape[1])])
    model = build_model({"name": "mlp", "hidden_size": 16, "epochs": 30, "lr": 0.01},
                        seed=0)
    assert (model.fit(frame, pd.Series(y)).predict(frame) == y).mean() > 0.9


@requires_torch
def test_early_stopping_restores_best_weights():
    import torch

    X, y = _toy()
    model = build_model({"name": "mlp", "hidden_size": 16, "epochs": 50, "patience": 1,
                         "lr": 0.5}, seed=0).fit(X, y)
    assert model.n_epochs_ <= 50
    rng = np.random.default_rng(0)
    order = rng.permutation(len(X))
    val = order[: round(0.1 * len(X))]
    with torch.no_grad():
        logits = model(torch.from_numpy(X[val]))
    counts = np.bincount(y[order[len(val):]], minlength=2)
    weights = torch.tensor(len(order[len(val):]) / (2 * counts), dtype=torch.float32)
    loss = torch.nn.CrossEntropyLoss(weight=weights)(logits, torch.from_numpy(y[val]))
    assert loss.item() == pytest.approx(model.best_val_loss_, rel=1e-5)


@requires_torch
def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        build_model({"name": "mlp"}, seed=0).predict(np.zeros((2, 3)))


@requires_torch
def test_rejects_non_binary_labels():
    X, _ = _toy()
    with pytest.raises(ValueError, match="binary"):
        _small_mlp().fit(X, np.full(len(X), 2))


@requires_torch
def test_finetune_requires_fit_and_is_reproducible():
    X, y = _toy()
    with pytest.raises(RuntimeError, match="not fitted"):
        _small_mlp().finetune(X, y, epochs=1, lr=1e-3)
    a = _small_mlp(seed=2).fit(X, y).finetune(X[:20], y[:20], epochs=3, lr=0.01)
    b = _small_mlp(seed=2).fit(X, y).finetune(X[:20], y[:20], epochs=3, lr=0.01)
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))
