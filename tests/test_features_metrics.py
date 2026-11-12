import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from drift.features import mask, prune
from drift.metrics import aut, detection_metrics


@pytest.fixture
def X():
    return pd.DataFrame({"A": [1, 0, 1], "B": [1, 1, 0], "C": [0, 1, 1]})


def test_mask_zeroes_only_listed(X):
    out = mask(X, ["A"])
    assert (out["A"] == 0).all()
    pd.testing.assert_frame_equal(out[["B", "C"]], X[["B", "C"]])
    assert X["A"].tolist() == [1, 0, 1]  # input untouched


def test_prune_drops_only_listed(X):
    assert list(prune(X, ["A", "C"]).columns) == ["B"]


@pytest.mark.parametrize("fn", [mask, prune])
def test_unknown_permission_raises(X, fn):
    with pytest.raises(KeyError):
        fn(X, ["Z"])


def test_detection_metrics_match_sklearn():
    y = np.array([1, 1, 0, 0, 1])
    p = np.array([1, 0, 0, 1, 1])
    m = detection_metrics(y, p)
    assert m["f1"] == pytest.approx(f1_score(y, p))
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["fpr"] == pytest.approx(1 / 2)


def test_aut_trapezoid():
    assert aut([1.0, 0.0]) == pytest.approx(0.5)
    assert aut([0.8, 0.6, 0.4]) == pytest.approx(0.6)


def test_aut_rejects_bad_input():
    with pytest.raises(ValueError):
        aut([0.9])
    with pytest.raises(ValueError):
        aut([0.9, float("nan")])


def test_timed_mask_zeroes_only_from_change_year():
    X = pd.DataFrame({"A": [1, 1, 1, 1], "B": [1, 1, 1, 1], "year": [2015, 2016, 2017, 2018]})
    out = mask(X, ["A", "B"], change_years={"A": 2017})
    assert list(out["A"]) == [1, 1, 0, 0]
    assert list(out["B"]) == [1, 1, 1, 1]  # no change year: never masked
    assert list(X["A"]) == [1, 1, 1, 1]  # input untouched


def test_timed_mask_validates_inputs():
    X = pd.DataFrame({"A": [1], "year": [2017]})
    with pytest.raises(KeyError):
        mask(X, ["A"], change_years={"MISSING": 2017})
    with pytest.raises(KeyError, match="year"):
        mask(X.drop(columns="year"), ["A"], change_years={"A": 2017})
