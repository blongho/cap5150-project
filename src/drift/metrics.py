"""Per-year detection metrics and AUT (Pendlebury et al., TESSERACT, 2019)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def detection_metrics(y_true, y_pred) -> dict[str, float]:
    """Metrics with malware (label 1) as the positive class."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "malware_share": float(np.mean(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else float("nan"),
    }


def aut(values: list[float]) -> float:
    """Area Under Time: trapezoidal mean of a metric over N >= 2 ordered test periods.

    AUT(f, N) = 1/(N-1) * sum_{k=1}^{N-1} (f(x_k) + f(x_{k+1})) / 2
    """
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        raise ValueError("AUT needs at least two test periods")
    if np.isnan(v).any():
        raise ValueError("AUT is undefined when a period has no value")
    return float(np.mean((v[:-1] + v[1:]) / 2))
