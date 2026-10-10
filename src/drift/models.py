"""Model factory. All models expose fit / predict / predict_proba."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier


def build_model(model_cfg: dict[str, Any], seed: int):
    name = model_cfg["name"]
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=model_cfg.get("n_estimators", 100),
            class_weight=model_cfg.get("class_weight"),
            n_jobs=model_cfg.get("n_jobs", -1),
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {name}")
