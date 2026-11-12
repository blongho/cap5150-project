"""Model factory. All models expose fit / predict / predict_proba."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier

# Keys accepted under an `mlp` model config besides `name`; defaults live in drift.mlp.MLP.
MLP_KEYS = frozenset({"hidden_size", "dropout", "epochs", "lr", "patience", "batch_size",
                      "val_fraction"})


def build_model(model_cfg: dict[str, Any], seed: int):
    """Build an unfitted model from its config block, seeded with `seed`."""
    name = model_cfg["name"]
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=model_cfg.get("n_estimators", 100),
            class_weight=model_cfg.get("class_weight"),
            n_jobs=model_cfg.get("n_jobs", -1),
            random_state=seed,
        )
    if name == "mlp":
        unknown = set(model_cfg) - MLP_KEYS - {"name"}
        if unknown:
            raise ValueError(f"Unknown mlp config keys: {sorted(unknown)}")
        # Imported here so the random forest path works where torch is not installed
        # (PyTorch publishes no wheels for Intel macOS with Python 3.13).
        from drift.mlp import MLP

        params = {k: model_cfg[k] for k in MLP_KEYS if k in model_cfg}
        return MLP(random_state=seed, **params)
    raise ValueError(f"Unknown model: {name}")
