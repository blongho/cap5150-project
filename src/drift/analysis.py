"""Supporting analyses for the report.

Replication: RF on all 166 permissions with a random stratified split, compared with the
accuracy reported by Sabbah et al. (sanity check of the pipeline).

Masking check: for each listed permission, the share of malware and benign apps requesting it
before and after its change year. If request rates do not fall after the change, masking
models deployment-time feature loss rather than drift visible in app manifests.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from drift.data import LABEL, YEAR
from drift.metrics import detection_metrics


def replication(df: pd.DataFrame, perms: list[str], rep: dict[str, Any], n_jobs: int = -1
                ) -> dict[str, float]:
    """Default RF on a random stratified split; metrics plus the reference accuracy."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[perms], df[LABEL], test_size=rep["test_size"], stratify=df[LABEL],
        random_state=rep["seed"],
    )
    model = RandomForestClassifier(random_state=rep["seed"], n_jobs=n_jobs).fit(X_tr, y_tr)
    m = detection_metrics(y_te.to_numpy(), model.predict(X_te))
    m["reference_accuracy"] = rep["reference_accuracy"]
    m["difference"] = m["accuracy"] - rep["reference_accuracy"]
    return m


def masking_check(df: pd.DataFrame, years: dict[str, int]) -> pd.DataFrame:
    """Request rate of each permission before and after its change year, by class."""
    rates = df.groupby([YEAR, LABEL])[list(years)].mean()
    rows = []
    for perm, change_year in sorted(years.items()):
        for label, name in ((1, "malware"), (0, "benign")):
            r = rates.xs(label, level=LABEL)[perm]
            before, after = r[r.index < change_year], r[r.index >= change_year]
            rows.append({
                "permission": perm, "change_year": change_year, "class": name,
                "rate_before": before.mean() if len(before) else float("nan"),
                "rate_after": after.mean() if len(after) else float("nan"),
                "years_after_in_data": int(len(after)),
            })
    return pd.DataFrame(rows).round(4)
