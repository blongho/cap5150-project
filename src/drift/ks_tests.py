"""Kolmogorov-Smirnov drift tests between the training window and each test year.

Per feature: one two-sample KS test per permission flag, with Benjamini-Hochberg correction
across the permissions of a year. On scores: one KS test on the detector's P(malware) for
training apps versus test-year apps.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, ks_2samp

BH_METHOD = "Benjamini-Hochberg (scipy.stats.false_discovery_control, method='bh')"


def ks_per_feature(train_df: pd.DataFrame, test_df: pd.DataFrame, perms: list[str]
                   ) -> pd.DataFrame:
    """One KS test per permission; columns permission, statistic, p_value."""
    rows = []
    for perm in perms:
        res = ks_2samp(train_df[perm].to_numpy(), test_df[perm].to_numpy())
        rows.append({"permission": perm, "statistic": float(res.statistic),
                     "p_value": float(res.pvalue)})
    return pd.DataFrame(rows, columns=["permission", "statistic", "p_value"])


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    return false_discovery_control(np.asarray(pvalues, dtype=float), method="bh")


def bh_correct(pvalues: np.ndarray, alpha: float) -> np.ndarray:
    """Boolean mask of p-values rejected at false discovery rate `alpha` (Benjamini-Hochberg)."""
    return bh_adjust(pvalues) <= alpha


def ks_on_scores(train_scores: np.ndarray, test_scores: np.ndarray) -> dict[str, float]:
    """KS test on model output scores; returns statistic and p_value."""
    res = ks_2samp(np.asarray(train_scores), np.asarray(test_scores))
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue)}


def run_ks_analysis(train_df: pd.DataFrame, test_dfs: dict[int, pd.DataFrame],
                    perms: list[str], model, results_dir: Path, alpha: float,
                    run_info: dict | None = None) -> pd.DataFrame:
    """Per-feature and score KS tests for each test year.

    Writes results/ks_tests.csv (year, permission, statistic, p_raw, p_adj, flagged),
    results/ks_scores.csv (year, statistic, p_value) and results/ks_run.json (method, alpha,
    provenance). `model` must be fitted on train_df[perms]. Returns the per-feature table.
    """
    results_dir = Path(results_dir)
    train_scores = model.predict_proba(train_df[perms])[:, 1]
    feature_rows, score_rows = [], []
    for year in sorted(test_dfs):
        test = test_dfs[year]
        t = ks_per_feature(train_df, test, perms).rename(columns={"p_value": "p_raw"})
        t["p_adj"] = bh_adjust(t["p_raw"].to_numpy())
        t["flagged"] = t["p_adj"] <= alpha
        t.insert(0, "year", year)
        feature_rows.append(t)
        score_rows.append({"year": year,
                           **ks_on_scores(train_scores, model.predict_proba(test[perms])[:, 1])})
    features = pd.concat(feature_rows, ignore_index=True)
    features.to_csv(results_dir / "ks_tests.csv", index=False, float_format="%.6g")
    pd.DataFrame(score_rows).to_csv(results_dir / "ks_scores.csv", index=False,
                                    float_format="%.6g")
    flagged = features.groupby("year")["flagged"].sum()
    meta = {"correction": BH_METHOD, "alpha": alpha, "n_permissions": len(perms),
            "flagged_per_year": {str(y): int(n) for y, n in flagged.items()},
            **(run_info or {})}
    (results_dir / "ks_run.json").write_text(json.dumps(meta, indent=2))
    return features
