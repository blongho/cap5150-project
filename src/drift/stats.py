"""Paired Wilcoxon signed-rank tests on per-seed AUT for the three hypotheses.

Each hypothesis is directional, so tests are one-sided with alternative "a > b":

    H1  masking hurts:        none_unmasked  > none_masked
    H2  pruning helps:        pruning_masked > none_masked
    H3  retraining helps more: retrain_masked > pruning_masked

Seeds pair the runs. With 5 seeds the smallest attainable one-sided p-value is 1/32 = 0.031,
so results are reported with an effect size (matched-pairs rank-biserial correlation).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

log = logging.getLogger(__name__)

MIN_SEEDS_FOR_POWER = 5
HYPOTHESES = (
    ("H1", "masking lowers AUT", "none_unmasked", "none_masked"),
    ("H2", "pruning recovers AUT under masking", "pruning_masked", "none_masked"),
    ("H3", "retraining beats pruning under masking", "retrain_masked", "pruning_masked"),
)


def rank_biserial(a, b) -> float:
    """Matched-pairs rank-biserial correlation: (R+ - R-) / (R+ + R-) over nonzero diffs."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = rankdata(np.abs(d))
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


def wilcoxon_aut(a: list[float], b: list[float], alpha: float = 0.05) -> dict:
    """One-sided paired Wilcoxon signed-rank test of a > b.

    Returns statistic, p_value, effect_size_r, conclusion and limited_power (fewer than
    MIN_SEEDS_FOR_POWER pairs). When every difference is zero the test is undefined; this
    returns p_value 1.0 and effect size 0.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("a and b must be non-empty 1-D sequences of equal length")
    if np.all(a == b):
        statistic, p_value = 0.0, 1.0
    else:
        res = wilcoxon(a, b, alternative="greater")
        statistic, p_value = float(res.statistic), float(res.pvalue)
    reject = p_value < alpha
    return {
        "statistic": statistic,
        "p_value": p_value,
        "effect_size_r": rank_biserial(a, b),
        "conclusion": f"{'reject' if reject else 'fail to reject'} H0 at alpha={alpha}",
        "limited_power": bool(a.size < MIN_SEEDS_FOR_POWER),
    }


def run_hypotheses(results_dir: Path, models=("rf", "mlp"), metric: str = "aut_f1",
                   alpha: float = 0.05) -> pd.DataFrame:
    """Test H1-H3 for each model whose per-seed AUT CSVs exist; write hypothesis_tests.csv.

    Models with any missing CSV are skipped with a warning; raises FileNotFoundError if no
    model has all the CSVs.
    """
    results_dir = Path(results_dir)
    rows = []
    for model in models:
        paths = {cell: results_dir / f"{model}_{cell}_aut.csv"
                 for _, _, a, b in HYPOTHESES for cell in (a, b)}
        missing = sorted(p.name for p in paths.values() if not p.exists())
        if missing:
            log.warning("model=%s skipped; missing %s", model, missing)
            continue
        auts = {cell: pd.read_csv(p).set_index("seed")[metric] for cell, p in paths.items()}
        for hid, claim, a_cell, b_cell in HYPOTHESES:
            pair = pd.concat([auts[a_cell].rename("a"), auts[b_cell].rename("b")],
                             axis=1, join="inner").sort_index()
            rows.append({
                "hypothesis": hid, "claim": claim, "model": model, "metric": metric,
                "a": a_cell, "b": b_cell, "n_seeds": len(pair),
                "mean_a": pair["a"].mean(), "mean_b": pair["b"].mean(),
                **wilcoxon_aut(pair["a"].tolist(), pair["b"].tolist(), alpha),
            })
    if not rows:
        raise FileNotFoundError(f"No model has all per-seed AUT CSVs in {results_dir}")
    table = pd.DataFrame(rows)
    table.to_csv(results_dir / "hypothesis_tests.csv", index=False, float_format="%.6g")
    return table
