"""Retraining defense: label a small share of each new year and update the detector.

For every test year in order, a stratified `frac` of the year's apps is labelled (the budget)
and the rest is held out for scoring (the eval set). Budgets accumulate over years. The RF is
refitted on the original training data plus all budgets so far; the MLP is fine-tuned on all
budgets so far.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.base import clone

from drift.data import HASH, LABEL

log = logging.getLogger(__name__)

MIN_PER_CLASS = 2


def _enough(df: pd.DataFrame) -> bool:
    """True when df holds at least MIN_PER_CLASS samples of each class."""
    counts = df[LABEL].value_counts()
    return all(counts.get(label, 0) >= MIN_PER_CLASS for label in (0, 1))


def sample_budget(df_year: pd.DataFrame, frac: float, seed: int
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one year into (budget_df, eval_df) with a label-stratified budget of `frac`.

    Each class contributes round(frac * class count) apps, at least one. If the year has fewer
    than MIN_PER_CLASS apps of either class, logs a warning and returns (empty, full year).
    """
    if not 0 < frac < 1:
        raise ValueError("frac must lie in (0, 1)")
    if not _enough(df_year):
        log.warning("year has fewer than %d apps of a class; no labelled budget", MIN_PER_CLASS)
        return df_year.iloc[0:0].copy(), df_year.copy()
    parts = [g.sample(n=max(1, round(frac * len(g))), random_state=seed)
             for _, g in df_year.groupby(LABEL, sort=True)]
    budget = pd.concat(parts)
    eval_df = df_year[~df_year[HASH].isin(budget[HASH])]
    return budget, eval_df


def cumulative_budget(budgets: list[pd.DataFrame]) -> pd.DataFrame:
    """All budgets so far as one frame; an empty frame when there are none."""
    if not budgets:
        return pd.DataFrame()
    return pd.concat(budgets)


def retrain_rf(model, X_train: pd.DataFrame, y_train: pd.Series, budget_df: pd.DataFrame,
               perms: list[str], seed: int):
    """Fit a fresh copy of `model` on the training data plus the budget, seeded with `seed`.

    Returns `model` unchanged (with a warning) when the budget is too small.
    """
    if budget_df.empty or not _enough(budget_df):
        log.warning("budget has fewer than %d apps of a class; RF not retrained", MIN_PER_CLASS)
        return model
    X = pd.concat([X_train[perms], budget_df[perms]], ignore_index=True)
    y = pd.concat([y_train, budget_df[LABEL]], ignore_index=True)
    return clone(model).set_params(random_state=seed).fit(X, y)


def finetune_mlp(mlp, budget_df: pd.DataFrame, perms: list[str], epochs: int, lr: float):
    """Fine-tune `mlp` in place on the budget and return it.

    Returns `mlp` unchanged (with a warning) when the budget is too small.
    """
    if budget_df.empty or not _enough(budget_df):
        log.warning("budget has fewer than %d apps of a class; MLP not fine-tuned", MIN_PER_CLASS)
        return mlp
    return mlp.finetune(budget_df[perms], budget_df[LABEL], epochs=epochs, lr=lr)
