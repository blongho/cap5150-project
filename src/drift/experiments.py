"""Full experiment grid: model x defense x condition x seed.

Each run trains on years <= split.train_end_year and scores every test year. It writes one
Run_Record JSON per (model, defense, condition, seed); runs whose record already exists are
skipped. Once every seed of a cell is present, two aggregate CSVs are written:

    {model}_{defense}_{condition}.csv      per-year mean and std over seeds
    {model}_{defense}_{condition}_aut.csv  AUT(F1), AUT(recall), AUT(FPR) per seed

Conditions: `unmasked` scores the test data as is; `masked` zeroes the listed permissions at
test time. Optional conditions (config `experiment`): `separate_masking` adds
`masked_deprecated` and `masked_restricted`, which zero one category only; `timed_masking`
zeroes each listed permission only in test years on or after its change year.
Optional variant: `malware_share_10pct` reruns the core conditions with each test year's
malware cut to `malware_share_target` (cells suffixed `_10pct`).

Defenses: `none`; `pruning` removes the listed permissions from training and test data;
`retrain` labels a stratified `retrain.budget_fraction` of each test year (in year order),
updates the model on all budgets so far (RF: refit on training data + budgets; MLP: fine-tune a
copy of the base model on the budgets) and scores the rest of that year. Under a masking
condition the budget is masked too, as it comes from the same deployment. Each year records
`n_labelled`, the cumulative budget size (0 for the other defenses).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from drift.config import resolve
from drift.data import LABEL, YEAR, permission_columns, subsample_malware, time_split
from drift.defenses import cumulative_budget, finetune_mlp, retrain_rf, sample_budget
from drift.features import mask, prune
from drift.metrics import aut, detection_metrics
from drift.models import build_model
from drift.permissions import change_years, select
from drift.runinfo import run_info

log = logging.getLogger(__name__)

MODELS = ("rf", "mlp")
DEFENSES = ("none", "pruning", "retrain")
CORE_CONDITIONS = ("unmasked", "masked")
SEPARATE_CONDITIONS = ("masked_deprecated", "masked_restricted")
TIMED_CONDITION = "timed_masking"
CONDITIONS = (*CORE_CONDITIONS, *SEPARATE_CONDITIONS, TIMED_CONDITION)
# Masking plan: condition -> (permissions to zero, change years for timed masking or None).
Plan = dict[str, tuple[list[str], "dict[str, int] | None"]]
# Config section holding each model's build_model settings.
MODEL_SECTIONS = {"rf": "model", "mlp": "mlp"}
METRICS = ("f1", "recall", "precision", "fpr", "accuracy", "n", "n_labelled")
AUT_METRICS = ("f1", "recall", "fpr")
RETRAIN_KEYS = ("budget_fraction", "mlp_finetune_epochs", "mlp_finetune_lr")


def cell_name(model: str, defense: str, condition: str, variant: str = "") -> str:
    return f"{model}_{defense}_{condition}{variant}"


def record_path(results_dir: Path, model: str, defense: str, condition: str, seed: int,
                variant: str = "") -> Path:
    return results_dir / f"{cell_name(model, defense, condition, variant)}_seed{seed}.json"


def default_conditions(cfg: dict[str, Any]) -> list[str]:
    """Core conditions plus the optional ones enabled in config `experiment`."""
    exp = cfg["experiment"]
    out = list(CORE_CONDITIONS)
    if exp.get("separate_masking", False):
        out += SEPARATE_CONDITIONS
    if exp.get("timed_masking", False):
        out.append(TIMED_CONDITION)
    return out


def masking_plan(table: pd.DataFrame, cfg: dict[str, Any]) -> tuple[list[str], Plan]:
    """The listed permissions and what each condition masks."""
    p = cfg["permissions"]
    listed = select(table, p["categories"], p["min_api"], p["max_api"])
    plan: Plan = {
        "unmasked": ([], None),
        "masked": (listed, None),
        "masked_deprecated": (select(table, ["deprecated"], p["min_api"], p["max_api"]), None),
        "masked_restricted": (select(table, ["restricted"], p["min_api"], p["max_api"]), None),
        TIMED_CONDITION: (listed, change_years(table, listed, p["categories"], p["min_api"],
                                               p["max_api"])),
    }
    return listed, plan


def features(df: pd.DataFrame, perms: list[str], listed: list[str], plan: Plan,
             condition: str, defense: str) -> pd.DataFrame:
    """Permission features of df after the test condition (masking) and the defense (pruning)."""
    to_mask, years = plan[condition]
    if years is not None:
        X = mask(df[[*perms, YEAR]], to_mask, years).drop(columns=YEAR)
    elif to_mask:
        X = mask(df[perms], to_mask)
    else:
        X = df[perms]
    if defense == "pruning":
        X = prune(X, listed)
    return X


def _validate(models, defenses, conditions, cfg: dict[str, Any]) -> None:
    exp = cfg["experiment"]
    if exp.get("malware_share_10pct", False) and "malware_share_target" not in exp:
        raise ValueError("experiment.malware_share_10pct needs experiment.malware_share_target")
    for kind, given, allowed in (("model", models, MODELS), ("defense", defenses, DEFENSES),
                                 ("condition", conditions, CONDITIONS)):
        unknown = sorted(set(given) - set(allowed))
        if unknown:
            raise ValueError(f"Unknown {kind}(s) {unknown}; choose from {list(allowed)}")
    for m in models:
        if MODEL_SECTIONS[m] not in cfg:
            raise ValueError(f"Config has no `{MODEL_SECTIONS[m]}` section for model {m!r}")
    if "retrain" in defenses:
        missing = [k for k in RETRAIN_KEYS if k not in cfg.get("retrain", {})]
        if missing:
            raise ValueError(f"Config `retrain` section is missing keys {missing}")
    if "mlp" in models and importlib.util.find_spec("torch") is None:
        raise RuntimeError(
            "The mlp model needs PyTorch, which is not installed here (no wheels for Intel "
            "macOS). Run in Docker or Colab, or pass --models rf."
        )


def _score(model, tests: dict[int, pd.DataFrame], perms, listed, plan, condition,
           defense) -> dict:
    out = {}
    for year, test in tests.items():
        X = features(test, perms, listed, plan, condition, defense)
        out[str(year)] = {**detection_metrics(test[LABEL].to_numpy(), model.predict(X)),
                          "n_labelled": 0}
    return out


def _retrain(model_name: str, base, train, tests, perms, listed, plan, condition, seed,
             cfg: dict[str, Any]) -> dict:
    """Score each test year after updating `base` on the labelled budgets so far."""
    rcfg = cfg["retrain"]
    budgets: list[pd.DataFrame] = []
    out = {}
    for year in sorted(tests):
        # Distinct, reproducible sampling seed per (seed, year).
        sample_seed = seed * 10_000 + year
        budget, eval_df = sample_budget(tests[year], rcfg["budget_fraction"], sample_seed)
        budget = budget.copy()
        budget[perms] = features(budget, perms, listed, plan, condition, "retrain")
        budgets.append(budget)
        labelled = cumulative_budget(budgets)
        if model_name == "rf":
            model = retrain_rf(base, train[perms], train[LABEL], labelled, perms, seed)
        else:
            model = finetune_mlp(copy.deepcopy(base), labelled, perms,
                                 rcfg["mlp_finetune_epochs"], rcfg["mlp_finetune_lr"])
        X = features(eval_df, perms, listed, plan, condition, "retrain")
        out[str(year)] = {**detection_metrics(eval_df[LABEL].to_numpy(), model.predict(X)),
                          "n_labelled": int(len(labelled))}
    return out


def _subsampled(tests: dict[int, pd.DataFrame], target: float, seed: int
                ) -> dict[int, pd.DataFrame]:
    """Test years with malware cut to `target`; years already at or below it are kept."""
    out = {}
    for year, test in tests.items():
        if test[LABEL].mean() <= target:
            log.warning("year %d malware share %.3f <= %.3f; kept as is",
                        year, test[LABEL].mean(), target)
            out[year] = test
        else:
            out[year] = subsample_malware(test, target, seed * 10_000 + year)
    return out


def run_experiments(
    cfg: dict[str, Any],
    models=MODELS,
    defenses=DEFENSES,
    conditions=None,
) -> list[Path]:
    """Run every missing (model, defense, condition, seed[, variant]); return records written.

    `conditions` defaults to the core conditions plus those enabled in the config.
    """
    conditions = list(conditions) if conditions is not None else default_conditions(cfg)
    _validate(models, defenses, conditions, cfg)
    exp = cfg["experiment"]
    results_dir = resolve(exp["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    seeds = exp["seeds"]

    df = pd.read_parquet(resolve(cfg["data"]["processed_path"]))
    perms = permission_columns(df.columns)
    table = pd.read_csv(resolve(cfg["permissions"]["list_path"]))
    listed, plan = masking_plan(table, cfg)
    split = cfg["split"]
    train, tests = time_split(df, split["train_end_year"], split["test_years"])
    info = run_info()
    if TIMED_CONDITION in conditions:
        years = plan[TIMED_CONDITION][1]
        for year in tests:
            n = sum(y <= year for y in years.values())
            log.info("timed masking: %d of %d listed permissions masked in %d",
                     n, len(listed), year)

    # (variant suffix, conditions) pairs; the share variant covers the core conditions only.
    variants = [("", conditions)]
    if exp.get("malware_share_10pct", False):
        variants.append(("_10pct", [c for c in conditions if c in CORE_CONDITIONS]))

    written: list[Path] = []
    for model_name in models:
        model_cfg = cfg[MODEL_SECTIONS[model_name]]
        for defense in defenses:
            for seed in seeds:
                todo = [(v, c) for v, conds in variants for c in conds
                        if not record_path(results_dir, model_name, defense, c, seed, v).exists()]
                if not todo:
                    log.info("model=%s defense=%s seed=%d exists, skipped",
                             model_name, defense, seed)
                    continue
                X_train = features(train, perms, listed, plan, "unmasked", defense)
                model = build_model(model_cfg, seed).fit(X_train, train[LABEL])
                for variant, condition in todo:
                    test_sets = tests if not variant else _subsampled(
                        tests, exp["malware_share_target"], seed)
                    if defense == "retrain":
                        per_year = _retrain(model_name, model, train, test_sets, perms, listed,
                                            plan, condition, seed, cfg)
                    else:
                        per_year = _score(model, test_sets, perms, listed, plan, condition,
                                          defense)
                    record = {
                        "model": model_name, "defense": defense, "condition": condition,
                        "variant": variant.lstrip("_") or "full", "seed": seed, **info,
                        "listed_permissions": listed,
                        "masked_permissions": plan[condition][0],
                        "change_years": plan[condition][1],
                        "n_features": int(X_train.shape[1]), "n_train": int(len(train)),
                        "aut": {m: aut([per_year[str(y)][m] for y in tests])
                                for m in AUT_METRICS},
                        "per_year": per_year, "config": cfg,
                    }
                    path = record_path(results_dir, model_name, defense, condition, seed,
                                       variant)
                    path.write_text(json.dumps(record, indent=2, default=float))
                    written.append(path)
                    log.info("model=%s defense=%s condition=%s%s seed=%d done",
                             model_name, defense, condition, variant, seed)

    for model_name in models:
        for defense in defenses:
            for variant, conds in variants:
                for condition in conds:
                    write_aggregates(results_dir, model_name, defense, condition, seeds,
                                     variant)
    return written


def write_aggregates(results_dir: Path, model: str, defense: str, condition: str,
                     seeds: list[int], variant: str = "") -> bool:
    """Write the cell's per-year and per-seed AUT CSVs; False if a seed's record is missing."""
    name = cell_name(model, defense, condition, variant)
    paths = [record_path(results_dir, model, defense, condition, s, variant) for s in seeds]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        log.warning("%s: missing %s; aggregates not written", name, missing)
        return False
    records = [json.loads(p.read_text()) for p in paths]

    # Records written before n_labelled existed come from defenses that label nothing.
    rows = [{"seed": r["seed"], "year": int(y), **{m: v.get(m, 0) for m in METRICS}}
            for r in records for y, v in r["per_year"].items()]
    per_year = pd.DataFrame(rows).groupby("year")[list(METRICS)].agg(["mean", "std"])
    per_year.columns = [f"{m}_{stat}" for m, stat in per_year.columns]
    per_year.round(6).to_csv(results_dir / f"{name}.csv")

    auts = pd.DataFrame([{"seed": r["seed"], **{f"aut_{m}": r["aut"][m] for m in AUT_METRICS}}
                         for r in records])
    auts.round(6).to_csv(results_dir / f"{name}_aut.csv", index=False)
    return True
