"""Integration smoke test for the experiment runner on a tiny synthetic dataset."""

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from drift.config import load_config, resolve
from drift.data import EXPECTED_PERMISSIONS, FIRST_PERMISSION, HASH, LABEL, LAST_PERMISSION, YEAR
from drift.experiments import cell_name, record_path, run_experiments

PERMISSION_LIST = resolve("data/deprecated_permissions.csv")
SEEDS = [0, 1]
YEARS = [2010, 2011, 2012, 2013, 2014]
DEFENSES = ["none", "pruning", "retrain"]
CONDITIONS = ["unmasked", "masked"]
MODELS = ["rf", pytest.param("mlp", marks=pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch is not installed"))]


def _permission_names() -> list[str]:
    """166 KronoDroid-style names that include every listed permission."""
    listed = sorted(set(pd.read_csv(PERMISSION_LIST)["permission"])
                    - {FIRST_PERMISSION, LAST_PERMISSION})
    filler = [f"FILLER_{i}" for i in range(EXPECTED_PERMISSIONS - 2 - len(listed))]
    return [FIRST_PERMISSION, *listed, *filler, LAST_PERMISSION]


@pytest.fixture
def cfg(tmp_path):
    """Config pointing at a 50-sample synthetic parquet (10 per year, both classes)."""
    rng = np.random.default_rng(0)
    perms = _permission_names()
    n = 50
    labels = np.tile([0, 1], n // 2)
    X = rng.integers(0, 2, size=(n, len(perms)))
    X[:, 1] = labels  # one learnable signal
    df = pd.DataFrame(X, columns=perms).astype("int8")
    df.insert(0, HASH, [f"h{i}" for i in range(n)])
    df.insert(1, LABEL, labels)
    df[YEAR] = np.repeat(YEARS, n // len(YEARS))
    parquet = tmp_path / "data.parquet"
    df.to_parquet(parquet)

    base = load_config(resolve("configs/base.yaml"))
    base["data"].update(processed_path=str(parquet), min_year=YEARS[0], max_year=YEARS[-1])
    base["permissions"]["list_path"] = str(PERMISSION_LIST)
    base["split"] = {"train_end_year": 2012, "test_years": [2013, 2014]}
    base["model"].update(n_estimators=5, n_jobs=1)
    base["mlp"].update(hidden_size=8, epochs=3)
    base["experiment"].update(seeds=SEEDS, results_dir=str(tmp_path / "results"),
                              separate_masking=False, timed_masking=False,
                              malware_share_10pct=False)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(base))
    return load_config(path)


@pytest.mark.parametrize("model", MODELS)
def test_runner_writes_records_and_skips_existing(cfg, model):
    written = run_experiments(cfg, [model], DEFENSES, CONDITIONS)
    results = resolve(cfg["experiment"]["results_dir"])
    assert len(written) == len(DEFENSES) * len(CONDITIONS) * len(SEEDS)

    for defense in DEFENSES:
        for condition in CONDITIONS:
            for seed in SEEDS:
                rec = json.loads(record_path(results, model, defense, condition, seed).read_text())
                for key in ("git_commit", "seed", "model", "condition", "defense",
                            "python_version", "sklearn_version", "torch_version",
                            "pandas_version", "config", "per_year", "aut"):
                    assert key in rec
                assert (rec["model"], rec["defense"], rec["condition"], rec["seed"]) == (
                    model, defense, condition, seed)
                assert sorted(rec["per_year"]) == ["2013", "2014"]
            name = cell_name(model, defense, condition)
            per_year = pd.read_csv(results / f"{name}.csv")
            assert list(per_year["year"]) == [2013, 2014]
            assert {"f1_mean", "f1_std", "recall_mean", "fpr_mean"} <= set(per_year.columns)
            assert sorted(pd.read_csv(results / f"{name}_aut.csv")["seed"]) == SEEDS

    before = {p.name: p.stat().st_mtime_ns for p in results.glob("*.json")}
    assert run_experiments(cfg, [model], DEFENSES, CONDITIONS) == []
    after = {p.name: p.stat().st_mtime_ns for p in results.glob("*.json")}
    assert after == before


def test_pruning_drops_listed_features_and_ignores_masking(cfg):
    run_experiments(cfg, ["rf"], ["none", "pruning"], CONDITIONS)
    results = resolve(cfg["experiment"]["results_dir"])
    none = json.loads(record_path(results, "rf", "none", "unmasked", 0).read_text())
    pruned_u = json.loads(record_path(results, "rf", "pruning", "unmasked", 0).read_text())
    pruned_m = json.loads(record_path(results, "rf", "pruning", "masked", 0).read_text())
    assert none["n_features"] == EXPECTED_PERMISSIONS
    assert pruned_u["n_features"] == EXPECTED_PERMISSIONS - len(none["listed_permissions"])
    # With the listed permissions removed, masking them changes nothing.
    assert pruned_u["per_year"] == pruned_m["per_year"]


def test_retrain_records_cumulative_labels_and_scores_held_out_apps(cfg):
    run_experiments(cfg, ["rf"], ["none", "retrain"], ["masked"])
    results = resolve(cfg["experiment"]["results_dir"])
    none = json.loads(record_path(results, "rf", "none", "masked", 0).read_text())
    retrain = json.loads(record_path(results, "rf", "retrain", "masked", 0).read_text())
    assert [none["per_year"][y]["n_labelled"] for y in ("2013", "2014")] == [0, 0]
    labelled = [retrain["per_year"][y]["n_labelled"] for y in ("2013", "2014")]
    assert 0 < labelled[0] < labelled[1]  # budgets accumulate over years
    for year in ("2013", "2014"):
        # Every defense is scored on the same held-out apps, never on the labelled budget.
        assert retrain["per_year"][year]["n"] == none["per_year"][year]["n"]
        assert retrain["per_year"][year]["n"] < 10  # 10 synthetic apps per year
    assert pd.read_csv(results / "rf_retrain_masked.csv")["n_labelled_mean"].iloc[-1] == (
        labelled[1])


def test_budget_fraction_is_required_for_every_defense(cfg):
    del cfg["retrain"]
    for defense in ("none", "retrain"):
        with pytest.raises(ValueError, match="retrain"):
            run_experiments(cfg, ["rf"], [defense], ["unmasked"])


def test_all_defenses_share_the_evaluation_set(cfg):
    run_experiments(cfg, ["rf"], ["none", "pruning", "retrain"], ["unmasked"])
    results = resolve(cfg["experiment"]["results_dir"])
    n = {d: [json.loads(record_path(results, "rf", d, "unmasked", s).read_text())
             ["per_year"]["2014"]["n"] for s in SEEDS] for d in ("none", "pruning", "retrain")}
    assert n["none"] == n["pruning"] == n["retrain"]


def test_unknown_names_are_rejected(cfg):
    with pytest.raises(ValueError, match="Unknown model"):
        run_experiments(cfg, ["svm"], ["none"], ["unmasked"])
    with pytest.raises(ValueError, match="Unknown condition"):
        run_experiments(cfg, ["rf"], ["none"], ["timed"])


def test_optional_conditions_and_share_variant(cfg):
    cfg["experiment"].update(separate_masking=True, timed_masking=True, malware_share_10pct=True)
    written = run_experiments(cfg, ["rf"], ["none"])
    results = resolve(cfg["experiment"]["results_dir"])
    names = {p.name for p in written}
    for cond in ("unmasked", "masked", "masked_deprecated", "masked_restricted",
                 "timed_masking"):
        assert f"rf_none_{cond}_seed0.json" in names
        assert (results / f"rf_none_{cond}_aut.csv").exists()
    for cond in ("unmasked", "masked"):
        assert f"rf_none_{cond}_10pct_seed0.json" in names
    assert "rf_none_timed_masking_10pct_seed0.json" not in names  # variant: core only
    assert len(written) == (5 + 2) * len(SEEDS)

    def rec(name):
        return json.loads((results / f"{name}_seed0.json").read_text())

    dep, res = rec("rf_none_masked_deprecated"), rec("rf_none_masked_restricted")
    assert set(dep["masked_permissions"]) | set(res["masked_permissions"]) == set(
        dep["listed_permissions"])
    # No listed permission changes before 2014, so timed masking leaves 2013 untouched.
    timed, unmasked = rec("rf_none_timed_masking"), rec("rf_none_unmasked")
    assert min(timed["change_years"].values()) == 2014
    assert timed["per_year"]["2013"] == unmasked["per_year"]["2013"]
    share = rec("rf_none_unmasked_10pct")
    assert share["variant"] == "10pct"
    for year in ("2013", "2014"):
        assert share["per_year"][year]["n"] < unmasked["per_year"][year]["n"]
        assert share["per_year"][year]["malware_share"] < 0.5
