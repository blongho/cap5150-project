"""Tests for the Wilcoxon hypothesis helpers."""

import pandas as pd
import pytest

from drift.stats import HYPOTHESES, rank_biserial, run_hypotheses, wilcoxon_aut


def test_identical_samples_give_p_one():
    res = wilcoxon_aut([0.9] * 5, [0.9] * 5)
    assert res["p_value"] == 1.0
    assert res["effect_size_r"] == 0.0
    assert res["conclusion"].startswith("fail to reject")


def test_clear_improvement_is_significant():
    res = wilcoxon_aut([0.9, 0.85, 0.88, 0.87, 0.86], [0.5, 0.48, 0.52, 0.49, 0.51])
    assert res["p_value"] < 0.05
    assert res["effect_size_r"] > 0
    assert res["conclusion"].startswith("reject")
    assert res["limited_power"] is False


def test_one_sided_direction():
    res = wilcoxon_aut([0.5, 0.48, 0.52, 0.49, 0.51], [0.9, 0.85, 0.88, 0.87, 0.86])
    assert res["p_value"] > 0.5
    assert res["effect_size_r"] == -1.0


def test_few_pairs_flag_limited_power():
    assert wilcoxon_aut([0.8, 0.7], [0.5, 0.4])["limited_power"] is True


def test_rank_biserial_mixed_signs():
    # diffs +1, -2, +3 -> ranks 1, 2, 3 -> (1 + 3 - 2) / 6
    assert rank_biserial([1, 0, 3], [0, 2, 0]) == pytest.approx(2 / 6)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        wilcoxon_aut([0.1, 0.2], [0.1])


def test_run_hypotheses_writes_table(tmp_path):
    seeds = [0, 1, 2, 3, 4]
    level = {"none_unmasked": 0.88, "none_masked": 0.79, "pruning_masked": 0.87,
             "retrain_masked": 0.90}
    for cell, value in level.items():
        pd.DataFrame({"seed": seeds, "aut_f1": [value + 0.001 * s for s in seeds]}).to_csv(
            tmp_path / f"rf_{cell}_aut.csv", index=False)
    table = run_hypotheses(tmp_path, models=("rf", "mlp"))  # mlp has no CSVs: skipped
    assert list(table["hypothesis"]) == [h[0] for h in HYPOTHESES]
    assert set(table["model"]) == {"rf"}
    assert (table["p_value"] < 0.05).all()
    assert (tmp_path / "hypothesis_tests.csv").exists()


def test_run_hypotheses_without_results_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_hypotheses(tmp_path)
