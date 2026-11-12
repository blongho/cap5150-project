"""Tests for the figure and table builder on synthetic results CSVs."""

import pandas as pd
import pytest

from drift.experiments import CORE_CONDITIONS, DEFENSES, MODELS, cell_name
from drift.figures import EXTRA_CONDITIONS, core_cells, make_all

YEARS = [2014, 2015, 2016]


def _write_cell(results, cell: str, level: float, labelled: int = 0) -> None:
    pd.DataFrame({"year": YEARS, "f1_mean": [level] * 3, "f1_std": [0.01] * 3,
                  "recall_mean": [level] * 3, "recall_std": [0.01] * 3,
                  "n_labelled_mean": [labelled // 3, labelled // 2, labelled]}).to_csv(
        results / f"{cell}.csv", index=False)
    pd.DataFrame({"seed": [0, 1, 2], "aut_f1": [level - 0.01, level, level + 0.01]}).to_csv(
        results / f"{cell}_aut.csv", index=False)


@pytest.fixture
def results(tmp_path):
    out = tmp_path / "results"
    out.mkdir()
    for m in MODELS:
        for d in DEFENSES:
            for c in CORE_CONDITIONS:
                _write_cell(out, cell_name(m, d, c), 0.85, 1200 if d == "retrain" else 0)
    return out


def test_builds_every_figure_and_table(results, tmp_path):
    figures = tmp_path / "figures"
    written = {p.name for p in make_all(results, figures)}
    assert {"f1_decay_rf.pdf", "f1_decay_rf.png", "f1_decay_mlp.pdf", "f1_decay_mlp.png",
            "recall_decay_rf.pdf", "recall_decay_mlp.pdf",
            "aut_summary.pdf", "aut_summary.png", "aut_table.tex"} <= written
    assert "aut_extras_table.tex" not in written  # extras not run
    table = (figures / "aut_table.tex").read_text()
    assert "0.850 $\\pm$ 0.010" in table
    assert "1,200" in table


def test_extras_table_when_extra_cells_exist(results, tmp_path):
    for m in MODELS:
        for d in DEFENSES:
            for c in EXTRA_CONDITIONS:
                _write_cell(results, cell_name(m, d, c), 0.8)
    written = {p.name for p in make_all(results, tmp_path / "figures")}
    assert "aut_extras_table.tex" in written


def test_missing_core_csv_is_named(results, tmp_path):
    victim = core_cells()[3]
    (results / f"{victim}.csv").unlink()
    with pytest.raises(FileNotFoundError, match=victim):
        make_all(results, tmp_path / "figures")


def test_report_tables_from_results(results, tmp_path):
    import json

    from drift.tables import make_tables

    pd.DataFrame({"benign": [10, 20], "malware": [5, 5], "total": [15, 25]},
                 index=pd.Index([2014, 2015], name="year")).to_csv(results / "year_counts.csv")
    (results / "replication.json").write_text(json.dumps(
        {"accuracy": 0.9529, "reference_accuracy": 0.954}))
    pd.DataFrame({"hypothesis": ["H1", "H1"], "model": ["rf", "mlp"],
                  "a": ["none_unmasked"] * 2, "b": ["none_masked"] * 2,
                  "mean_a": [0.9, 0.88], "mean_b": [0.8, 0.85], "p_value": [0.031, 0.2],
                  "effect_size_r": [1.0, 0.4],
                  "conclusion": ["reject H0 at alpha=0.05", "fail to reject H0"]}).to_csv(
        results / "hypothesis_tests.csv", index=False)
    pd.DataFrame({"year": [2014, 2014], "permission": ["GET_TASKS", "OTHER"],
                  "flagged": [True, False]}).to_csv(results / "ks_tests.csv", index=False)
    pd.DataFrame({"year": [2014], "statistic": [0.4]}).to_csv(results / "ks_scores.csv",
                                                              index=False)
    (results / "rf_none_masked_seed0.json").write_text(
        json.dumps({"listed_permissions": ["GET_TASKS"]}))
    pd.DataFrame({"permission": ["GET_TASKS"] * 2, "change_year": [2014] * 2,
                  "class": ["malware", "benign"], "rate_before": [0.19, 0.1],
                  "rate_after": [0.57, float("nan")]}).to_csv(
        results / "masking_check.csv", index=False)

    out = tmp_path / "tables"
    out.mkdir()
    names = {p.name for p in make_tables(results, out)}
    assert names == {"numbers.tex", "hypothesis_table.tex", "ks_table.tex", "masking_table.tex"}
    assert "\\newcommand{\\NApps}{40}" in (out / "numbers.tex").read_text()
    assert "1 / 2 & 1 / 1 & 0.400" in (out / "ks_table.tex").read_text()
    hyp = (out / "hypothesis_table.tex").read_text()
    assert ("H1 & None, unmasked $>$ None, masked & \\textbf{0.900} & 0.800 & \\textbf{0.031} & "
            "$+1.00$ & \\textbf{0.880} & 0.850 & 0.200 & $+0.40$ \\\\") in hyp
    masking = (out / "masking_table.tex").read_text()
    assert "\\texttt{GET\\_TASKS} & 2014 & 19 $\\to$ 57 & 10 $\\to$ --" in masking


def test_aut_table_bolds_best_defense_per_column(results, tmp_path):
    for m in MODELS:
        for c in CORE_CONDITIONS:
            _write_cell(results, cell_name(m, "retrain", c), 0.9, 1200)
    _write_cell(results, cell_name("mlp", "pruning", "masked"), 0.95)
    make_all(results, tmp_path / "figures")
    rows = {line.split(" & ")[0]: line for line in
            (tmp_path / "figures" / "aut_table.tex").read_text().splitlines() if " & " in line}
    assert rows["5\\% retraining"].count("\\textbf") == 3  # all but MLP masked
    assert rows["Pruning"].count("\\textbf") == 1  # MLP masked
    assert "\\textbf" not in rows["None"]
    assert rows["5\\% retraining"].endswith("& 1,200 \\\\")


def test_bold_best_direction_and_ties():
    from drift.tables import bold_best

    s = {k: pd.Series(v) for k, v in {"a": [0.9, 0.9], "b": [0.8, 0.8], "c": [0.9, 0.9]}.items()}
    hi = bold_best(s)
    assert hi["a"].startswith("\\textbf") and hi["c"].startswith("\\textbf")
    assert not hi["b"].startswith("\\textbf")
    lo = bold_best(s, higher_is_better=False)  # e.g. FPR
    assert lo["b"].startswith("\\textbf") and not lo["a"].startswith("\\textbf")
