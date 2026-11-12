"""Tests for the figure and table builder on synthetic results CSVs."""

import pandas as pd
import pytest

from drift.experiments import CORE_CONDITIONS, DEFENSES, MODELS, cell_name
from drift.figures import EXTRA_CONDITIONS, core_cells, make_all

YEARS = [2014, 2015, 2016]


def _write_cell(results, cell: str, level: float, labelled: int = 0) -> None:
    pd.DataFrame({"year": YEARS, "f1_mean": [level] * 3, "f1_std": [0.01] * 3,
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
