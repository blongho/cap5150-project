"""LaTeX tables and number macros for the report, built only from results/ files.

Each builder returns the path it wrote, or None when its inputs are missing (that analysis
was not run). Written to the figures directory next to the figures:

    numbers.tex             \\newcommand macros: dataset size, replication accuracy
    hypothesis_table.tex    H1-H3 Wilcoxon results     (results/hypothesis_tests.csv)
    share10_table.tex       10% malware-share variant  (results/*_10pct_aut.csv)
    ks_table.tex            KS drift per test year     (results/ks_tests.csv, ks_scores.csv)
    masking_table.tex       request rates around each change year (results/masking_check.csv)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from drift.experiments import CORE_CONDITIONS, DEFENSES, MODELS, cell_name

MODEL_LABELS = {"rf": "RF", "mlp": "MLP"}
DEFENSE_LABELS = {"none": "None", "pruning": "Pruning", "retrain": "5\\% retraining"}
CONDITION_LABELS = {"unmasked": "Unmasked", "masked": "Masked"}


def _tex(s: str) -> str:
    return s.replace("_", "\\_").replace("%", "\\%")


def _write(path: Path, header: str, cols: str, rows: list[str]) -> Path:
    body = [f"\\begin{{tabular}}{{{cols}}}", "\\toprule", header + " \\\\", "\\midrule",
            *rows, "\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(body) + "\n")
    return path


def _ms(s: pd.Series) -> str:
    return f"{s.mean():.3f} $\\pm$ {s.std():.3f}"


def numbers(results_dir: Path, out_dir: Path) -> Path | None:
    counts_path, rep_path = results_dir / "year_counts.csv", results_dir / "replication.json"
    if not counts_path.exists() or not rep_path.exists():
        return None
    counts = pd.read_csv(counts_path, index_col=0)
    rep = json.loads(rep_path.read_text())
    macros = {
        "NApps": f"{int(counts['total'].sum()):,}",
        "NMalware": f"{int(counts['malware'].sum()):,}",
        "ReplicationAccuracy": f"{rep['accuracy']:.3f}",
        "ReferenceAccuracy": f"{rep['reference_accuracy']:.3f}",
    }
    path = out_dir / "numbers.tex"
    path.write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
    return path


def hypothesis_table(results_dir: Path, out_dir: Path) -> Path | None:
    src = results_dir / "hypothesis_tests.csv"
    if not src.exists():
        return None
    t = pd.read_csv(src)
    rows = [f"{r.hypothesis} & {MODEL_LABELS.get(r.model, r.model)} & {r.mean_a:.3f} & "
            f"{r.mean_b:.3f} & {r.p_value:.3f} & {r.effect_size_r:+.2f} \\\\"
            for r in t.itertuples()]
    return _write(out_dir / "hypothesis_table.tex",
                  "Hyp. & Model & AUT $a$ & AUT $b$ & $p$ & $r$", "llcccc", rows)


def share10_table(results_dir: Path, out_dir: Path) -> Path | None:
    cells = [(m, d, c) for m in MODELS for d in DEFENSES for c in CORE_CONDITIONS]
    paths = {k: results_dir / f"{cell_name(*k, '_10pct')}_aut.csv" for k in cells}
    if any(not p.exists() for p in paths.values()):
        return None
    rows = []
    for (m, d, c), p in paths.items():
        a = pd.read_csv(p)
        rows.append(f"{MODEL_LABELS[m]} & {DEFENSE_LABELS[d]} & {CONDITION_LABELS[c]} & "
                    f"{_ms(a['aut_f1'])} & {_ms(a['aut_recall'])} & {_ms(a['aut_fpr'])} \\\\")
    return _write(out_dir / "share10_table.tex",
                  "Model & Defense & Condition & AUT(F1) & AUT(recall) & AUT(FPR)",
                  "lllccc", rows)


def ks_table(results_dir: Path, out_dir: Path) -> Path | None:
    features, scores = results_dir / "ks_tests.csv", results_dir / "ks_scores.csv"
    records = sorted(results_dir.glob("*_seed*.json"))
    if not features.exists() or not scores.exists() or not records:
        return None
    listed = set(json.loads(records[0].read_text())["listed_permissions"])
    f, s = pd.read_csv(features), pd.read_csv(scores).set_index("year")
    n_perms = f.groupby("year")["permission"].nunique().iloc[0]
    rows = []
    for year, g in f.groupby("year"):
        rows.append(f"{year} & {int(g['flagged'].sum())} / {n_perms} & "
                    f"{int(g[g['permission'].isin(listed)]['flagged'].sum())} / {len(listed)} & "
                    f"{s.loc[year, 'statistic']:.3f} \\\\")
    return _write(out_dir / "ks_table.tex",
                  "Year & Flagged (all) & Flagged (listed) & KS on RF scores", "lccc", rows)


def masking_table(results_dir: Path, out_dir: Path) -> Path | None:
    src = results_dir / "masking_check.csv"
    if not src.exists():
        return None
    t = pd.read_csv(src)
    wide = t.pivot(index=["permission", "change_year"], columns="class",
                   values=["rate_before", "rate_after"])

    def pct(v: float) -> str:
        return "--" if pd.isna(v) else f"{100 * v:.0f}"

    rows = [f"\\texttt{{{_tex(perm)}}} & {year} & "
            f"{pct(r[('rate_before', 'malware')])} $\\to$ {pct(r[('rate_after', 'malware')])} & "
            f"{pct(r[('rate_before', 'benign')])} $\\to$ {pct(r[('rate_after', 'benign')])} \\\\"
            for (perm, year), r in wide.sort_index(level=[1, 0]).iterrows()]
    return _write(out_dir / "masking_table.tex",
                  "Permission & Change & Malware \\% & Benign \\%", "lccc", rows)


def make_tables(results_dir: Path, out_dir: Path) -> list[Path]:
    built = [b(results_dir, out_dir) for b in
             (numbers, hypothesis_table, share10_table, ks_table, masking_table)]
    return [p for p in built if p is not None]
