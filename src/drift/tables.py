"""LaTeX tables and number macros for the report, built only from results/ files.

Each builder returns the path it wrote, or None when its inputs are missing (that analysis
was not run). Written to the figures directory next to the figures:

    numbers.tex             \\newcommand macros: dataset size, replication accuracy
    dataset_table.tex       apps per year, malware share and role (results/year_counts.csv)
    settings_table.tex      model, defense and protocol settings (config in a run record)
    hypothesis_table.tex    RQ1-RQ3 Wilcoxon results    (results/hypothesis_tests.csv)
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


def bold_best(values: dict, higher_is_better: bool = True) -> dict:
    """Mean $\\pm$ s.d. strings for a comparison group, with the best mean in bold.

    `values` maps any key to a per-seed Series. Means are compared at the displayed precision
    (3 decimals), so ties are all bold.
    """
    means = {k: round(float(v.mean()), 3) for k, v in values.items()}
    best = (max if higher_is_better else min)(means.values())
    return {k: f"\\textbf{{{_ms(v)}}}" if means[k] == best else _ms(v)
            for k, v in values.items()}


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


def dataset_table(results_dir: Path, out_dir: Path) -> Path | None:
    counts_path, records = results_dir / "year_counts.csv", sorted(
        results_dir.glob("*_seed*.json"))
    if not counts_path.exists() or not records:
        return None
    record = json.loads(records[0].read_text())
    if "config" not in record:
        return None
    split = record["config"]["split"]
    counts = pd.read_csv(counts_path, index_col=0)
    rows = []
    for year, r in counts.iterrows():
        role = ("train" if year <= split["train_end_year"]
                else "test" if year in split["test_years"] else "--")
        rows.append(f"{year} & {int(r['benign']):,} & {int(r['malware']):,} & "
                    f"{100 * r['malware'] / r['total']:.0f} & {role} \\\\")
    return _write(out_dir / "dataset_table.tex",
                  "Year & Benign & Malware & Malware \\% & Role", "rrrrl", rows)


def settings_table(results_dir: Path, out_dir: Path) -> Path | None:
    records = sorted(results_dir.glob("*_seed*.json"))
    if not records:
        return None
    record = json.loads(records[0].read_text())
    if "config" not in record:
        return None
    cfg = record["config"]
    rf, mlp, rt, split = cfg["model"], cfg["mlp"], cfg["retrain"], cfg["split"]
    seeds = cfg["experiment"]["seeds"]
    test_years = split["test_years"]
    rows = [
        f"Split & train $\\le$ {split['train_end_year']}; test {test_years[0]}--"
        f"{test_years[-1]}, each year \\\\",
        f"Seeds & {len(seeds)} ({seeds[0]}--{seeds[-1]}) \\\\",
        f"RF & {rf['n_estimators']} trees, class weight {_tex(str(rf['class_weight']))} \\\\",
        f"MLP & 2 $\\times$ {mlp['hidden_size']} ReLU, dropout {mlp['dropout']} \\\\",
        f"MLP training & Adam, lr {mlp['lr']:g}, batch {mlp['batch_size']}, "
        f"$\\le${mlp['epochs']} epochs \\\\",
        f"Early stopping & {100 * mlp['val_fraction']:.0f}\\% validation, patience "
        f"{mlp['patience']} \\\\",
        f"Retraining & {100 * rt['budget_fraction']:.0f}\\% of each test year, stratified "
        f"\\\\",
        f"MLP fine-tuning & {rt['mlp_finetune_epochs']} epochs, lr {rt['mlp_finetune_lr']:g} "
        f"\\\\",
    ]
    return _write(out_dir / "settings_table.tex", "Setting & Value", "ll", rows)


def hypothesis_table(results_dir: Path, out_dir: Path) -> Path | None:
    src = results_dir / "hypothesis_tests.csv"
    if not src.exists():
        return None
    t = pd.read_csv(src)
    models = [m for m in MODELS if m in set(t["model"])]
    by = t.set_index(["hypothesis", "model"])

    def side(cell: str) -> str:
        defense, condition = cell.split("_", 1)
        return f"{DEFENSE_LABELS[defense]}, {condition}"

    def stats(h: str, m: str) -> str:
        r = by.loc[(h, m)]
        p = f"{r.p_value:.3f}"
        if str(r.conclusion).startswith("reject"):  # significant at the configured alpha
            p = f"\\textbf{{{p}}}"
        a, b = round(r.mean_a, 3), round(r.mean_b, 3)  # bold the better side (ties: both)
        a_txt, b_txt = (f"\\textbf{{{v:.3f}}}" if v == max(a, b) else f"{v:.3f}" for v in (a, b))
        return f"{a_txt} & {b_txt} & {p} & ${r.effect_size_r:+.2f}$"

    rows = []
    for h, g in t.groupby("hypothesis", sort=True):
        first = g.iloc[0]
        label = h.replace("H", "RQ", 1)  # stored IDs H1-H3 are shown as research questions
        rows.append(f"{label} & {side(first.a)} $>$ {side(first.b)} & "
                    + " & ".join(stats(h, m) for m in models) + " \\\\")
    groups = " & ".join(f"\\multicolumn{{4}}{{c}}{{{MODEL_LABELS[m]}}}" for m in models)
    rules = " ".join(f"\\cmidrule(lr){{{3 + 4 * i}-{6 + 4 * i}}}" for i in range(len(models)))
    sub = " & ".join("$a$ & $b$ & $p$ & $r$" for _ in models)
    path = out_dir / "hypothesis_table.tex"
    path.write_text("\n".join([
        "\\begin{tabular}{ll" + "cccc" * len(models) + "}", "\\toprule",
        f" & & {groups} \\\\", rules,
        f"RQ & Comparison ($a > b$) & {sub} \\\\", "\\midrule",
        *rows, "\\bottomrule", "\\end{tabular}",
    ]) + "\n")
    return path


def share10_table(results_dir: Path, out_dir: Path) -> Path | None:
    cells = [(m, d, c) for m in MODELS for d in DEFENSES for c in CORE_CONDITIONS]
    paths = {k: results_dir / f"{cell_name(*k, '_10pct')}_aut.csv" for k in cells}
    if any(not p.exists() for p in paths.values()):
        return None
    data = {k: pd.read_csv(p) for k, p in paths.items()}
    metrics = (("aut_f1", True), ("aut_recall", True), ("aut_fpr", False))
    shown = {}
    for m in MODELS:  # compare the six settings of each model, per column
        group = [k for k in cells if k[0] == m]
        for col, higher in metrics:
            for k, text in bold_best({k: data[k][col] for k in group}, higher).items():
                shown[(k, col)] = text
    rows = []
    for k in cells:
        m, d, c = k
        rows.append(f"{MODEL_LABELS[m]} & {DEFENSE_LABELS[d]} & {CONDITION_LABELS[c]} & "
                    + " & ".join(shown[(k, col)] for col, _ in metrics) + " \\\\")
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
             (numbers, dataset_table, settings_table, hypothesis_table, share10_table,
              ks_table, masking_table)]
    return [p for p in built if p is not None]
