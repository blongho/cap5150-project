"""Figures and LaTeX tables for the final report, built only from results/ CSVs.

    f1_decay_{model}.pdf/.png  per-year F1 (mean +/- 1 std over seeds): unmasked and masked
                               with no defense, masked with pruning, masked with retraining
    aut_summary.pdf/.png       AUT(F1) of the 12 core cells (dot plot, error bars = std)
    aut_table.tex              core AUT table with the labels each defense used
    aut_extras_table.tex       separate and timed masking (written when those CSVs exist)
    other tables / numbers.tex see drift.tables (each written when its inputs exist)

Colors are the first categorical slots of a CVD-validated palette, in fixed order. Aqua and
yellow are below 3:1 contrast on white, so every series also has its own marker and line
style, and the tables carry the exact values.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from drift.experiments import CORE_CONDITIONS, DEFENSES, MODELS, cell_name  # noqa: E402
from drift.tables import make_tables  # noqa: E402

MODEL_LABELS = {"rf": "RF", "mlp": "MLP"}
DEFENSE_LABELS = {"none": "None", "pruning": "Pruning", "retrain": "5\\% retraining"}
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # blue, orange, aqua, yellow
INK, MUTED = "#1f1f1e", "#6b6a64"
# Lines in the decay figure: (defense, condition, label, marker, line style).
DECAY_LINES = (
    ("none", "unmasked", "Unmasked, no defense", "o", "-"),
    ("none", "masked", "Masked, no defense", "s", "-"),
    ("pruning", "masked", "Masked, pruning", "^", "--"),
    ("retrain", "masked", "Masked, 5% retraining", "D", ":"),
)
EXTRA_CONDITIONS = ("masked_deprecated", "masked_restricted", "timed_masking")


def core_cells() -> list[str]:
    return [cell_name(m, d, c) for m in MODELS for d in DEFENSES for c in CORE_CONDITIONS]


def require(results_dir: Path, names: list[str]) -> None:
    """Raise FileNotFoundError naming every missing results file."""
    missing = [n for n in names if not (results_dir / n).exists()]
    if missing:
        raise FileNotFoundError(f"Missing results in {results_dir}: {missing}; "
                                "run `make experiments` first")


def _style(ax) -> None:
    ax.grid(alpha=0.3, lw=0.5)
    ax.tick_params(labelsize=7, colors=MUTED, labelcolor=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)


def _save(fig, path: Path) -> list[Path]:
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [path.with_suffix(".pdf"), path.with_suffix(".png")]


def f1_decay(results_dir: Path, figures_dir: Path, model: str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for color, (defense, condition, label, marker, ls) in zip(PALETTE, DECAY_LINES,
                                                               strict=True):
        g = pd.read_csv(results_dir / f"{cell_name(model, defense, condition)}.csv")
        ax.plot(g["year"], g["f1_mean"], ls, marker=marker, ms=3.5, lw=1.4, color=color,
                label=label)
        ax.fill_between(g["year"], g["f1_mean"] - g["f1_std"], g["f1_mean"] + g["f1_std"],
                        color=color, alpha=0.15, lw=0)
    ax.set_xlabel("Test year", fontsize=8, color=INK)
    ax.set_ylabel(f"Malware F1 ({MODEL_LABELS[model]})", fontsize=8, color=INK)
    ax.set_ylim(0.5, 1.0)
    _style(ax)
    # Above the plot: inside, it would cover the masked curve's dips.
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0),
              handlelength=2.2, columnspacing=1.0, borderaxespad=0.2)
    return _save(fig, figures_dir / f"f1_decay_{model}")


def aut_summary(results_dir: Path, figures_dir: Path) -> list[Path]:
    rows = [(d, c) for d in DEFENSES for c in CORE_CONDITIONS]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    offsets = {"rf": -0.14, "mlp": 0.14}
    for color, marker, model in zip(PALETTE, ("o", "s"), MODELS, strict=False):
        for i, (d, c) in enumerate(rows):
            auts = pd.read_csv(results_dir / f"{cell_name(model, d, c)}_aut.csv")["aut_f1"]
            ax.errorbar(auts.mean(), i + offsets[model], xerr=auts.std(), fmt=marker,
                        ms=4, color=color, ecolor=color, elinewidth=1, capsize=2,
                        label=MODEL_LABELS[model] if i == 0 else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{DEFENSE_LABELS[d].replace(chr(92), '')}, {c}" for d, c in rows])
    ax.invert_yaxis()
    ax.set_xlabel("AUT(F1), mean $\\pm$ std over seeds", fontsize=8, color=INK)
    _style(ax)
    ax.grid(axis="y", alpha=0)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    return _save(fig, figures_dir / "aut_summary")


def _mean_std(results_dir: Path, cell: str, col: str = "aut_f1") -> str:
    s = pd.read_csv(results_dir / f"{cell}_aut.csv")[col]
    return f"{s.mean():.3f} $\\pm$ {s.std():.3f}"


def aut_table(results_dir: Path, figures_dir: Path) -> Path:
    """LaTeX tabular: AUT(F1) unmasked / masked per model and defense, plus labels used."""
    lines = [
        "\\begin{tabular}{llccr}", "\\toprule",
        "Model & Defense & Unmasked & Masked & Labels \\\\", "\\midrule",
    ]
    for model in MODELS:
        for defense in DEFENSES:
            per_year = pd.read_csv(results_dir / f"{cell_name(model, defense, 'masked')}.csv")
            labels = int(round(per_year["n_labelled_mean"].iloc[-1]))
            lines.append(
                f"{MODEL_LABELS[model]} & {DEFENSE_LABELS[defense]} & "
                f"{_mean_std(results_dir, cell_name(model, defense, 'unmasked'))} & "
                f"{_mean_std(results_dir, cell_name(model, defense, 'masked'))} & "
                f"{labels:,} \\\\")
        lines.append("\\midrule" if model != MODELS[-1] else "\\bottomrule")
    lines.append("\\end{tabular}")
    path = figures_dir / "aut_table.tex"
    path.write_text("\n".join(lines) + "\n")
    return path


def aut_extras_table(results_dir: Path, figures_dir: Path) -> Path | None:
    """LaTeX tabular for the separate and timed masking conditions, if they were run."""
    cells = [cell_name(m, d, c) for m in MODELS for d in DEFENSES for c in EXTRA_CONDITIONS]
    if any(not (results_dir / f"{c}_aut.csv").exists() for c in cells):
        return None
    lines = [
        "\\begin{tabular}{llccc}", "\\toprule",
        "Model & Defense & Deprecated only & Restricted only & Timed \\\\", "\\midrule",
    ]
    for model in MODELS:
        for defense in DEFENSES:
            values = " & ".join(_mean_std(results_dir, cell_name(model, defense, c))
                                for c in EXTRA_CONDITIONS)
            lines.append(f"{MODEL_LABELS[model]} & {DEFENSE_LABELS[defense]} & {values} \\\\")
        lines.append("\\midrule" if model != MODELS[-1] else "\\bottomrule")
    lines.append("\\end{tabular}")
    path = figures_dir / "aut_extras_table.tex"
    path.write_text("\n".join(lines) + "\n")
    return path


def make_all(results_dir: Path, figures_dir: Path) -> list[Path]:
    """Build every figure and table; raises FileNotFoundError if a core CSV is missing."""
    cells = core_cells()
    require(results_dir, [f"{c}.csv" for c in cells] + [f"{c}_aut.csv" for c in cells])
    figures_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for model in MODELS:
        out += f1_decay(results_dir, figures_dir, model)
    out += aut_summary(results_dir, figures_dir)
    out.append(aut_table(results_dir, figures_dir))
    extras = aut_extras_table(results_dir, figures_dir)
    if extras is not None:
        out.append(extras)
    return out + make_tables(results_dir, figures_dir)
