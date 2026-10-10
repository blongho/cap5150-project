"""Milestone experiments (RF only).

1. Replication: RF on all 166 permissions, random stratified 80/20 split (Sabbah et al.).
2. Time-aware decay: train on years <= train_end_year, test each later year, under three
   conditions: unmasked, forward-masked (listed permissions zeroed at test time), and the
   pruning defense (listed permissions removed from training and test data).
3. Masking check: yearly request rate of each listed permission, by class.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from drift.config import load_config, resolve  # noqa: E402
from drift.data import LABEL, YEAR, permission_columns, time_split  # noqa: E402
from drift.features import mask, prune  # noqa: E402
from drift.metrics import aut, detection_metrics  # noqa: E402
from drift.models import build_model  # noqa: E402
from drift.permissions import select  # noqa: E402

log = logging.getLogger("run_milestone")
CONDITIONS = ("unmasked", "masked", "pruned")


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def replication(df: pd.DataFrame, perms: list[str], cfg: dict) -> dict:
    rep = cfg["replication"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[perms], df[LABEL], test_size=rep["test_size"], stratify=df[LABEL],
        random_state=rep["seed"],
    )
    model = RandomForestClassifier(random_state=rep["seed"], n_jobs=-1).fit(X_tr, y_tr)
    m = detection_metrics(y_te.to_numpy(), model.predict(X_te))
    m["reference_accuracy"] = rep["reference_accuracy"]
    m["difference"] = m["accuracy"] - rep["reference_accuracy"]
    return m


def decay(df: pd.DataFrame, perms: list[str], listed: list[str], cfg: dict) -> pd.DataFrame:
    split = cfg["split"]
    train, tests = time_split(df, split["train_end_year"], split["test_years"])
    rows = []
    for seed in cfg["experiment"]["seeds"]:
        full = build_model(cfg["model"], seed).fit(train[perms], train[LABEL])
        pruned_train = prune(train[perms], listed)
        pruned = build_model(cfg["model"], seed).fit(pruned_train, train[LABEL])
        for year, test in tests.items():
            y = test[LABEL].to_numpy()
            preds = {
                "unmasked": full.predict(test[perms]),
                "masked": full.predict(mask(test[perms], listed)),
                "pruned": pruned.predict(prune(test[perms], listed)),
            }
            for cond, pred in preds.items():
                rows.append({"seed": seed, "condition": cond, "year": year,
                             **detection_metrics(y, pred)})
        log.info("seed %d done", seed)
    return pd.DataFrame(rows)


def aut_table(per_year: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cond, seed), g in per_year.groupby(["condition", "seed"]):
        g = g.sort_values("year")
        rows.append({"condition": cond, "seed": seed, "aut_f1": aut(g["f1"].tolist()),
                     "aut_recall": aut(g["recall"].tolist()), "aut_fpr": aut(g["fpr"].tolist())})
    t = pd.DataFrame(rows).groupby("condition")[["aut_f1", "aut_recall", "aut_fpr"]]
    return t.agg(["mean", "std"]).round(4).reindex(list(CONDITIONS))


def masking_check(df: pd.DataFrame, table: pd.DataFrame, listed: list[str]) -> pd.DataFrame:
    rates = df.groupby([YEAR, LABEL])[listed].mean()
    rows = []
    first_change = table[table.permission.isin(listed)].groupby("permission")["year"].min()
    for perm, change_year in first_change.items():
        for label, name in ((1, "malware"), (0, "benign")):
            r = rates.xs(label, level=LABEL)[perm]
            before, after = r[r.index < change_year], r[r.index >= change_year]
            rows.append({
                "permission": perm, "change_year": change_year, "class": name,
                "rate_before": before.mean() if len(before) else float("nan"),
                "rate_after": after.mean() if len(after) else float("nan"),
                "years_after_in_data": int(len(after)),
            })
    return pd.DataFrame(rows).round(4)


def plot(per_year: pd.DataFrame, out_path) -> None:
    labels = {"unmasked": "Unmasked", "masked": "Masked (no defense)", "pruned": "Pruned"}
    styles = {"unmasked": "-o", "masked": "-s", "pruned": "--^"}
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for cond in CONDITIONS:
        g = per_year[per_year.condition == cond].groupby("year")["f1"].agg(["mean", "std"])
        ax.plot(g.index, g["mean"], styles[cond], ms=3, lw=1.2, label=labels[cond])
        ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], alpha=0.2)
    ax.set_xlabel("Test year")
    ax.set_ylabel("Malware F1")
    ax.set_ylim(0.5, 1.0)
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(resolve(args.config))
    results = resolve(cfg["experiment"]["results_dir"])
    figures = resolve(cfg["experiment"]["figures_dir"])
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(resolve(cfg["data"]["processed_path"]))
    perms = permission_columns(df.columns)
    table = pd.read_csv(resolve(cfg["permissions"]["list_path"]))
    p = cfg["permissions"]
    listed = select(table, p["categories"], p["min_api"], p["max_api"])
    log.info("%d listed permissions: %s", len(listed), listed)

    rep = replication(df, perms, cfg)
    per_year = decay(df, perms, listed, cfg)
    per_year.to_csv(results / "milestone_per_year.csv", index=False)
    summary = per_year.groupby(["condition", "year"])[["f1", "recall", "precision", "fpr"]]
    summary = summary.agg(["mean", "std"]).round(4)
    summary.to_csv(results / "milestone_per_year_summary.csv")
    auts = aut_table(per_year)
    auts.to_csv(results / "milestone_aut.csv")
    check = masking_check(df, table, listed)
    check.to_csv(results / "masking_check.csv", index=False)
    plot(per_year, figures / "milestone_f1_by_year.pdf")

    run_info = {
        "config": cfg, "listed_permissions": listed, "replication": rep,
        "git_commit": git_commit(), "python": platform.python_version(),
        "sklearn": sklearn.__version__, "pandas": pd.__version__,
    }
    (results / "milestone_run.json").write_text(json.dumps(run_info, indent=2, default=float))
    print("Replication:", json.dumps(rep, default=float))
    print("\nAUT (mean, std over seeds):\n", auts.to_string())
    print("\nF1 by year (mean):\n", per_year.groupby(["year", "condition"])["f1"].mean()
          .unstack().round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
