# Permission-based Android malware detection under permission deprecation

This project measures how losing deprecated or restricted Android permission features at
deployment affects malware detection, and compares two defenses: pruning those permissions
(no labels) and retraining on 5% of newly labelled apps per year. It uses the KronoDroid
real-device dataset (2008–2020, 166 permission flags) with a random forest (RF) and an MLP.

## Run

Requires Python 3.10–3.13, `make`, and internet access on the first run.

```bash
git clone https://github.com/blongho/cap5150-project.git
cd cap5150-project
make reproduce
```

This installs pinned dependencies, downloads and verifies data, runs every experiment, the
statistical tests and the figures, checks results against `configs/expected_results.json`, and
runs lint and tests. A clean run takes about 30–40 minutes on a laptop CPU. To select Python,
use `make reproduce PYTHON=python3.12`.

PyTorch (for the MLP) has no wheels for Intel macOS. There, use Docker, which needs no local
Python, or Google Colab (`pip install -r requirements.lock`, then the `make` steps):

```bash
make docker-build
make docker-reproduce
```

Individual steps (each runs the steps it depends on):

| Command | Does | Writes |
| --- | --- | --- |
| `make data` | Download KronoDroid and AOSP manifests; verify SHA-256 | `data/raw/` |
| `make permissions` | Derive deprecated and restricted permissions (API 21–34) | `data/deprecated_permissions.csv` |
| `make prepare` | Clean the data | `data/processed/real_device.parquet`, `results/year_counts.csv` |
| `make experiments` | Model × defense × condition × seed grid | `results/*_seed*.json`, `results/*.csv` |
| `make stats` | Wilcoxon tests of H1–H3 | `results/hypothesis_tests.csv` |
| `make ks_tests` | KS drift tests per permission (Benjamini–Hochberg) and on RF scores | `results/ks_*.csv`, `results/ks_run.json` |
| `make analysis` | Replication check and permission request rates around each change | `results/replication.json`, `results/masking_check.csv` |
| `make figures` | Report figures and LaTeX tables | `figures/` |
| `make verify` | Compare results with the reference | — |
| `make check` | Lint and unit tests | — |

Settings and random seeds are in `configs/base.yaml`. `results/` and `figures/` are created
automatically and ignored by Git; `make clean` empties both. Interrupted experiment runs resume: existing run records are
reused. [GitHub Actions](https://github.com/blongho/cap5150-project/actions) runs the pipeline
on pushes and pull requests and uploads the outputs as artifacts.

## Tracing results to figures

Every figure and table is built by `scripts/make_figures.py` from `results/` only. A cell is
named `{model}_{defense}_{condition}`, with model `rf`/`mlp`, defense
`none`/`pruning`/`retrain` and condition `unmasked`/`masked` (plus `masked_deprecated`,
`masked_restricted`, `timed_masking`, and the `_10pct` malware-share variant).

| Output in `figures/` | Source in `results/` |
| --- | --- |
| `f1_decay_rf.pdf`, `f1_decay_mlp.pdf` | `{model}_none_unmasked.csv`, `{model}_none_masked.csv`, `{model}_pruning_masked.csv`, `{model}_retrain_masked.csv` (columns `f1_mean`, `f1_std`) |
| `aut_summary.pdf` | `{model}_{defense}_{condition}_aut.csv` for the 12 core cells |
| `aut_table.tex` | the same `_aut.csv` files, plus `n_labelled_mean` from `{model}_{defense}_masked.csv` |
| `aut_extras_table.tex` | `{model}_{defense}_{masked_deprecated,masked_restricted,timed_masking}_aut.csv` |
| `share10_table.tex` | `{model}_{defense}_{condition}_10pct_aut.csv` |
| `hypothesis_table.tex` | `hypothesis_tests.csv` |
| `ks_table.tex` | `ks_tests.csv`, `ks_scores.csv` |
| `masking_table.tex`, `numbers.tex` | `masking_check.csv`, `replication.json`, `year_counts.csv` |

Each `{cell}_seed{n}.json` run record holds the config, seed, git commit, library versions
and per-year metrics behind these files.

## Method and results

After cleaning, 76,708 apps remain. AOSP manifests (API 21–34) identify 20 distinct deprecated
or restricted permissions. Models train on 2008–2013 and are evaluated on each year from
2014–2020 across five seeds. **Masked** sets the 20 permissions to zero at test time;
**pruning** removes them from training and test data; **retraining** labels a stratified 5%
of each test year (cumulatively) and updates the model. For each seed and year the same 5% is
held back from scoring, so every defense is evaluated on the identical remaining 95%.

Area under the time curve (AUT), mean ± standard deviation over five seeds:

| Model | Defense | Condition | AUT(F1) | AUT(recall) | AUT(FPR) |
| --- | --- | --- | --- | --- | --- |
| RF | None | Unmasked | 0.885 ± 0.012 | 0.888 ± 0.016 | 0.089 ± 0.001 |
| RF | None | Masked | 0.797 ± 0.006 | 0.731 ± 0.008 | 0.063 ± 0.002 |
| RF | Pruning | Either | 0.879 ± 0.001 | 0.879 ± 0.001 | 0.093 ± 0.002 |
| RF | Retrain | Unmasked | 0.901 ± 0.003 | 0.917 ± 0.008 | 0.091 ± 0.008 |
| RF | Retrain | Masked | 0.890 ± 0.003 | 0.878 ± 0.004 | 0.072 ± 0.003 |
| MLP | None | Unmasked | 0.886 ± 0.009 | 0.870 ± 0.019 | 0.085 ± 0.009 |
| MLP | None | Masked | 0.821 ± 0.012 | 0.744 ± 0.019 | 0.059 ± 0.003 |
| MLP | Pruning | Either | 0.883 ± 0.004 | 0.863 ± 0.010 | 0.086 ± 0.004 |
| MLP | Retrain | Unmasked | 0.904 ± 0.006 | 0.884 ± 0.011 | 0.073 ± 0.003 |
| MLP | Retrain | Masked | 0.869 ± 0.008 | 0.836 ± 0.013 | 0.080 ± 0.007 |

A random forest on a random 80/20 split reaches 0.953 accuracy, against 0.954 reported by
Sabbah et al. Pruning gives identical results with and without masking because the masked permissions are
no longer features. Retraining uses 1,378 labelled apps by 2020. Permission request rates do
not fall after the API changes in KronoDroid, so masking tests feature loss at deployment
rather than reproducing observed drift.

Results were produced with Python 3.11.10, scikit-learn 1.5.2, PyTorch 2.7.1 (CPU) and
pandas 2.2.3 (Docker image from this repository).

## Data citation

A. Guerra-Manzanares, H. Bahsi, S. Nõmm. *KronoDroid: Time-based Hybrid-featured Dataset for
Effective Android Malware Detection and Characterization.* Computers & Security 110 (2021),
102399. The dataset contains extracted features only, no application binaries.
