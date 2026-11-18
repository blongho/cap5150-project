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
| `make stats` | Wilcoxon tests of RQ1–RQ3 | `results/hypothesis_tests.csv` |
| `make ks_tests` | KS drift tests per permission (Benjamini–Hochberg) and on RF scores | `results/ks_*.csv`, `results/ks_run.json` |
| `make analysis` | Replication check and permission request rates around each change | `results/replication.json`, `results/masking_check.csv` |
| `make figures` | Report figures and LaTeX tables | `figures/` |
| `make verify` | Compare results with the reference | — |
| `make check` | Lint and unit tests | — |

Settings and random seeds are in `configs/base.yaml`. `results/` and `figures/` are created
automatically and ignored by Git; `make clean` empties both. Interrupted experiment runs resume: existing run records are
reused.

## Data citation

A. Guerra-Manzanares, H. Bahsi, S. Nõmm. *KronoDroid: Time-based Hybrid-featured Dataset for
Effective Android Malware Detection and Characterization.* Computers & Security 110 (2021),
102399. The dataset contains extracted features only, no application binaries.
