# Permission-based Android malware detection under permission deprecation

CAP5150 course project (UCF, Fall 2026) by Longho Bernard Che.

When Android deprecates or restricts a permission, a deployed malware detector trained on
permission features can lose those inputs. This project measures how much detection degrades
when deprecated and restricted permissions disappear at deployment time, and compares two
defenses: **pruning** those permissions (no new labels) and **retraining on 5% of newly
labelled samples**. Data: the KronoDroid real-device subset (2008–2020, 166 permission flags).

## Quick start

Requirements: Python 3.10–3.12 (pinned numpy 1.26 has no 3.13 wheels), `make`, and internet access for the first run.

```bash
make setup        # create .venv and install pinned dependencies (requirements.txt)
make all          # download + verify data, build permission list, prepare data, run experiments
make check        # ruff lint + pytest
```

Everything is driven by `configs/base.yaml`. Use another config with `make all CONFIG=...`, or
another interpreter or environment location with `make setup PYTHON=python3.11 VENV=/path`.

## Pipeline

| Step | Command | Output |
| --- | --- | --- |
| Download and verify KronoDroid and 14 AOSP permission manifests (API 21–34) | `make data` | `data/raw/` (SHA-256 checked) |
| Derive deprecated / restricted permissions from AOSP | `make permissions` | `data/deprecated_permissions.csv` |
| Clean data, assign years | `make prepare` | `data/processed/real_device.parquet`, `results/year_counts.csv` |
| Milestone experiments (RF) | `make milestone` | `results/milestone_*.csv`, `results/masking_check.csv`, `figures/milestone_f1_by_year.pdf` |

Each run writes `results/milestone_run.json` with the full config, the permission list,
library versions and the git commit, so every figure traces back to a run.

## Method (implemented so far)

- **Cleaning:** year from `HighestModDate`; years outside 2008–2020 and unparseable dates
  are dropped; apps whose SHA-256 appears with both labels are dropped; duplicates removed.
  76,708 apps remain.
- **Permission list:** a permission is *deprecated* at the first API level whose AOSP
  framework manifest marks it `@deprecated`, and *restricted* at the first level that gives
  it the `hardRestricted` or `softRestricted` flag. API 21–34 yields 10 deprecated (matching
  Sabbah et al.) and 11 restricted entries, 20 distinct permissions.
- **Time-aware evaluation (TESSERACT C1/C2):** train on 2008–2013, test each year 2014–2020
  separately; 5 seeds; summarise each curve with AUT (Pendlebury et al., 2019).
- **Conditions:** *unmasked*; *masked* (listed permissions set to 0 at test time);
  *pruned* (listed permissions removed from training and test data).

## Preliminary results (RF, 5 seeds)

| Condition | AUT(F1) | AUT(recall) | AUT(FPR) |
| --- | --- | --- | --- |
| Unmasked | 0.885 ± 0.012 | 0.889 ± 0.016 | 0.089 ± 0.001 |
| Masked, no defense | 0.796 ± 0.006 | 0.731 ± 0.008 | 0.063 ± 0.002 |
| Pruned | 0.878 ± 0.001 | 0.880 ± 0.002 | 0.092 ± 0.002 |

Replication of Sabbah et al. (RF, random 80/20): accuracy 0.953 against the reported 0.954.

Request rates of the listed permissions do not fall after their API change in KronoDroid
(see `results/masking_check.csv`), so masking is a stress test of feature loss at deployment,
not a reconstruction of observed drift.

## Repository layout

```
configs/      experiment configuration
data/         downloaded data (git-ignored), permission list, AOSP checksums
src/drift/    library code: config, data, permissions, features, metrics, models, sources
scripts/      pipeline entry points called by the Makefile
tests/        unit tests (pytest)
results/      generated tables and run metadata
figures/      generated figures
paper/        LaTeX source for the reports
```

## Data and citation

KronoDroid is distributed by its authors for research use; cite: A. Guerra-Manzanares,
H. Bahsi, S. Nõmm. *KronoDroid: Time-based Hybrid-featured Dataset for Effective Android
Malware Detection and Characterization.* Computers & Security 110 (2021) 102399.
The dataset contains extracted features only, no application binaries.
