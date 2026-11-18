# Deprecated permissions and Android malware detection

Code for the CAP5150 project *Do Deprecated Permissions Break Android Malware Detectors?
Measuring and Mitigating Deployment-Time Feature Loss*.

A random forest and an MLP are trained on KronoDroid apps from 2008–2013 and tested on each
year from 2014–2020. The 20 permissions that Android deprecated or restricted between API 21
and 34 are hidden at test time, and two defenses are compared: pruning those permissions
before training, and retraining on 5% of newly labelled apps each year.

One command downloads the data, runs every experiment, builds every table and figure, and
checks the results against stored reference values.

## 1. Requirements

Choose **one** of these setups:

| Setup | You need | Use it when |
| --- | --- | --- |
| **A. Local** | Python 3.10–3.13, `make`, `git` | Linux, Apple-silicon Mac, or Windows with WSL |
| **B. Docker** | Docker, `make`, `git` | Intel Mac (PyTorch has no Intel-Mac builds), or you want the exact environment used for the paper |

Both need internet access on the first run (about 100 MB of data) and about 2 GB of disk.
No GPU is needed.

> **Intel Mac users: use Docker (setup B).** PyTorch publishes no builds for Intel macOS, so
> setup A cannot install it there. `make reproduce` then installs everything else, runs the
> random forest, and stops with `ModuleNotFoundError: No module named 'torch'` when it reaches
> the MLP. Docker runs the whole pipeline in a Linux container with the same pinned packages
> and commands; it is the setup used to verify this repository on an Intel Mac. Start Docker
> Desktop before running the commands in setup B.

## 2. Reproduce everything

### A. Local

```bash
git clone https://github.com/blongho/cap5150-project.git
cd cap5150-project
make reproduce
```

If your default `python3` is not 3.10–3.13, name one that is:

```bash
make reproduce PYTHON=python3.12
```

### B. Docker

```bash
git clone https://github.com/blongho/cap5150-project.git
cd cap5150-project
make docker-build       # once: builds the image with Python 3.11 and pinned packages
make docker-reproduce   # runs `make reproduce` inside the container
```

### What happens

`make reproduce` runs these stages in order:

1. Creates a virtual environment in `.venv/` and installs the pinned packages from
   `requirements.lock`.
2. Downloads KronoDroid and the Android framework manifests, and checks their SHA-256 hashes.
3. Derives the 20 deprecated or restricted permissions and cleans the data.
4. Runs all 210 experiment runs (2 models × 3 defenses × test conditions × 5 seeds).
5. Runs the statistical tests and analyses, then builds the figures and LaTeX tables.
6. Compares the results with `configs/expected_results.json`.
7. Runs the linter and unit tests.

A full run takes about 30 minutes on a 2019 MacBook Pro (8-core Intel Core i9, 32 GB RAM).

### How to tell it worked

The run ends with these lines:

```
OK: all <n> checks reproduce the reference ...
Reproduction complete: results/ and figures/ match configs/expected_results.json.
```

If any result differs, `make verify` prints `FAIL` with a list of the mismatched values and
stops with an error.

## 3. Outputs

| Folder | Contents |
| --- | --- |
| `results/` | One JSON record per run (settings, seed, library versions, per-year metrics), per-cell CSV summaries, and the statistical test results |
| `figures/` | Figures (`.pdf`, `.png`) and LaTeX tables (`.tex`) used in the report |
| `data/raw/`, `data/processed/` | Downloaded and cleaned data |

All three are created by the pipeline and ignored by Git.

## 4. Run one stage at a time

Each command runs the stages it depends on first. Downloaded files and finished experiment
runs are reused, so repeating a command is quick.

| Command | What it does |
| --- | --- |
| `make setup` | Create `.venv/` and install pinned packages |
| `make data` | Download the data and verify checksums |
| `make permissions` | Build `data/deprecated_permissions.csv` from the Android manifests |
| `make prepare` | Clean the data into `data/processed/real_device.parquet` |
| `make experiments` | Run the full experiment grid |
| `make stats` | Wilcoxon tests for research questions RQ1–RQ3 |
| `make ks_tests` | Kolmogorov–Smirnov drift tests |
| `make analysis` | Replication check and permission request rates |
| `make figures` | Build figures and tables |
| `make verify` | Compare results with the reference values |
| `make check` | Lint and unit tests |
| `make help` | List all commands |

## 5. Settings

All settings and random seeds are in [`configs/base.yaml`](configs/base.yaml): the train and
test years, model settings, the 5% labelling budget, the seeds and the optional analyses.
To try other settings, copy the file and pass it in. Run `make clean` first, because finished
runs in `results/` are reused regardless of the settings that produced them:

```bash
make clean
make all CONFIG=configs/my_settings.yaml
```

`make verify` checks against the paper's settings only, so it will report differences for a
changed configuration.

## 6. Troubleshooting

| Problem | Fix |
| --- | --- |
| `Python 3.10-3.13 is required` | Install a supported version and pass it: `make reproduce PYTHON=python3.12` |
| `No module named 'torch'` (Intel Mac) | PyTorch has no Intel-Mac builds; use the Docker setup (B) |
| `Cannot connect to the Docker daemon` | Start Docker Desktop, wait until it is running, and repeat the command |
| The run was interrupted | Run the same command again; finished runs are reused |
| You want to start over | `make clean` deletes `results/` and `figures/`; `make clean-data` deletes the downloaded data |
| A download fails or a checksum does not match | Check your connection and run `make clean-data`, then `make data` |

## 7. Repository layout

```
configs/    settings (base.yaml) and reference results (expected_results.json)
data/       permission list and manifest checksums; downloads go to data/raw/
scripts/    one entry point per pipeline stage
src/drift/  library code: data, permissions, models, defenses, metrics, statistics, figures
tests/      unit tests
Makefile    pipeline commands
Dockerfile  pinned environment (Python 3.11)
```

## Data

KronoDroid real-device subset: A. Guerra-Manzanares, H. Bahsi and S. Nõmm. *KronoDroid:
Time-based Hybrid-featured Dataset for Effective Android Malware Detection and
Characterization.* Computers & Security 110 (2021), 102399. The dataset contains extracted
features only, not app binaries.
