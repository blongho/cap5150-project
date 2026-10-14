# Permission-based Android malware detection under permission deprecation

This project measures how losing deprecated or restricted Android permission features affects
malware detection. It uses the KronoDroid real-device dataset (2008–2020, 166 permission flags)
to evaluate feature pruning, with retraining on 5% of newly labelled samples planned.

## Run

Requires Python 3.10–3.13, `make`, and internet access on the first run.

```bash
git clone https://github.com/blongho/cap5150-project.git
cd cap5150-project
make reproduce
```

This installs pinned dependencies, downloads and verifies data, runs experiments, checks
results against `configs/expected_results.json`, and runs lint and tests. To select Python,
use `make reproduce PYTHON=python3.12`.

Alternatively, from the cloned repository, use Docker without local Python:

```bash
make docker-build
make docker-reproduce
```

Settings and random seeds are in `configs/base.yaml`. Outputs go to `results/` (tables and
run metadata) and `figures/` (plots); both directories are created automatically.
`results/` is ignored by Git. Run `make help` for individual pipeline commands.

[GitHub Actions](https://github.com/blongho/cap5150-project/actions) runs the pipeline on
pushes and pull requests and uploads the outputs as artifacts.

## Method and preliminary results

After cleaning, 76,708 apps remain. AOSP manifests (API 21–34) identify 20 distinct deprecated
or restricted permissions. A random forest is trained on 2008–2013 and evaluated separately
on each year from 2014–2020 across five seeds.

Conditions: **unmasked** uses all features; **masked** sets listed permissions to zero at
test time; **pruned** removes them from both training and testing. Results below show mean
± standard deviation of area under the time curve (AUT).

| Condition | AUT(F1) | AUT(recall) | AUT(FPR) |
| --- | --- | --- | --- |
| Unmasked | 0.885 ± 0.012 | 0.889 ± 0.016 | 0.089 ± 0.001 |
| Masked | 0.796 ± 0.006 | 0.731 ± 0.008 | 0.063 ± 0.002 |
| Pruned | 0.878 ± 0.001 | 0.880 ± 0.002 | 0.092 ± 0.002 |

Permission request rates do not fall after the API changes in KronoDroid, so masking tests
feature loss at deployment rather than reproducing observed drift.

## Data citation

A. Guerra-Manzanares, H. Bahsi, S. Nõmm. *KronoDroid: Time-based Hybrid-featured Dataset for
Effective Android Malware Detection and Characterization.* Computers & Security 110 (2021),
102399. The dataset contains extracted features only, no application binaries.
