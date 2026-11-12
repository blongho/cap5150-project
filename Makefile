# Reproducible pipeline for the CAP5150 project.
#
#   make reproduce   ONE COMMAND: environment, data, experiments, verification, tests
#
# Individual steps:
#   make setup       create the virtual environment from requirements.lock (Python 3.10-3.13)
#   make data        download KronoDroid + AOSP manifests and verify SHA-256 checksums
#   make permissions build data/deprecated_permissions.csv from the AOSP manifests
#   make prepare     clean the data -> data/processed/real_device.parquet
#   make experiments run the full grid (model x defense x condition x seed) -> results/
#   make stats       Wilcoxon tests of H1-H3 -> results/hypothesis_tests.csv
#   make ks_tests    KS drift tests per permission and on RF scores -> results/ks_*.csv
#   make analysis    replication and masking checks -> results/replication.json, masking_check.csv
#   make figures     report figures and LaTeX tables from results/ -> figures/
#   make verify      compare results/ with configs/expected_results.json
#   make check       lint + unit tests
#   make lock        regenerate requirements.lock from requirements.txt
#
# Docker (no local Python needed):
#   make docker-build && make docker-reproduce
#
# Overrides: make setup PYTHON=python3.11 VENV=/path/to/venv CONFIG=configs/other.yaml

PYTHON      ?= python3
VENV        ?= .venv
CONFIG      ?= configs/base.yaml
PYTEST_ARGS ?=
IMAGE       ?= cap5150-drift:1.0

PY  := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
RUN := PYTHONHASHSEED=0 PYTHONPATH=src $(PY)

.PHONY: help reproduce setup check-python data permissions prepare experiments stats ks_tests analysis figures \
        verify \
        all \
        test lint check lock docker-build docker-reproduce clean clean-data

help:
	@sed -n '2,23p' Makefile | sed 's/^# \{0,1\}//'

reproduce: all verify check
	@echo "Reproduction complete: results/ and figures/ match configs/expected_results.json."

check-python:
	@$(PYTHON) -c 'import sys; v = sys.version_info[:2]; \
	ok = (3, 10) <= v <= (3, 13); \
	print("Using Python %d.%d" % v); \
	sys.exit(0 if ok else "Python 3.10-3.13 is required. Install one and run: make setup PYTHON=python3.12")'

$(VENV)/.installed: requirements.lock
	@$(MAKE) --no-print-directory check-python
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade "pip==24.2"
	$(PIP) install -r requirements.lock
	touch $@

setup: $(VENV)/.installed

data: setup
	$(RUN) scripts/download_data.py --config $(CONFIG)

permissions: data
	$(RUN) scripts/build_permission_list.py --config $(CONFIG)

prepare: data
	$(RUN) scripts/prepare_data.py --config $(CONFIG)

experiments: prepare permissions
	$(RUN) scripts/run_experiments.py --config $(CONFIG)

stats: experiments
	$(RUN) scripts/run_stats.py --config $(CONFIG)

ks_tests: prepare
	$(RUN) scripts/run_ks_tests.py --config $(CONFIG)

analysis: prepare permissions
	$(RUN) scripts/run_analysis.py --config $(CONFIG)

figures: experiments stats ks_tests analysis
	$(RUN) scripts/make_figures.py --config $(CONFIG)

all: experiments stats ks_tests analysis figures

verify: setup
	$(RUN) scripts/verify_results.py

test: setup
	PYTHONPATH=src $(VENV)/bin/pytest -q $(PYTEST_ARGS)

lint: setup
	$(VENV)/bin/ruff check src scripts tests

check: lint test

lock: check-python
	$(PYTHON) -m venv /tmp/cap5150-lock && /tmp/cap5150-lock/bin/pip install -q -r requirements.txt
	/tmp/cap5150-lock/bin/pip freeze | sort > requirements.lock.new
	@echo "Review requirements.lock.new, add version markers if needed, then replace requirements.lock"

docker-build:
	docker build -t $(IMAGE) .

docker-reproduce:
	docker run --rm -v "$(CURDIR)":/work -w /work $(IMAGE) sh -c "touch /opt/venv/.installed && make reproduce VENV=/opt/venv"

clean:
	rm -rf results/* figures/* .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

clean-data:
	rm -rf data/raw data/processed
