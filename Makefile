# Reproducible pipeline for the CAP5150 project.
#
#   make run         ONE COMMAND: picks local or Docker (Intel Mac) and runs everything
#   make reproduce   everything locally: environment, data, experiments, verification, tests
#
# Individual steps:
#   make setup       create the virtual environment from requirements.lock (Python 3.10-3.13)
#   make data        download KronoDroid + AOSP manifests and verify SHA-256 checksums
#   make permissions build data/deprecated_permissions.csv from the AOSP manifests
#   make prepare     clean the data -> data/processed/real_device.parquet
#   make experiments run the full grid (model x defense x condition x seed) -> results/
#   make stats       Wilcoxon tests of RQ1-RQ3 -> results/hypothesis_tests.csv
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

# Progress banner: $(call stage,N,text) prints "==> [time] Stage N/11: text".
STAGES := 11
stage = @printf '\n==> [%s] Stage %s/%s: %s\n' "$$(date +%H:%M:%S)" "$(1)" "$(STAGES)" "$(2)"

.PHONY: help run reproduce setup check-python data permissions prepare experiments stats \
        ks_tests analysis figures verify all test lint check lock docker-build \
        docker-reproduce clean clean-data

help:
	@sed -n '2,24p' Makefile | sed 's/^# \{0,1\}//'

# Intel Macs have no PyTorch builds, so they run the pipeline in Docker; others run locally.
run:
	@start=$$(date +%s); \
	if [ "$$(uname -s)-$$(uname -m)" = "Darwin-x86_64" ]; then \
	  echo "==> Intel Mac detected: running the pipeline in Docker"; \
	  docker info >/dev/null 2>&1 || { \
	    echo "Docker is not running. Start Docker Desktop, then run 'make run' again."; \
	    exit 1; }; \
	  $(MAKE) --no-print-directory docker-build docker-reproduce || exit $$?; \
	else \
	  echo "==> Running the pipeline locally"; \
	  $(MAKE) --no-print-directory reproduce || exit $$?; \
	fi; \
	echo "==> Finished in $$(( ($$(date +%s) - start) / 60 )) min"

reproduce: all verify check
	@echo "Reproduction complete: results/ and figures/ match configs/expected_results.json."

check-python:
	@$(PYTHON) -c 'import sys; v = sys.version_info[:2]; \
	ok = (3, 10) <= v <= (3, 13); \
	print("Using Python %d.%d" % v); \
	sys.exit(0 if ok else "Python 3.10-3.13 is required. Install one and run: make setup PYTHON=python3.12")'

$(VENV)/.installed: requirements.lock
	@echo "==> Creating the virtual environment and installing pinned packages (first run only)"
	@$(MAKE) --no-print-directory check-python
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade "pip==24.2"
	$(PIP) install -r requirements.lock
	touch $@

setup: $(VENV)/.installed
	$(call stage,1,Environment ready ($(VENV)))

data: setup
	$(call stage,2,Downloading data and verifying checksums)
	$(RUN) scripts/download_data.py --config $(CONFIG)

permissions: data
	$(call stage,3,Deriving deprecated and restricted permissions)
	$(RUN) scripts/build_permission_list.py --config $(CONFIG)

prepare: data
	$(call stage,4,Cleaning the data)
	$(RUN) scripts/prepare_data.py --config $(CONFIG)

experiments: permissions prepare
	$(call stage,5,Running experiments - the longest stage; progress shown as [run/total])
	$(RUN) scripts/run_experiments.py --config $(CONFIG)

stats: experiments
	$(call stage,6,Wilcoxon tests for RQ1-RQ3)
	$(RUN) scripts/run_stats.py --config $(CONFIG)

ks_tests: prepare
	$(call stage,7,Kolmogorov-Smirnov drift tests)
	$(RUN) scripts/run_ks_tests.py --config $(CONFIG)

analysis: prepare permissions
	$(call stage,8,Replication check and permission request rates)
	$(RUN) scripts/run_analysis.py --config $(CONFIG)

figures: experiments stats ks_tests analysis
	$(call stage,9,Building figures and tables)
	$(RUN) scripts/make_figures.py --config $(CONFIG)

all: experiments stats ks_tests analysis figures

verify: setup
	$(call stage,10,Comparing results with the reference values)
	$(RUN) scripts/verify_results.py

test: setup
	PYTHONPATH=src $(VENV)/bin/pytest -q $(PYTEST_ARGS)

lint: setup
	$(call stage,11,Lint and unit tests)
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
