# Reproducible pipeline for the CAP5150 project.
#   make setup      create the virtual environment and install pinned dependencies
#   make all        download data, build the permission list, prepare data, run experiments
#   make test lint  run unit tests and the linter
# Override the interpreter or environment location if needed:
#   make setup PYTHON=python3.11 VENV=/path/to/venv

PYTHON ?= python3
VENV   ?= .venv
CONFIG ?= configs/base.yaml
PYTEST_ARGS ?=

PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip
RUN  := PYTHONPATH=src $(PY)

.PHONY: help setup data permissions prepare milestone all test lint check clean clean-data

help:
	@grep -E '^#' Makefile | sed 's/^# \{0,1\}//'

$(VENV)/.installed: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	touch $@

setup: $(VENV)/.installed

data: setup
	$(RUN) scripts/download_data.py --config $(CONFIG)

permissions: data
	$(RUN) scripts/build_permission_list.py --config $(CONFIG)

prepare: data
	$(RUN) scripts/prepare_data.py --config $(CONFIG)

milestone: prepare permissions
	$(RUN) scripts/run_milestone.py --config $(CONFIG)

all: milestone

test: setup
	PYTHONPATH=src $(VENV)/bin/pytest -q $(PYTEST_ARGS)

lint: setup
	$(VENV)/bin/ruff check src scripts tests

check: lint test

clean:
	rm -rf results/* figures/* .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

clean-data:
	rm -rf data/raw data/processed
