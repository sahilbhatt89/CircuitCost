PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python

.PHONY: help setup notebooks lint test check

help:
	@printf '%s\n' \
	  'make setup  Create a Python virtual environment and install dev dependencies' \
	  'make notebooks Install JupyterLab and the notebook kernel' \
	  'make lint   Check Python style and formatting' \
	  'make test   Run automated tests' \
	  'make check  Run lint and automated tests'

$(VENV_PYTHON):
	$(PYTHON) -m venv $(VENV)

setup: $(VENV_PYTHON)
	$(VENV_PYTHON) -m pip install --editable '.[dev]'

notebooks: setup
	$(VENV_PYTHON) -m pip install --editable '.[notebooks]'

lint:
	$(VENV_PYTHON) -m ruff check src tests
	$(VENV_PYTHON) -m ruff format --check src tests

test:
	$(VENV_PYTHON) -m pytest

check: lint test
