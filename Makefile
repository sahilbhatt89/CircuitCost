PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python

.PHONY: help setup lint check

help:
	@printf '%s\n' \
	  'make setup  Create a Python virtual environment and install dev dependencies' \
	  'make lint   Check Python style and formatting' \
	  'make check  Run Python checks'

$(VENV_PYTHON):
	$(PYTHON) -m venv $(VENV)

setup: $(VENV_PYTHON)
	$(VENV_PYTHON) -m pip install --editable '.[dev]'

lint:
	$(VENV_PYTHON) -m ruff check src tests
	$(VENV_PYTHON) -m ruff format --check src tests

check: lint
