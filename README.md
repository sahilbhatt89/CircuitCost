# CircuitCost

A Python workspace for CircuitCost.

## Setup

Requirements: Python 3.11 or newer and GNU Make.

```sh
make setup
source .venv/bin/activate
python -c "import circuitcost; print(circuitcost.__file__)"
```

`make setup` creates `.venv` and installs this Python package in editable mode,
plus Pytest and Ruff. Make commands use `.venv` automatically. To select a
specific interpreter, use `make setup PYTHON=python3.13`.

## Develop and verify

```sh
make check   # Check Python style and formatting
```

- `src/circuitcost/`: Python package.
- `tests/`: Future Python tests; see `tests/README.md`.

Add Python tests under `tests/`, then run `.venv/bin/python -m pytest tests`.
There are currently no Python tests. Checks are scoped to `src/` and `tests/`
so separately cloned datasets are not included.

Set your editor's Python interpreter to `.venv/bin/python`.
