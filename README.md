# CircuitCost

Predict FPGA implementation LUT usage from high-level synthesis (HLS) features.
CircuitCost compares the original HLS LUT estimate, linear regression, and a
neural network implemented from scratch with Python and NumPy, including explicit
backpropagation. The completed experiment covers **531 Polybench GEMM designs**
on **xczu9eg-ffvb1156-2-i**.

The selected MLP has architecture **11 → 32 → 16 → 1**, ReLU hidden activations,
and a linear output. Training restores copied parameters from the epoch with
the lowest validation MSE.

## Setup

Requirements: Python 3.11+, GNU Make, and Git. From the project root:

```sh
make setup
source .venv/bin/activate
python -c "import circuitcost; print(circuitcost.__file__)"
```

Numerical dependencies (NumPy, Pandas, scikit-learn, Matplotlib, and Joblib) are
declared in `pyproject.toml`. `make setup` also installs Pytest and Ruff. To
select an interpreter, use `make setup PYTHON=python3.13`.

For notebooks:

```sh
make notebooks
python -m jupyterlab
```

Select `.venv/bin/python` as the notebook kernel in VS Code. To install into a
different active kernel, run `%pip install -e ".."` from either notebook folder,
then restart that kernel. Saving a notebook does not preserve its live variables.

The numerical versions used to verify the frozen experiment are pinned in
[`constraints-recorded.txt`](constraints-recorded.txt). To use those versions:

```sh
python -m pip install -c constraints-recorded.txt -e ".[dev,notebooks]"
```

The verification used Python 3.13.7. This constraints file pins numerical
libraries; it is not a complete operating-system or BLAS environment lock.

## Local dataset

Data comes from [UT-LCA/ML4Accel-Dataset](https://github.com/UT-LCA/ML4Accel-Dataset).
The dataset and experiment artifacts are ignored by Git. Clone the dataset from
the project root if it is not already present:

```sh
git clone https://github.com/UT-LCA/ML4Accel-Dataset.git ML4Accel-Dataset
```

Restore the two GEMM CSVs from the recorded historical revision:

```sh
csv_dir="fpga_ml_dataset/HLS_dataset/polybench/gemm/xczu9eg-ffvb1156-2-i"

git -C ML4Accel-Dataset restore \
  --source=e05bebec02ab3b80648e45377ffc91702ea8d2dd \
  --worktree -- \
  "$csv_dir/post_hls_info.csv" \
  "$csv_dir/post_implementation_info.csv"
```

The files reside under `ML4Accel-Dataset/$csv_dir/`. The first supplies HLS
features; the second supplies implementation `Total LUTs`.

[`load_designs`](src/circuitcost/data.py) explicitly joins the tables by `prj`
with one-to-one validation. It rejects duplicate, blank, or unmatched identifiers
and nonfinite measurements. HLS file order is preserved for reproducible splits;
the implementation CSV may be reordered without changing design/target pairing.
`prj` is retained as the index for tracking split membership, not as a feature.

Notebook paths are resolved relative to the checkout. Launch a notebook from
the project root or its notebook directory; no user-specific paths are needed.

## Reproduce the final workflow

Open [`Notebboks/finalExperiment.ipynb`](Notebboks/finalExperiment.ipynb) in a
fresh kernel and run its cells in order:

1. Load and validate matched designs.
2. Split using seed 42: 371 training, 80 validation, and 80 test designs.
3. Fit input `StandardScaler` on training data only; targets remain in LUT units.
4. Train the MLP or load the existing local frozen checkpoint and scaler.
5. Compare validation metrics with the HLS and linear-regression baselines.
6. Optionally repeat initialization seeds and reproduce the recorded test evaluation.

The notebook controls are:

| Setting | Default | Effect |
| --- | --- | --- |
| `RUN_TRAINING` | True when no local weights exist | Otherwise load the saved candidate; set True to retrain |
| `RUN_SEED_CHECKS` | False | Run two additional 50,000-epoch seed experiments |
| `EVALUATE_TEST` | False | Reproduce the already recorded frozen test evaluation |

Training uses seed 42, full-batch gradient descent, MSE loss, learning rate
`1e-6`, and a 50,000-epoch budget. The recorded candidate selected **epoch 2,336**.
Checkpoint selection restores the best parameters after completing the budget;
early stopping is not implemented. History entries are MSE in squared LUTs;
their square roots are RMSE in LUTs. The last history entry need not describe
the restored model.

New artifacts include weights, the fitted scaler, the linear baseline, feature
order, configuration, split IDs, CSV checksums, and package versions. Existing
checkpoints are checked against feature order, split membership, and the fitted
scaler before loading. To reproduce a frozen evaluation, keep the candidate
fixed and do not use test results to select another model.

The inputs are:

```text
clk_target(ns), clk_estimated(ns), clk_uncertainty(ns),
bram, dsp, ff, lut,
c_num_arith, c_num_logic, rtl_num_arith, rtl_num_logic
```

The other notebooks are retained for learning:

- [Dataset exploration](<Data Exploration/understandDataset.ipynb>): schemas,
  identifier checks, distributions, and HLS-versus-implementation plots.
- [Working experiment notes](Notebboks/dataPrep.ipynb): the original progression
  of diagnostics and training experiments. It imports the current model class;
  old experiment cells do not reconstruct earlier architectures automatically.

Notebook outputs are cleared to avoid displaying stale results after changing
data loading. Published results are recorded separately below.

## Final results

| Model | Validation MAE | Validation RMSE | Test MAE | Test RMSE | Test R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| HLS estimate | 5,758.98 | 9,836.98 | 5,025.46 | 8,018.33 | 0.637978 |
| Linear regression | 936.89 | 1,595.88 | 812.26 | **1,023.71** | **0.994099** |
| Custom NumPy MLP | **548.20** | **891.69** | **603.59** | 1,157.32 | 0.992458 |

Errors are in LUTs; lower MAE/RMSE and higher R² are better. R² is not percentage
accuracy. The MLP reduced test MAE by **25.7%** but increased test RMSE by
**13.1%** relative to linear regression. Neither model dominates across both
metrics. Recorded seed-7 and seed-21 validation RMSEs were 1,102.23 and 1,023.80.

The MLP has lower aggregate MAE and RMSE in the test groups below 15,000 actual
LUTs, while linear regression performs better in the group of 12 designs at or
above that size. This tail
explains why the simpler model remains competitive on RMSE. These descriptive
size groups were not used to choose predictions.

Read [`docs/FINAL_FINDINGS.md`](docs/FINAL_FINDINGS.md) for the range-level
analysis, interpretation, and limits. The experiment covers related GEMM
configurations on one FPGA. It does **not** establish generalization to unrelated
circuits, new kernels, different devices, or different synthesis tool versions.

Machine-readable summaries are in [`reports/`](reports/): verified aggregate
metrics and data fingerprints, descriptive test-range errors, and recorded
initialization-seed results. The numerical scores were rechecked against the
existing frozen checkpoint; no new model was selected for the final report.

## Automated checks

```sh
make test    # Dataset-independent tests
make check   # Ruff lint, format checks, and tests
```

Tests cover forward-output shapes; dense-layer and complete-network
finite-difference gradients; ReLU gradients; restoration of an earlier best
checkpoint's weights and biases after later updates; table matching with
reordered implementation rows; invalid design IDs and measurements; and
LUT-range summary boundaries. They use synthetic data and temporary CSVs, so
the external dataset and saved artifacts are not needed for `make check`.

## Project layout

```text
src/circuitcost/
    data.py          # Checked prj join and portable dataset discovery
    models.py        # Dense layers, ReLU, and 32/16 MLP
    train.py         # MSE, full-batch updates, best-checkpoint restoration
    evaluation.py    # Metrics and descriptive LUT-range summaries
Notebboks/finalExperiment.ipynb
Notebboks/dataPrep.ipynb
Data Exploration/understandDataset.ipynb
tests/
docs/FINAL_FINDINGS.md
reports/
ML4Accel-Dataset/     # Local, ignored
artifacts/           # Local, ignored
```

CircuitCost is licensed under [Apache License 2.0](LICENSE). Consult the source
dataset's documentation for its usage terms and attribution.
