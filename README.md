# CircuitCost

CircuitCost explores prediction of FPGA implementation resource usage from
high-level synthesis (HLS) estimates. The current experiment predicts the
implementation LUT count of Polybench GEMM designs on the
`xczu9eg-ffvb1156-2-i` FPGA.

The project compares the original HLS estimate, a scikit-learn linear regression
baseline, and a neural network implemented from scratch with Python and NumPy.
The neural network includes explicit forward propagation, backpropagation, and
gradient-descent updates.

## Setup

Requirements: Python 3.11 or newer and GNU Make.

From the project root:

```sh
make setup
source .venv/bin/activate
python -m pip install numpy pandas scikit-learn matplotlib jupyterlab ipykernel
python -c "import circuitcost; print(circuitcost.__file__)"
```

`make setup` creates `.venv` and installs this Python package in editable mode,
plus Pytest and Ruff. The notebook and numerical dependencies are installed
explicitly above because they are not yet declared in `pyproject.toml`.
To select a specific interpreter, use `make setup PYTHON=python3.13`.

Launch JupyterLab from the activated environment:

```sh
python -m jupyterlab
```

In VS Code, select `.venv/bin/python` as the notebook kernel. If using another
environment, install the project into that notebook's environment with
`%pip install -e "/absolute/path/to/CircuitCost"`, then restart the kernel and
rerun the cells. Notebook variables exist only in their running kernel.

## Dataset

Data comes from [UT-LCA/ML4Accel-Dataset](https://github.com/UT-LCA/ML4Accel-Dataset).
The dataset clone is local to this workspace and excluded from this repository
by `.gitignore`.

From the project root, clone the dataset if it is not already present:

```sh
git clone https://github.com/UT-LCA/ML4Accel-Dataset.git ML4Accel-Dataset
```

The GEMM CSVs used here are available in the dataset's Git history. Restore only
the two CSVs from the recorded revision:

```sh
csv_dir="fpga_ml_dataset/HLS_dataset/polybench/gemm/xczu9eg-ffvb1156-2-i"

git -C ML4Accel-Dataset restore \
  --source=e05bebec02ab3b80648e45377ffc91702ea8d2dd \
  --worktree -- \
  "$csv_dir/post_hls_info.csv" \
  "$csv_dir/post_implementation_info.csv"
```

This places both files under:

```text
ML4Accel-Dataset/fpga_ml_dataset/HLS_dataset/polybench/gemm/xczu9eg-ffvb1156-2-i/
```

| File | Contents |
| --- | --- |
| `post_hls_info.csv` | HLS resource estimates, timing estimates, and design features |
| `post_implementation_info.csv` | Implementation resource counts, timing, and power metrics |

The current pair contains 531 designs. The shared `prj` identifier is unique
within each CSV for this benchmark and device. Verify identifiers and row order
before selecting features and targets directly from separate tables:

```python
assert hls["prj"].equals(impl["prj"]), "Design rows are not aligned"
```

Both notebooks currently contain an absolute dataset path from the original
workspace. Update their `data_dir` cells to point to the directory above on your
machine before running them.

## Notebook workflow

Run cells in order, starting with:

1. [Dataset exploration](<Data Exploration/understandDataset.ipynb>): inspect
   schemas, missing values, design identifiers, summary statistics, histograms,
   and HLS-versus-implementation scatterplots.
2. [Data preparation and model comparison](Notebboks/dataPrep.ipynb): select
   features, split and standardize data, evaluate baselines, train the custom
   network, and inspect validation errors and learning curves.

The current target is implementation `Total LUTs`. Inputs are these 11 HLS
columns; `prj` is used for matching designs, not as a numerical feature:

```text
clk_target(ns), clk_estimated(ns), clk_uncertainty(ns),
bram, dsp, ff, lut,
c_num_arith, c_num_logic, rtl_num_arith, rtl_num_logic
```

The split is approximately 70% training, 15% validation, and 15% testing,
using `random_state=42`: 371 training designs, 80 validation designs, and
80 test designs. `StandardScaler` is fitted on training features and then
applied to validation and test features. Targets remain in original LUT units.

## Custom neural network

The model in [models.py](src/circuitcost/models.py) has this architecture:

```text
11 inputs -> Dense(16) -> ReLU -> Dense(8) -> ReLU -> Dense(1)
```

`DenseLayer` implements parameter initialization, forward and backward
calculations, and parameter updates. `ReLU` implements the activation and its
gradient. `NeuralNetworks` connects the layers and propagates values forward
and gradients backward; the output layer is linear.

[train.py](src/circuitcost/train.py) implements mean squared error (MSE), its
gradient, and full-batch gradient descent. Each epoch calculates gradients from
all training designs and updates parameters once. Training and validation
losses are recorded after the update; validation does not update parameters.

After preparing the notebook variables, the current training call is:

```python
import numpy as np
from circuitcost.models import NeuralNetworks
from circuitcost.train import train_model

np.random.seed(42)
model = NeuralNetworks(n_inputs=X_train_scaled.shape[1])

history = train_model(
    model,
    X_train_scaled,
    y_train,
    X_val_scaled,
    y_val,
    epochs=1000,
    learning_rate=0.000001,
)

predictions = model.forward(X_val_scaled).ravel()
```

`history["train_loss"]` and `history["val_loss"]` contain MSE in squared LUTs.
Their square roots give RMSE in LUTs. Recreating `model` initializes new
parameters, so retain the trained instance when evaluating it.

## Recorded validation results

These scores are saved outputs from `Notebboks/dataPrep.ipynb` for the
GEMM/device pair and split described above. The custom network used
`np.random.seed(42)`, 1,000 epochs, and a learning rate of `0.000001`.

| Model | MAE (LUTs) | RMSE (LUTs) | R² |
| --- | ---: | ---: | ---: |
| HLS estimate (`prediction = lut`) | 5,758.98 | 9,836.98 | 0.3846 |
| Linear regression | 936.89 | 1,595.88 | 0.9838 |
| Custom NumPy neural network | 3,044.42 | 5,796.02 | 0.7864 |

Lower MAE and RMSE are better; higher R² is better. R² is not percentage
accuracy. Linear regression currently performs best on this validation split.
The saved custom-network run has a final training RMSE of 4,833.22 LUTs and
validation RMSE of 5,796.02 LUTs.

These are exploratory results from one benchmark, FPGA device, and random
split. The test set is reserved for final evaluation. Further experiments
should compare models on the same validation examples and report errors in
original LUT units.

## Project structure

```text
CircuitCost/
├── Data Exploration/understandDataset.ipynb
├── Notebboks/dataPrep.ipynb
├── src/circuitcost/
│   ├── __init__.py
│   ├── models.py
│   └── train.py
├── tests/README.md
├── ML4Accel-Dataset/   # Local clone, ignored by Git
├── Makefile
├── pyproject.toml
└── README.md
```

## Development

```sh
make check   # Check Python style and formatting
```

Checks are scoped to `src/` and `tests/`. There are currently no automated
Python tests; `make check` does not train models or evaluate predictions.
Future tests can be placed under `tests/` and run with
`python -m pytest tests`.

The project is licensed under [Apache License 2.0](LICENSE). Consult the source
dataset's documentation for its usage terms and attribution.
