# CircuitCost final findings

## Question and scope

Can HLS measurements predict implementation LUT usage better than the original
HLS LUT estimate? This study uses **531 Polybench GEMM design configurations**
for **xczu9eg-ffvb1156-2-i**, with 11 numerical HLS features and implementation
`Total LUTs` as the target. It compares an unchanged HLS estimate, ordinary
least-squares linear regression, and a manually differentiated NumPy MLP.

## Frozen experiment

- Dataset CSV revision: `e05bebec02ab3b80648e45377ffc91702ea8d2dd`.
- Matched one-to-one `prj` identifiers; HLS input-file order preserved.
- Fixed split seed 42: 371 training, 80 validation, and 80 test designs.
- Input `StandardScaler` fitted on training features only; target remains in LUTs.
- MLP: `11 -> 32 -> ReLU -> 16 -> ReLU -> 1`, linear output.
- Random-normal weights and biases; initialization seed 42.
- Full-batch gradient descent, MSE objective, learning rate `1e-6`.
- Maximum 50,000 epochs; lowest-validation-MSE parameters copied and restored.
- Selected checkpoint: **epoch 2,336**. Training did not stop at this epoch.
- Seed 42 remained the predefined test candidate after the seed checks.

The numerical initialization and training recipe were preserved while adding
tests and reproducible loading. The published metrics below were verified by
loading the previously frozen parameters and scaler, without retraining or
choosing a replacement model. CSV checksums and numerical package versions used
for that verification are recorded in
[`reports/final_results.json`](../reports/final_results.json).

## Validation and initialization checks

| Model | Validation MAE | Validation RMSE | R² |
| --- | ---: | ---: | ---: |
| HLS estimate | 5,758.98 | 9,836.98 | 0.384620 |
| Linear regression | 936.89 | 1,595.88 | 0.983804 |
| Selected MLP, seed 42 | 548.20 | 891.69 | 0.994944 |

Recorded initialization experiments used the same data split and training recipe:

| Seed | Selected epoch | Validation MAE | Validation RMSE |
| --- | ---: | ---: | ---: |
| 42 | 2,336 | 548.20 | 891.69 |
| 7 | 13,148 | 626.29 | 1,102.23 |
| 21 | 4,322 | 692.81 | 1,023.80 |

Mean validation RMSE was approximately 1,005.90 LUTs, with sample standard
deviation 106.41 LUTs. These three successful runs support an initial stability
check on this split. They do not estimate variation across different training
sets or benchmarks. Seeds 7 and 21 in
[`recorded_seed_results.csv`](../reports/recorded_seed_results.csv) are recorded
notebook outputs, not newly rerun experiments; their displayed values were rounded.

## Held-out test results

| Model | Test MAE | Test RMSE | R² |
| --- | ---: | ---: | ---: |
| HLS estimate | 5,025.46 | 8,018.33 | 0.637978 |
| Linear regression | 812.26 | **1,023.71** | **0.994099** |
| Custom MLP | **603.59** | 1,157.32 | 0.992458 |

Compared with linear regression, the MLP reduced test MAE by **25.7%** but
increased test RMSE by **13.1%**. Its average absolute error improved, while its
larger errors outweighed that benefit under a squared-error metric. The
validation RMSE ranking did not carry over to the test set. R² is not percentage
accuracy and, on the same targets, adds no independent ranking beyond MSE.

The experiment therefore does **not** establish that the MLP is uniformly better.
Both learned models substantially outperform the uncorrected HLS estimate.

## How errors vary with design size

These are fixed, descriptive groups based on actual implementation LUT count.
They were used after the frozen evaluation to summarize errors, not to tune the
model or choose predictions. Actual implementation usage is unavailable at the
intended prediction time, so these ranges cannot serve as an inference-time
model-selection rule.

| Actual LUT range | Test designs | Linear MAE | MLP MAE | Linear RMSE | MLP RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| <5,000 | 29 | 519.28 | **247.75** | 645.49 | **303.95** |
| 5,000–14,999 | 39 | 932.87 | **408.46** | 1,125.78 | **531.98** |
| >=15,000 | 12 | **1,128.27** | 2,097.71 | **1,364.07** | 2,790.39 |

The MLP improves aggregate MAE and RMSE in the two lower-LUT groups, which
contain 68 test designs. Linear regression performs better in the group of
12 largest designs. In that group,
the MLP's maximum absolute error is 7,125.09 LUTs versus 2,244.94 for linear
regression. Those larger-design errors explain why the MLP has worse overall
RMSE despite lower overall MAE.

Mean signed error is `prediction - actual`. In the largest group, the MLP's bias
is +332.20 LUTs and linear regression's is -265.08 LUTs. Neither sign alone
describes every design; opposing errors can cancel in the mean. Complete counts,
metrics, bias, and maximum errors are in
[`test_errors_by_lut_range.csv`](../reports/test_errors_by_lut_range.csv).

## Why linear regression remains competitive

HLS features already describe resource usage, so a learned linear calibration
is a strong baseline. Here it achieves high test R² and better large-design
errors than the MLP. Its convex least-squares fit also avoids choosing among
thousands of neural-network training checkpoints.

The linear model has 11 input coefficients and an intercept, while the MLP has
929 weights and biases and is fitted on only 371 training designs. This makes
model complexity and selection variability relevant considerations, although
parameter counts alone do not prove the cause of the observed errors. A deeper
or wider network is not automatically preferable to the simpler baseline.

## Limits of the conclusion

1. **One kernel and one FPGA:** all examples are GEMM configurations on one
   device, not a sample of unrelated circuits. A random configuration split
   does not test transfer to new algorithms or circuit families.
2. **Related configurations:** training and test sets can share similar source
   structure and pragmas. Predicting another configuration is easier than
   predicting a completely new benchmark.
3. **Small evaluation sets:** each validation/test set has 80 designs, and the
   largest test group has only 12. No statistical significance or universal
   performance advantage is claimed.
4. **Validation-based selection:** architectures and many checkpoints were
   examined using validation results. The chosen validation score can be
   optimistic; a strong validation result does not guarantee the same test ranking.
5. **One test evaluation:** these recorded test results describe the frozen
   candidate. They should not be repeatedly used to choose new configurations.
6. **Limited seed check:** three initialization seeds do not establish robustness
   to different data splits, devices, tool versions, or workloads.

Generalization to different circuits would require independent benchmarks or a
grouped evaluation that holds out whole kernels/families, with preprocessing
fitted inside each training split. Cross-device and cross-tool-version claims
would require corresponding held-out data. Those experiments were not performed.

## Reproducing and checking the implementation

The final workflow is in
[`Notebboks/finalExperiment.ipynb`](../Notebboks/finalExperiment.ipynb).
Use the README setup and recorded numerical-version constraints. Fixed seeds,
CSV fingerprints, explicit design-ID joins, and saved split IDs make the
procedure reproducible; different BLAS implementations or platforms can still
introduce numerical differences in iterative training.

`make check` runs synthetic-data tests for forward shapes, dense and complete
network finite-difference gradients, ReLU gradients, and restoration of an
earlier best-validation parameter snapshot. Additional tests verify reordered
CSV matching and range-summary boundaries. These tests validate implementation
behavior; they do not establish predictive generalization.
