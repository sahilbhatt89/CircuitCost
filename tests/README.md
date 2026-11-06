Run the automated tests from the repository root with:

```sh
make test
```

Tests use synthetic data and temporary CSVs; the external dataset and saved
experiment artifacts are not required. They cover forward output shapes,
finite-difference gradients for dense layers and the complete ReLU network,
restoration of earlier weights and biases after validation deteriorates,
design-ID joins with reordered input files, and LUT-range error summaries.

`make check` runs both Python lint/format checks and this test suite.
