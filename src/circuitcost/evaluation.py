"""Regression metrics and descriptive error summaries for the frozen experiment."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def errors_by_lut_range(actual, predictions):
    """Summarize errors in fixed actual-LUT ranges for descriptive reporting.

    Actual LUTs are used only to analyze errors, never to select predictions.
    Bias is predicted minus actual: positive values mean overestimation.
    """
    actual = np.asarray(actual, dtype=float).reshape(-1)
    groups = pd.cut(
        actual,
        bins=[0, 5000, 15000, np.inf],
        labels=["<5,000", "5,000–14,999", ">=15,000"],
        right=False,
    )
    rows = []
    for name, predicted in predictions.items():
        predicted = np.asarray(predicted, dtype=float).reshape(-1)
        if predicted.shape != actual.shape:
            raise ValueError("Predictions and targets must have equal lengths")
        for label in groups.categories:
            mask = np.asarray(groups == label)
            if not mask.any():
                continue
            error = predicted[mask] - actual[mask]
            rows.append(
                {
                    "LUT range": label,
                    "Model": name,
                    "Count": int(mask.sum()),
                    "MAE": float(np.abs(error).mean()),
                    "RMSE": float(np.sqrt(np.mean(error**2))),
                    "Bias": float(error.mean()),
                    "Max absolute error": float(np.abs(error).max()),
                }
            )
    return pd.DataFrame(rows)
