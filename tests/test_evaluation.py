import numpy as np
import pytest

from circuitcost.evaluation import errors_by_lut_range


def test_range_boundaries_counts_and_signed_bias():
    result = errors_by_lut_range(
        [1000, 5000, 15000], {"model": [900, 5200, 14500]}
    ).set_index("LUT range")
    assert result["Count"].tolist() == [1, 1, 1]
    np.testing.assert_allclose(result["MAE"], [100, 200, 500])
    np.testing.assert_allclose(result["RMSE"], [100, 200, 500])
    np.testing.assert_allclose(result["Bias"], [-100, 200, -500])


def test_range_summary_rejects_misaligned_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        errors_by_lut_range([1000, 5000], {"model": [900]})
