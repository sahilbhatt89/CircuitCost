"""Ensure table matching is independent of implementation CSV row order."""

import numpy as np
import pandas as pd
import pytest

from circuitcost.data import FEATURE_COLUMNS, find_project_root, load_designs


@pytest.fixture
def csv_pair(tmp_path):
    hls = pd.DataFrame(
        {
            "prj": ["design_b", "design_a", "design_c"],
            **{name: [1, 2, 3] for name in FEATURE_COLUMNS},
        }
    )
    impl = pd.DataFrame(
        {"prj": ["design_a", "design_c", "design_b"], "Total LUTs": [100, 300, 200]}
    )
    hls.to_csv(tmp_path / "post_hls_info.csv", index=False)
    impl.to_csv(tmp_path / "post_implementation_info.csv", index=False)
    return tmp_path, hls, impl


def test_join_matches_prj_and_preserves_hls_order(csv_pair):
    directory, _, _ = csv_pair
    designs = load_designs(directory)
    assert designs.index.tolist() == ["design_b", "design_a", "design_c"]
    assert designs["Total LUTs"].tolist() == [200, 100, 300]
    assert designs.loc["design_a", "lut"] == 2


@pytest.mark.parametrize(
    "filename,which", [("post_hls_info.csv", 1), ("post_implementation_info.csv", 2)]
)
def test_rejects_duplicate_designs(csv_pair, filename, which):
    table = csv_pair[which].copy()
    table.loc[1, "prj"] = table.loc[0, "prj"]
    table.to_csv(csv_pair[0] / filename, index=False)
    with pytest.raises(ValueError, match="duplicate prj"):
        load_designs(csv_pair[0])


def test_rejects_unmatched_designs(csv_pair):
    directory, _, impl = csv_pair
    impl.loc[0, "prj"] = "unknown_design"
    impl.to_csv(directory / "post_implementation_info.csv", index=False)
    with pytest.raises(ValueError, match="Design IDs do not match"):
        load_designs(directory)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_rejects_invalid_measurements(csv_pair, value):
    directory, hls, _ = csv_pair
    hls["lut"] = hls["lut"].astype(float)
    hls.loc[0, "lut"] = value
    hls.to_csv(directory / "post_hls_info.csv", index=False)
    with pytest.raises(ValueError, match="finite, nonmissing"):
        load_designs(directory)


def test_rejects_missing_identifiers(csv_pair):
    directory, _, impl = csv_pair
    impl.loc[0, "prj"] = " "
    impl.to_csv(directory / "post_implementation_info.csv", index=False)
    with pytest.raises(ValueError, match="missing or blank prj"):
        load_designs(directory)


def test_find_project_from_notebook_directory(tmp_path):
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "src" / "circuitcost").mkdir(parents=True)
    notebook_dir = tmp_path / "notebooks"
    notebook_dir.mkdir()
    assert find_project_root(notebook_dir) == tmp_path
