"""Load matched HLS features and implementation targets by design identifier."""

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = (
    "clk_target(ns)",
    "clk_estimated(ns)",
    "clk_uncertainty(ns)",
    "bram",
    "dsp",
    "ff",
    "lut",
    "c_num_arith",
    "c_num_logic",
    "rtl_num_arith",
    "rtl_num_logic",
)
TARGET_COLUMN = "Total LUTs"
DATASET_REVISION = "e05bebec02ab3b80648e45377ffc91702ea8d2dd"
DATASET_SUBDIR = Path(
    "ML4Accel-Dataset/fpga_ml_dataset/HLS_dataset/polybench/gemm/xczu9eg-ffvb1156-2-i"
)


def find_project_root(start=None):
    """Locate the checkout from its root or either notebook directory."""
    start = Path.cwd() if start is None else Path(start).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "circuitcost"
        ).is_dir():
            return candidate
    raise FileNotFoundError(
        "Open the notebook from the CircuitCost checkout or a directory inside it."
    )


def load_designs(data_dir):
    """Join complete, unique design IDs while preserving HLS file order.

    Implementation data contributes only the target. Unmatched IDs, duplicate
    IDs, missing measurements, and nonfinite numerical values are rejected.
    The design ID is retained as both a column and the index for split tracking.
    """
    data_dir = Path(data_dir)
    try:
        hls = pd.read_csv(data_dir / "post_hls_info.csv", dtype={"prj": "string"})
        impl = pd.read_csv(
            data_dir / "post_implementation_info.csv", dtype={"prj": "string"}
        )
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Missing dataset CSV in {data_dir}. Follow the README dataset setup."
        ) from error

    for name, table, required in (
        ("HLS", hls, ("prj", *FEATURE_COLUMNS)),
        ("implementation", impl, ("prj", TARGET_COLUMN)),
    ):
        missing = set(required) - set(table.columns)
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")
        if table.empty:
            raise ValueError(f"{name}: no designs found")
        ids = table["prj"]
        if ids.isna().any() or ids.str.strip().eq("").any():
            raise ValueError(f"{name}: missing or blank prj identifiers")
        if ids.duplicated().any():
            raise ValueError(f"{name}: duplicate prj identifiers")

    hls_ids, impl_ids = set(hls["prj"]), set(impl["prj"])
    if hls_ids != impl_ids:
        raise ValueError(
            f"Design IDs do not match: {len(hls_ids - impl_ids)} HLS-only, "
            f"{len(impl_ids - hls_ids)} implementation-only"
        )

    designs = hls[["prj", *FEATURE_COLUMNS]].merge(
        impl[["prj", TARGET_COLUMN]],
        on="prj",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    numerical = [*FEATURE_COLUMNS, TARGET_COLUMN]
    designs[numerical] = designs[numerical].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(designs[numerical].to_numpy(dtype=float)).all():
        raise ValueError("Features and targets must be finite, nonmissing numbers")
    return designs.set_index("prj", drop=False)
