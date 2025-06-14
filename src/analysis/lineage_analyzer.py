# src/analysis/lineage_analyzer.py

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

from ..core.shared_types import (
    HexCoord,
    Phenotype,
)  # Keep if direct type use, else remove
from ..grid.coordinate_utils import axial_to_cartesian


def parse_snapshot_data(snapshot_filepath: Path, hex_size: float) -> pd.DataFrame:
    """
    Loads a grid snapshot JSON, pre-processes, and returns a DataFrame.
    Adds cartesian coordinates and angular positions.
    Assumes 'q', 'r', 'phenotype', 'lineage_id', 'generation', 'birth_time', 'is_frontier'
    fields exist in the snapshot JSON.
    """
    with open(snapshot_filepath, "r") as f:
        # Check if file is empty to avoid json.JSONDecodeError
        content = f.read()
        if not content.strip():
            return pd.DataFrame()
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {snapshot_filepath}: {e}")
            return pd.DataFrame()

    if not data:  # Handles case where JSON is valid but empty list '[]'
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Ensure essential columns exist
    required_cols = [
        "q",
        "r",
        "lineage_id",
        "phenotype",
        "generation",
        "birth_time",
        "is_frontier",
    ]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(
            f"Warning: Snapshot {snapshot_filepath.name} missing required columns: {missing}. Skipping."
        )
        return pd.DataFrame()

    # Type conversions for safety, can be expanded
    df["q"] = df["q"].astype(int)
    df["r"] = df["r"].astype(int)
    df["lineage_id"] = df["lineage_id"].astype(str)  # UUIDs become strings

    df["generation"] = pd.to_numeric(df["generation"], errors="coerce")
    df["birth_time"] = pd.to_numeric(df["birth_time"], errors="coerce")

    df.dropna(subset=["generation", "birth_time"], inplace=True)
    if df.empty:  # If all rows were dropped due to conversion errors
        print(
            f"Warning: All rows in {snapshot_filepath.name} resulted in NaN for generation/birth_time after conversion."
        )
        return pd.DataFrame()

    # Calculate Cartesian coordinates and angle (relative to origin 0,0)
    cart_coords = [
        axial_to_cartesian(HexCoord(row["q"], row["r"]), hex_size)
        for _, row in df.iterrows()
    ]
    df["cart_x"] = [c[0] for c in cart_coords]
    df["cart_y"] = [c[1] for c in cart_coords]

    df["angle_rad_norm"] = (np.arctan2(df["cart_y"], df["cart_x"]) + 2 * np.pi) % (
        2 * np.pi
    )

    return df


def calculate_P_lineage_snapshot_metrics(
    frontier_P_cells_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Calculates summary statistics for P-lineages on the frontier from a single snapshot.
    """
    if frontier_P_cells_df.empty:
        # Return a structure with NaNs for all expected metrics
        return {
            "num_unique_P_lineages": 0,
            "P_lineage_sizes": [],
            "mean_P_lineage_size": np.nan,
            "median_P_lineage_size": np.nan,
            "std_P_lineage_size": np.nan,
            "min_P_lineage_size": np.nan,
            "max_P_lineage_size": np.nan,
            "mean_P_lineage_min_gen_proxy": np.nan,
            "mean_P_lineage_min_birth_time_proxy": np.nan,
            "mean_P_lineage_angular_span_rad": np.nan,
            "P_lineage_angular_spans_rad": [],
        }

    lineage_groups = frontier_P_cells_df.groupby("lineage_id")

    P_lineage_sizes = lineage_groups.size().tolist()
    min_gens_per_lineage = lineage_groups["generation"].min().tolist()
    min_birth_times_per_lineage = lineage_groups["birth_time"].min().tolist()

    P_lineage_angular_spans_rad = []
    for _, group_df in lineage_groups:
        if len(group_df) <= 1:  # Single cell or empty group
            P_lineage_angular_spans_rad.append(0.0)
        else:
            sorted_angles = np.sort(group_df["angle_rad_norm"].values)
            gaps = np.diff(sorted_angles)
            # The largest gap also considers the wrap-around from 2pi to 0
            largest_gap = (sorted_angles[0] + 2 * np.pi) - sorted_angles[-1]
            # Span is 2pi minus the largest gap found
            actual_largest_gap = max(np.max(gaps) if gaps.size > 0 else 0, largest_gap)
            span = 2 * np.pi - actual_largest_gap
            P_lineage_angular_spans_rad.append(
                span if span >= 0 else 0.0
            )  # Ensure non-negative from float issues

    return {
        "num_unique_P_lineages": lineage_groups.ngroups,
        "P_lineage_sizes": P_lineage_sizes,
        "mean_P_lineage_size": np.mean(P_lineage_sizes) if P_lineage_sizes else np.nan,
        "median_P_lineage_size": (
            np.median(P_lineage_sizes) if P_lineage_sizes else np.nan
        ),
        "std_P_lineage_size": (
            np.std(P_lineage_sizes) if len(P_lineage_sizes) > 1 else 0.0
        ),
        "min_P_lineage_size": np.min(P_lineage_sizes) if P_lineage_sizes else np.nan,
        "max_P_lineage_size": np.max(P_lineage_sizes) if P_lineage_sizes else np.nan,
        "mean_P_lineage_min_gen_proxy": (
            np.mean(min_gens_per_lineage) if min_gens_per_lineage else np.nan
        ),
        "mean_P_lineage_min_birth_time_proxy": (
            np.mean(min_birth_times_per_lineage)
            if min_birth_times_per_lineage
            else np.nan
        ),
        "mean_P_lineage_angular_span_rad": (
            np.mean(P_lineage_angular_spans_rad)
            if P_lineage_angular_spans_rad
            else np.nan
        ),
        "P_lineage_angular_spans_rad": P_lineage_angular_spans_rad,
    }


def calculate_P_lineage_dynamics_between_snapshots(
    frontier_P_lineage_ids_t1: set,  # Set of P-lineage IDs on frontier at time t1
    frontier_P_lineage_ids_t2: set,  # Set of P-lineage IDs on frontier at time t2
) -> Dict[str, Any]:
    """
    Calculates P-lineage persistence and origination between two sets of lineage IDs.
    """
    surviving_lineages = frontier_P_lineage_ids_t1.intersection(
        frontier_P_lineage_ids_t2
    )
    num_survived = len(surviving_lineages)

    survival_rate = np.nan
    if frontier_P_lineage_ids_t1:  # Avoid division by zero if no lineages at t1
        survival_rate = num_survived / len(frontier_P_lineage_ids_t1)
    elif not frontier_P_lineage_ids_t1 and not frontier_P_lineage_ids_t2:  # Both empty
        survival_rate = np.nan  # Or 1.0 if defined as "all zero lineages survived"
    elif (
        not frontier_P_lineage_ids_t1 and frontier_P_lineage_ids_t2
    ):  # None to survive from
        survival_rate = np.nan  # Or could be seen as 0% "of potential survivors"

    originated_lineages = frontier_P_lineage_ids_t2.difference(
        frontier_P_lineage_ids_t1
    )
    num_originated = len(originated_lineages)

    return {
        "num_P_lineages_survived": num_survived,
        "P_lineage_survival_rate": survival_rate,
        "num_P_lineages_originated": num_originated,
        # "surviving_lineage_ids": list(surviving_lineages), # Optional: if needed for further tracing
        # "originated_lineage_ids": list(originated_lineages), # Optional
    }
