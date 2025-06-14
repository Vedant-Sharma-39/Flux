# src/visualization/plotter.py

import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union  # Added Union
import pandas as pd

from ..core.shared_types import HexCoord, Phenotype, Nutrient, SimulationParameters

# from ..core.cell import Cell # Only needed if grid is directly Cell objects
from ..grid.grid import Grid
from ..grid.coordinate_utils import axial_to_cartesian
from ..environment.environment_rules import EnvironmentRules

# --- Configuration for Plotting ---
PHENOTYPE_COLORS = {
    Phenotype.G_UNPREPARED: "skyblue",
    Phenotype.P_PREPARED: "salmon",
    "OTHER_P_LINEAGE": "lightcoral",  # For lineage plotting
    "UNKNOWN_PHENOTYPE": "grey",  # Fallback for unknown phenotype values
}
NUTRIENT_ZONE_COLORS_BACKGROUND = {
    Nutrient.N1_PREFERRED: "#c8e6c9",
    Nutrient.N2_CHALLENGING: "#ffcdd2",
    Nutrient.NONE: "#f5f5f5",
}
DEFAULT_CELL_COLOR = PHENOTYPE_COLORS["UNKNOWN_PHENOTYPE"]
HEX_EDGE_COLOR = "black"
CELL_EDGE_COLOR = "dimgray"
HEX_ALPHA = 0.9
INITIAL_PLOT_RADIUS_CARTESIAN_FACTOR = 20.0


def plot_nutrient_background(
    ax: plt.Axes,
    environment_rules: EnvironmentRules,
    # Can be SimulationParameters object or a dict from loaded JSON
    sim_params_obj_or_dict: Union[SimulationParameters, Dict[str, Any]],
    current_plot_max_radius_cartesian: float,
):
    """Draws nutrient zones as background."""
    # Extract nutrient_bands carefully
    if isinstance(sim_params_obj_or_dict, SimulationParameters):
        nutrient_bands_def = sim_params_obj_or_dict.nutrient_bands
    elif isinstance(sim_params_obj_or_dict, dict):
        # Assuming bands in dict are [(radius_sq_float, nutrient_name_str), ...]
        # and need conversion to enum if plotter expects enums
        raw_bands = sim_params_obj_or_dict.get("nutrient_bands", [])
        nutrient_bands_def = []
        for r_sq, nt_name in raw_bands:
            try:
                nutrient_bands_def.append((float(r_sq), Nutrient[nt_name]))
            except KeyError:  # Should not happen if sim_params.json is well-formed
                nutrient_bands_def.append((float(r_sq), Nutrient.NONE))  # Fallback
            except ValueError:  # for r_sq if 'inf'
                nutrient_bands_def.append(
                    (
                        float("inf") if str(r_sq).lower() == "inf" else float(r_sq),
                        Nutrient[nt_name],
                    )
                )

    else:
        nutrient_bands_def = []  # No bands if sim_params type is unexpected

    # Ensure bands are sorted by radius_sq (already done in SimParams, ensure if from dict)
    # The plotter itself shouldn't re-sort if it comes from SimParams object.
    # If from dict, it might need sorting if not guaranteed.
    # For simplicity, assuming EnvironmentRules handles sorted bands correctly.
    # The environment_rules object passed should be initialized with sorted bands.

    # Logic for determining fallback_nutrient and drawing circles:
    # This part uses environment_rules.params.nutrient_bands, which should be the authoritative sorted list.
    # So, the sim_params_obj_or_dict passed here is mainly for hex_size if needed,
    # but nutrient background relies on the bands defined within environment_rules.
    # Let's clarify: plot_nutrient_background should ideally just take `environment_rules`
    # and `current_plot_max_radius_cartesian`.
    # The `sim_params_obj_or_dict` is mainly for `plot_colony_snapshot` to get `hex_size`.

    # The existing logic in plotter.py used sim_params.nutrient_bands.
    # If EnvironmentRules is the source of truth for bands, use that.
    # Let's assume environment_rules.params has the correctly processed bands.

    bands_to_plot = (
        environment_rules.params.nutrient_bands
    )  # This should be List[Tuple[float, Nutrient]]

    fallback_nutrient = Nutrient.NONE
    if bands_to_plot:
        # Check for an infinite band first
        infinite_band_nutrient = next(
            (nt for r_sq, nt in bands_to_plot if r_sq == float("inf")), None
        )
        if infinite_band_nutrient:
            fallback_nutrient = infinite_band_nutrient
        else:
            # If no infinite band, the fallback beyond the largest finite band is NONE
            # unless the plot radius is within the first band.
            # This logic can be complex. A simpler approach:
            # The EnvironmentRules.get_nutrient_at_coord for a very large radius gives the effective fallback.
            # However, for plotting, we draw layers.

            # Revised fallback logic:
            # The nutrient for the outermost region (up to current_plot_max_radius_cartesian)
            # is determined by the bands. If current_plot_max_radius_cartesian is beyond all defined
            # finite bands, and no infinite band exists, it's Nutrient.NONE.

            # Default to Nutrient.NONE if outside all defined bands
            current_plot_r_sq = current_plot_max_radius_cartesian**2
            determined_outer_nutrient = Nutrient.NONE
            for max_r_sq, nt in bands_to_plot:  # bands_to_plot is sorted
                if current_plot_r_sq < max_r_sq:
                    determined_outer_nutrient = nt
                    break
            # If loop finishes and determined_outer_nutrient is still NONE, it means
            # current_plot_r_sq is beyond the largest finite radius_sq, or bands_to_plot is empty.
            # If an infinite band exists, it would have been caught by `get_nutrient_at_coord` logic effectively.
            # The environment_rules should handle this.

            # Let's use a simpler layering: Draw the largest circle with Nutrient.NONE first (or a default background)
            # then layer on top. The original logic was okay.

            # Simplified from original plotter for clarity:
            # Determine nutrient at the very edge of the plot for the base circle
            # This relies on environment_rules.get_nutrient_at_coord which is robust
            # (need a dummy HexCoord far away) - this is overthinking.
            # The original layering approach from outer to inner is fine.

            # Using original fallback logic structure from your plotter:
            largest_finite_band_max_r_sq = -1.0
            found_infinite_band = False
            for max_r_sq, nt in reversed(bands_to_plot):
                if max_r_sq == float("inf"):
                    fallback_nutrient = nt
                    found_infinite_band = True
                    break
                if largest_finite_band_max_r_sq < 0:
                    largest_finite_band_max_r_sq = max_r_sq

            if not found_infinite_band:
                if (
                    largest_finite_band_max_r_sq > 0
                    and current_plot_max_radius_cartesian**2
                    > largest_finite_band_max_r_sq
                ):
                    fallback_nutrient = Nutrient.NONE  # Beyond largest finite band
                elif bands_to_plot:  # Within some defined bands
                    # Find which band current_plot_max_radius_cartesian falls into for the fallback
                    for max_r_sq_b, nt_b in bands_to_plot:  # Assumed sorted
                        if current_plot_max_radius_cartesian**2 < max_r_sq_b:
                            fallback_nutrient = nt_b
                            break
                # If still Nutrient.NONE and bands_to_plot is not empty, it implies it's within the first band.
                # This specific case is complex, relying on sorted bands and layering.

    # Base circle for the entire plot area
    ax.add_artist(
        plt.Circle(
            (0, 0),
            current_plot_max_radius_cartesian * 1.05,  # Ensure it covers plot edges
            color=NUTRIENT_ZONE_COLORS_BACKGROUND.get(fallback_nutrient, "white"),
            alpha=0.25,
            zorder=-20,
        )
    )

    # Draw defined bands (finite radii) from largest to smallest radius
    # This ensures smaller bands are drawn on top of larger ones.
    # Filter for finite bands and sort them descending by radius for drawing.
    finite_bands = sorted(
        [(r_sq, nt) for r_sq, nt in bands_to_plot if r_sq != float("inf")],
        key=lambda x: x[0],
        reverse=True,
    )

    for max_r_sq_band, nutrient_type in finite_bands:
        radius_to_draw = np.sqrt(max_r_sq_band)
        if radius_to_draw <= 0:
            continue

        color = NUTRIENT_ZONE_COLORS_BACKGROUND.get(nutrient_type, "white")
        # This circle represents the area *up to* this band's outer limit
        circle = plt.Circle(
            (0, 0),
            radius_to_draw,
            color=color,
            alpha=0.25,
            zorder=-19 + (1 / (radius_to_draw + 1e-6)),
        )  # Vary zorder slightly
        ax.add_artist(circle)


def plot_colony_snapshot(
    # Non-default arguments FIRST
    sim_params_obj_or_dict: Union[SimulationParameters, Dict[str, Any]],
    current_sim_time: float,
    # Optional/default arguments AFTER
    grid: Optional[Grid] = None,
    cells_df: Optional[pd.DataFrame] = None,
    environment_rules: Optional[EnvironmentRules] = None,
    ax: Optional[plt.Axes] = None,
    show_grid_lines: bool = False,
    color_by: str = "phenotype",
    p_lineage_color_map: Optional[Dict[str, Any]] = None,
    custom_legend_handles: Optional[List[plt.Rectangle]] = None,
    custom_legend_labels: Optional[List[str]] = None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.get_figure()
    ax.clear()

    # Extract hex_size consistently
    if isinstance(sim_params_obj_or_dict, SimulationParameters):
        hex_size = sim_params_obj_or_dict.hex_size
    elif isinstance(sim_params_obj_or_dict, dict):
        hex_size = float(
            sim_params_obj_or_dict.get("hex_size", 1.0)
        )  # Default if missing
    else:
        raise TypeError("sim_params_obj_or_dict must be SimulationParameters or dict")

    # --- Determine plot limits & max visible radius ---
    cell_coords_for_bounds: List[Tuple[int, int]] = []
    num_total_cells = 0

    # Prioritize cells_df if provided, as it's likely from a specific snapshot file
    if cells_df is not None and not cells_df.empty:
        num_total_cells = len(cells_df)
        # Ensure q, r are integers if coming from DataFrame
        for _, row in cells_df.iterrows():
            cell_coords_for_bounds.append((int(row["q"]), int(row["r"])))
    elif grid is not None:
        live_cells_with_coords = grid.get_all_cells_with_coords()
        num_total_cells = len(live_cells_with_coords)
        for hc, _ in live_cells_with_coords:
            cell_coords_for_bounds.append((hc.q, hc.r))

    # ... (Plot boundary calculation logic as in your previous refined version, using hex_size and cell_coords_for_bounds) ...
    initial_plot_radius_cartesian = INITIAL_PLOT_RADIUS_CARTESIAN_FACTOR * hex_size
    min_x_plot, max_x_plot = (
        -initial_plot_radius_cartesian,
        initial_plot_radius_cartesian,
    )
    min_y_plot, max_y_plot = (
        -initial_plot_radius_cartesian,
        initial_plot_radius_cartesian,
    )
    max_coord_radius_cartesian = 0.0

    if cell_coords_for_bounds:
        all_cart_coords = [
            axial_to_cartesian(HexCoord(q, r), hex_size)
            for q, r in cell_coords_for_bounds
        ]
        if all_cart_coords:  # Ensure not empty
            all_x_cell_centers = [c[0] for c in all_cart_coords]
            all_y_cell_centers = [c[1] for c in all_cart_coords]

            padding = hex_size * 3
            min_x_plot = min(min_x_plot, min(all_x_cell_centers) - padding)
            max_x_plot = max(max_x_plot, max(all_x_cell_centers) + padding)
            min_y_plot = min(min_y_plot, min(all_y_cell_centers) - padding)
            max_y_plot = max(max_y_plot, max(all_y_cell_centers) + padding)

            for x_cart, y_cart in all_cart_coords:
                dist_sq = x_cart**2 + y_cart**2
                if dist_sq > max_coord_radius_cartesian**2:
                    max_coord_radius_cartesian = (
                        np.sqrt(dist_sq) if dist_sq > 0 else 0.0
                    )

    effective_bg_radius = max(
        initial_plot_radius_cartesian, max_coord_radius_cartesian + hex_size * 5
    )
    ax.set_xlim(min_x_plot, max_x_plot)
    ax.set_ylim(min_y_plot, max_y_plot)

    # --- Plot Nutrient Background (if environment_rules provided) ---
    if environment_rules:
        plot_nutrient_background(
            ax, environment_rules, sim_params_obj_or_dict, effective_bg_radius
        )
    else:  # Default plain background if no env_rules
        ax.set_facecolor("#f0f0f0")  # Light grey

    # --- Plot Cells ---
    patches = []
    hex_radius_draw = hex_size * 0.90  # Small gaps for visual distinction

    # Prepare cell data iterator
    cell_data_iterator = []
    if cells_df is not None and not cells_df.empty:
        for _, row in cells_df.iterrows():
            # Convert phenotype string from DF to Enum if necessary
            phenotype_val = row["phenotype"]
            current_phenotype_enum = None
            if isinstance(phenotype_val, str):
                try:
                    current_phenotype_enum = Phenotype[phenotype_val]
                except KeyError:
                    print(
                        f"Warning: Unknown phenotype string '{phenotype_val}' in cells_df."
                    )
            elif isinstance(phenotype_val, Phenotype):
                current_phenotype_enum = phenotype_val

            cell_data_iterator.append(
                {
                    "q": int(row["q"]),
                    "r": int(row["r"]),
                    "phenotype": current_phenotype_enum,
                    "lineage_id": str(
                        row.get("lineage_id", "N/A")
                    ),  # Ensure lineage_id is string
                }
            )
    elif grid is not None:
        for (
            hc,
            cell_obj,
        ) in grid.get_all_cells_with_coords():  # Use previously fetched if available
            cell_data_iterator.append(
                {
                    "q": hc.q,
                    "r": hc.r,
                    "phenotype": cell_obj.phenotype,
                    "lineage_id": str(cell_obj.lineage_id),
                }
            )

    for cell_data in cell_data_iterator:
        coord = HexCoord(cell_data["q"], cell_data["r"])
        cartesian_coord = axial_to_cartesian(coord, hex_size)

        current_phenotype = cell_data["phenotype"]
        face_color = PHENOTYPE_COLORS.get(
            current_phenotype, DEFAULT_CELL_COLOR
        )  # Default if phenotype is None

        if color_by == "lineage_p" and p_lineage_color_map is not None:
            if current_phenotype == Phenotype.P_PREPARED:
                lineage_id_str = cell_data["lineage_id"]
                face_color = p_lineage_color_map.get(
                    lineage_id_str, PHENOTYPE_COLORS["OTHER_P_LINEAGE"]
                )
            elif current_phenotype == Phenotype.G_UNPREPARED:
                face_color = PHENOTYPE_COLORS[Phenotype.G_UNPREPARED]
            # else keep default for unknown/None phenotypes
        elif color_by == "nutrient_at_cell" and environment_rules:
            nutrient = environment_rules.get_nutrient_at_coord(coord)
            face_color = NUTRIENT_ZONE_COLORS_BACKGROUND.get(
                nutrient, DEFAULT_CELL_COLOR
            )
        # Default is already phenotype color (or DEFAULT_CELL_COLOR if phenotype was None)

        hexagon = RegularPolygon(
            cartesian_coord,
            numVertices=6,
            radius=hex_radius_draw,
            orientation=np.pi / 6,  # Pointy topped
            facecolor=face_color,
            edgecolor=HEX_EDGE_COLOR if show_grid_lines else CELL_EDGE_COLOR,
            alpha=HEX_ALPHA,
            linewidth=0.3 if show_grid_lines else 0.15,
            zorder=10,
        )
        patches.append(hexagon)

    if patches:
        ax.add_collection(PatchCollection(patches, match_original=True))

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Colony Time: {current_sim_time:.2f} (Cells: {num_total_cells})")
    ax.set_xlabel(f"X (hex_size={hex_size})")  # Shorter labels
    ax.set_ylabel(f"Y (hex_size={hex_size})")

    # Legend Handling
    if custom_legend_handles and custom_legend_labels:
        final_handles, final_labels = custom_legend_handles, custom_legend_labels
        legend_title = "Legend"
    else:  # Default phenotype legend
        final_handles = [
            plt.Rectangle(
                (0, 0), 1, 1, facecolor=color
            )  # Removed edgecolor for cleaner legend patch
            for phenotype_enum, color in PHENOTYPE_COLORS.items()
            if isinstance(phenotype_enum, Phenotype)  # Only actual enums
        ]
        final_labels = [
            phenotype_enum.name
            for phenotype_enum in PHENOTYPE_COLORS.keys()
            if isinstance(phenotype_enum, Phenotype)
        ]
        legend_title = "Phenotypes"

    if final_handles:  # Only show legend if there's something to show
        ax.legend(
            final_handles,
            final_labels,
            title=legend_title,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize="small",
            title_fontsize="medium",
        )

    fig.tight_layout(rect=[0, 0, 0.83, 1])  # Adjust for legend
    return fig, ax


def plot_population_dynamics(
    time_points: List[float],
    total_population: List[int],
    phenotype_g_population: Optional[List[int]] = None,  # Changed A to G
    phenotype_p_population: Optional[List[int]] = None,  # Changed B to P
    ax: Optional[plt.Axes] = None,
    title: str = "Population Dynamics",
):
    """Plots population counts over time."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()  # Get parent figure if ax is passed

    ax.plot(
        time_points,
        total_population,
        label="Total Population",
        color="black",
        linewidth=2,
    )

    if phenotype_g_population is not None:
        ax.plot(
            time_points,
            phenotype_g_population,
            label=Phenotype.G_UNPREPARED.name,
            color=PHENOTYPE_COLORS[Phenotype.G_UNPREPARED],
            linestyle="--",
        )
    if phenotype_p_population is not None:
        ax.plot(
            time_points,
            phenotype_p_population,
            label=Phenotype.P_PREPARED.name,
            color=PHENOTYPE_COLORS[Phenotype.P_PREPARED],
            linestyle=":",
        )

    ax.set_xlabel("Time (simulation units)")
    ax.set_ylabel("Number of Cells")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.7)
    if fig:  # Ensure fig object exists if ax was passed
        fig.tight_layout()
    return fig, ax
