# src/visualization/plotter.py

import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Circle
from matplotlib.collections import PatchCollection
import numpy as np
from typing import List, Tuple, Optional

# Assuming these imports are correct based on your project structure
from ..core.shared_types import HexCoord, Phenotype, Nutrient, SimulationParameters
from ..core.cell import Cell  # Assuming Cell is defined correctly for type hinting
from ..grid.grid import Grid
from ..grid.coordinate_utils import axial_to_cartesian
from ..environment.environment_rules import EnvironmentRules


# --- Configuration for Plotting ---
PHENOTYPE_COLORS = {
    Phenotype.G_UNPREPARED: "skyblue",
    Phenotype.P_PREPARED: "salmon",
}
NUTRIENT_ZONE_COLORS_BACKGROUND = {
    Nutrient.N1_PREFERRED: "#c8e6c9",  # Light Green
    Nutrient.N2_CHALLENGING: "#ffcdd2",  # Light Red/Pink
    Nutrient.NONE: "#f5f5f5",  # Light Grey
}
DEFAULT_CELL_COLOR = "darkgrey"  # Changed for better contrast if phenotype unknown
HEX_EDGE_COLOR = "black"  # For grid lines if shown
CELL_EDGE_COLOR = "dimgray"  # For cell outlines
HEX_ALPHA = 0.9
INITIAL_PLOT_RADIUS_CARTESIAN_FACTOR = (
    20.0  # Factor to multiply by hex_size for initial view
)


def plot_nutrient_background(
    ax: plt.Axes,
    environment_rules: EnvironmentRules,  # Used to get nutrient info
    sim_params: SimulationParameters,
    current_plot_max_radius_cartesian: float,  # Max Cartesian radius visible or encompassing cells
):
    """
    Draws the nutrient zones as background using concentric circles.
    Bands are defined in sim_params.nutrient_bands as (max_squared_cartesian_radius, NutrientType).
    It's crucial that sim_params.nutrient_bands is sorted by increasing radius.
    """
    # Bands are (max_radius_sq, Nutrient_type) and should be pre-sorted by radius
    # in EnvironmentRules or SimulationParameters

    # Start with a background for the entire visible area, using the nutrient type
    # that would apply at the very edge of the visible area or beyond defined bands.
    fallback_nutrient = Nutrient.NONE  # Default if no bands or outside all bands

    # Determine the nutrient for the region from the outermost defined band up to current_plot_max_radius_cartesian
    # If current_plot_max_radius_cartesian is beyond the last defined band (not float('inf')), it's Nutrient.NONE
    # If current_plot_max_radius_cartesian is within a band, that band's nutrient applies to its outer region.

    if sim_params.nutrient_bands:
        # Check if the view extends beyond the largest finite band
        largest_finite_band_max_r_sq = -1.0
        found_infinite_band = False
        for max_r_sq, nt in reversed(sim_params.nutrient_bands):  # Check from largest
            if max_r_sq == float("inf"):
                fallback_nutrient = nt
                found_infinite_band = True
                break
            if largest_finite_band_max_r_sq < 0:  # first finite band from outside
                largest_finite_band_max_r_sq = max_r_sq

        if (
            not found_infinite_band
            and current_plot_max_radius_cartesian**2 > largest_finite_band_max_r_sq
        ):
            fallback_nutrient = Nutrient.NONE
        elif (
            not found_infinite_band
        ):  # View is within the largest finite band or smaller bands
            # Find which band current_plot_max_radius_cartesian falls into
            for max_r_sq, nt in sim_params.nutrient_bands:  # Assumed sorted
                if current_plot_max_radius_cartesian**2 < max_r_sq:
                    fallback_nutrient = nt
                    break
                if max_r_sq == float("inf"):  # Should have been caught above
                    fallback_nutrient = nt
                    break
        # If still fallback_nutrient = Nutrient.NONE and there are bands, it means it's within the first band
        elif not sim_params.nutrient_bands and fallback_nutrient == Nutrient.NONE:
            pass  # No bands, truly none
        elif (
            fallback_nutrient == Nutrient.NONE and sim_params.nutrient_bands
        ):  # Must be within the first band up to its limit
            if current_plot_max_radius_cartesian**2 < sim_params.nutrient_bands[0][0]:
                fallback_nutrient = sim_params.nutrient_bands[0][1]

    # Draw the largest circle first with the fallback_nutrient color
    ax.add_artist(
        plt.Circle(
            (0, 0),
            current_plot_max_radius_cartesian * 1.05,  # Slightly larger for safety
            color=NUTRIENT_ZONE_COLORS_BACKGROUND.get(fallback_nutrient, "white"),
            alpha=0.25,
            zorder=-20,
        )
    )

    # Now draw defined bands on top, from largest defined radius to smallest
    # This ensures correct layering of smaller circles on top of larger ones.
    # Bands are (max_radius_sq, Nutrient_type)
    # We want to draw the circle for nutrient_type up to its max_radius_sq
    sorted_defined_bands = sorted(
        [b for b in sim_params.nutrient_bands if b[0] != float("inf")],
        key=lambda x: x[0],
        reverse=True,
    )

    for max_r_sq_band, nutrient_type in sorted_defined_bands:
        radius_to_draw = np.sqrt(max_r_sq_band)
        if radius_to_draw <= 0:
            continue

        color = NUTRIENT_ZONE_COLORS_BACKGROUND.get(nutrient_type, "white")
        # This circle represents the area *up to* this band's outer limit,
        # filled with this band's nutrient color.
        # Smaller radius bands (processed later in this loop) will draw over this.
        circle = plt.Circle(
            (0, 0),
            radius_to_draw,
            color=color,
            alpha=0.25,
            zorder=-19 + (1 / (radius_to_draw + 1)),
        )  # Vary zorder slightly
        ax.add_artist(circle)


def plot_colony_snapshot(
    grid: Grid,
    environment_rules: EnvironmentRules,
    sim_params: SimulationParameters,
    current_sim_time: float,
    ax: Optional[plt.Axes] = None,
    show_grid_lines: bool = False,  # if True, shows black edges for hexes
    color_by: str = "phenotype",
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.get_figure()
    ax.clear()

    # --- Determine plot limits & max visible radius for background ---
    cells_with_coords = grid.get_all_cells_with_coords()

    # Default initial plot radius, scaled by hex_size
    initial_plot_radius_cartesian = (
        INITIAL_PLOT_RADIUS_CARTESIAN_FACTOR * sim_params.hex_size
    )

    min_x_plot = -initial_plot_radius_cartesian
    max_x_plot = initial_plot_radius_cartesian
    min_y_plot = -initial_plot_radius_cartesian
    max_y_plot = initial_plot_radius_cartesian

    max_coord_radius_cartesian = 0.0  # Max cartesian distance of any cell from origin

    if cells_with_coords:
        all_cart_coords = [
            axial_to_cartesian(hc, sim_params.hex_size) for hc, _ in cells_with_coords
        ]
        all_x_cell_centers = [c[0] for c in all_cart_coords]
        all_y_cell_centers = [c[1] for c in all_cart_coords]

        # Bounding box of cell centers
        current_min_x_cells = min(all_x_cell_centers)
        current_max_x_cells = max(all_x_cell_centers)
        current_min_y_cells = min(all_y_cell_centers)
        current_max_y_cells = max(all_y_cell_centers)

        # Expand plot limits to include all cells with padding
        # Padding should be relative to hex_size for visual consistency
        padding = sim_params.hex_size * 3
        min_x_plot = min(min_x_plot, current_min_x_cells - padding)
        max_x_plot = max(max_x_plot, current_max_x_cells + padding)
        min_y_plot = min(min_y_plot, current_min_y_cells - padding)
        max_y_plot = max(max_y_plot, current_max_y_cells + padding)

        # Calculate max_coord_radius_cartesian for nutrient background extent
        for x, y in all_cart_coords:
            dist_sq = x**2 + y**2
            if dist_sq > max_coord_radius_cartesian**2:
                max_coord_radius_cartesian = np.sqrt(dist_sq)

    # The nutrient background should cover the determined plot view
    # Max radius needed for background is the largest extent of the plot axes from origin
    bg_extent_radius = np.sqrt(
        max(min_x_plot**2, max_x_plot**2) + max(min_y_plot**2, max_y_plot**2)
    )
    # Or simply, the largest radius visible among cells, plus padding
    effective_bg_radius = max(
        initial_plot_radius_cartesian,
        max_coord_radius_cartesian + sim_params.hex_size * 5,
    )

    ax.set_xlim(min_x_plot, max_x_plot)
    ax.set_ylim(min_y_plot, max_y_plot)

    # --- Plot Nutrient Background ---
    plot_nutrient_background(ax, environment_rules, sim_params, effective_bg_radius)

    # --- Plot Cells ---
    patches = []
    hex_radius_draw = sim_params.hex_size * 0.90  # Create small gaps

    for coord, cell in cells_with_coords:
        cartesian_coord = axial_to_cartesian(coord, sim_params.hex_size)
        face_color = DEFAULT_CELL_COLOR
        edge_color_cell = CELL_EDGE_COLOR

        if color_by == "phenotype":
            face_color = PHENOTYPE_COLORS.get(cell.phenotype, DEFAULT_CELL_COLOR)
        elif color_by == "nutrient_at_cell":
            # This might be less informative now that background shows nutrients
            nutrient = environment_rules.get_nutrient_at_coord(coord)
            face_color = NUTRIENT_ZONE_COLORS_BACKGROUND.get(
                nutrient, DEFAULT_CELL_COLOR
            )
        # Add more coloring options...

        hexagon = RegularPolygon(
            cartesian_coord,
            numVertices=6,
            radius=hex_radius_draw,
            orientation=0,  # Pointy topped
            facecolor=face_color,
            edgecolor=HEX_EDGE_COLOR if show_grid_lines else edge_color_cell,
            alpha=HEX_ALPHA,
            linewidth=0.3 if show_grid_lines else 0.15,
            zorder=10,  # Ensure cells are on top of nutrient background
        )
        patches.append(hexagon)

    if patches:
        collection = PatchCollection(patches, match_original=True)
        ax.add_collection(collection)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Colony Time: {current_sim_time:.2f} (Cells: {grid.count_cells()})")
    ax.set_xlabel(f"X coordinate (units relative to hex_size={sim_params.hex_size})")
    ax.set_ylabel(f"Y coordinate (units relative to hex_size={sim_params.hex_size})")

    # Create a custom legend for phenotypes
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=CELL_EDGE_COLOR)
        for phenotype, color in PHENOTYPE_COLORS.items()
    ]
    legend_labels = [phenotype.name for phenotype in PHENOTYPE_COLORS.keys()]
    if legend_patches:
        ax.legend(
            legend_patches,
            legend_labels,
            title="Phenotypes",
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize="small",
            title_fontsize="medium",
        )

    fig.tight_layout(rect=[0, 0, 0.83, 1])  # Adjust rect for legend if it's outside
    return fig, ax


def plot_population_dynamics(
    time_points: List[float],
    total_population: List[int],
    phenotype_a_population: Optional[List[int]] = None,
    phenotype_b_population: Optional[List[int]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Population Dynamics",
):
    """Plots population counts over time."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    ax.plot(
        time_points,
        total_population,
        label="Total Population",
        color="black",
        linewidth=2,
    )
    if phenotype_a_population is not None:  # Check for None explicitly
        ax.plot(
            time_points,
            phenotype_a_population,
            label=f"{Phenotype.A_UNPREPARED.name}",
            color=PHENOTYPE_COLORS[Phenotype.A_UNPREPARED],
            linestyle="--",
        )
    if phenotype_b_population is not None:  # Check for None explicitly
        ax.plot(
            time_points,
            phenotype_b_population,
            label=f"{Phenotype.B_PREPARED.name}",
            color=PHENOTYPE_COLORS[Phenotype.B_PREPARED],
            linestyle=":",
        )

    ax.set_xlabel("Time (simulation units)")
    ax.set_ylabel("Number of Cells")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.7)
    fig.tight_layout()
    return fig, ax
