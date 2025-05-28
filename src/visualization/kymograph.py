# microbial_colony_sim/src/visualization/kymograph.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from src.core.data_structures import SimulationConfig  # For config access if needed


class KymographGenerator:
    def __init__(
        self,
        config: SimulationConfig,
        output_dir_base: str = "visualizations/kymographs",
    ):
        self.config = config
        # Output path is per experiment, similar to ColonyVisualizer
        self.output_path = Path(output_dir_base) / config.experiment_name
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(
            f"sim.{config.experiment_name}.KymographGenerator"
        )  # Use logger

    def load_kymograph_data(
        self, npz_filepath: Path
    ) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
        """Loads raw kymograph data from an .npz file."""
        if not npz_filepath.exists():
            self.logger.error(f"Kymograph data file not found: {npz_filepath}")
            return None
        try:
            loaded_data_raw = np.load(
                npz_filepath, allow_pickle=True
            )  # allow_pickle if complex objects were stored

            # Reconstruct the kymograph_perimeter_data structure
            # Expected keys in npz: f"{attr_name}_times", f"{attr_name}_values"
            reconstructed_data = {}
            attribute_keys = set()
            for key in loaded_data_raw.keys():
                if key.endswith("_times"):
                    attribute_keys.add(key.replace("_times", ""))
                elif key.endswith("_values"):
                    attribute_keys.add(key.replace("_values", ""))

            for attr_base_name in attribute_keys:
                times_key = f"{attr_base_name}_times"
                values_key = f"{attr_base_name}_values"
                if times_key in loaded_data_raw and values_key in loaded_data_raw:
                    reconstructed_data[attr_base_name] = {
                        "times": loaded_data_raw[times_key],
                        "values_matrix": loaded_data_raw[
                            values_key
                        ],  # This is Time x AngleBins
                    }
                else:
                    self.logger.warning(
                        f"Missing times or values for attribute base '{attr_base_name}' in {npz_filepath}"
                    )

            self.logger.info(
                f"Loaded kymograph data for attributes: {list(reconstructed_data.keys())} from {npz_filepath}"
            )
            return reconstructed_data
        except Exception as e:
            self.logger.error(
                f"Error loading kymograph data from {npz_filepath}: {e}", exc_info=True
            )
            return None

    def plot_perimeter_kymograph(
        self,
        attribute_name: str,
        kymo_data: Dict[
            str, np.ndarray
        ],  # Expects {"times": np.array, "values_matrix": np.array}
        cmap_name: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        aspect_ratio: str = "auto",
        title_prefix: str = "Perimeter Kymograph",
    ):
        """
        Plots a single perimeter kymograph (Time vs. Angle).
        kymo_data contains 'times' and 'values_matrix' (Time x AngleBins).
        """
        times = kymo_data.get("times")
        values_matrix = kymo_data.get("values_matrix")

        if (
            times is None
            or values_matrix is None
            or times.size == 0
            or values_matrix.size == 0
        ):
            self.logger.warning(
                f"Cannot plot kymograph for '{attribute_name}': Missing or empty data."
            )
            return

        if values_matrix.ndim != 2:
            self.logger.error(
                f"Kymograph data for '{attribute_name}' is not 2D. Shape: {values_matrix.shape}"
            )
            return
        if len(times) != values_matrix.shape[0]:
            self.logger.error(
                f"Time dimension mismatch for '{attribute_name}'. Times: {len(times)}, Values rows: {values_matrix.shape[0]}"
            )
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Transpose values_matrix so that time is on X and angle on Y
        # imshow expects (row, col), so if values_matrix is (Time, Angle), imshow(values_matrix.T)
        # makes Angle the rows (Y-axis) and Time the columns (X-axis).
        # extent=[time_min, time_max, angle_min, angle_max]
        num_angular_bins = values_matrix.shape[1]

        im = ax.imshow(
            values_matrix.T,
            aspect=aspect_ratio,
            origin="lower",
            cmap=cmap_name,
            extent=[times.min(), times.max(), 0, 360],  # Assuming angles 0-360 degrees
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_xlabel("Time (simulation units)")
        ax.set_ylabel(f"Angular Bin (0-{num_angular_bins-1}) / Angle (degrees)")
        ax.set_title(f"{title_prefix}: {attribute_name.replace('_', ' ').title()}")

        try:
            cbar = fig.colorbar(
                im, ax=ax, label=attribute_name.replace("_", " ").title()
            )
        except Exception as e_cbar:
            self.logger.warning(
                f"Could not create colorbar for {attribute_name}: {e_cbar}"
            )

        plt.tight_layout()
        output_filename = (
            self.output_path
            / f"kymo_perimeter_{self.config.experiment_name}_{attribute_name}.png"
        )
        fig.savefig(output_filename, dpi=150)
        plt.close(fig)
        self.logger.info(f"Saved perimeter kymograph to {output_filename}")

    def plot_all_loaded_perimeter_kymographs(
        self, loaded_data: Dict[str, Dict[str, np.ndarray]]
    ):
        """Plots all kymographs from the loaded data dictionary."""
        if not loaded_data:
            self.logger.info("No kymograph data loaded to plot.")
            return

        for attribute_name, data_dict in loaded_data.items():
            # Determine appropriate colormap and normalization if desired
            # For simplicity, using viridis for all now.
            # Could have a mapping:
            # if "lag" in attribute_name: cmap = "YlOrRd" else: cmap = "viridis"
            self.plot_perimeter_kymograph(
                attribute_name, data_dict, cmap_name="viridis"
            )

    # Placeholder for plot_radial_kymograph (Time vs. Radius, averaged over angle)
    # def plot_radial_kymograph(self, attribute_name: str, ...):
    #     pass
