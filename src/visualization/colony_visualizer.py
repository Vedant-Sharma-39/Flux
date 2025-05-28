# microbial_colony_sim/src/visualization/colony_visualizer.py
import matplotlib

matplotlib.use("Agg")  # USE NON-INTERACTIVE BACKEND AT THE VERY TOP
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Circle
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Callable  # Added Callable
from enum import Enum
from collections import Counter
import matplotlib.animation as animation
import traceback  # For detailed error printing

from src.core.data_structures import SimulationConfig, HexCoord
from src.core.enums import Phenotype, Nutrient
from src.agents.cell import Cell  # Assuming Cell class is defined
from src.agents.population_manager import PopulationManager
from src.grid.nutrient_environment import NutrientEnvironment


class CellColorMode(Enum):
    PHENOTYPE = "phenotype"
    REMAINING_LAG = "remaining_lag_time"
    INHERENT_LAG_GL = "inherent_T_lag_GL"
    INHERENT_GROWTH_G = "inherent_growth_rate_G"


PHENOTYPE_COLORS: Dict[Phenotype, str] = {
    Phenotype.G_SPECIALIST: "blue",
    Phenotype.L_SPECIALIST: "green",
    Phenotype.SWITCHING_GL: "orange",
}
DEFAULT_CELL_COLOR = "gray"
NUTRIENT_AREA_COLORS: Dict[Nutrient, Tuple[float, float, float, float]] = {
    Nutrient.GLUCOSE: mcolors.to_rgba("lightblue", alpha=0.25),
    Nutrient.GALACTOSE: mcolors.to_rgba("lightgreen", alpha=0.25),
}
DEFAULT_NUTRIENT_COLOR = mcolors.to_rgba("lightgray", alpha=0.25)


class ColonyVisualizer:
    def __init__(
        self,
        config: SimulationConfig,
        population_manager: PopulationManager,
        nutrient_env: NutrientEnvironment,
        hex_render_size: float = 1.0,
        output_dir_base: str = "visualizations",
    ):
        self.config = config
        self.population_manager = population_manager
        self.nutrient_env = nutrient_env
        self.hex_render_size = hex_render_size

        self.output_path = Path(output_dir_base) / config.experiment_name
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.cbars: Dict[str, plt.colorbar.Colorbar] = {}

        self.plot_radius_factor = 1.1
        num_bands_to_visualize = self.config.visualization.num_bands_to_visualize

        self.max_vis_radius_data_units = max(
            self.config.W_band * num_bands_to_visualize, 30.0
        )
        if (
            self.config.max_grid_radius > 0
            and self.config.max_grid_radius < self.max_vis_radius_data_units
        ):
            self.max_vis_radius_data_units = self.config.max_grid_radius
        self.plot_extent = (
            self.max_vis_radius_data_units
            * self.hex_render_size
            * self.plot_radius_factor
        )

        # Colormap normalization ranges
        self.norm_growth_g = mcolors.Normalize(
            vmin=min(self.config.g_rate_prototype_1, self.config.g_rate_prototype_2)
            * 0.8,
            vmax=max(self.config.g_rate_prototype_1, self.config.g_rate_prototype_2)
            * 1.2,
        )
        # Estimate max lag based on the highest G-growth prototype on the trade-off curve
        max_g_growth_for_lag_calc = max(
            self.config.g_rate_prototype_1, self.config.g_rate_prototype_2
        )
        self.max_expected_lag_from_prototypes = (
            self.config.trade_off_params.T_lag_min
            + self.config.trade_off_params.slope * max_g_growth_for_lag_calc
        )

        self.norm_inherent_lag = mcolors.Normalize(
            vmin=self.config.trade_off_params.T_lag_min * 0.8,
            vmax=max(1.0, self.max_expected_lag_from_prototypes * 1.1),
        )
        self.norm_remaining_lag = mcolors.Normalize(
            vmin=0, vmax=max(1.0, self.max_expected_lag_from_prototypes * 0.8)
        )  # Current lag usually less

        self.animation_frames_data: List[Tuple[float, List[Cell], CellColorMode]] = []
        self.current_animation_color_mode: CellColorMode = (
            CellColorMode.PHENOTYPE
        )  # Default for animation filename

    def _setup_plot_internal(self, title_info: str):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(11, 10))

        self.ax.clear()
        for cbar_key in list(self.cbars.keys()):
            if self.cbars.get(cbar_key) is not None:
                try:
                    self.fig.delaxes(self.cbars[cbar_key].ax)
                except (AttributeError, KeyError, ValueError, TypeError):
                    pass  # More robustly catch errors
            if cbar_key in self.cbars:
                del self.cbars[cbar_key]

        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlim(-self.plot_extent, self.plot_extent)
        self.ax.set_ylim(-self.plot_extent, self.plot_extent)
        self.ax.set_title(
            f"Colony: {self.config.experiment_name}\n{title_info}", fontsize=10
        )
        self.ax.set_xlabel("X coordinate")
        self.ax.set_ylabel("Y coordinate")

    def _draw_nutrient_bands_internal(self):
        if not self.ax:
            return
        max_radius_for_bands_data_units = self.plot_extent / self.hex_render_size
        transitions_data_units = self.nutrient_env.get_band_transitions(
            max_radius_for_bands_data_units
        )
        current_outer_radius_plot_units = self.plot_extent
        band_idx_outer = (
            int(max_radius_for_bands_data_units // self.config.W_band)
            if self.config.W_band > 0
            else 0
        )
        color_outer = NUTRIENT_AREA_COLORS.get(
            Nutrient.GLUCOSE if band_idx_outer % 2 == 0 else Nutrient.GALACTOSE,
            DEFAULT_NUTRIENT_COLOR,
        )
        self.ax.add_patch(
            Circle(
                (0, 0),
                current_outer_radius_plot_units,
                facecolor=color_outer,
                edgecolor="none",
                zorder=0,
            )
        )
        sorted_transitions_plot_units = sorted(
            [t * self.hex_render_size for t in transitions_data_units], reverse=True
        )
        for r_transition_plot_units in sorted_transitions_plot_units:
            if r_transition_plot_units <= 1e-6:
                continue
            radius_just_inside_data_units = (
                r_transition_plot_units / self.hex_render_size
            ) - 1e-3
            band_idx = (
                int(radius_just_inside_data_units // self.config.W_band)
                if self.config.W_band > 0
                else 0
            )
            color = NUTRIENT_AREA_COLORS.get(
                Nutrient.GLUCOSE if band_idx % 2 == 0 else Nutrient.GALACTOSE,
                DEFAULT_NUTRIENT_COLOR,
            )
            self.ax.add_patch(
                Circle(
                    (0, 0),
                    r_transition_plot_units,
                    facecolor=color,
                    edgecolor="none",
                    zorder=0,
                )
            )

    def _draw_cells_internal(self, cells: List[Cell], color_mode: CellColorMode):
        if not self.ax or not cells:
            return
        patches = []
        face_colors_data = []
        for cell_obj in cells:
            cart_x = self.hex_render_size * (
                np.sqrt(3) * cell_obj.coord.q + np.sqrt(3) / 2.0 * cell_obj.coord.r
            )
            cart_y = self.hex_render_size * (3.0 / 2.0 * cell_obj.coord.r)
            hexagon = RegularPolygon(
                (cart_x, cart_y),
                numVertices=6,
                radius=self.hex_render_size * 0.9,
                orientation=np.radians(0),
            )
            patches.append(hexagon)
            if color_mode == CellColorMode.PHENOTYPE:
                face_colors_data.append(
                    PHENOTYPE_COLORS.get(cell_obj.current_phenotype, DEFAULT_CELL_COLOR)
                )
            elif color_mode == CellColorMode.REMAINING_LAG:
                face_colors_data.append(cell_obj.remaining_lag_time)
            elif color_mode == CellColorMode.INHERENT_LAG_GL:
                face_colors_data.append(cell_obj.inherent_T_lag_GL)
            elif color_mode == CellColorMode.INHERENT_GROWTH_G:
                face_colors_data.append(cell_obj.inherent_growth_rate_G)
        if not patches:
            return
        collection = PatchCollection(
            patches, match_original=(color_mode == CellColorMode.PHENOTYPE)
        )
        collection.set_edgecolor("black")
        collection.set_linewidth(0.3)
        collection.set_alpha(0.85)
        if color_mode == CellColorMode.PHENOTYPE:
            collection.set_facecolor(face_colors_data)
        else:
            scalar_data = np.array(face_colors_data)
            if not scalar_data.size:  # Handle empty scalar data
                self.ax.add_collection(
                    collection, autolim=False
                )  # Add empty collection if no data to map
                return
            if color_mode == CellColorMode.REMAINING_LAG:
                cmap_name = "YlOrRd"
                norm = self.norm_remaining_lag
                cbar_label = "Remaining Lag"
            elif color_mode == CellColorMode.INHERENT_LAG_GL:
                cmap_name = "coolwarm"
                norm = self.norm_inherent_lag
                cbar_label = "Inherent T_lag_GL"
            elif color_mode == CellColorMode.INHERENT_GROWTH_G:
                cmap_name = "viridis"
                norm = self.norm_growth_g
                cbar_label = "Inherent G-Growth"
            else:
                cmap_name = "viridis"
                norm = mcolors.Normalize(
                    vmin=np.min(scalar_data) if scalar_data.size else 0,
                    vmax=np.max(scalar_data) if scalar_data.size else 1,
                )
                cbar_label = "Scalar Value"
            cmap = cm.get_cmap(cmap_name)
            collection.set_array(scalar_data)
            collection.set_cmap(cmap)
            collection.set_norm(norm)
            if self.fig and self.ax:
                try:
                    cbar = self.fig.colorbar(
                        collection,
                        ax=self.ax,
                        orientation="vertical",
                        fraction=0.046,
                        pad=0.04,
                    )
                    cbar.set_label(cbar_label)
                    self.cbars[cbar_label] = cbar
                except Exception as e_cbar:
                    print(
                        f"VISUALIZER: Error creating colorbar for {cbar_label}: {e_cbar}"
                    )
        self.ax.add_collection(collection, autolim=False)

    def _update_animation_frame(self, frame_index: int) -> Tuple:
        if frame_index >= len(self.animation_frames_data):
            return tuple()
        sim_time, cells_data, color_mode = self.animation_frames_data[frame_index]
        title = f"Time: {sim_time:.2f} (Frame: {frame_index + 1}/{len(self.animation_frames_data)}) | Mode: {color_mode.name}"
        self._setup_plot_internal(title)
        self._draw_nutrient_bands_internal()
        self._draw_cells_internal(cells_data, color_mode)
        counts = {}
        info_text = f"Cells: {len(cells_data)}\n"
        if cells_data:
            pheno_counts = Counter(c.current_phenotype for c in cells_data)
            for p_type in Phenotype:
                info_text += (
                    f"{p_type.name.split('_')[0]}: {pheno_counts.get(p_type, 0)} "
                )
        if self.ax:
            self.ax.text(
                0.01,
                0.98,
                info_text.strip(),
                transform=self.ax.transAxes,
                fontsize=7,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.6),
            )
        drawn_artists = []
        if self.ax:
            drawn_artists.extend(self.ax.collections)
            drawn_artists.extend(self.ax.patches)
            drawn_artists.extend(self.ax.texts)
        return tuple(drawn_artists)

    def record_animation_frame(
        self,
        current_sim_time: float,
        color_mode: CellColorMode = CellColorMode.PHENOTYPE,
    ):
        current_cells = list(self.population_manager.get_all_cells())
        self.animation_frames_data.append((current_sim_time, current_cells, color_mode))
        self.current_animation_color_mode = color_mode

    def save_animation(
        self,
        filename_suffix: str = "",
        fps: int = 10,
        writer_name: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        if not self.animation_frames_data:
            print("VISUALIZER: No frames recorded for animation.")
            return
        if self.fig is None:
            self._setup_plot_internal("Initial Animation Setup")

        mode_suffix = self.current_animation_color_mode.value.lower().replace(" ", "_")
        if filename_suffix:
            mode_suffix = f"{mode_suffix}_{filename_suffix}"
        output_filename_base = f"animation_{mode_suffix}"

        actual_writer_name = writer_name
        if actual_writer_name is None:
            available_writers_list = animation.writers.list()
            if "ffmpeg" in available_writers_list:
                actual_writer_name = "ffmpeg"
            elif "pillow" in available_writers_list:
                actual_writer_name = "pillow"
            elif "imagemagick" in available_writers_list:
                actual_writer_name = "imagemagick"
            else:
                actual_writer_name = "ffmpeg"

        default_ext = ".mp4"
        if actual_writer_name in ["pillow", "imagemagick"]:
            default_ext = ".gif"
        output_filename = self.output_path / f"{output_filename_base}{default_ext}"

        print(
            f"VISUALIZER: Preparing to save animation to {output_filename} ({len(self.animation_frames_data)} frames) using '{actual_writer_name}' at {fps} FPS."
        )
        try:
            interval_ms = max(1, int(1000 / fps))  # Must be > 0
            ani = animation.FuncAnimation(
                self.fig,
                self._update_animation_frame,
                frames=len(self.animation_frames_data),
                interval=interval_ms,
                blit=False,
                repeat=False,
            )
            if actual_writer_name and actual_writer_name in animation.writers.list():
                print(
                    f"VISUALIZER: Writer '{actual_writer_name}' confirmed available. Saving..."
                )
                ani.save(
                    str(output_filename),
                    writer=actual_writer_name,
                    fps=fps,
                    dpi=150,
                    progress_callback=progress_callback,
                )
                print(f"VISUALIZER: Animation saved to {output_filename}")
            else:
                print(
                    f"VISUALIZER: Writer '{actual_writer_name}' not available or not specified correctly. Animation not saved. Available: {animation.writers.list()}"
                )
        except Exception as e:
            print(f"VISUALIZER: Error saving animation: {e}\n{traceback.format_exc()}")
        finally:
            print("VISUALIZER: save_animation method finished.")

    def plot_colony_state_to_file(
        self,
        current_sim_time: float,
        step_count: int,
        color_mode: CellColorMode = CellColorMode.PHENOTYPE,
        extra_title_info: str = "",
    ):
        title = f"T={current_sim_time:.2f} S={step_count}"  # Shorter title for file name consistency
        if extra_title_info:
            title += f" {extra_title_info}"
        self._setup_plot_internal(title)
        self._draw_nutrient_bands_internal()
        all_cells = self.population_manager.get_all_cells()
        self._draw_cells_internal(all_cells, color_mode)
        if color_mode == CellColorMode.PHENOTYPE:
            legend_patches = [
                plt.Rectangle((0, 0), 1, 1, color=PHENOTYPE_COLORS[p])
                for p in PHENOTYPE_COLORS
            ]
            legend_labels = [p.name for p in PHENOTYPE_COLORS]
            if self.ax:
                self.ax.legend(
                    legend_patches,
                    legend_labels,
                    loc="upper right",
                    title="Phenotypes",
                    fontsize=8,
                )
        counts = {}
        info_text = f"Cells: {len(all_cells)}\n"
        if all_cells:
            pheno_counts = Counter(cell.current_phenotype for cell in all_cells)
            for p_type in Phenotype:
                info_text += (
                    f"{p_type.name.split('_')[0]}: {pheno_counts.get(p_type, 0)} "
                )
        info_text += f"\nColor: {color_mode.name}"
        if self.ax:
            self.ax.text(
                0.01,
                0.98,
                info_text.strip(),
                transform=self.ax.transAxes,
                fontsize=7,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.6),
            )
        mode_suffix = color_mode.value.lower().replace(" ", "_")
        time_str = f"{current_sim_time:07.2f}".replace(
            ".", "_"
        )  # Ensure time is filename-safe
        filename_base = f"snapshot_t{time_str}_step{step_count:05d}_{mode_suffix}.png"
        filename = self.output_path / filename_base
        try:
            if self.fig:
                self.fig.savefig(str(filename), dpi=150)
            # print(f"VISUALIZER: Saved snapshot to {filename}") # Use logger from engine
        except Exception as e:
            print(f"VISUALIZER: Error saving snapshot {filename}: {e}")

    def close_plot(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.cbars.clear()
