# microbial_colony_sim/src/core/data_structures.py
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from .enums import Nutrient, Phenotype


@dataclass(frozen=True)
class HexCoord:
    q: int
    r: int

    def __add__(self, other: "HexCoord") -> "HexCoord":
        if not isinstance(other, HexCoord):
            return NotImplemented
        return HexCoord(self.q + other.q, self.r + other.r)

    def __sub__(self, other: "HexCoord") -> "HexCoord":
        if not isinstance(other, HexCoord):
            return NotImplemented
        return HexCoord(self.q - other.q, self.r - other.r)

    def __hash__(self):
        return hash((self.q, self.r))


@dataclass(frozen=True)
class TradeOffParams:
    T_lag_min: float
    slope: float


@dataclass
class VisualizationParams:
    visualization_enabled: bool = False
    animation_save_path: str = (
        "animations/"  # Base path, specific experiment path appended by visualizer
    )
    animation_frame_interval: int = 20  # Record frame every N simulation steps
    hex_pixel_size: float = 10.0  # For rendering size of hexes in plots
    animation_color_mode: str = (
        "PHENOTYPE"  # Default: PHENOTYPE, REMAINING_LAG_TIME, INHERENT_LAG_GL, INHERENT_GROWTH_G
    )
    animation_writer: str = (
        "ffmpeg"  # Preferred: ffmpeg (mp4), pillow (gif), imagemagick (gif)
    )
    save_key_snapshots: bool = (
        False  # True to also save individual PNGs at animation intervals
    )
    num_bands_to_visualize: int = 6  # For plot extent calculation
    kymo_angular_bins: int = 60  # Number of angular bins for kymograph
    kymo_radial_shell_width_factor: float = (
        0.2  # Factor of W_band for perimeter kymograph shell
    )


@dataclass
class SimulationConfig:
    experiment_name: str = "default_experiment"
    W_band: float = 10.0
    grid_type: str = "hexagonal"  # Currently only hexagonal is implemented
    max_grid_radius: float = (
        100.0  # Max extent of grid if bounded (not strictly enforced currently)
    )
    g_rate_prototype_1: float = 0.1
    g_rate_prototype_2: float = 0.5
    prob_daughter_inherits_prototype_1: float = 0.5
    lambda_L_fixed_rate: float = 0.1
    trade_off_params: TradeOffParams = field(
        default_factory=lambda: TradeOffParams(T_lag_min=1.0, slope=20.0)
    )
    dt: float = 0.1
    max_simulation_time: float = 200.0
    initial_cell_count: int = 5
    initial_colony_radius: float = 1.5  # Used by initializer to find initial coords
    metrics_interval_time: float = 1.0  # How often (sim time) to record metrics
    data_output_path: str = "results/"
    visualization: VisualizationParams = field(default_factory=VisualizationParams)
    max_cells_safety_threshold: int = 75000
    log_level: str = "INFO"
