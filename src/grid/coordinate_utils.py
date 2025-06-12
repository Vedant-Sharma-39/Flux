# src/grid/coordinate_utils.py

import math
from typing import (
    List,
    Tuple,
    Set as TypingSet,
)  # Alias Set to avoid potential name clash if Set is used as var
from ..core.shared_types import HexCoord  # Import HexCoord from shared_types
import numpy as np

# Axial directions for neighbors
AXIAL_DIRECTIONS: List[HexCoord] = [
    HexCoord(1, 0),
    HexCoord(1, -1),
    HexCoord(0, -1),
    HexCoord(-1, 0),
    HexCoord(-1, 1),
    HexCoord(0, 1),
]

# Cube directions (for conversions and distances)
CUBE_DIRECTIONS: List[Tuple[int, int, int]] = [
    (1, -1, 0),
    (1, 0, -1),
    (0, 1, -1),
    (-1, 1, 0),
    (-1, 0, 1),
    (0, -1, 1),
]


def axial_to_cube(coord: HexCoord) -> Tuple[int, int, int]:
    """Converts axial coordinates (q, r) to cube coordinates (x, y, z)."""
    x = coord.q
    z = coord.r
    y = -x - z
    return x, y, z


def cube_to_axial(cube_coord: Tuple[int, int, int]) -> HexCoord:
    """Converts cube coordinates (x, y, z) to axial coordinates (q, r)."""
    q = cube_coord[0]
    r = cube_coord[2]
    return HexCoord(q, r)


def cube_distance(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> int:
    """Calculates the distance between two hexes in cube coordinates."""
    return (abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])) // 2


def axial_distance(a: HexCoord, b: HexCoord) -> int:
    """Calculates the distance between two hexes in axial coordinates."""
    ac = axial_to_cube(a)
    bc = axial_to_cube(b)
    return cube_distance(ac, bc)


def get_neighbors(coord: HexCoord) -> List[HexCoord]:
    """Returns a list of all 6 neighbors for a given axial coordinate."""
    return [coord + d for d in AXIAL_DIRECTIONS]


def get_ring(center: HexCoord, radius: int) -> List[HexCoord]:
    """Gets all hexes in a ring of a given radius around a center hex."""
    if radius < 0:
        return []
    if radius == 0:
        return [center]
    results: List[HexCoord] = []
    cube_center = axial_to_cube(center)
    # Start with the hex `radius` steps in one direction
    current_cube_coord_list = list(
        axial_to_cube(center + AXIAL_DIRECTIONS[4] * radius)
    )  # Example starting direction

    for i in range(6):
        for _ in range(radius):
            results.append(cube_to_axial(tuple(current_cube_coord_list)))
            for j in range(3):  # Add CUBE_DIRECTIONS component-wise
                current_cube_coord_list[j] += CUBE_DIRECTIONS[i][j]
    return results


def get_filled_hexagon(center: HexCoord, radius: int) -> List[HexCoord]:
    """Gets all hexes within a hexagonal area of a given radius from the center."""
    if radius < 0:
        return []
    results: List[HexCoord] = [center]
    for r_idx in range(1, radius + 1):
        results.extend(get_ring(center, r_idx))
    return results


def axial_to_cartesian_sq_distance_from_origin(
    coord: HexCoord, hex_size: float
) -> float:
    """
    Calculates the squared Euclidean distance from the origin HexCoord(0,0)
    to the center of a hex defined by axial coordinates.
    Assumes pointy-topped hexes. hex_size is distance from center to corner.
    """
    x_cart = hex_size * (math.sqrt(3) * coord.q + math.sqrt(3) / 2.0 * coord.r)
    y_cart = hex_size * (3.0 / 2.0 * coord.r)
    return x_cart * x_cart + y_cart * y_cart


def axial_to_cartesian_sq_distance_from_origin_batch(
    coords_q: np.ndarray, coords_r: np.ndarray, hex_size: float
) -> np.ndarray:
    """Vectorized version: coords_q and coords_r are 1D arrays of q and r values."""
    x_cart = hex_size * (math.sqrt(3) * coords_q + math.sqrt(3) / 2.0 * coords_r)
    y_cart = hex_size * (3.0 / 2.0 * coords_r)
    return x_cart**2 + y_cart**2


def axial_to_cartesian(coord: HexCoord, hex_size: float) -> Tuple[float, float]:
    """
    Converts axial hex coordinates to Cartesian (pixel) coordinates.
    Assumes pointy-topped hexes.
    hex_size is the distance from the center to any corner.
    """
    x = hex_size * (math.sqrt(3) * coord.q + math.sqrt(3) / 2.0 * coord.r)
    y = hex_size * (3.0 / 2.0 * coord.r)
    return x, y
