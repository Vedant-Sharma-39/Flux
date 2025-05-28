# microbial_colony_sim/src/grid/coordinate_utils.py
import math
from typing import Tuple, List, Set

# CORRECTED IMPORTS:
from src.core.data_structures import HexCoord
from src.core.exceptions import GridError


# Axial directions as HexCoord objects for easy addition/subtraction
# Order: E, SE, SW, W, NW, NE (clockwise starting East for pointy top hexes)
AXIAL_DIRECTIONS: List[HexCoord] = [
    HexCoord(1, 0),
    HexCoord(1, -1),
    HexCoord(0, -1),
    HexCoord(-1, 0),
    HexCoord(-1, 1),
    HexCoord(0, 1),
]

# Cube directions (dx, dy, dz) corresponding to the axial directions above
# if axial (q,r) maps to cube (q, -q-r, r)
# For reference, from Red Blob Games, standard cube directions:
CUBE_DIRECTIONS: List[Tuple[int, int, int]] = [
    (1, -1, 0),
    (1, 0, -1),
    (0, 1, -1),  # E, SE, S (if +y is south-ish)
    (-1, 1, 0),
    (-1, 0, 1),
    (0, -1, 1),  # W, NW, N (if -y is north-ish)
]
# Note: The order in CUBE_DIRECTIONS corresponds to a common iteration pattern,
# not necessarily a direct 1-to-1 mapping with AXIAL_DIRECTIONS's specific order above
# unless explicitly stated for an algorithm.


def axial_to_cube(coord: HexCoord) -> Tuple[int, int, int]:
    """
    Converts axial coordinates (q, r) to cube coordinates (x, y, z).
    Standard mapping: x = q, z = r, so y = -q - r.
    Ensures x + y + z = 0.
    """
    x = coord.q
    z = coord.r
    y = -x - z
    # Sanity check, should always hold if conversion is correct.
    # Not strictly necessary to check here if logic is sound, but good for dev.
    # if x + y + z != 0:
    #     raise GridError(f"Internal error: Cube coordinates ({x},{y},{z}) derived from axial {coord} do not sum to 0.")
    return x, y, z


def cube_to_axial(x: int, y: int, z: int) -> HexCoord:
    """
    Converts cube coordinates (x, y, z) to axial coordinates (q, r).
    Standard mapping: q = x, r = z.
    Requires x + y + z = 0.
    """
    if x + y + z != 0:
        raise GridError(
            f"Cube coordinates ({x},{y},{z}) must sum to 0 to be valid for axial conversion."
        )
    return HexCoord(x, z)


def hex_distance(c1: HexCoord, c2: HexCoord) -> int:
    """Calculates the grid distance (number of steps) between two hexes in axial coordinates."""
    cube1 = axial_to_cube(c1)
    cube2 = axial_to_cube(c2)
    return (
        abs(cube1[0] - cube2[0]) + abs(cube1[1] - cube2[1]) + abs(cube1[2] - cube2[2])
    ) // 2


def get_neighbors(coord: HexCoord) -> List[HexCoord]:
    """Returns a list of 6 neighbor coordinates for a given axial coordinate."""
    return [HexCoord(coord.q + d.q, coord.r + d.r) for d in AXIAL_DIRECTIONS]


def _cube_add(
    cube: Tuple[int, int, int], vec: Tuple[int, int, int]
) -> Tuple[int, int, int]:
    """Helper function to add two cube coordinate tuples."""
    return (cube[0] + vec[0], cube[1] + vec[1], cube[2] + vec[2])


def get_ring(center: HexCoord, radius: int) -> List[HexCoord]:
    """
    Gets all hexes in a ring of a given 'radius' around a 'center' hex.
    A radius of 0 is the center hex itself.
    A radius of 1 is the 6 direct neighbors.
    Algorithm inspired by Red Blob Games (Hexagonal Grids - Range - Ring).
    """
    if radius < 0:
        raise GridError("Radius cannot be negative for get_ring.")
    if radius == 0:
        return [center]

    results: List[HexCoord] = []
    center_cube = axial_to_cube(center)

    # Start at one "corner" of the hexagon ring: radius steps in one cube direction.
    # CUBE_DIRECTIONS[4] is (-1, 0, 1)
    current_cube = _cube_add(
        center_cube,
        (
            CUBE_DIRECTIONS[4][0] * radius,
            CUBE_DIRECTIONS[4][1] * radius,
            CUBE_DIRECTIONS[4][2] * radius,
        ),
    )

    for i in range(6):  # For each of the 6 sides of the hexagon ring
        # The order in CUBE_DIRECTIONS matters for ring traversal
        # CUBE_DIRECTIONS = [(1,-1,0), (1,0,-1), (0,1,-1), (-1,1,0), (-1,0,1), (0,-1,1)]
        # This order ensures a counter-clockwise traversal of the ring.
        for _ in range(radius):  # For each step along that side
            results.append(
                cube_to_axial(current_cube[0], current_cube[1], current_cube[2])
            )
            current_cube = _cube_add(current_cube, CUBE_DIRECTIONS[i])

    return results


def get_filled_disk(center: HexCoord, radius: int) -> List[HexCoord]:
    """
    Gets all hexes within a given 'radius' (inclusive) of a 'center' hex.
    Algorithm inspired by Red Blob Games (Hexagonal Grids - Range - Filled Hexagon/Disk).
    Iterates over a cubic area and filters by distance.
    """
    if radius < 0:
        raise GridError("Radius cannot be negative for get_filled_disk.")

    results: List[HexCoord] = []
    center_x, center_y, center_z = axial_to_cube(center)

    # Iterate over a bounding box in cube coordinates
    for dx in range(-radius, radius + 1):
        # Constrain dy based on dx to stay within the hexagonal shape
        for dy in range(max(-radius, -dx - radius), min(radius, -dx + radius) + 1):
            dz = -dx - dy  # The third cube coordinate is constrained
            # The hex formed by (dx, dy, dz) is a valid offset from center
            # if hex_distance(HexCoord(0,0), cube_to_axial(dx,dy,dz)) <= radius
            # The loops above already ensure this for cube coordinates.

            results.append(cube_to_axial(center_x + dx, center_y + dy, center_z + dz))

    return results


def euclidean_distance(
    c1: HexCoord, c2: HexCoord, hex_pixel_size: float = 1.0
) -> float:
    """
    Calculates the Euclidean distance between the geometric centers of two hex cells.
    Assumes 'pointy top' orientation for hexagons where 'q' aligns with x-ish and 'r' with y-ish component.
    'hex_pixel_size' is the distance from the center to a vertex (outer radius).
    """
    # Cartesian conversion for pointy top hexagons:
    # x_cart = size * (sqrt(3) * q + sqrt(3)/2 * r)
    # y_cart = size * (            3./2 * r)

    x1_cart = hex_pixel_size * (math.sqrt(3) * c1.q + math.sqrt(3) / 2.0 * c1.r)
    y1_cart = hex_pixel_size * (3.0 / 2.0 * c1.r)

    x2_cart = hex_pixel_size * (math.sqrt(3) * c2.q + math.sqrt(3) / 2.0 * c2.r)
    y2_cart = hex_pixel_size * (3.0 / 2.0 * c2.r)

    return math.sqrt((x1_cart - x2_cart) ** 2 + (y1_cart - y2_cart) ** 2)


def euclidean_distance_to_origin(coord: HexCoord, hex_pixel_size: float = 1.0) -> float:
    """
    Calculates the Euclidean distance from the geometric center of a hex cell
    at (q,r) to the grid origin (0,0).
    """
    origin = HexCoord(0, 0)
    return euclidean_distance(coord, origin, hex_pixel_size)


# --- Line Drawing (from Red Blob Games, useful for various grid algorithms) ---


def _hex_lerp_cube(
    a_cube: Tuple[float, float, float], b_cube: Tuple[float, float, float], t: float
) -> Tuple[float, float, float]:
    """Linearly interpolate between two hexes (in fractional cube coordinates)."""
    return (
        a_cube[0] + (b_cube[0] - a_cube[0]) * t,
        a_cube[1] + (b_cube[1] - a_cube[1]) * t,
        a_cube[2] + (b_cube[2] - a_cube[2]) * t,
    )


def _cube_round(cube_float: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """Rounds fractional cube coordinates to the nearest integer hex cube coordinate."""
    rx = round(cube_float[0])
    ry = round(cube_float[1])
    rz = round(cube_float[2])

    # Constraint: rx + ry + rz must be 0. Adjust the one with the largest diff.
    x_diff = abs(rx - cube_float[0])
    y_diff = abs(ry - cube_float[1])
    z_diff = abs(rz - cube_float[2])

    if x_diff > y_diff and x_diff > z_diff:
        rx = -ry - rz
    elif y_diff > z_diff:
        ry = -rx - rz
    else:
        rz = -rx - ry

    return int(rx), int(ry), int(rz)


def hex_line_draw(c1: HexCoord, c2: HexCoord) -> List[HexCoord]:
    """
    Returns a list of hexes forming a line between c1 and c2 (inclusive).
    Algorithm from Red Blob Games.
    """
    n = hex_distance(c1, c2)
    if n == 0:
        return [c1]

    results: List[HexCoord] = []
    c1_cube_float = tuple(float(x_i) for x_i in axial_to_cube(c1))
    c2_cube_float = tuple(float(x_i) for x_i in axial_to_cube(c2))

    for i in range(n + 1):
        t = 1.0 / n * i if n > 0 else 0.0  # Ensure t goes from 0 to 1
        # Add a tiny epsilon to endpoints for lerp to avoid floating point issues at edges
        # current_cube_float = _hex_lerp_cube(c1_cube_float, c2_cube_float, t * (1 + 1e-6) - 0.5e-6)
        # This epsilon adjustment is often for sampling pixel centers, maybe not needed for grid hexes.
        # Standard lerp:
        current_cube_float = _hex_lerp_cube(c1_cube_float, c2_cube_float, t)

        axial_coord = cube_to_axial(*_cube_round(current_cube_float))

        # Ensure unique hexes in the line, though rounding should typically handle this.
        if not results or results[-1] != axial_coord:
            results.append(axial_coord)
    return results
