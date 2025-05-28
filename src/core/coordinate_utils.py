# microbial_colony_sim/src/grid/coordinate_utils.py
import math
from typing import Tuple, List, Set
from src.core.data_structures import HexCoord
from src.core.exceptions import GridError

# Axial directions: (dq, dr)
# Corresponding to: E, NE, NW, W, SW, SE (for pointy top, q=col, r=row in skewed grid)
# Or more abstractly: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
# Let's use the common q (column, pointy top "x") and r (row, pointy top "y" component)
# Pointy top:
# q changes by +/-1 for E/W movements primarily
# r changes by +/-1 for NW/SE movements primarily
# Axial directions as HexCoord objects for easy addition/subtraction
AXIAL_DIRECTIONS: List[HexCoord] = [
    HexCoord(1, 0),
    HexCoord(1, -1),
    HexCoord(0, -1),
    HexCoord(-1, 0),
    HexCoord(-1, 1),
    HexCoord(0, 1),
]
# Order: E, SE, SW, W, NW, NE (clockwise starting East for pointy top)
# RedBlobGames often uses: (1,0) (0,1) (-1,1) (-1,0) (0,-1) (1,-1) -> E, NE, NW, W, SW, SE (different order)
# The specific order of AXIAL_DIRECTIONS matters for algorithms like `get_ring` if they iterate through directions.
# For now, this list is for `get_neighbors`.

# Cube directions (corresponding to AXIAL_DIRECTIONS if we map axial(q,r) to cube(q, -q-r, r))
# Or, more standard cube directions from RedBlobGames:
CUBE_DIRECTIONS: List[Tuple[int, int, int]] = [
    (1, -1, 0),
    (1, 0, -1),
    (0, 1, -1),
    (-1, 1, 0),
    (-1, 0, 1),
    (0, -1, 1),
]


def axial_to_cube(coord: HexCoord) -> Tuple[int, int, int]:
    """Converts axial coordinates (q, r) to cube coordinates (x, y, z).
    x = q
    y = -q - r
    z = r
    This is a common mapping. RedBlobGames uses x=q, z=r, y=-x-z. It's equivalent.
    """
    x = coord.q
    z = coord.r
    y = -x - z
    if x + y + z != 0:  # Sanity check, should always hold if conversion is correct
        raise GridError("Cube coordinates derived from axial must sum to 0.")
    return x, y, z


def cube_to_axial(x: int, y: int, z: int) -> HexCoord:
    """Converts cube coordinates (x, y, z) to axial coordinates (q, r).
    Assumes x=q, z=r standard mapping.
    """
    if x + y + z != 0:
        raise GridError(f"Cube coordinates ({x},{y},{z}) must sum to 0 to be valid.")
    return HexCoord(x, z)  # q = x, r = z


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
    return (cube[0] + vec[0], cube[1] + vec[1], cube[2] + vec[2])


def get_ring(center: HexCoord, radius: int) -> List[HexCoord]:
    """
    Gets all hexes in a ring of a given 'radius' around a 'center' hex.
    A radius of 0 is the center hex itself.
    A radius of 1 is the 6 direct neighbors.
    Algorithm from Red Blob Games (Hexagonal Grids - Range - Ring).
    """
    if radius < 0:
        raise GridError("Radius cannot be negative for get_ring.")
    if radius == 0:
        return [center]

    results: List[HexCoord] = []
    center_cube = axial_to_cube(center)

    # Start at one "corner" of the hexagon ring: radius steps in one direction.
    # CUBE_DIRECTIONS[4] is (-1, 0, 1) which corresponds to axial (-1, 1) (NW direction for instance)
    current_cube = _cube_add(
        center_cube,
        (
            CUBE_DIRECTIONS[4][0] * radius,
            CUBE_DIRECTIONS[4][1] * radius,
            CUBE_DIRECTIONS[4][2] * radius,
        ),
    )

    for i in range(6):  # For each of the 6 sides of the hexagon ring
        for _ in range(radius):  # For each step along that side
            results.append(
                cube_to_axial(current_cube[0], current_cube[1], current_cube[2])
            )
            current_cube = _cube_add(current_cube, CUBE_DIRECTIONS[i])

    return results


def get_filled_disk(center: HexCoord, radius: int) -> List[HexCoord]:
    """
    Gets all hexes within a given 'radius' (inclusive) of a 'center' hex.
    Algorithm from Red Blob Games (Hexagonal Grids - Range - Filled Hexagon/Disk).
    """
    if radius < 0:
        raise GridError("Radius cannot be negative for get_filled_disk.")

    results: List[HexCoord] = []
    center_x, center_y, center_z = axial_to_cube(center)

    for dx in range(-radius, radius + 1):
        for dy in range(max(-radius, -dx - radius), min(radius, -dx + radius) + 1):
            dz = -dx - dy
            # Check if the derived cube coordinate is within the main cube defined by radius
            # This check is inherently handled by the loop bounds for dy.
            # The condition |dx|+|dy|+|dz| <= 2*radius is for a different kind of range.
            # For a simple distance-based disk:
            # current_coord = cube_to_axial(center_x + dx, center_y + dy, center_z + dz)
            # if hex_distance(center, current_coord) <= radius:
            #    results.append(current_coord)
            # The loop above is more direct for cube coordinates.

            results.append(cube_to_axial(center_x + dx, center_y + dy, center_z + dz))

    return results


def euclidean_distance(
    c1: HexCoord, c2: HexCoord, hex_pixel_size: float = 1.0
) -> float:
    """
    Calculates the Euclidean distance between the geometric centers of two hex cells.
    Assumes 'pointy top' orientation for hexagons.
    'hex_pixel_size' is the distance from the center to a vertex (outer radius).
    """
    # Convert hex coordinates to pixel/Cartesian coordinates first
    # Pointy top orientation:
    # x_cart = size * (sqrt(3) * q + sqrt(3)/2 * r)
    # y_cart = size * (          3./2 * r)

    x1 = hex_pixel_size * (math.sqrt(3) * c1.q + math.sqrt(3) / 2.0 * c1.r)
    y1 = hex_pixel_size * (3.0 / 2.0 * c1.r)

    x2 = hex_pixel_size * (math.sqrt(3) * c2.q + math.sqrt(3) / 2.0 * c2.r)
    y2 = hex_pixel_size * (3.0 / 2.0 * c2.r)

    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def euclidean_distance_to_origin(coord: HexCoord, hex_pixel_size: float = 1.0) -> float:
    """
    Calculates the Euclidean distance from the geometric center of a hex cell
    at (q,r) to the grid origin (0,0).
    Assumes 'pointy top' orientation for hexagons.
    'hex_pixel_size' is the distance from the center to a vertex.
    """
    origin = HexCoord(0, 0)
    return euclidean_distance(coord, origin, hex_pixel_size)


# Helper for line drawing, could be useful for frontier analysis or other things
def hex_lerp(
    a: HexCoord, b: HexCoord, t: float
) -> Tuple[float, float, float]:  # Returns cube floats
    """Linearly interpolate between two hexes (in cube coordinates)."""
    cube_a = axial_to_cube(a)
    cube_b = axial_to_cube(b)
    return (
        cube_a[0] + (cube_b[0] - cube_a[0]) * t,
        cube_a[1] + (cube_b[1] - cube_a[1]) * t,
        cube_a[2] + (cube_b[2] - cube_a[2]) * t,
    )


def cube_round(
    cube_float: Tuple[float, float, float],
) -> Tuple[int, int, int]:  # Returns cube ints
    """Rounds fractional cube coordinates to the nearest hex cube coordinate."""
    rx = round(cube_float[0])
    ry = round(cube_float[1])
    rz = round(cube_float[2])

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
    """Returns a list of hexes forming a line between c1 and c2."""
    n = hex_distance(c1, c2)
    if n == 0:
        return [c1]

    results: List[HexCoord] = []
    for i in range(n + 1):
        t = 1.0 / n * i if n > 0 else 0.0
        cube_float = hex_lerp(c1, c2, t)
        axial_coord = cube_to_axial(*cube_round(cube_float))
        # Due to rounding, sometimes the same hex can be added.
        # A set can be used if uniqueness is critical, or check last element.
        if not results or results[-1] != axial_coord:
            results.append(axial_coord)
    return results
