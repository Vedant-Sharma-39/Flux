# microbial_colony_sim/src/core/exceptions.py


class SimulationError(Exception):
    """Base class for exceptions in this simulation."""

    pass


class ConfigurationError(SimulationError):
    """Exception raised for errors in configuration."""

    pass


class GridError(SimulationError):
    """Exception raised for errors related to the grid (e.g., out of bounds, occupied)."""

    pass


class CellError(SimulationError):
    """Exception raised for errors in cell state or operations."""

    pass


class DynamicsError(SimulationError):
    """Exception raised for errors during simulation dynamics (e.g., invalid transition)."""

    pass


# Add more specific exceptions as needed
