# microbial_colony_sim/src/dynamics/trade_off_functions.py
from src.core.data_structures import TradeOffParams
from src.core.exceptions import DynamicsError


def calculate_inherent_T_lag_GL(
    inherent_growth_rate_G: float, params: TradeOffParams
) -> float:
    """
    Calculates the inherent lag time (inherent_T_lag_GL) a cell would experience
    if it, as a G_specialist, encounters Galactose, based on its
    inherent_growth_rate_G.

    The relationship must be monotonically increasing: higher growth_rate_G leads to longer lag.

    Args:
        inherent_growth_rate_G: The cell's specific, inherited maximum potential
                                 growth rate on Glucose.
        params: An object containing parameters for the trade-off function
                       (e.g., T_lag_min, slope).

    Returns:
        The corresponding inherent_T_lag_GL.
    """
    if inherent_growth_rate_G < 0:
        # Although cell constructor might catch this, good to have specific check here too.
        raise DynamicsError(
            "inherent_growth_rate_G cannot be negative for trade-off calculation."
        )

    # Example linear trade-off: T_lag = T_lag_min + slope * inherent_g_rate_G
    # Ensure T_lag_min and slope are positive or zero for monotonicity if slope is positive.
    # Or ensure T_lag_min is high enough if slope is negative (though conceptual model implies positive slope)

    lag_time = params.T_lag_min + params.slope * inherent_growth_rate_G

    # Lag time should not be negative.
    return max(0.0, lag_time)
