"""User-specified proper-step bound for the requested TV target of 1e-6.

All logarithms are natural. This implements the supplied expression; it does
not independently establish its hypotheses or account for retry-cap bias.
"""

import math


def required_steps(h, d, D, log_M):
    """ceil((12 ln(20) + ln(M)) / ln(1 + h*(ln 2)^2/(4*d*D^2))).

    D and h must use the same coordinates (h has squared-length units).
    Accept log_M rather than M to avoid overflow in high dimensions.
    """
    if not isinstance(d, int) or isinstance(d, bool) or d < 1:
        raise ValueError("d must be a positive integer.")
    if not all(math.isfinite(x) for x in (h, D, log_M)) or h <= 0 or D <= 0 or log_M < 0:
        raise ValueError("Require finite h>0, D>0, and log_M>=0.")
    denominator = math.log1p(h * math.log(2)**2 / (4*d*D**2))
    if denominator == 0:
        raise ValueError("h is too small for floating-point evaluation of the bound.")
    return math.ceil((12*math.log(20) + log_M) / denominator)


# Retained import for existing experiment callers.
from geometry import example_geometry
