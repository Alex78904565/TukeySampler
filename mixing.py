"""Sufficient endpoint mixing budget for the ideal uncapped In-and-Out chain."""

import math


def mixing_steps(h, diameter, log_M, tv_distance):
    """Return ceil((log(M-1) - 2 log(2 epsilon))/log(1+h*pi^2/D^2)).

    D must bound the diameter, and log_M must bound the log warmness of
    the initial distribution. The guarantee is for the final endpoint.
    Zero steps suffice when the initial chi-squared bound already meets
    the target. Geometry and arithmetic are evaluated in floating point.
    """
    if not all(math.isfinite(x) for x in (h, diameter, log_M, tv_distance)):
        raise ValueError("Mixing parameters must be finite.")
    if h <= 0 or diameter <= 0 or log_M < 0 or not 0 < tv_distance < 1:
        raise ValueError("Require h>0, diameter>0, log_M>=0, and 0<tv_distance<1.")
    if log_M == 0:
        return 0
    # Stable even for M close to one, or too large to exponentiate.
    log_chi2 = log_M + math.log(-math.expm1(-log_M))
    numerator = log_chi2 - 2*(math.log(2) + math.log(tv_distance))
    if numerator <= 0:
        return 0
    try:
        ratio = h * (math.pi / diameter)**2
    except OverflowError as error:
        raise ValueError("Mixing parameters exceed floating-point range.") from error
    denominator = math.log1p(ratio)
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("Mixing parameters exceed floating-point range.")
    budget = numerator / denominator
    if not math.isfinite(budget):
        raise ValueError("Step budget exceeds floating-point range.")
    return math.ceil(budget)
