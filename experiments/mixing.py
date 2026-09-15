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


def example_geometry(body, d):
    """Analytic inner ball, enclosing radius D, and log warmth for examples.

    Warm start = Uniform(inner ball), so M = volume(K)/volume(ball).
    Geometry is evaluated in the original, unscaled coordinates.
    The enclosing ball uses the same center as the inner ball.
    """
    if d < 1:
        raise ValueError("d must be positive.")
    if body == "box":
        center, r, D, log_volume = [0.0]*d, 1.0, math.sqrt(d), d*math.log(2)
    elif body == "simplex":
        r = 1/(d+math.sqrt(d))
        center, log_volume = [r]*d, -math.lgamma(d+1)
        # The vertices are 0,e1,...,ed; a convex hull lies in any ball
        # containing every vertex. These are their distances from (r,...,r).
        D = max(math.sqrt(d)*r, math.sqrt(1-2*r+d*r*r))
    else:
        raise ValueError("Analytic geometry supports box and simplex.")
    log_ball = d/2*math.log(math.pi) - math.lgamma(d/2+1) + d*math.log(r)
    return center, r, D, max(0.0, log_volume-log_ball)
