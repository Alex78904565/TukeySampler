"""Inscribed-ball warm start, following the ideas in fast-tukey's helpers."""

import math
from functools import lru_cache

import numpy as np
from scipy.optimize import linprog


def enclosing_balls(A, b):
    """Return c, r, R with B(c,r) inside K and K inside B(c,R).

    A Chebyshev-center LP finds the inner ball. Coordinate minimum/maximum
    LPs give a bounding box and hence a conservative outer radius.
    Reuses solved geometry for the 16 most recent constraint sets in this
    process. Keys use constraint values, so editing A or b triggers a new solve.
    Returns a copy of the center to protect the cache from caller edits.
    These are floating-point estimates, not certified geometric bounds.
    """
    A, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
    if A.ndim != 2 or min(A.shape) == 0 or b.shape != (A.shape[0],):
        raise ValueError("Expected A with shape (m, d) and b with shape (m,).")
    if not np.isfinite(A).all() or not np.isfinite(b).all():
        raise ValueError("A and b must be finite.")
    norms = np.linalg.norm(A, axis=1)
    if np.any(norms == 0):
        raise ValueError("Remove zero-normal constraints before calling enclosing_balls.")
    # Normalize rows so the LP's radius is measured in Euclidean distance.
    A, b = A / norms[:, None], b / norms
    c, r, R = _solve_enclosing_balls(A.shape, A.tobytes(), b.tobytes())
    return c.copy(), r, R


@lru_cache(maxsize=16)
def _solve_enclosing_balls(shape, A_bytes, b_bytes):
    """Solve once per normalized constraint set; immutable bytes form the key."""
    A = np.frombuffer(A_bytes, dtype=float).reshape(shape)
    b = np.frombuffer(b_bytes, dtype=float)
    d = A.shape[1]
    result = linprog(
        np.r_[np.zeros(d), -1.0],
        A_ub=np.column_stack((A, np.ones(len(b)))), b_ub=b,
        bounds=[(None, None)] * d + [(0, None)], method="highs",
    )
    if not result.success:
        raise ValueError(f"Inner-ball LP failed: {result.message}")
    c = result.x[:d]
    # Recompute the feasible radius from slacks and shrink slightly for roundoff.
    r = float(np.min(b - A @ c)) * (1 - 1e-10)
    if r <= 0:
        raise ValueError("The polytope must have nonempty interior.")

    lo, hi = np.empty(d), np.empty(d)
    for j in range(d):
        direction = np.eye(d)[j]
        for sign, bounds in ((1, lo), (-1, hi)):
            result = linprog(sign * direction, A_ub=A, b_ub=b,
                             bounds=[(None, None)] * d, method="highs")
            if not result.success:
                raise ValueError(f"Bounding-box LP failed (is K bounded?): {result.message}")
            bounds[j] = result.x[j]
    R = float(np.linalg.norm(np.maximum(np.abs(lo - c), np.abs(hi - c))))
    return c, r, R


def uniform_ball(center, radius, rng):
    """One uniform point in a ball, using fast-tukey's dropped-coordinate idea."""
    center = np.asarray(center, dtype=float)
    if center.ndim != 1 or center.size == 0 or not np.isfinite(center).all():
        raise ValueError("center must be a finite, nonempty vector.")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive.")
    z = rng.normal(size=center.size + 2)
    return center + radius * z[:center.size] / np.linalg.norm(z)


def log_warmness_bound(d, r, R):
    """log M <= d log(R/r) for Unif(B(c,r)) relative to Unif(K).

    M = vol(K)/vol(B(c,r)); this bound follows from K subset B(c,R).
    It can be large: an inner-ball start need not be O(1)-warm.
    """
    if not 0 < r <= R or not math.isfinite(R):
        raise ValueError("Require finite radii with 0 < r <= R.")
    return d * math.log(R / r)


def example_geometry(body, d):
    """Analytic inner ball, enclosing radius R, and log warmth for examples.

    Warm start = Uniform(inner ball), so M = volume(K)/volume(ball).
    Geometry is evaluated in the original, unscaled coordinates.
    The enclosing ball uses the same center as the inner ball.
    """
    if d < 1:
        raise ValueError("d must be positive.")
    if body == "box":
        center, r, R, log_volume = [0.0]*d, 1.0, math.sqrt(d), d*math.log(2)
    elif body == "simplex":
        r = 1/(d+math.sqrt(d))
        center, log_volume = [r]*d, -math.lgamma(d+1)
        # The vertices are 0,e1,...,ed; a convex hull lies in any ball
        # containing every vertex. These are their distances from (r,...,r).
        R = max(math.sqrt(d)*r, math.sqrt(1-2*r+d*r*r))
    else:
        raise ValueError("Analytic geometry supports box and simplex.")
    log_ball = d/2*math.log(math.pi) - math.lgamma(d/2+1) + d*math.log(r)
    return center, r, R, max(0.0, log_volume-log_ball)
