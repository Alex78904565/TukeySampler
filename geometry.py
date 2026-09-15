"""Inscribed-ball warm start, following the ideas in fast-tukey's helpers."""

import math

import numpy as np
from scipy.optimize import linprog


def enclosing_balls(A, b):
    """Return c, r, R with B(c,r) inside K and K inside B(c,R).

    A Chebyshev-center LP finds the inner ball. Coordinate minimum/maximum
    LPs give a bounding box and hence a conservative outer radius.
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
