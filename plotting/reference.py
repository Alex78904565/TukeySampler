"""Exact projected supports and bin probabilities for boxes and simplices."""
import numpy as np


def reference_body(A, b):
    """Recognize supported bodies from constraints, never from samples."""
    A, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
    d = A.shape[1]
    if np.all(np.count_nonzero(A, axis=1) == 1):
        lo, hi = np.full(d, -np.inf), np.full(d, np.inf)
        for row, bound in zip(A, b):
            j = np.flatnonzero(row)[0]
            if row[j] > 0:
                hi[j] = min(hi[j], bound/row[j])
            else:
                lo[j] = max(lo[j], bound/row[j])
        if np.isfinite(lo).all() and np.isfinite(hi).all() and np.all(hi > lo):
            return "box", lo, hi
    expected = np.column_stack((np.vstack((-np.eye(d), np.ones(d))), np.r_[np.zeros(d), 1]))
    rows = np.column_stack((A, b)) / np.max(np.abs(A), axis=1)[:, None]
    if len(rows) == d+1 and all(any(np.allclose(row, other) for other in rows) for row in expected):
        return "simplex", np.zeros(d), np.ones(d)
    raise ValueError("Exact nD references support axis-aligned boxes and standard simplices only.")


def marginal_reference(kind, d, group, lo, hi, bins=25):
    """Exact expected histogram heights and the projected outline.

    Simplex survival: P(X_i>a, X_j>b)=(1-a-b)_+^d.
    Four differences yield a rectangular bin's probability mass.
    """
    edges = [np.linspace(lo[j], hi[j], bins+1) for j in group]
    if len(group) == 1:
        mass = np.ones(bins)/bins if kind == "box" else (1-edges[0][:-1])**d - (1-edges[0][1:])**d
        return edges, mass/np.diff(edges[0]), None
    x, y = edges
    if kind == "box":
        density = np.full((bins, bins), 1/((x[-1]-x[0])*(y[-1]-y[0])))
        boundary = np.array([[x[0], y[0]], [x[-1], y[0]], [x[-1], y[-1]], [x[0], y[-1]]])
    else:
        survival = np.maximum(1-x[:, None]-y[None, :], 0)**d
        mass = survival[:-1, :-1]-survival[1:, :-1]-survival[:-1, 1:]+survival[1:, 1:]
        density = np.maximum(mass, 0)/np.outer(np.diff(x), np.diff(y))
        boundary = np.array([[0, 0], [1, 0], [0, 1]])
    return edges, density, boundary
