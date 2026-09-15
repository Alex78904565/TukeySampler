"""A simple Gaussian mixture restricted to a 2D polygon, evaluated on a grid."""

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.spatial.distance import cdist

from geometry import enclosing_balls


def make_grid(A, b, resolution=85):
    """Return polygon, cell centers, membership mask, cell area, extent, radius."""
    center, radius, _ = enclosing_balls(A, b)
    vertices = HalfspaceIntersection(np.column_stack((A, -b)), center).intersections
    polygon = vertices[ConvexHull(vertices).vertices]
    lo, hi = polygon.min(axis=0), polygon.max(axis=0)
    margin = 0.08 * (hi-lo)
    lo, hi = lo-margin, hi+margin
    dx, dy = (hi-lo) / resolution
    xx, yy = np.meshgrid(lo[0] + (np.arange(resolution)+0.5)*dx,
                         lo[1] + (np.arange(resolution)+0.5)*dy)
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    mask = np.all(grid @ np.asarray(A).T <= b, axis=1)
    return polygon, grid, mask, dx*dy, [lo[0], hi[0], lo[1], hi[1]], radius


def restricted_density(samples, grid, mask, cell_area, bandwidth):
    """Average Gaussian kernels after restricting EACH kernel to K.

    The normalization integral is approximated by a grid sum. Each point has
    equal total weight, including points near an edge. Off-body values are zero.
    This is a visualization estimate, not a correction to the sampling chain.
    """
    if not np.isfinite(bandwidth) or bandwidth <= 0 or len(samples) == 0:
        raise ValueError("Need a positive bandwidth and at least one sample.")
    total = np.zeros(len(grid))
    for start in range(0, len(samples), 256):
        distance = cdist(grid[mask], samples[start:start+256], metric="sqeuclidean")
        log_weights = -distance / (2*bandwidth**2)
        log_weights -= log_weights.max(axis=0)  # stable even for tiny bandwidths
        weights = np.exp(log_weights)
        weights /= weights.sum(axis=0) * cell_area
        total[mask] += weights.sum(axis=1)
    return total / len(samples)
