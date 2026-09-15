"""Turn 2D input points into a convex polygon and its uniform moments."""

import numpy as np
from scipy.spatial import ConvexHull


def polygon_body(points):
    """Return A, b, mean, covariance for the convex hull of the input points.

    Interior points do not change the hull. Moments are computed by splitting
    the polygon into triangles and weighting their moments by area.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("Supply at least three finite 2D points spanning an area.")
    hull = ConvexHull(points)
    vertices = points[hull.vertices]
    center = vertices.mean(axis=0)
    areas, means, seconds = [], [], []
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        triangle = np.array([center, a, b])
        total = triangle.sum(axis=0)
        areas.append(abs(np.linalg.det(np.array([a-center, b-center]))) / 2)
        means.append(total / 3)
        seconds.append((triangle.T @ triangle + np.outer(total, total)) / 12)
    weights = np.array(areas) / sum(areas)
    mean = weights @ np.array(means)
    second = np.einsum("i,ijk->jk", weights, seconds)
    return hull.equations[:, :2], -hull.equations[:, 2], mean, second - np.outer(mean, mean)


def shape_points(name, count=9, seed=7):
    """Regular polygon, stretched polygon, or hull of a random point cloud."""
    if count < 3:
        raise ValueError("Use at least three points.")
    angles = np.linspace(0, 2*np.pi, count, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    if name == "ellipse":
        return points @ np.array([[2.2, 0.5], [0, 0.75]])
    if name == "random":
        return np.random.default_rng(seed).normal(size=(count, 2))
    return points
