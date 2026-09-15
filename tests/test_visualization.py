"""Numerical checks for polygon moments and the displayed density."""

import unittest
import numpy as np

from plotting.density import make_grid, restricted_density
from shapes import polygon_body, shape_points
from plotting.plot_pairs import coordinate_groups
from plotting.reference import reference_body, marginal_reference
from demo import example_body


class VisualizationTests(unittest.TestCase):
    def test_exact_marginal_references(self):
        for kind in ("box", "simplex"):
            A, b, _, _ = example_body(kind, 5)
            name, lo, hi = reference_body(A, b)
            self.assertEqual(name, kind)
            for group in ((0, 1), (4,)):
                edges, density, _ = marginal_reference(name, 5, group, lo, hi)
                widths = np.diff(edges[0])
                if len(group) == 2:
                    widths = np.outer(widths, np.diff(edges[1]))
                self.assertAlmostEqual(np.sum(density*widths), 1)
                self.assertTrue(np.all(density >= 0))
            edges, density, boundary = marginal_reference(name, 5, (0, 1), lo, hi)
            if kind == "simplex":
                self.assertEqual(boundary.shape, (3, 2))
                self.assertGreater(density[0, 0], density[10, 10])
                # P(X1<=.5, X2<=.5) = 1 - 2*(.5)^5.
                e, v, _ = marginal_reference(name, 5, (0, 1), lo, hi, bins=2)
                self.assertAlmostEqual(v[0, 0]*0.25, 1-2*0.5**5)
            else:
                np.testing.assert_allclose(density, 0.25)

    def test_coordinate_pairs_cover_every_dimension(self):
        self.assertEqual(coordinate_groups(5), [(0, 1), (2, 3), (4,)])
        self.assertEqual(coordinate_groups(1), [(0,)])
        self.assertEqual(len(coordinate_groups(10)), 5)
        for d in range(1, 12):
            self.assertEqual([i for group in coordinate_groups(d) for i in group], list(range(d)))

    def test_polygon_moments(self):
        A, b, mean, cov = polygon_body([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, 0]])
        self.assertEqual(len(b), 4)  # Interior point is not an additional facet.
        np.testing.assert_allclose(mean, [0, 0], atol=1e-14)
        np.testing.assert_allclose(cov, np.eye(2)/3, atol=1e-14)

    def test_restricted_kernels_and_accumulation(self):
        A, b, _, _ = polygon_body(shape_points("polygon", 7))
        _, grid, mask, area, _, r = make_grid(A, b, resolution=35)
        samples = np.array([[0., 0.], [0.9, 0.]])
        full = restricted_density(samples, grid, mask, area, 0.15*r)
        left = restricted_density(samples[:1], grid, mask, area, 0.15*r)
        right = restricted_density(samples[1:], grid, mask, area, 0.15*r)
        self.assertTrue(np.all(full[~mask] == 0))
        self.assertTrue(np.all(full >= 0))
        self.assertAlmostEqual(full.sum()*area, 1)
        self.assertAlmostEqual(right.sum()*area, 1)  # Edge sample retains full mass.
        np.testing.assert_allclose(full, (left+right)/2)


if __name__ == "__main__":
    unittest.main()
