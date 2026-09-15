"""Run with: python -m unittest -v. Only standard-library unittest is needed."""

import unittest

import numpy as np

from demo import example_body, parse_max_attempts
from geometry import enclosing_balls, uniform_ball
from sampler import SamplingFailure, in_and_out
from functools import partial
# These scripted draws exercise the original scalar transition explicitly.
in_and_out = partial(in_and_out, prefetch_out=False, batch_in=False)


class ScriptedNormal:
    """Predetermined Gaussian draws make rejection behavior directly testable."""

    def __init__(self, values):
        self.values = iter(values)
        self.calls = 0

    def normal(self, size):
        self.calls += 1
        return np.full(size, next(self.values))


class SamplerTests(unittest.TestCase):
    def test_progress_during_uncapped_rejections(self):
        rng = ScriptedNormal([1] + [1]*256 + [-0.75])
        reports = []
        path, attempts = in_and_out([[1], [-1]], [1, 1], [0], h=4, steps=1,
                                    max_attempts=None, rng=rng,
                                    progress=lambda *counts: reports.append(counts))
        self.assertIn((0, 256, 256), reports)
        self.assertEqual(reports[-1], (1, 257, 0))
        np.testing.assert_array_equal(path, [[0], [0.5]])
        np.testing.assert_array_equal(attempts, [257])

    def test_uncapped_retries(self):
        rng = ScriptedNormal([1, 1, 1, -0.75])
        path, attempts = in_and_out([[1], [-1]], [1, 1], [0],
                                    h=4, steps=1, max_attempts=None, rng=rng)
        np.testing.assert_array_equal(path, [[0], [0.5]])
        np.testing.assert_array_equal(attempts, [3])
        self.assertEqual(rng.calls, 4)
        self.assertIsNone(parse_max_attempts("inf"))
        self.assertIsNone(parse_max_attempts("infinity"))
        self.assertEqual(parse_max_attempts("100"), 100)

    def test_fixed_outward_point_and_variance(self):
        # h=4 -> scale=2. y=2 (outside); proposals 4 (reject), 0.5 (accept).
        rng = ScriptedNormal([1, 1, -0.75])
        path, attempts = in_and_out([[1], [-1]], [1, 1], [0],
                                    h=4, steps=1, max_attempts=2, rng=rng)
        np.testing.assert_array_equal(path, [[0], [0.5]])
        np.testing.assert_array_equal(attempts, [2])
        self.assertEqual(rng.calls, 3)

    def test_cap_stops_without_redrawing_outward_point(self):
        rng = ScriptedNormal([10, 0, 0])
        with self.assertRaises(SamplingFailure):
            in_and_out([[1], [-1]], [1, 1], [0], h=1, steps=2,
                       max_attempts=2, rng=rng)
        self.assertEqual(rng.calls, 3)

    def test_zero_steps_and_invalid_inputs(self):
        path, attempts = in_and_out([[1], [-1]], [1, 1], [0], h=1, steps=0)
        self.assertEqual(path.shape, (1, 1))
        self.assertEqual(attempts.size, 0)
        for changes in ({"h": 0}, {"h": np.nan}, {"steps": -1},
                        {"steps": 1.5}, {"max_attempts": 0}, {"x0": [2]}):
            args = dict(A=[[1], [-1]], b=[1, 1], x0=[0], h=1, steps=1)
            args.update(changes)
            with self.assertRaises(ValueError):
                in_and_out(**args)

    def test_geometry_and_uniform_ball(self):
        # Shifted rectangle exercises unrestricted LP variables and unequal scales.
        A = np.vstack((np.eye(2), -np.eye(2)))
        b = np.array([-1, 4, 3, -2])  # [-3,-1] x [2,4]
        c, r, R = enclosing_balls(A, b)
        np.testing.assert_allclose(c, [-2, 3])
        self.assertAlmostEqual(r, 1)
        self.assertAlmostEqual(R, np.sqrt(2))
        rng = np.random.default_rng(11)
        points = np.array([uniform_ball(c, r, rng) for _ in range(10_000)])
        self.assertTrue(np.all(points @ A.T <= b))
        # A uniform d-ball has E[||X-c||^2] = r^2*d/(d+2).
        self.assertAlmostEqual(np.mean(np.sum((points-c)**2, axis=1)), 0.5, delta=0.015)

    def test_unsuitable_bodies(self):
        bodies = [([[1], [-1]], [0, -1]),  # empty
                  ([[1], [-1]], [0, 0]),   # no interior
                  ([[1, 0], [-1, 0]], [1, 1])]  # unbounded strip
        for A, b in bodies:
            with self.assertRaises(ValueError):
                enclosing_balls(A, b)

    def test_reproducibility(self):
        A, b, _, _ = example_body("box", 2)
        runs = [in_and_out(A, b, [0, 0], h=0.02, steps=100,
                           rng=np.random.default_rng(13)) for _ in range(2)]
        for left, right in zip(*runs):
            np.testing.assert_array_equal(left, right)

    def test_uniform_moments(self):
        # Broad regression checks on correlated samples, not statistical proofs.
        for name in ("box", "simplex"):
            with self.subTest(body=name):
                A, b, mean, cov = example_body(name, 2)
                c, r, _ = enclosing_balls(A, b)
                rng = np.random.default_rng(23)
                path, _ = in_and_out(A, b, uniform_ball(c, r, rng),
                                     h=0.1*r*r/4, steps=30_000, rng=rng)
                samples = path[3_001:]
                self.assertTrue(np.all(path @ A.T <= b))
                np.testing.assert_allclose(samples.mean(axis=0), mean, atol=0.06)
                np.testing.assert_allclose(np.cov(samples, rowvar=False), cov, atol=0.04)


if __name__ == "__main__":
    unittest.main()
