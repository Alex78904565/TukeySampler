"""Independent streams, Windows-spawn execution, and batch failure behavior."""
import unittest
from unittest.mock import patch

import numpy as np

from batch import sample_endpoints
from demo import example_body
from geometry import enclosing_balls
from sampler import SamplingFailure


class ParallelTests(unittest.TestCase):
    def test_worker_count_preserves_streams_order_and_feasibility(self):
        A, b, _, _ = example_body('simplex', 3)
        options = dict(samples=5, steps=1100, h=1e-4, seed=42)
        serial = sample_endpoints(A, b, workers=1, **options)
        reports = []
        with patch('geometry.enclosing_balls', wraps=enclosing_balls) as geometry:
            parallel = sample_endpoints(A, b, workers=2,
                progress=lambda done, total: reports.append((done, total)), **options)
            self.assertEqual(geometry.call_count, 1)
        np.testing.assert_array_equal(serial[0], parallel[0])
        np.testing.assert_array_equal(serial[1], parallel[1])
        self.assertTrue(np.all(parallel[0] @ A.T <= b))
        self.assertEqual(len(np.unique(parallel[0], axis=0)), 5)
        self.assertEqual(reports[-1], (5, 5))

    def test_zero_steps_returns_fresh_ball_draws(self):
        A, b, _, _ = example_body('box', 2)
        points, proposals = sample_endpoints(A, b, samples=3, steps=0, h=.01, workers=2)
        self.assertTrue(np.all(np.linalg.norm(points, axis=1) <= 1))
        np.testing.assert_array_equal(proposals, [0, 0, 0])
        self.assertEqual(len(np.unique(points, axis=0)), 3)

    def test_cap_failure_propagates_from_worker(self):
        A, b, _, _ = example_body('box', 2)
        with self.assertRaises(SamplingFailure):
            sample_endpoints(A, b, samples=2, steps=1, h=1e20,
                             workers=2, max_attempts=1)

    def test_ball_intersection_and_provided_geometry(self):
        A, b, _, _ = example_body('box', 2)
        for workers in (1, 2):
            with patch('geometry.enclosing_balls', side_effect=AssertionError('Unexpected LP solve')):
                points, _ = sample_endpoints(A, b, samples=6, steps=20, h=.01,
                    start_ball=([0., 0.], .2), ball=([0., 0.], .4), workers=workers)
            self.assertTrue(np.all(np.linalg.norm(points, axis=1) <= .4))
        for start_ball, ball in [(([0., 0.], 2.), None),
                                 (([0., 0.], .5), ([0., 0.], .4)),
                                 (([2., 0.], .1), None)]:
            with self.assertRaises(ValueError):
                sample_endpoints(A, b, samples=2, steps=0, h=.01,
                                 start_ball=start_ball, ball=ball)

    def test_invalid_budgets(self):
        A, b, _, _ = example_body('box', 2)
        for update in ({'samples': 0}, {'workers': 0}, {'steps': -1}, {'h': float('nan')}):
            options = dict(samples=2, steps=1, h=.01)
            options.update(update)
            with self.assertRaises(ValueError):
                sample_endpoints(A, b, **options)


if __name__ == '__main__':
    unittest.main()
