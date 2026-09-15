import math
import unittest

import numpy as np

from experiments.benchmark_h import timed_run
from experiments.mixing import example_geometry, required_steps
from sampler import in_and_out, SamplingTimeout


class MixingTests(unittest.TestCase):
    def test_formula_and_scaling(self):
        h, d, D, log_M = 0.01, 10, math.sqrt(2), 9.732979594483439
        k = required_steps(h, d, D, log_M)
        self.assertEqual(k, 760668)
        ratio = (12*math.log(20)+log_M)/math.log1p(h*math.log(2)**2/(4*d*D**2))
        self.assertGreaterEqual(k, ratio)
        self.assertLess(k-1, ratio)
        self.assertEqual(k, required_steps(4*h, d, 2*D, log_M))
        self.assertGreater(required_steps(h/2, d, D, log_M), k)

    def test_geometry(self):
        _, r, D, log_M = example_geometry("simplex", 2)
        self.assertAlmostEqual(r, 1/(2+math.sqrt(2)))
        self.assertAlmostEqual(D, math.sqrt((1-r)**2+r*r))
        c10, r10, D10, _ = example_geometry("simplex", 10)
        distances = np.linalg.norm(np.vstack((np.zeros(10), np.eye(10)))-c10, axis=1)
        self.assertAlmostEqual(D10, distances.max())
        self.assertEqual(required_steps(0.01, 10, D10, example_geometry("simplex", 10)[3]), 344509)
        self.assertAlmostEqual(example_geometry("box", 10)[2], math.sqrt(10))
        self.assertAlmostEqual(math.exp(log_M), 0.5/(math.pi*r*r))
        self.assertAlmostEqual(example_geometry("box", 1)[3], 0)

    def test_timeout_is_not_a_sample(self):
        with self.assertRaises(SamplingTimeout) as caught:
            in_and_out([[1], [-1]], [1, 1], [0], h=0.01, steps=10, deadline=0)
        self.assertEqual(caught.exception.completed_steps, 0)
        self.assertEqual(caught.exception.proposals, 0)

    def test_chunking_matches_original_chain(self):
        # Reproduce the same ball start and random draws across a chunk boundary.
        from geometry import uniform_ball
        rng = np.random.default_rng(77)
        x0 = uniform_ball([0.], 1., rng)
        path, attempts = in_and_out([[1], [-1]], [1, 1], x0, h=0.001,
                                    steps=1100, max_attempts=None, rng=rng,
                                    prefetch_out=False, batch_in=False)
        run = timed_run([[1], [-1]], [1, 1], [0.], 1., 0.001, 1100, 77, 0, None,
                        prefetch_out=False, batch_in=False)
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["inward_proposals"], attempts.sum())
        np.testing.assert_array_equal(run["endpoint"], path[-1])


if __name__ == "__main__":
    unittest.main()
