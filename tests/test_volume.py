"""Membership, telescoping, retention, and failure checks for DFK volume."""
import math
import unittest
from unittest.mock import patch
import numpy as np
from demo import example_body
from sampler import in_and_out, SamplingFailure
from tests.test_batching import Draws
from volume import estimate_volume, volume_plan


class VolumeTests(unittest.TestCase):
    def test_ball_rejection_keeps_outward_point_in_both_modes(self):
        for batched in (False, True):
            inward = Draws([[0], [-1.5, -2]] if batched else [[0], [-1.5]])
            path, attempts = in_and_out([[1], [-1]], [10, 10], [1], h=1, steps=1,
                rng=inward, out_rng=Draws([[2]]), prefetch_out=True,
                batch_in=batched, batch_floor=2, ball=([1], 1))
            np.testing.assert_array_equal(path, [[1], [1.5]])
            np.testing.assert_array_equal(attempts, [2])

    def test_invalid_ball_and_outside_start(self):
        for ball in [([0, 0], 1), ([0], 0), ([0], math.inf), ([math.nan], 1), ([3], 1)]:
            with self.assertRaises(ValueError):
                in_and_out([[1], [-1]], [10, 10], [0], h=.1, steps=1, ball=ball)

    def test_schedule_and_translation_scaling(self):
        A, b, _, _ = example_body('simplex', 3)
        original = volume_plan(A, b)
        shift = np.array([2., -3., 4.])
        moved = volume_plan(A, 3*b + A@shift)
        np.testing.assert_allclose(moved['center'], 3*np.array(original['center'])+shift)
        self.assertAlmostEqual(moved['log_ball_volume']-original['log_ball_volume'], 3*math.log(3))
        radii = np.array(original['radii'])
        self.assertGreaterEqual(radii[-1], original['outer_radius'])
        self.assertTrue(np.all((radii[1:]/radii[:-1])**3 <= 2+1e-12))

    def test_ratio_and_thinning_across_chunks(self):
        plan = dict(dimension=1, center=[0.], inner_radius=1., outer_radius=2.,
                    radii=[1., 2.], phases=1, log_ball_volume=math.log(2))
        completed = 0
        def scripted(A, b, x, *, steps, diagnostics, **kwargs):
            nonlocal completed
            # Alternating inside/outside; exactly half the retained points hit.
            indices = completed + np.arange(1, steps+1)
            points = np.where(indices % 2, 0., 1.5).reshape(-1, 1)
            completed += steps
            diagnostics['generated_inward'] = steps
            return np.vstack([x, points]), np.ones(steps, dtype=int)
        with patch('volume.volume_plan', return_value=plan), patch('volume.in_and_out', side_effect=scripted):
            result = estimate_volume([[1], [-1]], [2, 2], samples=400, burn_in=2, thin=3)
        self.assertEqual(completed, 1202)
        self.assertEqual(result['phase_results'][0]['samples'], 400)
        self.assertEqual(result['phase_results'][0]['hits'], 200)
        self.assertAlmostEqual(result['volume'], 4.)

    def test_sampler_failure_is_not_replaced(self):
        A, b, _, _ = example_body('box', 2)
        with patch('volume.in_and_out', side_effect=SamplingFailure('test cap')) as sampler:
            with self.assertRaises(SamplingFailure):
                estimate_volume(A, b, samples=2, burn_in=0, thin=1)
        self.assertEqual(sampler.call_count, 1)

    def test_zero_overlap_fails(self):
        plan = dict(dimension=1, center=[0.], inner_radius=1., outer_radius=2.,
                    radii=[1., 2.], phases=1, log_ball_volume=math.log(2))
        def outside(A, b, x, *, steps, diagnostics, **kwargs):
            diagnostics['generated_inward'] = steps
            return np.vstack([x, np.full((steps, 1), 1.5)]), np.ones(steps, dtype=int)
        with patch('volume.volume_plan', return_value=plan), patch('volume.in_and_out', side_effect=outside):
            with self.assertRaises(SamplingFailure):
                estimate_volume([[1], [-1]], [2, 2], samples=2, burn_in=0, thin=1)


if __name__ == '__main__':
    unittest.main()
