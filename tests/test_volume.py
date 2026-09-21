"""Membership, telescoping, independent endpoints, and failure checks for DFK volume."""
import math
import geometry
import unittest
from unittest.mock import patch
import numpy as np
from demo import example_body
from sampler import in_and_out, SamplingFailure
from tests.test_batching import Draws
from volume import estimate_volume, volume_plan, sampling_plan, dfk_sample_count


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

    def test_repeated_volume_runs_reuse_geometry(self):
        geometry._solve_enclosing_balls.cache_clear()
        self.addCleanup(geometry._solve_enclosing_balls.cache_clear)
        A, b, _, _ = example_body('box', 2)
        def endpoints(A, b, *, samples, start_ball, **kwargs):
            return np.tile(start_ball[0], (samples, 1)), np.ones(samples, dtype=int)
        with patch('geometry.linprog', wraps=geometry.linprog) as solve, \
             patch('volume.sample_endpoints', side_effect=endpoints):
            volume_plan(A, b)
            for seed in [7, 8, 9]:
                estimate_volume(A, b, samples=2, steps=1, experimental=True, seed=seed)
            self.assertEqual(solve.call_count, 5)

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

    def test_independent_phase_calls_and_telescoping(self):
        plan = dict(dimension=1, center=[0.], inner_radius=1., outer_radius=4.,
                    radii=[1., 2., 4.], phases=2, log_ball_volume=math.log(2))
        seeds = []
        def endpoints(A, b, *, samples, start_ball, ball, seed, **kwargs):
            seeds.append(seed.spawn_key)
            np.testing.assert_array_equal(start_ball[0], [0.])
            self.assertEqual(start_ball[1], 1.)
            # Exactly half hit the previous body in EACH phase.
            points = np.tile([[0.], [.75*ball[1]]], (samples//2, 1))
            return points, np.full(samples, 7)
        with patch('volume.volume_plan', return_value=plan), \
             patch('volume.sample_endpoints', side_effect=endpoints) as sampler:
            result = estimate_volume([[1], [-1]], [4, 4], samples=10, steps=3, experimental=True)
        self.assertEqual(sampler.call_count, 2)
        self.assertEqual(seeds, [(0,), (1,)])
        self.assertAlmostEqual(result['volume'], 8.)
        self.assertEqual(result['total_proper_steps'], 60)
        self.assertEqual([p['hits'] for p in result['phase_results']], [5, 5])
        self.assertEqual([p['proposals'] for p in result['phase_results']], [70, 70])

    def test_sampler_failure_is_not_replaced(self):
        A, b, _, _ = example_body('box', 2)
        with patch('volume.sample_endpoints', side_effect=SamplingFailure('test cap')) as sampler:
            with self.assertRaises(SamplingFailure):
                estimate_volume(A, b, samples=2, steps=1, experimental=True)
        self.assertEqual(sampler.call_count, 1)

    def test_zero_overlap_fails(self):
        plan = dict(dimension=1, center=[0.], inner_radius=1., outer_radius=2.,
                    radii=[1., 2.], phases=1, log_ball_volume=math.log(2))
        with patch('volume.volume_plan', return_value=plan), \
             patch('volume.sample_endpoints', return_value=(np.full((2, 1), 1.5), np.ones(2))):
            with self.assertRaises(SamplingFailure):
                estimate_volume([[1], [-1]], [2, 2], samples=2, steps=1, experimental=True)

    def test_explicit_count_against_high_precision_inequality(self):
        from decimal import Decimal, localcontext
        for m, eps, delta in [(53, .1, .25), (3, .5, .125), (53, .01, .125)]:
            n = dfk_sample_count(m, eps, delta)
            with localcontext() as ctx:
                ctx.prec = 70
                one = Decimal(1)
                a = Decimal(str(eps))/(one+Decimal(str(eps)))
                bound = Decimal(str(delta))*a*a
                self.assertLessEqual((one+one/n)**m-one, bound)
                self.assertGreater((one+one/(n-1))**m-one, bound)
        self.assertEqual(dfk_sample_count(53, .1, .25), 25678)
        self.assertEqual(dfk_sample_count(0, .1, .25), 0)

    def test_default_plan_accounts_for_all_endpoint_errors(self):
        A, b, _, _ = example_body('simplex', 3)
        plan = sampling_plan(A, b, h=.001)
        m, n = plan['phases'], plan['samples_per_phase']
        a = plan['relative_error']/(1+plan['relative_error'])
        statistical = math.expm1(m*math.log1p(1/n))/a**2
        joint_tv = m*n*plan['endpoint_tv']
        self.assertLessEqual(statistical+joint_tv, plan['failure_probability']+1e-14)
        for phase in plan['phase_plans']:
            rate = math.log1p(plan['h']*math.pi**2/phase['diameter_bound']**2)
            log_tv = -.5*phase['steps_per_sample']*rate + .5*math.log(math.expm1(phase['log_M_bound']))-math.log(2)
            self.assertLessEqual(log_tv, math.log(plan['endpoint_tv']))
        self.assertFalse(plan['experimental'])

    def test_underbudgets_require_explicit_experimental_mode(self):
        A, b, _, _ = example_body('box', 2)
        for overrides in [dict(samples=2), dict(steps=0), dict(tv_distance=.1),
                          dict(max_attempts=100), dict(failure_probability=1.2),
                          dict(relative_error=0), dict(experimental=True)]:
            with self.assertRaises(ValueError):
                sampling_plan(A, b, **overrides)
        result = sampling_plan(A, b, samples=2, tv_distance=.1, experimental=True)
        self.assertTrue(result['experimental'])
        self.assertIn('No volume-error guarantee', result['guarantee'])

    def test_serial_and_parallel_volume_agree(self):
        A, b, _, _ = example_body('box', 2)
        settings = dict(samples=12, steps=4, h=.01, experimental=True, seed=99)
        serial = estimate_volume(A, b, workers=1, **settings)
        parallel = estimate_volume(A, b, workers=2, **settings)
        self.assertEqual(serial['volume'], parallel['volume'])
        self.assertEqual([r['hits'] for r in serial['phase_results']],
                         [r['hits'] for r in parallel['phase_results']])
        self.assertEqual([r['proposals'] for r in serial['phase_results']],
                         [r['proposals'] for r in parallel['phase_results']])

    def test_zero_phase_body_uses_known_volume_without_sampling(self):
        plan = dict(dimension=1, center=[0.], inner_radius=1., outer_radius=1.,
                    radii=[1.], phases=0, log_ball_volume=math.log(2))
        with patch('volume.volume_plan', return_value=plan), \
             patch('volume.sample_endpoints') as sampler:
            result = estimate_volume([[1], [-1]], [1, 1])
        sampler.assert_not_called()
        self.assertEqual(result['total_proper_steps'], 0)
        self.assertEqual(result['volume'], 2.)

    def test_cli_rejects_legacy_and_underbudget_runs(self):
        import subprocess
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        for flags in [('--burn-in', '10'), ('--thin', '2'), ('--samples', '2')]:
            result = subprocess.run([sys.executable, 'volume.py', '--plan-only', *flags],
                                    cwd=root, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
        result = subprocess.run([sys.executable, 'volume.py', '--plan-only', '--experimental',
                                 '--samples', '2', '--steps', '1'],
                                cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('No volume-error guarantee', result.stdout)


if __name__ == '__main__':
    unittest.main()
