"""Behavior checks for independent OUT-prefetch and IN-batching switches."""
import unittest
import numpy as np
from sampler import in_and_out as optimized_sampler, SamplingFailure, add_optimization_arguments, optimization_options
from functools import partial
in_and_out = partial(optimized_sampler, prefetch_out=False, batch_in=False)


class Draws:
    def __init__(self, values):
        self.values = iter(values)
        self.shapes = []

    def normal(self, size):
        self.shapes.append(size)
        return np.asarray(next(self.values), dtype=float).reshape(size)


class BatchingTests(unittest.TestCase):
    def test_defaults_and_cli_opt_outs(self):
        import argparse
        parser = argparse.ArgumentParser()
        add_optimization_arguments(parser)
        for flags, expected in [([], (True, True)),
                                (['--no-prefetch-out'], (False, True)),
                                (['--no-batch-in'], (True, False)),
                                (['--no-prefetch-out', '--no-batch-in'], (False, False))]:
            options = optimization_options(parser.parse_args(flags))
            self.assertEqual((options['prefetch_out'], options['batch_in']), expected)
        runs = [optimized_sampler([[1], [-1]], [1, 1], [0], h=.01, steps=50,
                                  rng=np.random.default_rng(5), **options)
                for options in [{}, dict(prefetch_out=True, batch_in=True)]]
        for a, b in zip(*runs):
            np.testing.assert_array_equal(a, b)

    def test_floor_jumps_then_doubles_and_resets(self):
        inward = Draws([[0], [0]*8, [0]*3+[-2]+[0]*12, [-2]])
        path, attempts = in_and_out([[1], [-1]], [1, 1], [0], h=1, steps=2,
                                    rng=inward, out_rng=Draws([[2], [2]]),
                                    batch_in=True, batch_floor=8)
        self.assertEqual(inward.shapes, [(1, 1), (8, 1), (16, 1), (1, 1)])
        np.testing.assert_array_equal(attempts, [13, 1])
        np.testing.assert_array_equal(path, [[0], [0], [0]])

    def test_prefetch_matches_scalar_with_separate_streams(self):
        args = ([[1], [-1]], [1, 1], [0])
        results = []
        for prefetch in [False, True]:
            results.append(in_and_out(*args, h=.1, steps=200, max_attempts=None,
                                     rng=np.random.default_rng(11), out_rng=np.random.default_rng(12),
                                     prefetch_out=prefetch))
        for a, b in zip(*results):
            np.testing.assert_array_equal(a, b)

    def test_batch_first_success_fixed_y_and_counts(self):
        out = Draws([[2]])
        inward = Draws([[0], [0, 0], [0, -1.5, -2, 0]])
        stats = {}
        path, attempts = in_and_out([[1], [-1]], [1, 1], [0], h=1, steps=1,
                                    rng=inward, out_rng=out, batch_in=True, diagnostics=stats, batch_floor=2)
        np.testing.assert_array_equal(path, [[0], [.5]])
        self.assertEqual(attempts[0], 5)
        self.assertEqual(stats['generated_inward'], 7)
        self.assertEqual(stats['unused_inward'], 2)
        self.assertEqual(inward.shapes, [(1, 1), (2, 1), (4, 1)])
        self.assertEqual(out.shapes, [1])

    def test_batch_limit_is_not_retry_cap(self):
        inward = Draws([[0], [0, 0], [0, 0], [-2, 0]])
        path, attempts = in_and_out([[1], [-1]], [1, 1], [0], h=1, steps=1,
                                    rng=inward, out_rng=Draws([[2]]), batch_in=True,
                                    batch_limit=2, max_attempts=None, batch_floor=2)
        self.assertEqual(attempts[0], 6)
        self.assertEqual(path[-1, 0], 0)

    def test_cap_does_not_accept_past_budget(self):
        inward = Draws([[0], [0, 0], [0]])
        with self.assertRaises(SamplingFailure) as error:
            in_and_out([[1], [-1]], [1, 1], [0], h=1, steps=1,
                       rng=inward, out_rng=Draws([[2]]), batch_in=True, max_attempts=4, batch_floor=2)
        self.assertEqual(error.exception.proposals, 4)
        self.assertEqual(inward.shapes, [(1, 1), (2, 1), (1, 1)])


if __name__ == '__main__':
    unittest.main()
