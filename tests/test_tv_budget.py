"""Endpoint budget and command-line integration checks."""
import math
from pathlib import Path
import subprocess
import sys
import unittest

from geometry import example_geometry
from mixing import mixing_steps

ROOT = Path(__file__).resolve().parents[1]

class TVBudgetTests(unittest.TestCase):
    def test_budget_is_sufficient_and_previous_step_is_not(self):
        for h, diameter, log_M, epsilon in [(1e-4, math.sqrt(2), 9.732979594483439, 1e-6),
                                           (.1, 2, math.log(3), .1)]:
            k = mixing_steps(h, diameter, log_M, epsilon)
            log_initial = math.log(math.expm1(log_M))
            rate = math.log1p(h*math.pi**2/diameter**2)
            def log_tv(n):
                return -math.log(2) + (log_initial-n*rate)/2
            self.assertLessEqual(log_tv(k), math.log(epsilon))
            self.assertGreater(log_tv(k-1), math.log(epsilon))

    def test_zero_steps_and_scaling(self):
        self.assertEqual(mixing_steps(.1, 2, 0, 1e-6), 0)
        self.assertEqual(mixing_steps(.1, 2, math.log(1.01), .1), 0)
        self.assertEqual(mixing_steps(.1, 2, 3, .01), mixing_steps(.4, 4, 3, .01))
        self.assertEqual(example_geometry("simplex", 1)[3], 0)

    def test_invalid_parameters(self):
        for values in [(0, 2, 3, .1), (.1, -2, 3, .1), (.1, 2, -1, .1),
                       (.1, 2, 3, 0), (.1, 2, 3, 1), (math.inf, 2, 3, .1)]:
            with self.assertRaises(ValueError):
                mixing_steps(*values)

    def cli(self, *args):
        return subprocess.run([sys.executable, "demo.py", *args], cwd=ROOT,
                              capture_output=True, text=True)

    def test_plan_and_incompatible_options(self):
        plan = self.cli("--body", "simplex", "--dim", "10", "--h", "1e-4",
                        "--tv-distance", "1e-6", "--plan-only")
        self.assertEqual(plan.returncode, 0, plan.stderr)
        self.assertIn("72,924", plan.stdout)
        self.assertNotIn("Progress:", plan.stdout)
        for extra in [("--steps", "20"), ("--start", "center"),
                      ("--max-attempts", "10"), ("--body", "random")]:
            result = self.cli("--tv-distance", ".1", "--plan-only", *extra)
            self.assertNotEqual(result.returncode, 0)

    def test_short_run_and_zero_step_endpoint(self):
        for dim in ["1", "2"]:
            result = self.cli("--body", "box", "--dim", dim, "--h", ".1",
                              "--tv-distance", ".1")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Final endpoint:", result.stdout)
            self.assertIn("All iterates feasible: True", result.stdout)

if __name__ == "__main__":
    unittest.main()
