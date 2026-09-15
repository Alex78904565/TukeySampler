"""Run a small experiment: python demo.py --body box --steps 20000."""

import argparse

import numpy as np

from geometry import enclosing_balls, log_warmness_bound, uniform_ball
from sampler import SamplingFailure, in_and_out, add_optimization_arguments, optimization_options
from shapes import polygon_body, shape_points


def example_body(name, d, count=9, seed=7):
    """Return A, b and the exact uniform mean/covariance for a simple body."""
    if name == "box":
        # [-1, 1]^d
        return np.vstack((np.eye(d), -np.eye(d))), np.ones(2 * d), np.zeros(d), np.eye(d) / 3
    if name != "simplex":
        if d != 2:
            raise ValueError("Polygon examples require --dim 2.")
        return polygon_body(shape_points(name, count, seed))
    # Standard simplex: x_j >= 0, sum(x_j) <= 1.
    A = np.vstack((-np.eye(d), np.ones(d)))
    b = np.r_[np.zeros(d), 1.0]
    mean = np.ones(d) / (d + 1)
    cov = ((d + 1) * np.eye(d) - np.ones((d, d))) / ((d + 1)**2 * (d + 2))
    return A, b, mean, cov


def parse_max_attempts(value):
    """Accept a positive integer or 'inf' for an uncapped inward step."""
    if value.lower() in ("inf", "infinity"):
        return None
    try:
        limit = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use a positive integer or inf.") from error
    if limit < 1:
        raise argparse.ArgumentTypeError("Use a positive integer or inf.")
    return limit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body", choices=("box", "simplex", "polygon", "ellipse", "random"), default="box")
    parser.add_argument("--points", help="CSV of x,y points (no header); sample their convex hull, overriding --body")
    parser.add_argument("--vertices", type=int, default=9, help="Polygon vertices or random input point count")
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20_000, help="Total proper transitions, including burn-in")
    parser.add_argument("--burn-in", type=int, default=2_000)
    parser.add_argument("--h", type=float, help="Variance in original coordinates; default 0.1*r^2/d^2")
    parser.add_argument("--max-attempts", type=parse_max_attempts, default=100_000,
                        help="Inward retry cap per transition; inf removes the cap")
    parser.add_argument("--start", choices=("ball", "center"), default="ball")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save", help="Optional .npz file containing the full run")
    parser.add_argument("--plot", help="Distribution figure; pairs of coordinates when dim > 2")
    parser.add_argument("--animate", help="Cumulative animation, e.g. evolution.gif; supports coordinate pairs")
    parser.add_argument("--smoothing", type=float, default=1.0, help="Plot bandwidth multiplier; larger means smoother")
    add_optimization_arguments(parser)
    args = parser.parse_args()
    options = optimization_options(args)
    if args.dim < 1 or not 0 <= args.burn_in <= args.steps - 2:
        parser.error("Require dim >= 1 and 0 <= burn-in <= steps - 2.")
    if (args.plot or args.animate) and args.steps - args.burn_in < 3:
        parser.error("Plotting requires at least three retained points.")
    if (args.plot or args.animate) and (not np.isfinite(args.smoothing) or args.smoothing <= 0):
        parser.error("smoothing must be finite and positive.")
    if args.vertices < 3 or ((args.points or args.body not in ("box", "simplex")) and args.dim != 2):
        parser.error("Polygon shapes require dim=2 and at least three input points.")
    if args.animate and not args.animate.lower().endswith(".gif"):
        parser.error("Use a .gif filename for --animate.")

    if args.points:
        A, b, mean, cov = polygon_body(np.loadtxt(args.points, delimiter=","))
    else:
        A, b, mean, cov = example_body(args.body, args.dim, args.vertices, args.seed)
    center, r, R = enclosing_balls(A, b)
    rng = np.random.default_rng(args.seed)
    x0 = uniform_ball(center, r, rng) if args.start == "ball" else center
    h = 0.1 * r**2 / args.dim**2 if args.h is None else args.h
    print(f"{args.points or args.body}, d={args.dim}, constraints={len(b)}, start={args.start}, seed={args.seed}")
    print(f"Inner radius r={r:.6g}, outer radius bound R={R:.6g}, h={h:.6g}")
    if args.start == "ball":
        print(f"Geometric log(M) upper-bound estimate: {log_warmness_bound(args.dim, r, R):.6g}")
    else:
        print("A fixed center is feasible but is not a finite-M warm distribution.")
    try:
        path, attempts = in_and_out(A, b, x0, h=h, steps=args.steps,
                                    max_attempts=args.max_attempts, rng=rng, **options)
    except (SamplingFailure, ValueError) as error:
        parser.exit(1, f"{error}\n")

    # Discard x0 and the first burn_in transitions; keep every later iterate.
    samples = path[args.burn_in + 1:]
    empirical_cov = np.atleast_2d(np.cov(samples, rowvar=False, ddof=0))
    print(f"Transitions: {len(attempts)}; retained iterates: {len(samples)}")
    print(f"Inward proposals: {attempts.sum()}; mean/step: {attempts.mean():.3f}; max: {attempts.max()}")
    print(f"All iterates feasible: {np.all(path @ A.T <= b)}")
    print("Empirical mean:", samples.mean(axis=0), " exact:", mean)
    print(f"Largest covariance entry error: {np.max(np.abs(empirical_cov - cov)):.6g}")
    print("These are correlated iterates. Moment agreement is not a mixing certificate.")
    if args.save:
        np.savez(args.save, path=path, attempts=attempts, A=A, b=b, h=h,
                 burn_in=args.burn_in, seed=args.seed, start=args.start,
                 max_attempts=np.inf if args.max_attempts is None else args.max_attempts,
                 center=center, r=r, R=R, **options, batch_floor=32, batch_limit=1048576)
    if args.plot:
        from plotting.plot_samples import plot_distribution
        plot_distribution(path, A, b, args.burn_in, args.plot, args.smoothing)
        print(f"Saved plot to {args.plot}")
    if args.animate:
        from plotting.plot_samples import animate_distribution
        animate_distribution(path, A, b, args.burn_in, args.animate, args.smoothing)
        print(f"Saved animation to {args.animate}")


if __name__ == "__main__":
    main()
