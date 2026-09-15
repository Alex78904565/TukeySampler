"""DFK volume estimation (main.pdf, Algorithm 35) using our in-and-out sampler.

K_i = K intersect B(c, r * 2**(i/d)). Sample K_{i+1}, count membership
in K_i, then divide the known inner-ball volume by the product of fractions.
This is an experimental MCMC implementation, not a certified error bound.
"""
import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from geometry import enclosing_balls, uniform_ball
from sampler import (SamplingFailure, in_and_out, add_optimization_arguments,
                     optimization_options)


def volume_plan(A, b):
    """Find the inner ball and the expanding radii, in original coordinates."""
    center, r, R = enclosing_balls(A, b)
    d = len(center)
    m = max(0, math.ceil(d * math.log2(R / r)))
    radii = r * np.exp2(np.arange(m + 1) / d)
    # Avoid undershooting the outer bound due to floating-point roundoff.
    radii[-1] = max(radii[-1], R)
    log_ball_volume = d/2 * math.log(math.pi) - math.lgamma(d/2 + 1) + d * math.log(r)
    return dict(dimension=d, center=center.tolist(), inner_radius=r,
                outer_radius=R, radii=radii.tolist(), phases=m,
                log_ball_volume=log_ball_volume)


def estimate_volume(A, b, *, samples=2000, burn_in=2000, thin=10, h=None,
                    seed=7, max_attempts=100000, prefetch_out=True, batch_in=True,
                    batch_floor=32, progress=None):
    """Estimate volume of a bounded, full-dimensional {x: A x <= b}.

    Each phase runs burn_in + samples*thin proper steps and keeps exactly
    samples iterates. Reuse the previous endpoint to start the next phase.
    If the previous phase were uniform, this start would be at most 2-warm.
    Finite burn-in and thinning do not certify mixing or independence.
    progress(phase, phase_count, completed_steps, phase_steps, current_rejections)
    is called during sampling. Retry failures propagate; no failed run is replaced.
    """
    for name, value, minimum in [('samples', samples, 1), ('burn_in', burn_in, 0), ('thin', thin, 1)]:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    plan = volume_plan(A, b)
    d, r = plan['dimension'], plan['inner_radius']
    h = 0.1 * r*r / d if h is None else float(h)
    if not math.isfinite(h) or h <= 0:
        raise ValueError('h must be finite and positive')
    rng = np.random.default_rng(seed)
    center = np.asarray(plan['center'])
    x = uniform_ball(center, r, rng)
    log_volume = plan['log_ball_volume']
    phase_steps = burn_in + samples * thin
    results = []
    start = perf_counter()
    for phase, (inner, outer) in enumerate(zip(plan['radii'][:-1], plan['radii'][1:]), 1):
        hits = retained = proposals = generated = completed = 0
        phase_start = perf_counter()
        # Chunking bounds path and OUT-prefetch memory; it does not restart the chain.
        while completed < phase_steps:
            count = min(1024, phase_steps-completed)
            def report(done, attempts, rejected):
                if progress is not None:
                    progress(phase, plan['phases'], completed+done, phase_steps, rejected)
            stats = {}
            path, attempts = in_and_out(
                A, b, x, h=h, steps=count, rng=rng, max_attempts=max_attempts,
                ball=(center, outer), prefetch_out=prefetch_out, batch_in=batch_in,
                batch_floor=batch_floor, diagnostics=stats, progress=report)
            indices = completed + np.arange(1, count+1)
            keep = (indices > burn_in) & ((indices-burn_in) % thin == 0)
            points = path[1:][keep]
            hits += int(np.count_nonzero(np.linalg.norm(points-center, axis=1) <= inner))
            retained += len(points)
            proposals += int(attempts.sum())
            generated += stats['generated_inward']
            x = path[-1].copy()
            completed += count
        if hits == 0:
            raise SamplingFailure(f'Phase {phase}: zero overlap hits; no finite volume estimate. '
                                  'Increase the sampling budget and check mixing.')
        fraction = hits / retained
        log_volume -= math.log(fraction)
        results.append(dict(phase=phase, inner_radius=inner, outer_radius=outer,
                            hits=hits, samples=retained, fraction=fraction,
                            steps=completed, proposals=proposals, generated_inward=generated,
                            seconds=perf_counter()-phase_start))
    return dict(**plan, volume=math.exp(log_volume), log_volume=log_volume,
                seconds=perf_counter()-start, samples_per_phase=samples, burn_in=burn_in,
                thin=thin, h=h, seed=seed, max_attempts=max_attempts,
                prefetch_out=prefetch_out, batch_in=batch_in, batch_floor=batch_floor,
                phase_results=results,
                note='Experimental correlated MCMC estimate; no certified relative error or confidence interval.')


def main():
    from demo import example_body, parse_max_attempts
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--body', choices=['box', 'simplex'], default='box')
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--samples', type=int, default=2000, help='Retained iterates per phase')
    parser.add_argument('--burn-in', type=int, default=2000, help='Discarded steps at EVERY phase')
    parser.add_argument('--thin', type=int, default=10, help='Proper steps per retained iterate')
    parser.add_argument('--h', type=float, help='Gaussian variance; heuristic default 0.1*r^2/d')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--repeats', type=int, default=1, help='Independent complete volume estimates')
    parser.add_argument('--max-attempts', type=parse_max_attempts, default=100000)
    parser.add_argument('--batch-floor', type=int, default=32)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--output', type=Path, help='Save settings and all phase results as JSON')
    add_optimization_arguments(parser)
    args = parser.parse_args()
    if min(args.dim, args.samples, args.thin, args.repeats) < 1 or args.burn_in < 0 or args.batch_floor < 2:
        parser.error('Require positive dimension/samples/thin/repeats, burn-in >= 0, floor >= 2')
    if args.h is not None and (not math.isfinite(args.h) or args.h <= 0):
        parser.error('h must be finite and positive')
    A, b, _, _ = example_body(args.body, args.dim)
    plan = volume_plan(A, b)
    steps = args.burn_in + args.samples*args.thin
    print(f"{args.body}, d={args.dim}: {plan['phases']} phases, {steps:,} steps/phase, "
          f"{plan['phases']*steps*args.repeats:,} total proper steps", flush=True)
    print('Experimental MCMC budgets; no certified volume error.', flush=True)
    if args.plan_only:
        return
    exact = 2.0**args.dim if args.body == 'box' else math.exp(-math.lgamma(args.dim+1))
    record = dict(body=args.body, exact_volume=exact, runs=[])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for repeat in range(args.repeats):
            last_report = -math.inf
            def report(phase, phases, done, total, rejected):
                nonlocal last_report
                now = perf_counter()
                if done == total or now-last_report >= 10:
                    print(f'Trial {repeat+1}/{args.repeats}, phase {phase}/{phases}: '
                          f'{done:,}/{total:,} steps; current rejections {rejected:,}', flush=True)
                    last_report = now
            result = estimate_volume(A, b, samples=args.samples, burn_in=args.burn_in,
                                     thin=args.thin, h=args.h, seed=args.seed+repeat,
                                     max_attempts=args.max_attempts, batch_floor=args.batch_floor,
                                     progress=report, **optimization_options(args))
            result['relative_error'] = result['volume']/exact - 1
            record['runs'].append(result)
            print(f"Volume {result['volume']:.8g}; exact {exact:.8g}; "
                  f"error {result['relative_error']:+.2%}; {result['seconds']:.2f}s", flush=True)
            if args.output:
                args.output.write_text(json.dumps(record, indent=2), encoding='utf-8')
    except (SamplingFailure, ValueError, OverflowError) as error:
        parser.exit(1, f'Volume run stopped: {error}\n')


if __name__ == '__main__':
    main()
