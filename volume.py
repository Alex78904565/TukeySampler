"""DFK (Algorithm 35) with independent In-and-Out endpoints in every phase.

Source: Lee and Vempala, Techniques in Optimization and Sampling, Aug. 7, 2026,
Section 11.1, Algorithm 35 and Theorem 11.7. See README for the explicit
sample-count derivation and the additional approximate-sampling error budget.
"""
import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from batch import sample_endpoints
from geometry import enclosing_balls
from mixing import mixing_steps
from sampler import SamplingFailure, add_optimization_arguments, optimization_options


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


def dfk_sample_count(phases, relative_error, failure_probability):
    """Smallest sufficient N from the independent exact-uniform variance bound.

    a = epsilon/(1+epsilon) handles the reciprocal volume estimator.
    Require ((1+1/N)**phases - 1) / a**2 <= failure_probability.
    This function budgets sampling noise ONLY; approximate uniformity is extra.
    """
    if isinstance(phases, bool) or not isinstance(phases, (int, np.integer)) or phases < 0:
        raise ValueError('phases must be a nonnegative integer')
    if not math.isfinite(relative_error) or relative_error <= 0:
        raise ValueError('relative_error must be finite and positive')
    if not math.isfinite(failure_probability) or not 0 < failure_probability < 1:
        raise ValueError('failure_probability must lie strictly between zero and one')
    if phases == 0:
        return 0
    a = relative_error/(1+relative_error)
    denominator = math.expm1(math.log1p(failure_probability*a*a)/phases)
    if denominator <= 0 or not math.isfinite(1/denominator):
        raise ValueError('Sample budget exceeds floating-point range')
    count = math.ceil(1/denominator)
    # Guard against rounding down at an integer boundary.
    if math.expm1(phases*math.log1p(1/count)) > failure_probability*a*a:
        count += 1
    return count


def sampling_plan(A, b, *, relative_error=.1, failure_probability=.25,
                  samples=None, steps=None, tv_distance=None, h=None,
                  experimental=False, max_attempts=None):
    """Plan independent samples and per-phase step budgets without drawing them.

    Theory mode splits failure_probability equally between ideal DFK sampling
    noise and joint TV error across all endpoints. Geometry and budgets use
    floating point; the guarantee assumes those geometric bounds are valid.
    Explicit experimental mode allows smaller budgets and caps, without a
    volume-accuracy guarantee. The algorithm still uses independent chains.
    """
    if not math.isfinite(failure_probability) or not 0 < failure_probability < 1:
        raise ValueError('failure_probability must lie strictly between zero and one')
    plan = volume_plan(A, b)
    m, d, r, R = plan['phases'], plan['dimension'], plan['inner_radius'], plan['outer_radius']
    statistical_budget = failure_probability/2
    required_samples = dfk_sample_count(m, relative_error, statistical_budget)
    for name, value, minimum in [('samples', samples, 1), ('steps', steps, 0)]:
        if value is not None and (isinstance(value, bool) or
                not isinstance(value, (int, np.integer)) or value < minimum):
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if max_attempts is not None and (isinstance(max_attempts, bool) or
            not isinstance(max_attempts, (int, np.integer)) or max_attempts < 1):
        raise ValueError('max_attempts must be a positive integer or None')
    if tv_distance is not None and (not math.isfinite(tv_distance) or not 0 < tv_distance < 1):
        raise ValueError('tv_distance must lie strictly between zero and one')
    if experimental and (samples is None or (steps is None and tv_distance is None)):
        raise ValueError('Experimental mode requires samples and either steps or tv_distance')
    if not experimental and max_attempts is not None:
        raise ValueError('Theory mode requires uncapped retries')
    samples = required_samples if samples is None else int(samples)
    if not experimental and samples < required_samples:
        raise ValueError(f'Require at least {required_samples} samples per phase, or explicit experimental mode')
    h = .1*r*r/d if h is None else float(h)
    if not math.isfinite(h) or h <= 0:
        raise ValueError('h must be finite and positive')
    total_samples = m*samples
    allowed_tv = failure_probability/(2*total_samples) if total_samples else None
    endpoint_tv = allowed_tv if tv_distance is None else tv_distance
    if not experimental and total_samples and endpoint_tv > allowed_tv:
        raise ValueError(f'Endpoint TV must be <= {allowed_tv:g} for the allocated joint error budget')
    phases = []
    for i, (inner, outer) in enumerate(zip(plan['radii'][:-1], plan['radii'][1:]), 1):
        # Every chain starts independently from the SAME known ball B(c,r).
        # K intersect B(c,outer) is inside B(c,min(outer,R)), giving these bounds.
        enclosing_radius = min(outer, R)
        diameter = 2*enclosing_radius
        log_M = d*math.log(enclosing_radius/r)
        required_steps = mixing_steps(h, diameter, log_M, endpoint_tv)
        phase_steps = required_steps if steps is None else int(steps)
        if not experimental and phase_steps < required_steps:
            raise ValueError(f'Phase {i} requires at least {required_steps} steps per endpoint; '
                             'use experimental mode for smaller budgets')
        phases.append(dict(phase=i, inner_radius=inner, outer_radius=outer,
                           diameter_bound=diameter, log_M_bound=log_M,
                           required_steps=required_steps, steps_per_sample=phase_steps))
    return dict(**plan, algorithm='dfk_independent_endpoints_v1',
                relative_error=relative_error, failure_probability=failure_probability,
                statistical_failure_budget=statistical_budget,
                sampling_tv_budget=failure_probability/2, endpoint_tv=endpoint_tv,
                required_samples_per_phase=required_samples, samples_per_phase=samples,
                total_samples=total_samples,
                total_proper_steps=samples*sum(p['steps_per_sample'] for p in phases),
                h=h, experimental=experimental, max_attempts=max_attempts, phase_plans=phases,
                guarantee=('No volume-error guarantee: explicit experimental budgets.' if experimental else
                  'Ideal-arithmetic bound: volume within factor 1+relative_error with failure at most '
                  'failure_probability, assuming valid numerical ball bounds and ideal independent randomness. '
                  'Floating-point implementation error is not certified.'))


def estimate_volume(A, b, *, relative_error=.1, failure_probability=.25,
                    samples=None, steps=None, tv_distance=None, h=None,
                    workers=1, seed=7, max_attempts=None, experimental=False,
                    prefetch_out=True, batch_in=True, batch_floor=32, progress=None):
    """Estimate volume with fresh independent chains within AND across phases.

    No thinning, shared endpoints, or phase-to-phase reuse occurs. One endpoint
    per chain is counted. See sampling_plan for the two theoretical budgets.
    progress(phase, phase_count, completed_samples, phase_samples) runs in parent.
    Failures propagate without retrying, replacing, or discarding failed chains.
    Use a __main__ guard for workers > 1 in scripts.
    """
    if isinstance(workers, bool) or not isinstance(workers, (int, np.integer)) or workers < 1:
        raise ValueError('workers must be a positive integer')
    if isinstance(batch_floor, bool) or not isinstance(batch_floor, (int, np.integer)) or batch_floor < 2:
        raise ValueError('batch_floor must be an integer >= 2')
    start = perf_counter()
    plan = sampling_plan(A, b, relative_error=relative_error, failure_probability=failure_probability,
                         samples=samples, steps=steps, tv_distance=tv_distance, h=h,
                         experimental=experimental, max_attempts=max_attempts)
    center = np.asarray(plan['center'])
    phase_seeds = np.random.SeedSequence(seed).spawn(plan['phases'])
    log_volume = plan['log_ball_volume']
    results = []
    for phase, child_seed in zip(plan['phase_plans'], phase_seeds):
        phase_start = perf_counter()
        def report(done, total):
            if progress is not None:
                progress(phase['phase'], plan['phases'], done, total)
        points, proposals = sample_endpoints(
            A, b, samples=plan['samples_per_phase'], steps=phase['steps_per_sample'],
            h=plan['h'], workers=workers, seed=child_seed, max_attempts=max_attempts,
            start_ball=(center, plan['inner_radius']), ball=(center, phase['outer_radius']),
            prefetch_out=prefetch_out, batch_in=batch_in, batch_floor=batch_floor,
            progress=report if progress is not None else None)
        hits = int(np.count_nonzero(np.linalg.norm(points-center, axis=1) <= phase['inner_radius']))
        if hits == 0:
            raise SamplingFailure(f"Phase {phase['phase']}: zero overlap hits; volume estimate failed. "
                                  'This run is not replaced or conditioned on success.')
        fraction = hits/plan['samples_per_phase']
        log_volume -= math.log(fraction)
        results.append(dict(**phase, hits=hits, samples=plan['samples_per_phase'], fraction=fraction,
                            steps=phase['steps_per_sample']*plan['samples_per_phase'],
                            proposals=int(proposals.sum()), seconds=perf_counter()-phase_start))
    return dict(**plan, volume=math.exp(log_volume), log_volume=log_volume,
                seconds=perf_counter()-start, seed=seed, workers=workers,
                prefetch_out=prefetch_out, batch_in=batch_in, batch_floor=batch_floor,
                phase_results=results)


def main():
    from demo import example_body, parse_max_attempts
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--body', choices=['box', 'simplex'], default='box')
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--relative-error', type=float, default=.1, help='Multiplicative volume tolerance epsilon')
    parser.add_argument('--failure-probability', type=float, default=.25)
    parser.add_argument('--samples', type=int, help='Override endpoints per phase (checked in theory mode)')
    parser.add_argument('--steps', type=int, help='Override proper steps per endpoint (checked in theory mode)')
    parser.add_argument('--tv-distance', type=float, help='Override TV per endpoint (checked in theory mode)')
    parser.add_argument('--experimental', action='store_true', help='Allow under-budget trials; no volume guarantee')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--h', type=float, help='Gaussian variance; heuristic default 0.1*r^2/d')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--max-attempts', type=parse_max_attempts, default=None)
    parser.add_argument('--batch-floor', type=int, default=32)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--output', type=Path)
    add_optimization_arguments(parser)
    args = parser.parse_args()
    if min(args.dim, args.repeats, args.workers) < 1 or args.batch_floor < 2:
        parser.error('Require positive dim/repeats/workers and batch-floor >= 2')
    settings = dict(relative_error=args.relative_error, failure_probability=args.failure_probability,
                    samples=args.samples, steps=args.steps, tv_distance=args.tv_distance, h=args.h,
                    experimental=args.experimental, max_attempts=args.max_attempts)
    record = dict(body=args.body, runs=[])
    try:
        A, b, _, _ = example_body(args.body, args.dim)
        plan = sampling_plan(A, b, **settings)
        record['plan'] = plan
        record['seed'] = args.seed
        record['workers'] = args.workers
        record['optimizations'] = optimization_options(args)
        print(f"{args.body}, d={args.dim}: {plan['phases']} phases; "
              f"{plan['samples_per_phase']:,} independent endpoints/phase; "
              f"{plan['total_proper_steps']:,} proper steps per volume estimate", flush=True)
        print(plan['guarantee'], flush=True)
        for phase in plan['phase_plans']:
            print(f"Phase {phase['phase']}: {phase['steps_per_sample']:,} steps/endpoint; "
                  f"TV target {plan['endpoint_tv']:.6g}", flush=True)
        if args.plan_only:
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(record, indent=2), encoding='utf-8')
            return
        exact = 2.**args.dim if args.body == 'box' else math.exp(-math.lgamma(args.dim+1))
        record['exact_volume'] = exact
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
        for repeat in range(args.repeats):
            last_report = -math.inf
            def report(phase, phases, done, total):
                nonlocal last_report
                now = perf_counter()
                if done == total or now-last_report >= 5:
                    print(f'Trial {repeat+1}/{args.repeats}, phase {phase}/{phases}: '
                          f'{done:,}/{total:,} endpoints complete', flush=True)
                    last_report = now
            result = estimate_volume(A, b, seed=args.seed+repeat, workers=args.workers,
                        batch_floor=args.batch_floor, progress=report,
                        **settings, **optimization_options(args))
            result['observed_relative_error'] = result['volume']/exact-1
            result['status'] = 'completed'
            record['runs'].append(result)
            print(f"Volume {result['volume']:.8g}; exact {exact:.8g}; "
                  f"error {result['observed_relative_error']:+.2%}; {result['seconds']:.2f}s", flush=True)
            if args.output:
                args.output.write_text(json.dumps(record, indent=2), encoding='utf-8')
    except (SamplingFailure, ValueError, OverflowError) as error:
        record['failure'] = str(error)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(record, indent=2), encoding='utf-8')
        parser.exit(1, f'Volume run stopped: {error}\n')


if __name__ == '__main__':
    main()
