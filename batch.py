"""Draw independent endpoints: python batch.py --samples 100 --workers 4."""

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing
import math
import os
from time import perf_counter

_stop = None


def _initialize_worker(stop):
    global _stop
    _stop = stop
    # Limit libraries loaded after this initializer. A calling script that
    # imports NumPy before spawning may already have initialized its BLAS pool.
    # --workers controls chain processes, not a guarantee about BLAS threads.
    for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
                 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '1'


def _endpoint(job):
    import numpy as np
    from geometry import uniform_ball
    from sampler import in_and_out, SamplingFailure

    A, b, center, radius, h, steps, seed, options = job
    rng = np.random.default_rng(seed)
    x = uniform_ball(center, radius, rng)
    proposals = 0
    def check_cancelled(*counts):
        if _stop is not None and _stop.is_set():
            raise SamplingFailure('Batch cancelled because another run failed or was interrupted.')
    # Keep memory bounded per worker; continue the same chain across chunks.
    for offset in range(0, steps, 1024):
        path, attempts = in_and_out(A, b, x, h=h, steps=min(1024, steps-offset),
                                    rng=rng, progress=check_cancelled if _stop is not None else None,
                                    **options)
        x = path[-1].copy()
        proposals += int(attempts.sum())
    return x, proposals


def sample_endpoints(A, b, *, samples, steps, h, workers=1, seed=7,
                     max_attempts=None, prefetch_out=True, batch_in=True,
                     batch_floor=32, batch_limit=1_048_576, ball=None,
                     start_ball=None, progress=None):
    """Return (endpoints, proposal_counts), ordered by sample index.

    Each endpoint uses an independent uniform inner-ball start and a spawned
    random stream. The seed and sample index determine the stream, independent
    of worker count or scheduling. Geometry is solved once in the parent.
    progress(completed, total) runs in the parent, including every five seconds
    while waiting for parallel jobs. Failures propagate without replacing runs.
    ball restricts the target to K intersect B(center, radius). start_ball
    optionally supplies already computed inscribed geometry, which is checked
    for containment in both targets. No previous chain endpoint is reused.
    Call from a __main__ guard in scripts when workers > 1 (Windows spawn).
    """
    import numpy as np
    from geometry import enclosing_balls

    for name, value, minimum in (('samples', samples, 1), ('steps', steps, 0),
                                 ('workers', workers, 1)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if not np.isfinite(h) or h <= 0:
        raise ValueError('h must be finite and positive')
    if max_attempts is not None and (isinstance(max_attempts, bool) or
            not isinstance(max_attempts, (int, np.integer)) or max_attempts < 1):
        raise ValueError('max_attempts must be a positive integer or None')
    for name, value, minimum in (('batch_floor', batch_floor, 2), ('batch_limit', batch_limit, 1)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    root_seed = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    seeds = root_seed.spawn(samples)
    A, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
    if A.ndim != 2 or min(A.shape) == 0 or b.shape != (len(A),):
        raise ValueError('Expected A with shape (m, d) and b with shape (m,)')
    if not np.isfinite(A).all() or not np.isfinite(b).all():
        raise ValueError('A and b must be finite')
    if start_ball is None:
        center, radius, _ = enclosing_balls(A, b)
    else:
        center, radius = np.asarray(start_ball[0], dtype=float), float(start_ball[1])
    if ball is not None:
        target_center, target_radius = np.asarray(ball[0], dtype=float), float(ball[1])
        if (target_center.shape != (A.shape[1],) or not np.isfinite(target_center).all()
                or not np.isfinite(target_radius) or target_radius <= 0):
            raise ValueError('Invalid target ball')
        if start_ball is None:
            radius = min(radius, target_radius-np.linalg.norm(center-target_center))
    if (center.shape != (A.shape[1],) or not np.isfinite(center).all()
            or not np.isfinite(radius) or radius <= 0):
        raise ValueError('Invalid starting ball')
    if np.any(A @ center + radius*np.linalg.norm(A, axis=1) > b):
        raise ValueError('Starting ball must be contained in the polytope')
    if ball is not None and np.linalg.norm(center-target_center)+radius > target_radius:
        raise ValueError('Starting ball must be contained in the target ball')
    options = dict(max_attempts=max_attempts, prefetch_out=prefetch_out, batch_in=batch_in,
                   batch_floor=batch_floor, batch_limit=batch_limit, ball=ball)
    def job(i):
        return A, b, center, radius, h, steps, seeds[i], options
    endpoints = np.empty((samples, len(center)))
    proposals = np.empty(samples, dtype=np.int64)
    completed = 0

    if workers == 1:
        for i in range(samples):
            endpoints[i], proposals[i] = _endpoint(job(i))
            if progress is not None:
                progress(i+1, samples)
    else:
        context = multiprocessing.get_context('spawn')
        stop = context.Event()
        with ProcessPoolExecutor(max_workers=min(workers, samples),
                mp_context=context, initializer=_initialize_worker, initargs=(stop,)) as pool:
            # Bound queued copies of geometry even for large DFK sample counts.
            next_index = min(samples, 2*workers)
            pending = {pool.submit(_endpoint, job(i)): i for i in range(next_index)}
            try:
                while pending:
                    done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                    for future in done:
                        i = pending.pop(future)
                        endpoints[i], proposals[i] = future.result()
                        completed += 1
                        if next_index < samples:
                            pending[pool.submit(_endpoint, job(next_index))] = next_index
                            next_index += 1
                    if progress is not None:
                        progress(completed, samples)
            except BaseException:
                stop.set()
                for future in pending:
                    future.cancel()
                raise
    return endpoints, proposals


def main():
    import numpy as np
    from demo import example_body, parse_max_attempts
    from geometry import enclosing_balls
    from mixing import mixing_steps
    from sampler import SamplingFailure, add_optimization_arguments, optimization_options

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--body', choices=('box', 'simplex'), default='simplex')
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--workers', type=int, default=1)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument('--steps', type=int)
    budget.add_argument('--tv-distance', type=float, help='TV target for EACH endpoint')
    parser.add_argument('--h', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--max-attempts', type=parse_max_attempts, default=None)
    parser.add_argument('--save', default='endpoints.npz')
    parser.add_argument('--plan-only', action='store_true')
    add_optimization_arguments(parser)
    args = parser.parse_args()
    if min(args.dim, args.samples, args.workers) < 1:
        parser.error('dim, samples, and workers must be positive')
    if args.tv_distance is not None and args.max_attempts is not None:
        parser.error('TV mode requires uncapped retries')
    try:
        A, b, _, _ = example_body(args.body, args.dim)
        # Use bounds for the actual numerical ball used by sample_endpoints.
        # This also populates the geometry cache before starting the workers.
        _, radius, _ = enclosing_balls(A, b)
        log_volume = args.dim*math.log(2) if args.body == 'box' else -math.lgamma(args.dim+1)
        log_ball = args.dim/2*math.log(math.pi) - math.lgamma(args.dim/2+1) + args.dim*math.log(radius)
        log_M = max(0., log_volume-log_ball)
        diameter = (2*math.sqrt(args.dim) if args.body == 'box'
                    else (math.sqrt(2) if args.dim > 1 else 1.))
        steps = (mixing_steps(args.h, diameter, log_M, args.tv_distance)
                 if args.tv_distance is not None else (20000 if args.steps is None else args.steps))
        if steps < 0 or not np.isfinite(args.h) or args.h <= 0:
            raise ValueError('Require steps >= 0 and finite h > 0')
        print(f'{args.samples} independent endpoints; {steps:,} steps each; '
              f'{min(args.workers, args.samples)} workers', flush=True)
        if args.tv_distance is not None:
            print('TV target is per endpoint; ideal uncapped-chain bound with numerical ball geometry.', flush=True)
        if args.plan_only:
            return
        started = perf_counter()
        def report(done, total):
            print(f'{done}/{total} endpoints completed; elapsed {perf_counter()-started:.1f}s', flush=True)
        points, proposals = sample_endpoints(A, b, samples=args.samples, steps=steps,
                h=args.h, workers=args.workers, seed=args.seed, max_attempts=args.max_attempts,
                progress=report, **optimization_options(args))
        seconds = perf_counter()-started
        np.savez(args.save, samples=points, proposals=proposals, A=A, b=b,
                 steps=steps, h=args.h, seed=args.seed, workers=args.workers,
                 tv_distance=np.nan if args.tv_distance is None else args.tv_distance,
                 log_M=log_M, diameter=diameter, seconds=seconds,
                 max_attempts=np.inf if args.max_attempts is None else args.max_attempts,
                 **optimization_options(args))
        print(f'Saved {args.save}; total wall time {seconds:.2f}s', flush=True)
    except (ValueError, SamplingFailure) as error:
        parser.exit(1, f'Batch stopped: {error}\n')


if __name__ == '__main__':
    main()
