"""Sample the standard 10D simplex using the Payne-Weinberger TV step bound.

Example: python -m experiments.run_payne_weinberger --h 5e-5 --seed 53
The guarantee concerns the final endpoint of the ideal uncapped chain.
Earlier saved states are correlated diagnostic data, not independent samples.
"""

import argparse
import json
import math
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

from demo import example_body
from geometry import uniform_ball, example_geometry
from mixing import mixing_steps
from sampler import in_and_out, add_optimization_arguments, optimization_options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h', type=float, default=5e-5)
    parser.add_argument('--seed', type=int, default=53)
    parser.add_argument('--output', default='payne_weinberger_10d')
    add_optimization_arguments(parser)
    args = parser.parse_args()
    options = optimization_options(args)
    if not math.isfinite(args.h) or args.h <= 0:
        parser.error('h must be finite and positive')

    d, epsilon = 10, 1e-6
    center, radius, enclosing_radius, log_M = example_geometry('simplex', d)
    diameter_squared = 2.0  # Largest vertex separation: ||e_i-e_j||^2.
    C_P = diameter_squared / math.pi**2
    steps = mixing_steps(args.h, math.sqrt(diameter_squared), log_M, epsilon)
    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(body='simplex', dimension=d, h=args.h, seed=args.seed,
                    requested_tv=epsilon, required_steps=steps, log_M=log_M,
                    inner_radius=radius, enclosing_radius=enclosing_radius,
                    diameter_squared=diameter_squared, poincare_upper_bound=C_P,
                    formula='ceil(ln((M-1)/(4*epsilon^2))/ln(1+h*pi^2/diameter^2))',
                    max_attempts=None, status='running', optimizations=options, batch_floor=32, batch_limit=1048576,
                    interpretation='TV bound for final endpoint of ideal uncapped chain; full path is diagnostic data')
    Path(str(prefix)+'.json').write_text(json.dumps(metadata, indent=2))
    print(f'10D simplex: h={args.h:g}, C_P <= {C_P:.9g}, k={steps:,}', flush=True)
    print('Uniform inner-ball start; uncapped retries; target TV 1e-6.', flush=True)
    A, b, _, _ = example_body('simplex', d)
    rng = np.random.default_rng(args.seed)
    start, cpu_start = perf_counter(), process_time()
    x0 = uniform_ball(center, radius, rng)
    next_report = start + 5

    def progress(done, proposals, current_rejections):
        nonlocal next_report
        now = perf_counter()
        if now >= next_report:
            print(f'{done:,}/{steps:,} steps ({100*done/steps:.1f}%) | '
                  f'{now-start:.1f}s | {proposals:,} proposals | '
                  f'current-step rejections {current_rejections:,}', flush=True)
            next_report = now + 5

    path, attempts = in_and_out(A, b, x0, h=args.h, steps=steps,
                               max_attempts=None, rng=rng, progress=progress, **options)
    seconds, cpu_seconds = perf_counter()-start, process_time()-cpu_start
    metadata.update(status='completed', completed_steps=len(attempts),
                    inward_proposals=int(attempts.sum()),
                    rejections=int(attempts.sum()-len(attempts)),
                    largest_attempt_count=int(attempts.max()),
                    seconds=seconds, cpu_seconds=cpu_seconds,
                    endpoint=path[-1].tolist(),
                    all_states_feasible=bool(np.all(path @ A.T <= b)))
    np.savez_compressed(str(prefix)+'.npz', path=path, attempts=attempts,
                        A=A, b=b, h=args.h, burn_in=0, seed=args.seed,
                        start='ball', max_attempts=np.inf,
                        center=center, r=radius, R=enclosing_radius, **options, batch_floor=32, batch_limit=1048576)
    Path(str(prefix)+'.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
