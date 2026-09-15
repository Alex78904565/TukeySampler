"""Compare scalar, OUT-only prefetch, and IN-only batching on the 10D simplex.

Run: python -m experiments.benchmark_batching --repeats 5
BLAS uses one thread for all variants to avoid small-matrix thread overhead.
"""
import os
for variable in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS']:
    os.environ[variable] = '1'

import argparse
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter, process_time
import numpy as np
from demo import example_body
from geometry import uniform_ball
from experiments.mixing import example_geometry
from sampler import in_and_out

VARIANTS = ['scalar', 'prefetch_out', 'batch_in']


def plot(record, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(record['h_values']), figsize=(10, 4.8),
                             layout='constrained', squeeze=False)
    for ax, h in zip(axes[0], record['h_values']):
        for i, variant in enumerate(VARIANTS):
            values = [r['seconds'] for r in record['runs'] if r['h'] == h and r['variant'] == variant]
            if not values:
                continue
            ax.scatter([i]*len(values), values, color='#849aa6', alpha=.8)
            median = float(np.median(values))
            ax.plot([i-.18, i+.18], [median, median], color='#007f83', lw=3)
            ax.annotate(f'{median:.2f}s', (i, median), xytext=(8, 6), textcoords='offset points')
        ax.set(xticks=range(3), xticklabels=['Scalar', 'OUT prefetch\nonly', 'IN batches\nonly'],
               yscale='log', ylabel='Completed-run elapsed seconds', title=f'h = {h:g}', xlim=(-.5, 2.65))
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('10D simplex: batching changes tested separately')
    fig.supxlabel(f"{len(record['runs'])}/{len(record['h_values'])*record['repeats']*3} trials | points: individual runtimes; bars: medians\n"
                  'PW TV bound 1e-6; uncapped retries; one BLAS thread. IN batching changes later trajectories.', fontsize=9)
    fig.savefig(str(prefix)+'.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h-values', type=float, nargs='+', default=[5e-5, 1e-4])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=431)
    parser.add_argument('--output', default='batching_comparison')
    parser.add_argument('--batch-limit', type=int, default=1_048_576)
    args = parser.parse_args()
    if args.repeats < 1 or args.batch_limit < 1 or any(not math.isfinite(h) or h <= 0 for h in args.h_values):
        parser.error('Require positive repeats and finite positive h')
    hs = sorted(set(args.h_values))
    A, b, _, _ = example_body('simplex', 10)
    center, radius, _, log_M = example_geometry('simplex', 10)
    numerator = log_M + math.log1p(-math.exp(-log_M)) - math.log(4) - 2*math.log(1e-6)
    prefix = Path(args.output)
    record = dict(h_values=hs, repeats=args.repeats, seed=args.seed, variants=VARIANTS,
                  blas_threads=1, batch_limit=args.batch_limit, requested_tv=1e-6, max_attempts=None,
                  timing='full sampler call including prefetched draws and allocations; plotting excluded',
                  rng='separate start, OUT, and IN streams; scalar and prefetch paths match; IN leftovers discarded', runs=[])
    order = np.random.default_rng([args.seed, 999])
    for repeat in range(args.repeats):
        jobs = [(index, v) for index in range(len(hs)) for v in VARIANTS]
        order.shuffle(jobs)
        for index, variant in jobs:
            h = hs[index]
            seed = [args.seed, index, repeat]
            x0 = uniform_ball(center, radius, np.random.default_rng(seed+[0]))
            out_rng = np.random.default_rng(seed+[1])
            rng = np.random.default_rng(seed+[2])
            steps = math.ceil(numerator/math.log1p(h*math.pi**2/2))
            label = f'[{variant}, h={h:g}, trial={repeat+1}]'
            print(f'{label} starting {steps:,} steps', flush=True)
            start, cpu_start = perf_counter(), process_time()
            next_report = start + 10
            def report(done, proposals, rejected):
                nonlocal next_report
                now = perf_counter()
                if now >= next_report:
                    print(f'{label} {done:,}/{steps:,} | {now-start:.1f}s | {proposals:,} proposals | current rejects {rejected:,}', flush=True)
                    next_report = now + 10
            stats = {}
            path, attempts = in_and_out(A, b, x0, h=h, steps=steps, max_attempts=None,
                                       rng=rng, out_rng=out_rng, prefetch_out=variant=='prefetch_out',
                                       batch_in=variant=='batch_in', batch_limit=args.batch_limit, batch_floor=2,
                                       diagnostics=stats, progress=report)
            seconds, cpu_seconds = perf_counter()-start, process_time()-cpu_start
            assert np.all(path @ A.T <= b)
            row = dict(variant=variant, h=h, repeat=repeat+1, seed=seed,
                       steps=steps, seconds=seconds, cpu_seconds=cpu_seconds,
                       proposals_to_first_success=int(attempts.sum()),
                       generated_inward=stats['generated_inward'], unused_inward=stats['unused_inward'],
                       endpoint=path[-1].tolist(), path_hash=hashlib.sha256(path.tobytes()).hexdigest())
            record['runs'].append(row)
            if variant != 'batch_in':
                other = [r for r in record['runs'] if r['h']==h and r['repeat']==repeat+1
                         and r['variant'] not in [variant, 'batch_in']]
                if other:
                    assert other[0]['path_hash'] == row['path_hash']
                    assert other[0]['generated_inward'] == row['generated_inward']
            print(f"{label} completed {seconds:.3f}s; generated {stats['generated_inward']:,} IN vectors", flush=True)
            Path(str(prefix)+'.json').write_text(json.dumps(record, indent=2))
            plot(record, prefix)
    print('MEDIANS', flush=True)
    for h in hs:
        for v in VARIANTS:
            rows = [r for r in record['runs'] if r['h']==h and r['variant']==v]
            print(h, v, np.median([r['seconds'] for r in rows]), flush=True)


if __name__ == '__main__':
    main()
