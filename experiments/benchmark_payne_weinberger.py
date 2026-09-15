"""Serial repeated runtime sweep for the 10D simplex using the PW TV bound."""

import argparse
import json
import math
from pathlib import Path
from time import process_time

import numpy as np

from experiments.benchmark_h import timed_run
from demo import example_body
from experiments.mixing import example_geometry
from sampler import add_optimization_arguments, optimization_options


def plot_runs(record, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout='constrained')
    summary = []
    for index, h in enumerate(record['h_values']):
        rows = [r for r in record['runs'] if r['h'] == h]
        if not rows:
            continue
        elapsed = [r['seconds'] for r in rows]
        cpu = [r['cpu_seconds'] for r in rows]
        summary.append(dict(h=h, trials=len(rows), required_steps=rows[0]['required_steps'],
                            median_seconds=float(np.median(elapsed)),
                            min_seconds=min(elapsed), max_seconds=max(elapsed)))
        for ax, values in zip(axes, [elapsed, cpu]):
            ax.scatter([h]*len(values), values, s=35, color='#8299a5', alpha=.8,
                       label='Individual completed trials' if index == 0 else None)
    for ax, key, label in zip(axes, ['seconds', 'cpu_seconds'], ['Elapsed seconds', 'CPU seconds']):
        hs = sorted({r['h'] for r in record['runs']})
        medians = [np.median([r[key] for r in record['runs'] if r['h'] == h]) for h in hs]
        ax.plot(hs, medians, 'o-', color='#007f83', label='Median')
        ax.set(xscale='log', yscale='log', xlabel='Gaussian variance h', ylabel=label)
        ax.set_xticks(record['h_values'], [f'{h:g}' for h in record['h_values']])
        ax.minorticks_off()
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('10D simplex: time to the Payne-Weinberger TV step bound')
    fig.supxlabel(f"TV target 1e-6 | uncapped retries | {len(record['runs'])}/{len(record['h_values'])*record['repeats']} completed trials\n"
                  'Measured completion times, not extrapolations. Medians do not establish expected runtime.', fontsize=9)
    fig.savefig(str(prefix)+'.png', dpi=180)
    plt.close(fig)
    record['summary'] = summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h-values', nargs='+', type=float, default=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=137)
    parser.add_argument('--output', default='pw_h_sweep')
    parser.add_argument('--resume', help='Completed results to retain; unfinished trial restarts from its original seed')
    add_optimization_arguments(parser)
    args = parser.parse_args()
    options = optimization_options(args)
    if args.repeats < 1 or any(not math.isfinite(h) or h <= 0 for h in args.h_values):
        parser.error('Require positive repeats and finite positive h values.')
    hs = sorted(set(args.h_values))
    center, radius, D, log_M = example_geometry('simplex', 10)
    C_P = 2/math.pi**2
    numerator = log_M + math.log1p(-math.exp(-log_M)) - math.log(4) - 2*math.log(1e-6)
    plan = {h: math.ceil(numerator/math.log1p(h/C_P)) for h in hs}
    A, b, _, _ = example_body('simplex', 10)
    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    record = dict(body='simplex', dimension=10, requested_tv=1e-6,
                  log_M=log_M, inner_radius=radius, enclosing_radius=D,
                  diameter_squared=2, poincare_upper_bound=C_P,
                  formula='ceil(ln((M-1)/(4*epsilon^2))/ln(1+h*pi^2/2))',
                  h_values=hs, repeats=args.repeats, base_seed=args.seed,
                  max_attempts=None, time_limit_seconds=0,
                  execution='serial, shuffled h order within each repeat',
                  timing='initialization and chunk bookkeeping included; saving and plotting excluded',
                  optimizations=options, batch_floor=32, batch_limit=1048576, runs=[])
    if args.resume:
        previous = json.loads(Path(args.resume).read_text())
        if previous.get('optimizations', dict(prefetch_out=False, batch_in=False)) != options:
            parser.error('Resume optimization settings differ. For old scalar results use --no-prefetch-out --no-batch-in.')
        for key in ['body', 'dimension', 'h_values', 'base_seed', 'formula', 'log_M', 'max_attempts', 'requested_tv']:
            if previous[key] != record[key]:
                parser.error(f'Resume settings differ: {key}')
        record['runs'] = previous['runs']
        if any(r['repeat'] > args.repeats for r in record['runs']):
            parser.error('Requested repeats must include every saved trial')
        record['resumed_from'] = str(Path(args.resume).resolve())
        record['interrupted_trial_restarted'] = previous.get('current_trial')
    finished = {(r['h'], r['repeat']) for r in record['runs']}
    if len(finished) != len(record['runs']):
        parser.error('Duplicate saved trials')
    order_rng = np.random.default_rng([args.seed, 987])
    for repeat in range(args.repeats):
        for index in order_rng.permutation(len(hs)):
            index = int(index)
            h = hs[index]
            if (h, repeat+1) in finished:
                continue
            seed = [args.seed, index, repeat]
            label = f'[h={h:g}, trial={repeat+1}/{args.repeats}]'
            print(f'{label} starting {plan[h]:,} proper steps', flush=True)
            record['current_trial'] = dict(h=h, repeat=repeat+1, seed=seed)
            Path(str(prefix)+'.json').write_text(json.dumps(record, indent=2))
            cpu_start = process_time()
            row = timed_run(A, b, center, radius, h, plan[h], seed, 0, None,
                            progress_every=5, progress_label=label, **options)
            row.update(cpu_seconds=process_time()-cpu_start, h=h, seed=seed,
                       repeat=repeat+1, required_steps=plan[h])
            assert row['status'] == 'completed' and row['completed_steps'] == plan[h]
            assert np.all(A @ np.asarray(row['endpoint']) <= b)
            row['rejections'] = row['inward_proposals'] - row['completed_steps']
            record['runs'].append(row)
            record.pop('current_trial')
            print(f"{label} completed in {row['seconds']:.3f}s; {row['inward_proposals']:,} proposals", flush=True)
            plot_runs(record, prefix)
            Path(str(prefix)+'.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record['summary'], indent=2), flush=True)
    from plotting.plot_pw_density import plot_density
    plot_density(record, prefix, animate=True)


if __name__ == '__main__':
    main()
