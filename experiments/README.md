# Experiments

Run these modules from the repository root with the project environment active.
Results are generated locally and are not included in Git. Each command supports
`--help` for its full set of options.

## Endpoint timing

```sh
python -m experiments.run_payne_weinberger --h 1e-4
python -m experiments.benchmark_payne_weinberger --h-values 5e-5 1e-4 --repeats 3 --output pw_sweep
```

These programs sample the standard 10D simplex from a uniform inner-ball start.
They calculate an endpoint step budget using `C_P = 2/pi^2`, target
`epsilon = 1e-6`, and warmness M:

```text
k = ceil(log((M - 1) / (4 * epsilon^2)) / log(1 + h / C_P)).
```

This budget concerns the ideal endpoint law, not independence of intermediate
states or a certificate for numerical implementation error. Trials run serially
with uncapped retries. The sweep saves completion times and final endpoints.

`benchmark_h` preserves the original, older geometric budget for reproducing
historical experiments, with per-trial time limits. It does not use the
Payne-Weinberger budget in `demo.py --tv-distance`:

```sh
python -m experiments.benchmark_h --plan-only
python -m experiments.benchmark_h --h-values 5e-5 1e-4 --seconds 10 --repeats 3 --output pilot
```

`--seconds 0` removes the time limit. Interrupted trials are recorded as
unfinished. `--progress-every` controls reporting. The formula is defined in
`experiments/mixing.py`; D is the same-center enclosing radius and `--log-M`
means the natural logarithm of warmness.

## Gaussian batching and batch floors

```sh
python -m experiments.benchmark_batching --repeats 5 --output batching_comparison_1m
python -m experiments.benchmark_floors --floors 2 8 16 32 64 128 256 --output in_floor_comparison_extended
python -m plotting.plot_advisor_optimizations
```

The batching experiment compares scalar, OUT-prefetch-only, and IN-batching-only
variants. The floor experiment varies the second IN batch size while leaving
OUT prefetch disabled. Both use one BLAS thread and shuffled serial trial order.
Timings include sampling and noise generation, excluding initialization and plots.

Changing IN batches changes which random draws are discarded and can change
subsequent paths. Compare repeated completed timings, accounting for rare long
rejection streaks. Medians and pilot projections are not expected-runtime guarantees.
The comparison plot reads the two JSON files named in the commands above.

## Visualization helpers

- `python -m experiments.example_10d` generates 1,000 independent uniform
  simplex points using Dirichlet weights, plus reference plots and an animation.
  This illustrates the target distribution rather than running in-and-out.
- `python -m plotting.plot_pw_density --help` describes how to plot endpoint-sweep
  JSON. It also saves an endpoint NPZ archive; its optional animation accumulates
  trial endpoints, not proper steps. Runtime sweeps save static endpoint plots by
  default. Use trajectory plotting to animate evolution along proper steps.
- `python -m plotting.plot_final_endpoints archive.npz` draws a static scatter,
  heatmap, and exact target comparison from that endpoint archive.
- `python -m experiments.compare_h first.json second.json` compares compatible
  pilots using constant-rate completion-time projections. These are projections,
  not measurements of completed runs.

Trajectory plotting is documented in the [main README](../README.md).
