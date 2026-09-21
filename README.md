# TukeySampler

A Python toolkit for polytope sampling and volume estimation, built toward
**Tukey-depth weighted sampling**. It implements uniform in-and-out sampling,
inner-ball warm starts, DFK volume estimation, and density visualizations.
Tukey-depth weighting is not implemented yet.

## Setup

Use Python 3.10 or newer. From the repository folder:

```sh
python -m venv .venv
```

Activate with `.\.venv\Scripts\Activate.ps1` in Windows PowerShell, or
`source .venv/bin/activate` on macOS/Linux, then install dependencies:

```sh
python -m pip install -r requirements.txt
```

## Sample a polytope

```sh
python demo.py --body box --steps 20000 --burn-in 2000 --save run.npz --plot density.png
python demo.py --body simplex --dim 10 --steps 30000 --burn-in 3000 --plot simplex.png
python demo.py --body polygon --vertices 9 --burn-in 0 --animate polygon.gif
```

The target is uniform on `K = {x : A @ x <= b}`. The in-and-out method [1]
performs each proper transition as follows:

1. Draw an OUT point from `Normal(x, h * I)`.
2. Hold that point fixed and draw IN proposals around it with the same variance.
3. Accept the first proposal inside K as the next state.

| Option | Meaning |
| --- | --- |
| `--steps` | Total proper transitions, including burn-in; default `20000` |
| `--burn-in` | Initial transitions omitted from plots and analysis; default `2000` |
| `--h` | Per-coordinate Gaussian variance; default `0.1 * r^2 / d^2` |
| `--max-attempts` | IN proposal cap per transition; default `100000`, or `inf` |
| `--seed` | Random seed; default `7` |
| `--save` | Save the path, proposal counts, and settings as an NPZ archive |

Here r is the inscribed-ball radius and d is the dimension. These default
budgets are exploratory; they do not establish a prescribed mixing accuracy.
Exhausting the retry cap stops the run. Unlimited retries can take arbitrarily
long. Retained states from a trajectory are correlated.

OUT noise prefetch and IN proposal batching are enabled by default. Disable them
with `--no-prefetch-out` and `--no-batch-in`. IN batches contain 1, 32, 64, ...
proposals, up to 1,048,576 rows. The first feasible row is accepted. This matrix
size cap is separate from the retry cap.

### Choose an endpoint TV target

```sh
python demo.py --body simplex --dim 10 --h 1e-4 --tv-distance 1e-6 --plan-only
python demo.py --body simplex --dim 10 --h 1e-4 --tv-distance 1e-6 --save run.npz
```

`--tv-distance` replaces `--steps`. The first command prints the budget without
sampling; the second runs it, reporting progress about every five seconds.
For this example the budget is 72,924 proper transitions.

The calculation combines the chi-squared contraction in [1, Theorem 14],
`chi_squared(initial) <= M - 1`, `TV <= sqrt(chi_squared)/2`, and the
Payne-Weinberger inequality `C_PI <= D^2 / pi^2` for a convex body:

```text
k = ceil((log(M - 1) - 2*log(2*tv_distance)) / log(1 + h*pi^2/D^2))
```

Here D is the diameter and M is the warmness of the initial distribution.
Automatic mode supports boxes and standard simplices, using their analytic
volumes, diameters, and uniform inner-ball starts, with `M = vol(K)/vol(ball)`.
It requires a ball start and uncapped IN retries, which it selects by default.
The formula gives zero steps when the initial bound already meets the target.
Geometry and arithmetic use floating point rather than certified arithmetic.

The target applies to the **final endpoint**, not to the pooled trajectory or
its plotted density. Burn-in defaults to zero in this mode and only controls
summaries and plots; it does not add transitions to k. The full path and OUT
noise are stored in memory, so inspect large budgets with `--plan-only` first.
Manual step mode keeps its 20,000-step, 2,000-burn-in, 100,000-retry defaults.

For a custom convex body, use `mixing.mixing_steps(h, diameter, log_M, tv_distance)`
with valid diameter and initial-warmness bounds, then pass the result to
`in_and_out(..., steps=k, max_attempts=None)`. A fixed initial point is not an
M-warm start for a continuous uniform target.

### Shapes and plots

Boxes and standard simplices support arbitrary dimensions. The `polygon`,
`ellipse`, and `random` bodies are 2D polygons; `--vertices` controls the number
of input points. `--points points.csv` takes the convex hull of headerless x,y
rows. Input points must span a nonzero area.

`--plot` produces a density figure and `--animate` produces a GIF. In higher
dimensions, coordinate pairs have empirical and exact uniform-target histograms;
an unpaired coordinate gets a 1D density line. Exact target comparisons support
boxes and standard simplices. The 2D views use Gaussian kernels restricted to
the polygon. Plots illustrate exploration and do not certify joint mixing.

To visualize a saved path:

```sh
python -m plotting.plot_samples run.npz --output evolution.gif --burn-in 0
```

### Draw independent endpoints in parallel

```sh
python batch.py --body simplex --dim 10 --samples 100 --h 1e-4 --tv-distance 1e-6 --workers 4 --plan-only
python batch.py --body simplex --dim 10 --samples 100 --h 1e-4 --tv-distance 1e-6 --workers 4 --save endpoints.npz
```

Each sample is the final endpoint of its own chain, with a fresh uniform
Chebyshev-ball start and a separate NumPy random stream. Geometry is solved once
in the parent process. `--workers 4` runs up to four chains at once using worker
processes; `--workers 1` (the default) runs serially. The same seed gives the same
sample streams and output order regardless of scheduling. Both sampling
optimizations remain enabled and can be disabled with the usual switches.

Use `--steps 30000` instead of `--tv-distance` for a manual budget. The TV target
applies to **each endpoint**, not the entire batch. A conservative joint target
of eta can be obtained by setting the per-endpoint target to eta / samples.
Automatic budgets use the exact built-in body volume and diameter together with
the numerical ball actually used for initialization. IN retries default to
uncapped; retry caps are allowed only in manual mode. Failures stop the batch.

The NPZ contains `samples` with shape `(samples, dim)`, proposal totals per
endpoint, settings, and total sampling wall time. Full trajectories are not
saved; chains run in chunks of 1,024 steps to bound per-worker path memory.
Chunking preserves the transition rule but changes random-number consumption
relative to `demo.py`. Progress reports completed endpoints, with a heartbeat
every five seconds while parallel jobs run. More workers use more memory and
may not help small batches because process startup has a cost.

The API is `batch.sample_endpoints(A, b, samples=100, steps=k, h=h, workers=4)`.
Place calls inside `if __name__ == '__main__':` in scripts using multiple
workers, as required by Windows process spawning. This batch mode produces
independent endpoints; it does not parallelize consecutive steps of one chain
or parallelize consecutive steps of a chain. DFK in `volume.py` uses this
independent-endpoint API and supports `--workers` within each phase.

## Use the Python API

```python
import numpy as np
from geometry import enclosing_balls, uniform_ball
from sampler import in_and_out

A = np.vstack([np.eye(2), -np.eye(2)])
b = np.ones(4)
center, r, R = enclosing_balls(A, b)
rng = np.random.default_rng(7)
x0 = uniform_ball(center, r, rng)

path, attempts = in_and_out(A, b, x0, h=0.025, steps=20000, rng=rng)
samples = path[2001:]  # Omit x0 and the first 2,000 transitions.
```

A has shape `(m, d)` and b has shape `(m,)`. The body must be bounded with
nonempty interior. The inner-ball start is uniform in an inscribed ball found
by linear programming; enclosing radii are conservative floating-point estimates.
Solved geometry is cached for the 16 most recent constraint sets within a Python
process, including repeated DFK runs. Changing the constraint values triggers a
new solve. Each warm-start draw uses fresh randomness in the cached ball; only
the geometry is reused. Separate command-line processes have separate caches.

`path` contains `steps + 1` states, including x0. `attempts` counts IN proposals
through each first success, excluding surplus rows generated by batching.
Optional `diagnostics` records the actual number of generated IN vectors.
`batch_floor`, `batch_limit`, and both optimization switches are API parameters.

## Estimate volume

`volume.py` follows the expanding-body estimator in Lee and Vempala [2,
Algorithm 35]. Every phase draws **independent chain endpoints**, with a fresh
uniform inscribed-ball start and fresh random stream for each endpoint. Different
phases also use separate streams. Previous endpoints are not reused, and no
thinned trajectory is treated as an independent sample collection.

```sh
python volume.py --body simplex --dim 10 --h 1e-4 --relative-error .1 --failure-probability .25 --workers 4 --plan-only
python volume.py --body box --dim 2 --relative-error .5 --workers 4 --output volume_box.json
```

Inspect `--plan-only` first: sufficient theoretical budgets can be very large.
The first command only prints the plan. `--relative-error epsilon` targets the
interval `[V/(1+epsilon), (1+epsilon)*V]`. The failure probability includes both
finite-sample noise and approximate-uniform sampling error. `--workers` runs
independent chains in parallel within each phase; phase execution order does
not create dependence because the starts and parameters do not use earlier data.
`--repeats` runs separate complete estimates. Geometry is cached, not sampled anew.

### DFK budgets and their assumptions

Let `m = ceil(d*log2(R/r))`, and `K_i = K intersect B(c,r*2**(i/d))`.
The overlap `p_i = vol(K_i)/vol(K_(i+1))` is at least 1/2 by convexity.
Algorithm 35 estimates each ratio by the fraction of endpoints in the smaller
body, then returns the known initial-ball volume divided by their product.
A zero product is a failed estimate, never silently replaced or retried.

The independent, exactly uniform reference experiment in [2, Lemmas 11.4–11.6]
has relative product variance at most `(1+1/N)**m - 1`. For final volume
factor `1+epsilon`, use product tolerance `a = epsilon/(1+epsilon)` to handle
the reciprocal. The implemented sufficient sample count solves Chebyshev's
inequality without the asymptotic notation in Theorem 11.7:

```text
statistical_budget = failure_probability / 2
sampling_budget = failure_probability / 2
a = epsilon / (1 + epsilon)
N = ceil(1 / expm1(log1p(statistical_budget * a*a) / m))
endpoint_tv = sampling_budget / (m*N)
```

The remaining sampling error is explicit: independent approximate endpoints
have joint TV at most the sum of their individual TV errors, at most
`m*N*endpoint_tv`. Thus ideal-estimator failure plus sampling error is at most
the requested failure probability. This extra error allocation is our stated
implementation of approximate sampling, not a constant quoted from Theorem 11.7.
For `m=0` the known ball volume is returned without sampling.

Each endpoint starts independently in the original ball `B(c,r)`. At a phase
with outer radius `s`, use `log(M) <= d*log(min(s,R)/r)` and diameter bound
`D <= 2*min(s,R)` in `mixing_steps`. The warmness is **not** assumed to be two:
that would require a uniform start from the preceding body, which we do not
have for these independent starts. Default retries are uncapped. The bound
assumes valid geometry, exact arithmetic, and ideal independent randomness;
LP solutions, membership tests, Gaussian draws, and budgets use floating point
and pseudorandom generators. This is not an interval-arithmetic certification.

### Explicit experimental budgets

For small performance trials or weaker accuracy experiments:

```sh
python volume.py --body box --dim 2 --experimental --samples 100 --steps 100 --workers 4 --output volume_pilot.json
python volume.py --body box --dim 2 --experimental --samples 100 --tv-distance .1 --h .01 --plan-only
```

These still use independent chains, but make **no volume-error guarantee**.
Without `--experimental`, an insufficient `--samples`, `--steps`, or
`--tv-distance` override is rejected, as is any finite retry cap. The former
`--burn-in` and `--thin` volume options are removed rather than reinterpreted.
Old DFK output files came from correlated trajectories and cannot be relabeled
as results of this implementation; rerun them for an independent-sample study.

`volume.estimate_volume(A, b, ...)` exposes the same settings. JSON includes the
algorithm version, phase budgets, overlaps, proposal counts, settings, timing,
and any failure. Saved observations and theoretical guarantees are separate.
See [the implementation audit](audit/README.md) for source locations, the
corrections, and remaining limitations.

## Project layout

| Location | Purpose |
| --- | --- |
| `sampler.py` | In-and-out transitions, batching, and rejection accounting |
| `mixing.py` | Sufficient endpoint step budget from TV, diameter, and warmness |
| `geometry.py` | Ball geometry and warm-start sampling |
| `shapes.py` | Polygon construction and exact polygon moments |
| `demo.py` | Sampling command-line interface and built-in bodies |
| `batch.py` | Independent endpoint batches with optional worker processes |
| `volume.py` | Volume estimator and command-line interface |
| `plotting/` | Density calculations, target comparisons, and plotting helpers |
| `experiments/` | Mixing-budget runners and performance experiments |
| `tests/` | Automated correctness checks |

See [experiments/README.md](experiments/README.md) for benchmark commands.
Run all tests from the repository root:

```sh
python -m unittest discover -s tests -v
```

Only source code, README files, and dependency/configuration files are tracked.
Results, plots, sample archives, PDFs, and logs stay local. Plotting helpers for
saved experiments require their input data to be generated first.

## References

[1] Yunbum Kook, Santosh S. Vempala, and Matthew S. Zhang,
*In-and-Out: Algorithmic Diffusion for Sampling Convex Bodies*,
[arXiv:2405.01425v4](https://arxiv.org/abs/2405.01425v4). Algorithm 1 specifies
the transitions; Theorem 14 bounds the uncapped-chain chi-squared divergence.

[2] Yin Tat Lee and Santosh S. Vempala, *Techniques in Optimization and Sampling*,
draft dated August 7, 2026, Section 11.1, Algorithm 35 and Lemmas 11.4–11.6
(printed page 179), Theorem 11.7 (printed page 180).
This is the DFK expanding-ball algorithm used by `volume.py`. The draft is a
local research reference and is not bundled with the repository.

[3] L. Esposito, C. Nitsch, and C. Trombetti, *Best constants in Poincare
inequalities for convex domains*, [arXiv:1110.2960v1](https://arxiv.org/abs/1110.2960v1),
Theorem 1.1 with p=2. In the variance convention of [1, Definition 9], this
implies `C_PI <= diameter**2/pi**2`. The simplex's diameter squared is 2;
`2/pi**2` is its geometry-specific bound, not a universal constant.
