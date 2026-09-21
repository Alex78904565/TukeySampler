# Implementation and source audit

Reviewed September 21, 2026. Scope: sampling transitions, ball starts, proper-step
budgets, independent endpoint batches, DFK estimation, and the experiment/plot
entry points. This is a traceable code review, not a claim of publication-level
validation of every historical PDF, experiment, or numerical implementation.

## Source checks

The supplied PDFs were read directly, including the typeset DFK pages.

| Source | Checked statement | Implementation consequence |
| --- | --- | --- |
| Kook, Vempala, Zhang, In-and-Out, arXiv:2405.01425v4, Algorithm 1 (PDF p. 3) | Draw OUT once; hold it fixed for Gaussian IN rejection; fail at a cap | `sampler.py` keeps OUT fixed, accepts the first feasible IN proposal, and raises on cap exhaustion |
| Same paper, Definition 9 and Theorem 14 | Variance PI convention; uncapped chi-squared contraction `(1+h/C_PI)^(-k)` | `mixing.py` uses this rate, `M-1`, and `TV <= sqrt(chi2)/2`; TV-budget modes require uncapped retries |
| Esposito, Nitsch, Trombetti, arXiv:1110.2960v1, Theorem 1.1 at p=2 | Diameter-based Poincare inequality | `C_PI <= D^2/pi^2`; D is diameter, not enclosing radius |
| Lee and Vempala, Techniques in Optimization and Sampling, August 7, 2026 draft, Algorithm 35, printed p. 179 / PDF p. 180 | Nested ball intersections, uniform samples, hit ratios, reciprocal product | `volume.py` implements these bodies and estimator |
| Same draft, Lemmas 11.4-11.6 and Theorem 11.7, printed pp. 179-180 / PDF pp. 180-181 | Bernoulli sample variance within phases, independent product variance across phases; approximate sampling/dependence require additional work | Fresh independent chains within and across phases; explicit endpoint-TV error allocation |

References and links are in the [main README](../README.md). The supplied draft
is not distributed in Git. The source itself acknowledges more sophisticated
approximate/dependent sampling implementations: independence is the assumption
of the elementary variance proof followed here, not a claim about every DFK
variant in the literature.

## Corrections

1. **DFK dependence:** the previous implementation retained thinned states of one
   chain and reused its endpoint in the next phase. It did not satisfy the
   independent-sample proof. That collection path is removed from `volume.py`.
   Each new sample gets its own uniform initial-ball draw and random stream;
   no observed endpoint is used to initialize another chain or phase.
2. **DFK budgets:** the old defaults of 2,000 samples, burn-in, and thinning were
   exploratory choices. Default budgets now derive N from the product variance
   with the reciprocal tolerance `epsilon/(1+epsilon)`. Half the failure budget
   covers ideal sample noise; half covers joint TV via the sum over all endpoint
   errors. The README gives the derivation. This allocation is an explicit
   additional argument, not a quotation of Theorem 11.7's constant.
3. **Manual runs:** reduced budgets and finite caps require `--experimental` and
   are saved without a volume-error guarantee. Legacy `--burn-in`/`--thin` flags
   fail instead of silently acquiring a different meaning.
4. **Endpoint plots:** the PW runtime sweep no longer automatically animates
   accumulated endpoints. Static endpoint/reference images are the default.
   An optional endpoint-accumulation animation remains explicitly named in the
   plotting helper; it is not an animation of proper steps.
5. **Parallel DFK:** `volume.py --workers` now uses independent endpoint workers.
   Geometry is supplied from the parent; changing worker count preserves sample
   streams and output order. Pending jobs are bounded to avoid queuing a copy of
   the geometry for every sample at once.

## Other requested behavior checked

| Request | Code and evidence |
| --- | --- |
| Prefetch OUT noise and batch IN proposals, each switchable | `sampler.py`, `tests/test_batching.py`: noise rather than positions; first feasible row; unused proposals discarded; defaults on |
| Distinguish batch allocation limits from retry limits | Batch cap 1,048,576 rows, floor 32; retry count is separate and may be unlimited; scripted tests exercise both |
| Track failed IN attempts between proper steps | `attempts` counts proposals through success; optional diagnostics count all generated vectors, including unused rows |
| Reuse Chebyshev geometry, not one initial point | `geometry.py` caches by constraint values and returns a center copy; batch and DFK draw fresh points; cache tests cover repeated calls and mutation |
| Use the current endpoint TV bound | `demo.py`, `batch.py`, PW experiments call `mixing_steps`; tests check k and k-1. The old formula remains explicitly historical in `experiments/benchmark_h.py` |
| Pair coordinates, show projected boundary and target density | `plotting/reference.py` uses exact box/simplex bin probabilities; `plot_pairs.py` handles odd dimensions; tests check mass and coordinate coverage |
| Random polygon | `shapes.py` takes the hull of Gaussian 2D input points, not uniformly bounded input points; polygon-only scope is explicit |
| Timing versus theoretical work | Completed elapsed times, timeouts, proper-step counts, and projections remain distinct. Parallel timings are wall times, not a promised linear speedup |
| Keep research artifacts out of Git | Only code, READMEs, dependencies, and repository configuration are published; results, PDFs, LaTeX, local environments, and screenshots remain ignored |

## Verification and remaining limits

Tests cover the DFK sample inequality with high-precision arithmetic at N and
N-1; total statistical plus sampling-TV budget; per-phase mixing inequalities;
separate phase streams; independent initialization through the batch sampler;
telescoping ratio arithmetic; zero-hit/cap failures; cached geometry; and equal
serial/parallel outputs. Existing transition, geometry, mixing, and visualization
tests are retained. Small CLI experiments test plumbing, not a theorem's success
probability. Large theoretical DFK budgets are inspected with `--plan-only`, not
silently replaced by cheaper runs.

The guarantees are mathematical guarantees for the stated ideal algorithm,
conditional on valid geometric bounds and ideal independent randomness. SciPy
linear programming, Gaussian generation, membership tests, and arithmetic use
floating point and pseudorandom streams; no interval-arithmetic or finite-bit
error proof is supplied. Ordinary trajectory exploration in `demo.py` remains
clearly labeled correlated; those trajectories are not used as DFK samples.

Historical DFK files and PDF descriptions of the old correlated prototype are
not retroactively valid independent-sample results. They must be distinguished
from `dfk_independent_endpoints_v1` and rerun before making claims about the new
implementation. This audit does not certify every theorem in the local study
documents, nor establish a measured runtime improvement for a full theoretical
DFK run. Tukey-depth weighting itself remains unimplemented, as stated at the
top of the main README.
