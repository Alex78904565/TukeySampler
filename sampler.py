"""Algorithm 1 of Kook, Vempala, Zhang (2405.01425v4), for K = {x: A x <= b}."""

from itertools import count
from time import perf_counter

import numpy as np


class SamplingFailure(RuntimeError):
    """The inward rejection sampler exhausted its cap; the run must stop."""

    def __init__(self, message, completed_steps=0, proposals=0):
        super().__init__(message)
        self.completed_steps = completed_steps
        self.proposals = proposals


class SamplingTimeout(SamplingFailure):
    """An external timing limit interrupted an unfinished run."""


def in_and_out(A, b, x0, *, h, steps, max_attempts=100_000, rng=None, deadline=None,
               progress=None, prefetch_out=True, batch_in=True, batch_limit=1_048_576,
               out_rng=None, diagnostics=None, batch_floor=32, ball=None):
    """Return (path, attempts) after exactly `steps` successful transitions.

    path has shape (steps + 1, d), including x0. attempts[i] counts inward
    proposals for transition i. h is the variance of EACH Gaussian step.
    A rejection cap raises SamplingFailure, without returning a partial run.
    Set max_attempts=None for unlimited inward retries.
    Optional deadline is an absolute time.perf_counter() value for benchmarks.
    An interrupted run raises SamplingTimeout and is not a completed sample.
    Optional progress(completed_steps, total_proposals, current_rejections)
    receives periodic counts, including during a long inward rejection loop.
    Assume K is bounded and has nonempty interior (see geometry.py).
    No burn-in, thinning, restarts, or adaptive parameter changes occur here.
    Optimization switches (independent; default on): prefetch_out draws all
    OUT noise at once; batch_in tries IN matrices of 1,floor,2*floor,... rows, bounded
    by batch_limit. Unused rows after the first success are discarded.
    batch_floor defaults to 32, selected empirically at h=1e-4 on the 10D
    simplex. Set 2 to recover the original 1,2,4,... schedule.
    out_rng optionally separates OUT from IN randomness for matched comparisons.
    diagnostics, if provided, receives actual generated IN-vector counts.
    Optional ball=(center, radius) restricts the target to K intersected with
    that Euclidean ball, as needed by DFK volume estimation.
    """
    A, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
    x = np.array(x0, dtype=float, copy=True)
    if A.ndim != 2 or min(A.shape) == 0 or b.shape != (A.shape[0],):
        raise ValueError("Expected A with shape (m, d) and b with shape (m,).")
    if x.shape != (A.shape[1],):
        raise ValueError("x0 must have shape (d,).")
    if not all(np.isfinite(v).all() for v in (A, b, x)):
        raise ValueError("A, b, and x0 must be finite.")
    if not np.all(A @ x <= b):
        raise ValueError("x0 must lie inside the polytope.")
    if ball is not None:
        center, radius = np.asarray(ball[0], dtype=float), float(ball[1])
        if center.shape != x.shape or not np.isfinite(center).all() or not np.isfinite(radius) or radius <= 0:
            raise ValueError('ball must contain a finite center of shape (d,) and positive radius.')
        if np.linalg.norm(x-center) > radius:
            raise ValueError('x0 must lie inside the ball.')
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive.")
    for name, value, minimum in (("steps", steps, 0), ("max_attempts", max_attempts, 1)):
        if name == "max_attempts" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")

    rng = np.random.default_rng() if rng is None else rng
    out_rng = rng if out_rng is None else out_rng
    if isinstance(batch_limit, bool) or not isinstance(batch_limit, (int, np.integer)) or batch_limit < 1:
        raise ValueError('batch_limit must be a positive integer.')
    if isinstance(batch_floor, bool) or not isinstance(batch_floor, (int, np.integer)) or batch_floor < 2:
        raise ValueError('batch_floor must be an integer >= 2.')
    scale = np.sqrt(h)
    path = np.empty((steps + 1, x.size))
    attempts = np.empty(steps, dtype=int)
    path[0] = x
    generated = np.empty(steps, dtype=np.int64) if diagnostics is not None else None
    # Store NOISE, not positions: each jump still starts at the current x.
    out_noise = out_rng.normal(size=(steps, x.size)) if prefetch_out else None

    for step in range(steps):
        if progress is not None and step % 256 == 0:
            progress(step, int(attempts[:step].sum()), 0)
        if deadline is not None and step % 256 == 0 and perf_counter() >= deadline:
            raise SamplingTimeout("Run reached its timing limit.", step, int(attempts[:step].sum()))
        # OUT: draw once, without checking membership.
        noise = out_noise[step] if prefetch_out else out_rng.normal(size=x.size)
        y = x + scale * noise

        if batch_in:
            rejected, batch_size = 0, 1
            while True:
                size = batch_size if max_attempts is None else min(batch_size, max_attempts-rejected)
                if size == 0:
                    raise SamplingFailure('Inward proposal cap exhausted.', step,
                                          int(attempts[:step].sum()) + rejected)
                proposals = y + scale * rng.normal(size=(size, x.size))
                # Row j contains proposal j. One matrix multiplication checks all.
                feasible = np.all(proposals @ A.T <= b, axis=1)
                if ball is not None:
                    feasible &= np.linalg.norm(proposals-center, axis=1) <= radius
                successes = np.flatnonzero(feasible)
                if len(successes):
                    first = int(successes[0])
                    x = proposals[first].copy()
                    path[step+1] = x
                    attempts[step] = rejected + first + 1
                    if generated is not None:
                        generated[step] = rejected + size
                    break
                rejected += size
                if deadline is not None and perf_counter() >= deadline:
                    raise SamplingTimeout('Timing limit during inward batches.', step,
                                          int(attempts[:step].sum()) + rejected)
                if progress is not None and rejected >= 256:
                    progress(step, int(attempts[:step].sum()) + rejected, rejected)
                batch_size = min(batch_floor if batch_size == 1 else 2*batch_size, batch_limit)
            continue

        # IN: keep y fixed; sample N(y, h I) until a proposal lies in K.
        trials = count(1) if max_attempts is None else range(1, max_attempts + 1)
        for trial in trials:
            proposal = y + scale * rng.normal(size=x.size)
            if np.all(A @ proposal <= b) and (ball is None or np.linalg.norm(proposal-center) <= radius):
                x = proposal
                attempts[step] = trial
                if generated is not None:
                    generated[step] = trial
                path[step + 1] = x
                break
            if deadline is not None and trial % 256 == 0 and perf_counter() >= deadline:
                raise SamplingTimeout("Run reached its timing limit during inward retries.",
                                      step, int(attempts[:step].sum()) + trial)
            if progress is not None and trial % 256 == 0:
                progress(step, int(attempts[:step].sum()) + trial, trial)
        else:
            raise SamplingFailure(
                f"Transition {step + 1} failed after {max_attempts} inward proposals. "
                "The run stopped; no replacement transition was made.",
                step, int(attempts[:step].sum()) + max_attempts,
            )

    if progress is not None:
        progress(steps, int(attempts.sum()), 0)
    if diagnostics is not None:
        diagnostics['generated_inward_per_step'] = generated
        diagnostics['generated_inward'] = int(generated.sum())
        diagnostics['unused_inward'] = int(generated.sum()-attempts.sum())
    return path, attempts


def add_optimization_arguments(parser):
    """Shared CLI switches for ordinary sampling commands."""
    import argparse
    parser.add_argument('--prefetch-out', action=argparse.BooleanOptionalAction, default=True,
                        help='Prefetch OUT noise (default on; --no-prefetch-out disables)')
    parser.add_argument('--batch-in', action=argparse.BooleanOptionalAction, default=True,
                        help='Batch IN proposals (default on; --no-batch-in disables)')


def optimization_options(args):
    return dict(prefetch_out=args.prefetch_out, batch_in=args.batch_in)
