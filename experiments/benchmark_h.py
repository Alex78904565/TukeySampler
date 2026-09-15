"""Time in-and-out runs using the supplied TV step bound, for several h values.

Example: python -m experiments.benchmark_h --dim 10 --h-values 0.0001 0.001 0.01
Use --seconds 0 to wait for completion without a wall-time limit.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from demo import example_body, parse_max_attempts
from geometry import uniform_ball
from experiments.mixing import example_geometry, required_steps
from sampler import in_and_out, SamplingFailure, SamplingTimeout, add_optimization_arguments, optimization_options


def timed_run(A, b, center, radius, h, steps, seed, seconds, max_attempts,
              progress_every=0, progress_label="", prefetch_out=True, batch_in=True):
    """One fresh warm start, then k transitions. Keep only the final endpoint.

    Small chunks limit memory. They continue the same chain and RNG, without
    restart, adaptation, burn-in removal, or changes to the transition rule.
    Timing includes initialization, bookkeeping, and progress printing, but
    excludes plotting and writing result files.
    """
    rng = np.random.default_rng(seed)
    start = perf_counter()
    deadline = start+seconds if seconds else None
    x = uniform_ball(center, radius, rng)
    completed, proposals, status = 0, 0, "completed"
    next_report = start + progress_every

    def report(chunk_steps, chunk_proposals, current_rejections):
        nonlocal next_report
        now = perf_counter()
        if now < next_report:
            return
        done = completed + chunk_steps
        total = proposals + chunk_proposals
        elapsed = now-start
        print(f"{progress_label} steps {done:,}/{steps:,} ({100*done/steps:.4f}%) | "
              f"elapsed {elapsed:.1f}s | proposals {total:,} | "
              f"current-step rejections {current_rejections:,}", flush=True)
        next_report = now + progress_every

    try:
        while completed < steps:
            path, attempts = in_and_out(A, b, x, h=h, steps=min(1000, steps-completed),
                                        max_attempts=max_attempts, rng=rng, deadline=deadline,
                                        progress=report if progress_every > 0 else None,
                                        prefetch_out=prefetch_out, batch_in=batch_in)
            x = path[-1]
            completed += len(attempts)
            proposals += int(attempts.sum())
    except SamplingFailure as error:
        completed += error.completed_steps
        proposals += error.proposals
        status = "timeout" if isinstance(error, SamplingTimeout) else "retry_cap_failure"
    elapsed = perf_counter()-start
    return dict(status=status, seconds=elapsed, completed_steps=completed,
                inward_proposals=proposals,
                # A failed/unfinished trajectory is deliberately not an output sample.
                endpoint=x.tolist() if status == "completed" else None)


def plot_results(rows, output, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    hs = sorted(set(row["h"] for row in rows))
    ks = [next(row["required_steps"] for row in rows if row["h"] == h) for h in hs]
    axes[0].plot(hs, ks, "o-", color="#216869")
    axes[0].set(ylabel="Required proper steps k", title="Your supplied step bound", yscale="log")
    styles = [("completed", "o", "#216869", "Completed"),
              ("timeout", "^", "#dd6b20", "Time limit: completion time > shown"),
              ("retry_cap_failure", "x", "#9e2947", "Retry cap: failed, not completed")]
    for status, marker, color, label in styles:
        selected = [r for r in rows if r["status"] == status]
        if selected:
            axes[1].scatter([r["h"] for r in selected], [r["seconds"] for r in selected],
                            marker=marker, color=color, label=label)
            axes[2].scatter([r["h"] for r in selected],
                            [100*r["completed_steps"]/r["required_steps"] for r in selected],
                            marker=marker, color=color)
    axes[1].set(ylabel="Elapsed sampling seconds", yscale="log", title="Measured times (each repeat)")
    axes[1].legend(fontsize=8, loc="best")
    axes[2].set(ylabel="Completed fraction of required steps (%)", title="Progress toward k", ylim=(0, 102))
    axes[2].set_yscale("symlog", linthresh=0.001)
    if not any(r["status"] == "completed" for r in rows):
        axes[1].text(0.5, 0.85, "No completed runs", transform=axes[1].transAxes,
                     ha="center", fontsize=11)
        # Do not magnify millisecond differences between identical time limits.
        middle = float(np.median([r["seconds"] for r in rows]))
        base = 10**round(np.log10(middle))
        axes[1].set_ylim(min(base/10, min(r["seconds"] for r in rows)/2),
                         max(base*10, max(r["seconds"] for r in rows)*2))
    for ax in axes:
        ax.set(xscale="log", xlabel="Gaussian variance h (original coordinates)")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=0.2)
    fig.supxlabel("Timeouts are unfinished runs, not completion measurements. No failed runs are discarded. Plotting time excluded.", fontsize=10)
    fig.suptitle(title)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_completion_times(rows, output):
    """Plot actual completed times only; unfinished runs never become times."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    completed = sorted((r for r in rows if r["status"] == "completed"), key=lambda r: r["h"])
    if not completed:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    hs = [r["h"] for r in completed]
    minutes = [r["seconds"]/60 for r in completed]
    ax.scatter(hs, minutes, color="#216869", s=50)
    if len(set(hs)) == len(hs):
        ax.plot(hs, minutes, color="#216869", alpha=0.6)
    for h, value in zip(hs, minutes):
        ax.annotate(f"{value:.1f} min", (h, value), xytext=(0, 8), textcoords="offset points", ha="center")
    ax.set(xscale="log", xlabel="Gaussian variance h", ylabel="Actual completion time (minutes)",
           title="Time to complete the required proper steps")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(sorted(set(hs)), [f"{h:g}" for h in sorted(set(hs))])
    ax.margins(x=0.2, y=0.25)
    ax.grid(alpha=0.2)
    unfinished = sum(r["status"] != "completed" for r in rows)
    fig.supxlabel(f"Individual completed runs, not expected runtimes. Unfinished recorded runs: {unfinished}.", fontsize=9)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body", choices=("box", "simplex"), default="simplex")
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--h-values", type=float, nargs="+", default=[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=10, help="Time limit PER run; 0 means unlimited")
    parser.add_argument("--progress-every", type=float, default=5,
                        help="Print progress every this many seconds; 0 disables it")
    parser.add_argument("--max-attempts", type=parse_max_attempts, default=None, help="Retry cap N; default inf (uncapped)")
    parser.add_argument("--D", type=float, help="Override enclosing radius D; default centered at the inner-ball center")
    parser.add_argument("--log-M", type=float, help="Override ln(M); default exact inner-ball volume ratio")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="h_sweep", help="Output prefix for JSON and PNG")
    parser.add_argument("--plan-only", action="store_true", help="Print step counts without sampling")
    add_optimization_arguments(parser)
    args = parser.parse_args()
    options = optimization_options(args)
    if args.dim < 1 or args.repeats < 1 or not np.isfinite(args.seconds) or args.seconds < 0:
        parser.error("Require dim/repeats >= 1 and finite seconds >= 0.")
    if not np.isfinite(args.progress_every) or args.progress_every < 0:
        parser.error("progress-every must be finite and nonnegative.")
    center, radius, enclosing_radius, log_M = example_geometry(args.body, args.dim)
    D = enclosing_radius if args.D is None else args.D
    log_M = log_M if args.log_M is None else args.log_M
    try:
        plan = [(h, required_steps(h, args.dim, D, log_M)) for h in args.h_values]
    except ValueError as error:
        parser.error(str(error))
    print(f"{args.body}, d={args.dim}, inner radius={radius:.9g}, D={D:.9g}, ln(M)={log_M:.9g}", flush=True)
    print("Requested TV target: 1e-6. Supplied expression used verbatim; no independent certificate.", flush=True)
    for h, k in plan:
        print(f"h={h:g}: k={k:,}", flush=True)
    if args.plan_only:
        return
    A, b, _, _ = example_body(args.body, args.dim)
    metadata = dict(body=args.body, dimension=args.dim, radius=radius, D=D, log_M=log_M,
                    D_convention="enclosing radius about inner-ball center" if args.D is None else "user override",
                    requested_tv=1e-6, formula="ceil((12*ln(20)+ln(M))/ln(1+h*ln(2)^2/(4*d*D^2)))",
                    max_attempts=args.max_attempts, time_limit_seconds=args.seconds,
                    progress_interval_seconds=args.progress_every, optimizations=options,
                    batch_floor=32, batch_limit=1048576, runs=[])
    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for index, (h, k) in enumerate(plan):
        for repeat in range(args.repeats):
            seed = [args.seed, index, repeat]
            label = f"[h={h:g}, repeat={repeat+1}/{args.repeats}]"
            print(f"{label} starting {k:,} proper steps", flush=True)
            result = timed_run(A, b, center, radius, h, k, seed, args.seconds, args.max_attempts,
                               progress_every=args.progress_every, progress_label=label, **options)
            result.update(h=h, required_steps=k, repeat=repeat+1, seed=seed)
            metadata["runs"].append(result)
            Path(str(prefix)+".json").write_text(json.dumps(metadata, indent=2))
            print(f"h={h:g}, repeat={repeat+1}: {result['status']}, {result['seconds']:.3f}s, "
                  f"{result['completed_steps']:,}/{k:,} steps, {result['inward_proposals']:,} proposals", flush=True)
            plot_results(metadata["runs"], str(prefix)+".png",
                         title=f"{args.dim}D {args.body}: enclosing radius D={D:.6g}, ln(M)={log_M:.6g}")
            plot_completion_times(metadata["runs"], str(prefix)+"_completion.png")
    plot_results(metadata["runs"], str(prefix)+".png",
                 title=f"{args.dim}D {args.body}: enclosing radius D={D:.6g}, ln(M)={log_M:.6g}")
    print(f"Saved {prefix}.json and {prefix}.png", flush=True)


if __name__ == "__main__":
    main()
