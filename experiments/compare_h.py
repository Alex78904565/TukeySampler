"""Compare equal-duration h pilots using observed progress toward required k.

Example: python -m experiments.compare_h small_h_pilot.json larger_h_pilot.json
The completion-time projection assumes a constant observed rate; it is not
a measured completion time, confidence bound, or finite-mean runtime claim.
"""

import argparse
import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", default="h_comparison")
    args = parser.parse_args()
    documents = [json.loads(Path(name).read_text()) for name in args.inputs]
    first = documents[0]
    for doc in documents[1:]:
        legacy = dict(prefetch_out=False, batch_in=False)
        if doc.get('optimizations', legacy) != first.get('optimizations', legacy):
            raise ValueError('Cannot combine experiments with different optimization settings.')
        for key in ("body", "dimension", "D", "log_M", "formula", "max_attempts", "time_limit_seconds"):
            if doc[key] != first[key]:
                raise ValueError(f"Cannot combine experiments with different {key}.")
    runs = [run for doc in documents for run in doc["runs"]]
    if any(run["status"] not in ("timeout", "completed") for run in runs):
        raise ValueError("Retry-cap failures cannot be treated as ordinary time-limited pilots.")
    summaries = []
    fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
    for h in sorted({run["h"] for run in runs}):
        group = [run for run in runs if run["h"] == h]
        if any(run["completed_steps"] == 0 for run in group):
            raise ValueError("A zero-progress run has an infinite rate projection; inspect it separately.")
        projections = [r["seconds"]*r["required_steps"]/r["completed_steps"]/3600 for r in group]
        progress = [100*r["completed_steps"]/r["required_steps"] for r in group]
        summary = dict(h=h, repeats=len(group), median_progress_percent=median(progress),
                       projected_hours_median=median(projections), projected_hours_min=min(projections),
                       projected_hours_max=max(projections),
                       proposals_per_completed_step=sum(r["inward_proposals"] for r in group)/sum(r["completed_steps"] for r in group))
        summaries.append(summary)
        ax.scatter([h]*len(group), projections, color="#788a94", alpha=0.7, s=28)
        print(f"h={h:g}: median progress {median(progress):.4f}%; "
              f"constant-rate projection {median(projections):.3f}h "
              f"(repeat range {min(projections):.3f}-{max(projections):.3f}h)")
    ax.plot([s["h"] for s in summaries], [s["projected_hours_median"] for s in summaries],
            "o-", color="#216869", label="Median rate-based projection")
    ax.scatter([], [], color="#788a94", label="Individual repeats")
    ax.set(xscale="log", yscale="log", xlabel="Gaussian variance h", ylabel="Projected completion time (hours)",
           title=f"{first['dimension']}D {first['body']}: {first['time_limit_seconds']:g}-second pilots")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=0.2)
    ax.legend()
    fig.supxlabel("Extrapolated from pilot progress, not completed runtimes. Long rejection episodes can change these rates.", fontsize=9)
    fig.savefig(args.output+".png", dpi=160)
    plt.close(fig)
    Path(args.output+".json").write_text(json.dumps(dict(sources=args.inputs, summaries=summaries,
        interpretation="Constant-rate projections from pilots, not measured completion times"), indent=2))


if __name__ == "__main__":
    main()
