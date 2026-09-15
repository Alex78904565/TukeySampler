"""Draw 1,000 independent uniform simplex samples and their paired marginals.

Run: python -m experiments.example_10d
This is a ground-truth illustration, not an in-and-out sampling run.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from plotting.reference import marginal_reference


def main():
    d, n = 10, 1000
    rng = np.random.default_rng(7)
    # Eleven nonnegative barycentric weights sum to one. Dropping the last
    # gives Uniform({x in R^10: x >= 0, sum(x) <= 1}). No burn-in is needed.
    samples = rng.dirichlet(np.ones(d+1), size=n)[:, :d]
    assert np.all(samples >= 0) and np.all(samples.sum(axis=1) <= 1)
    fig, axes = plt.subplots(5, 3, figsize=(13, 16), layout="constrained")
    panels = []
    for i in range(0, d, 2):
        edges, target, boundary = marginal_reference("simplex", d, (i, i+1),
                                                      np.zeros(d), np.ones(d), bins=25)
        counts, _, _ = np.histogram2d(samples[:, i], samples[:, i+1], bins=edges)
        areas = np.outer(np.diff(edges[0]), np.diff(edges[1]))
        empirical = counts / (n * areas)
        assert np.isclose(np.sum(empirical * areas), 1)
        panels.append((empirical, target, boundary))
    vmax = max(max(empirical.max(), target.max()) for empirical, target, _ in panels)
    artists = []
    for row, i in enumerate(range(0, d, 2)):
        left, middle, right = axes[row]
        empirical, target, boundary = panels[row]
        scatter = left.scatter(samples[:, i], samples[:, i+1], s=6, alpha=0.4, color="#216869")
        left.set_title(f"(x{i+1}, x{i+2}): collected points")
        for ax, density in ((middle, empirical), (right, target)):
            im = ax.imshow(density.T, origin="lower", extent=[0, 1, 0, 1],
                           vmin=0, vmax=vmax, cmap="viridis", interpolation="nearest")
            im.set_clip_path(Polygon(boundary, closed=True, transform=ax.transData))
            if ax is middle:
                empirical_image = im
        artists.append((scatter, empirical_image))
        middle.set_title(f"(x{i+1}, x{i+2}): sample heatmap")
        right.set_title(f"(x{i+1}, x{i+2}): exact reference")
        fig.colorbar(im, ax=[middle, right], shrink=0.8, label="Marginal probability density")
        for ax in (left, middle, right):
            outline = np.vstack((boundary, boundary[0]))
            ax.plot(*outline.T, color="#dd6b20", linewidth=1.5)
            ax.set(xlim=(-0.025, 1.025), ylim=(-0.025, 1.025),
                   xlabel=f"x{i+1}", ylabel=f"x{i+2}", aspect="equal")
    title = fig.suptitle("Uniform inside a 10D simplex: 1,000 independent samples", fontsize=15)
    fig.supxlabel("Each row projects the same collected points. Fixed bins and color scale; early peaks may saturate.\nOrange: projected boundary. Reference: exact bin averages of 90(1-x-y)^8. Sample heatmap: counts / (sample count x bin area).", fontsize=10)
    folder = Path(__file__).resolve().parents[1]
    fig.savefig(folder / "uniform_10d_example.png", dpi=140)
    # Early frames are closer together to reveal how the estimate builds up.
    frame_counts = np.unique(np.r_[np.geomspace(1, n, 40).astype(int), n])

    def update(frame):
        count = frame_counts[frame]
        for row, i in enumerate(range(0, d, 2)):
            scatter, empirical_image = artists[row]
            scatter.set_offsets(samples[:count, i:i+2])
            counts, _, _ = np.histogram2d(samples[:count, i], samples[:count, i+1], bins=edges)
            empirical_image.set_data((counts / (count * areas)).T)
        title.set_text(f"Uniform inside a 10D simplex: {count:,} / {n:,} independent samples")

    animation = FuncAnimation(fig, update, frames=len(frame_counts), interval=200, repeat=False)
    animation.save(folder / "uniform_10d_example.gif", writer=PillowWriter(fps=5), dpi=90)
    plt.close(fig)
    np.savez(folder / "uniform_10d_example.npz", samples=samples, seed=7,
             description="Independent uniform 10D simplex samples (Dirichlet), not MCMC")
    print(f"Saved {n} samples, uniform_10d_example.png, and uniform_10d_example.gif in {folder}")


if __name__ == "__main__":
    main()
