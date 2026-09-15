"""Plot a saved run: python -m plotting.plot_samples run.npz --output distribution.png."""

import argparse

import matplotlib
matplotlib.use("Agg")  # Save figures without requiring a desktop plotting window.
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon
from plotting.density import make_grid, restricted_density



def plot_distribution(path, A, b, burn_in, output, smoothing=1.0):
    """Plot restricted Gaussian kernels; smoothing sets bandwidth = 0.15*r*s."""
    path, A, b = np.asarray(path), np.asarray(A), np.asarray(b)
    if path.ndim == 2 and path.shape[1] != 2:
        from plotting.plot_pairs import plot_pairs
        return plot_pairs(path, burn_in, output, A=A, b=b)
    if path.ndim != 2:
        raise ValueError("Expected a 2D path array (steps+1, d).")
    if not 0 <= burn_in <= len(path) - 2:
        raise ValueError("Need at least one retained point.")
    if not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("smoothing must be finite and positive.")
    samples = path[burn_in + 1:]
    polygon, grid, mask, cell_area, extent, radius = make_grid(A, b)
    outline = np.vstack((polygon, polygon[0]))
    target_density = 1 / ConvexHull(polygon).volume
    density = restricted_density(samples, grid, mask, cell_area, 0.15*radius*smoothing).reshape(85, 85)
    target = np.where(mask, target_density, 0).reshape(85, 85)
    lo, hi = np.array([extent[0], extent[2]]), np.array([extent[1], extent[3]])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), layout="constrained")
    # Display at most 5,000 evenly spaced points; KDE uses ALL retained points.
    shown = samples[np.linspace(0, len(samples) - 1, min(5_000, len(samples)), dtype=int)]
    axes[0].scatter(shown[:, 0], shown[:, 1], s=3, alpha=0.22, color="#216869", rasterized=True)
    axes[0].set_title(f"Retained points\n{len(shown):,} shown / {len(samples):,} total")
    vmax = max(float(density.max()), target_density)
    extent = [lo[0], hi[0], lo[1], hi[1]]
    for ax, values, title in zip(axes[1:], (density, target),
                                 ("Gaussian kernels restricted to K", "Uniform target: 1 / area")):
        im = ax.imshow(values, origin="lower", extent=extent, cmap="viridis",
                       vmin=0, vmax=vmax, interpolation="nearest", aspect="equal")
        im.set_clip_path(Polygon(polygon, closed=True, transform=ax.transData))
        ax.set_title(title)
    fig.colorbar(im, ax=list(axes[1:]), shrink=0.8, label="Probability density")
    for ax in axes:
        ax.plot(outline[:, 0], outline[:, 1], color="#dd6b20", linewidth=1.5)
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), xlabel="x1", ylabel="x2")
        ax.set_aspect("equal")
    fig.suptitle(f"In-and-out sample distribution | burn-in: {burn_in:,} | smoothing: {smoothing:g}", fontsize=14)
    fig.supxlabel("Density is zero outside K; kernels are normalized on the grid. Visual agreement does not certify mixing.", fontsize=10)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def animate_distribution(path, A, b, burn_in, output, smoothing=1.0, frames=45):
    """Save a GIF of cumulative density; all retained samples contribute.

    Uneven frame times show early exploration in more detail. The density is
    an occupation estimate along one chain, not an ensemble endpoint law.
    """
    path = np.asarray(path)
    if path.ndim == 2 and path.shape[1] != 2:
        from plotting.plot_pairs import plot_pairs
        return plot_pairs(path, burn_in, output, animate=True, frames=frames, A=A, b=b)
    if path.ndim != 2 or not 0 <= burn_in < len(path)-1:
        raise ValueError("Need a 2D path and at least one retained point.")
    if not np.isfinite(smoothing) or smoothing <= 0 or frames < 2:
        raise ValueError("Need positive smoothing and at least two frames.")
    samples = path[burn_in+1:]
    polygon, grid, mask, cell_area, extent, radius = make_grid(A, b)
    target = 1 / ConvexHull(polygon).volume
    # Accumulate kernels once, keeping bandwidth fixed throughout the animation.
    counts = np.unique(np.geomspace(1, len(samples), frames).astype(int))
    counts = np.unique(np.r_[counts, len(samples)])
    densities, total, previous = [], np.zeros(len(grid)), 0
    for count in counts:
        batch = samples[previous:count]
        total += len(batch) * restricted_density(batch, grid, mask, cell_area, 0.15*radius*smoothing)
        densities.append((total/count).reshape(85, 85).copy())
        previous = count

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
    scatter = axes[0].scatter([], [], s=4, alpha=0.3, color="#216869")
    dot, = axes[0].plot([], [], "o", color="#dd6b20", markersize=7)
    axes[0].set_title("Collected points and current position")
    axes[1].set_title("Cumulative density / uniform target")
    # Fixed ratio scale: 1 is uniform; early concentrations saturate above 2.
    im = axes[1].imshow(densities[0]/target, origin="lower", extent=extent,
                        cmap="viridis", vmin=0, vmax=2, interpolation="nearest")
    im.set_clip_path(Polygon(polygon, closed=True, transform=axes[1].transData))
    fig.colorbar(im, ax=axes[1], shrink=0.8, extend="max", label="Density ratio (1 = uniform; 2+ saturated)")
    outline = np.vstack((polygon, polygon[0]))
    for ax in axes:
        ax.plot(outline[:, 0], outline[:, 1], color="#dd6b20", linewidth=1.5)
        ax.set(xlim=extent[:2], ylim=extent[2:], xlabel="x1", ylabel="x2", aspect="equal")
    title = fig.suptitle("")
    fig.supxlabel("One chain, cumulative samples. Fixed bandwidth and colors; zero density outside K.", fontsize=10)

    def update(frame):
        count = counts[frame]
        shown = samples[np.linspace(0, count-1, min(count, 5000), dtype=int)]
        scatter.set_offsets(shown)
        dot.set_data([samples[count-1, 0]], [samples[count-1, 1]])
        im.set_data(densities[frame]/target)
        title.set_text(f"Transition {burn_in+count:,} | retained samples: {count:,} | burn-in: {burn_in:,}")
        return scatter, dot, im, title

    animation = FuncAnimation(fig, update, frames=len(counts), interval=180, repeat=False)
    animation.save(output, writer=PillowWriter(fps=6), dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="Archive produced by demo.py --save")
    parser.add_argument("--output", default="distribution.png", help="PNG figure or GIF animation")
    parser.add_argument("--burn-in", type=int, help="Override the saved burn-in")
    parser.add_argument("--smoothing", type=float, default=1.0, help="Bandwidth multiplier; larger means smoother")
    args = parser.parse_args()
    with np.load(args.run) as run:
        burn_in = int(run["burn_in"]) if args.burn_in is None else args.burn_in
        plot = animate_distribution if args.output.lower().endswith(".gif") else plot_distribution
        plot(run["path"], run["A"], run["b"], burn_in, args.output, args.smoothing)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
