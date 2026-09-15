"""Paired projections with geometric outlines and exact uniform references."""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon
from plotting.reference import reference_body, marginal_reference


def coordinate_groups(d):
    return [tuple(range(i, min(i+2, d))) for i in range(0, d, 2)]


def plot_pairs(path, burn_in, output, animate=False, frames=45, *, A, b):
    """Compare empirical and exact marginal histograms on identical fixed bins.

    All retained points count. Pair panels share a color scale; 1D panels
    share a y scale. Animation keeps reference panels and scales fixed.
    """
    path = np.asarray(path)
    if path.ndim != 2 or path.shape[1] < 1 or not np.isfinite(path).all():
        raise ValueError("Expected a finite path with shape (steps+1, d).")
    if not 0 <= burn_in < len(path)-1 or frames < 2:
        raise ValueError("Need retained samples and at least two animation frames.")
    samples = path[burn_in+1:]
    d = samples.shape[1]
    kind, lo, hi = reference_body(A, b)
    groups = coordinate_groups(d)
    counts = np.unique(np.r_[np.geomspace(1, len(samples), frames).astype(int), len(samples)]) if animate else [len(samples)]
    fig, axes = plt.subplots(len(groups), 2, figsize=(9, 3.4*len(groups)),
                             squeeze=False, layout="constrained")
    artists = []
    for row, group in enumerate(groups):
        left, right = axes[row]
        edges, target, boundary = marginal_reference(kind, d, group, lo, hi)
        if len(group) == 2:
            values = [np.histogram2d(*samples[:n, group].T, bins=edges)[0]
                      / n / np.outer(np.diff(edges[0]), np.diff(edges[1])) for n in counts]
            # Early concentrations saturate in animations; scales never change.
            vmax = 2*target.max() if animate else max(target.max(), values[-1].max())
            extent = [edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]]
            images = []
            for ax, density in ((left, values[0]), (right, target)):
                im = ax.imshow(density.T, origin="lower", extent=extent, vmin=0, vmax=vmax,
                               cmap="viridis", interpolation="nearest", aspect="equal")
                im.set_clip_path(Polygon(boundary, closed=True, transform=ax.transData))
                outline = np.vstack((boundary, boundary[0]))
                ax.plot(*outline.T, color="#dd6b20", linewidth=1.5)
                padx, pady = (extent[1]-extent[0])*0.05, (extent[3]-extent[2])*0.05
                ax.set(xlim=(extent[0]-padx, extent[1]+padx), ylim=(extent[2]-pady, extent[3]+pady),
                       xlabel=f"x{group[0]+1}", ylabel=f"x{group[1]+1}")
                images.append(im)
            fig.colorbar(images[1], ax=list(axes[row]), shrink=0.8,
                         extend="max" if animate else "neither", label="Marginal density")
            artists.append((images[0], values, True))
        else:
            values = [np.histogram(samples[:n, group[0]], bins=edges[0])[0]/n/np.diff(edges[0]) for n in counts]
            ymax = max(target.max(), max(v.max() for v in values))*1.05
            line = left.stairs(values[0], edges[0], color="#216869")
            right.stairs(target, edges[0], color="#216869")
            for ax in (left, right):
                ax.axvline(edges[0][0], color="#dd6b20", linewidth=1)
                ax.axvline(edges[0][-1], color="#dd6b20", linewidth=1)
                ax.set(ylim=(0, ymax), xlabel=f"x{group[0]+1}", ylabel="Marginal density")
            artists.append((line, values, False))
        coords = ", ".join(f"x{i+1}" for i in group)
        left.set_title(f"{coords}: sampled distribution")
        right.set_title(f"{coords}: uniform-on-body reference")
    title = fig.suptitle("")
    fig.supxlabel("Orange: projected boundary. Reference: exact bin probabilities, not a fitted distribution.\nAll retained samples contribute. Marginal agreement does not certify joint mixing.", fontsize=9)

    def update(frame):
        for artist, values, is_image in artists:
            if is_image:
                artist.set_data(values[frame].T)
            else:
                artist.set_data(values=values[frame])
        title.set_text(f"{d}D {kind} | transition {burn_in+counts[frame]:,} | retained: {counts[frame]:,}")

    if animate:
        animation = FuncAnimation(fig, update, frames=len(counts), interval=180, repeat=False)
        animation.save(output, writer=PillowWriter(fps=6), dpi=100)
    else:
        update(0)
        fig.savefig(output, dpi=150)
    plt.close(fig)
