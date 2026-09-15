"""Compare independent chain endpoints with exact 10D simplex marginals."""
import argparse
import json
from pathlib import Path

import numpy as np
from plotting.reference import marginal_reference


def plot_density(record, prefix, animate=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.animation import FuncAnimation, PillowWriter

    rows = [r for r in record['runs'] if r['status'] == 'completed']
    samples = np.asarray([r['endpoint'] for r in rows])
    n = len(samples)
    if not n:
        return
    assert samples.shape == (n, 10)
    assert np.all(samples >= 0) and np.all(samples.sum(axis=1) <= 1)
    # Coarse bins are deliberate: 100 endpoints cannot resolve a fine grid.
    edges, target, boundary = marginal_reference('simplex', 10, (0, 1),
                                                  np.zeros(10), np.ones(10), bins=10)
    area = np.outer(np.diff(edges[0]), np.diff(edges[1]))
    def empirical(points, j):
        counts = np.histogram2d(points[:, j], points[:, j+1], bins=edges)[0]
        density = counts/(len(points)*area)
        assert np.isclose(np.sum(density*area), 1)
        return density
    densities = [empirical(samples, j) for j in range(0, 10, 2)]
    assert np.isclose(np.sum(target*area), 1)
    vmax = max(target.max(), max(x.max() for x in densities))
    fig, axes = plt.subplots(5, 2, figsize=(8, 15), layout='constrained')
    artists = []
    for i, j in enumerate(range(0, 10, 2)):
        for col, values in enumerate([densities[i], target]):
            ax = axes[i, col]
            im = ax.imshow(values.T, origin='lower', extent=[0, 1, 0, 1],
                           vmin=0, vmax=vmax, cmap='viridis', interpolation='nearest')
            im.set_clip_path(Polygon(boundary, closed=True, transform=ax.transData))
            ax.plot([0, 1, 0, 0], [0, 0, 1, 0], color='#dd6b20', linewidth=1.4)
            ax.set(xlabel=f'x{j+1}', ylabel=f'x{j+2}', xlim=(0, 1), ylim=(0, 1), aspect='equal')
            ax.set_title('Trial endpoint density' if col == 0 else 'Exact uniform-simplex marginal')
            if col == 0:
                artists.append(im)
        fig.colorbar(im, ax=list(axes[i]), shrink=.75, label='Marginal density')
    title = fig.suptitle(f'10D simplex: {n} independent trial endpoints, pooled across h')
    fig.supxlabel('One endpoint per completed chain; 10 x 10 bins, shared color scale.\n'
                  'Reference: exact bin averages of 90(1-x-y)^8. Noisy agreement is not a joint TV test.', fontsize=9)
    fig.savefig(str(prefix)+'_density.png', dpi=150)
    if animate:
        counts = np.unique(np.r_[np.geomspace(1, n, 30).astype(int), n])
        def update(count):
            for i, j in enumerate(range(0, 10, 2)):
                artists[i].set_data(empirical(samples[:count], j).T)
            title.set_text(f'10D simplex: {count}/{n} trial endpoints (early colors may saturate)')
            return artists
        animation = FuncAnimation(fig, update, frames=counts, interval=250, repeat=False)
        animation.save(str(prefix)+'_density.gif', writer=PillowWriter(fps=4), dpi=75)
    plt.close(fig)
    np.savez_compressed(str(prefix)+'_endpoints.npz', samples=samples,
                        h=np.asarray([r['h'] for r in rows]),
                        seeds=np.asarray([r['seed'] for r in rows]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results')
    parser.add_argument('--animate', action='store_true')
    args = parser.parse_args()
    source = Path(args.results)
    plot_density(json.loads(source.read_text()), source.with_suffix(''), args.animate)
