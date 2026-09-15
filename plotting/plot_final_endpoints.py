"""Static scatter, density, and exact-reference views of saved trial endpoints."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from plotting.reference import marginal_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', help='Endpoint NPZ produced by plot_pw_density.py')
    args = parser.parse_args()
    source = Path(args.archive)
    with np.load(source) as data:
        samples = data['samples']
    n, d = samples.shape
    if d != 10 or n == 0:
        parser.error('Expected nonempty 10D endpoint data')
    assert np.all(samples >= 0) and np.all(samples.sum(axis=1) <= 1)
    edges, target, boundary = marginal_reference('simplex', d, (0, 1),
                                                  np.zeros(d), np.ones(d), bins=10)
    area = np.outer(np.diff(edges[0]), np.diff(edges[1]))
    densities = [np.histogram2d(samples[:, j], samples[:, j+1], bins=edges)[0]/(n*area)
                 for j in range(0, d, 2)]
    vmax = max(target.max(), max(x.max() for x in densities))
    fig, axes = plt.subplots(5, 3, figsize=(12, 16), layout='constrained')
    for row, j in enumerate(range(0, d, 2)):
        scatter, empirical, reference = axes[row]
        scatter.scatter(samples[:, j], samples[:, j+1], s=13, alpha=.65, color='#007f83')
        for ax, values in [(empirical, densities[row]), (reference, target)]:
            im = ax.imshow(values.T, origin='lower', extent=[0, 1, 0, 1],
                           vmin=0, vmax=vmax, cmap='viridis', interpolation='nearest')
            im.set_clip_path(Polygon(boundary, closed=True, transform=ax.transData))
        for ax, title in zip(axes[row], [f'All {n} final points', 'Final endpoint density', 'Exact target marginal']):
            ax.plot([0, 1, 0, 0], [0, 0, 1, 0], color='#dd6b20', linewidth=1.2)
            ax.set(title=title, xlabel=f'x{j+1}', ylabel=f'x{j+2}',
                   xlim=(-.02, 1.02), ylim=(-.02, 1.02), aspect='equal')
        fig.colorbar(im, ax=[empirical, reference], shrink=.75, label='Marginal density')
    fig.suptitle(f'Final output distribution: {n} independently initialized 10D simplex trials', fontsize=16)
    fig.supxlabel('One final point per trial, pooled across h; each row projects the same points.\n'
                  'Heatmaps use matching 10 x 10 bins and a shared color scale. Reference: bin averages of 90(1-x-y)^8.', fontsize=10)
    output = source.with_name(source.stem+'_final.png')
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
