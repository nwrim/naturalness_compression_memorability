"""
Generates panels B and D of the Figure 2.
"""

import sys
from pathlib import Path

sys.path.append('../../src')

import numpy as np
from skimage.io import imread
from skimage.feature import canny
import matplotlib.pyplot as plt
import seaborn as sns

from grayscale import raw_avg
from plotting import save_inverted_image
from compressibility_metrics import calculate_gradient_magnitude, calculate_slope


def exclude_threshold_image(image, thresholds):
    excluded_image = image.copy()
    if len(thresholds) > 1:
        excluded_image[(excluded_image >= thresholds[0]) & (excluded_image < thresholds[1])] = 0
    else:
        excluded_image[excluded_image >= thresholds[0]] = 0
    return excluded_image


plt.rcParams['svg.fonttype'] = 'none'

IMAGE_DIR = Path('.')
OUT_DIR = Path('outputs') / 'fig2'
EXAMPLE_IMAGES = {
    'low': '0554.jpg',    # urban / low naturalness
    'high': '0998.jpg',  # natural / high naturalness
}
COLOR_DICT = {
    'low': "#ffa13c",
    'high': "#385a3a",
}
BINS = np.array(list(range(0, 101, 10)) + [500])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 3))
    for cat, image_name in EXAMPLE_IMAGES.items():
        img = imread(IMAGE_DIR / image_name)
        grayscale_img = raw_avg(img)
        gradient_magnitude = calculate_gradient_magnitude(grayscale_img)
        edges = canny(grayscale_img, sigma=3, low_threshold=0, high_threshold=0, mode='reflect')
        gradient_magnitude = gradient_magnitude * edges
        save_inverted_image(gradient_magnitude, OUT_DIR / f'b_{cat}_0.png')

        for i, thresholds in enumerate([(100,), (30, 40), (0, 10)]):
            excluded_gradient_magnitude = exclude_threshold_image(gradient_magnitude, thresholds)
            save_inverted_image(excluded_gradient_magnitude, OUT_DIR / f'b_{cat}_{i + 1}.png')

        edge_count_by_magnitude_bin, _ = np.histogram(gradient_magnitude[edges], bins=BINS)
        # express as percent of total edges so the y-axis matches fig2ac (panel C)
        pct_edge_count_by_magnitude_bin = edge_count_by_magnitude_bin / edge_count_by_magnitude_bin.sum() * 100
        reversed_bins = np.max(BINS[:-1]) - BINS[:-1]

        scatter_idxs = [[1, 2, 4, 5, 6, 7, 8, 9], [0], [3], [10]]
        scatter_shapes = ['o', 'p', 'v', 'P']
        for idx, marker in zip(scatter_idxs, scatter_shapes):
            sns.scatterplot(x=reversed_bins[idx], y=pct_edge_count_by_magnitude_bin[idx], ax=ax,
                            color=COLOR_DICT[cat], marker=marker, s=200, edgecolor='black', zorder=2)
        sns.regplot(x=reversed_bins, y=pct_edge_count_by_magnitude_bin, ax=ax,
                    ci=None, color=COLOR_DICT[cat], scatter=False, line_kws={'linewidth': 4, 'zorder': 1})
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xticklabels([100, 80, 60, 40, 20, 0])

        # print values reported alongside the figure
        beta = calculate_slope(np.array(pct_edge_count_by_magnitude_bin), reversed_bins)
        print(cat)
        print('% of edges: ', np.round(pct_edge_count_by_magnitude_bin[[10, 3, 0]], 2))
        print('beta :', np.round(beta, 2))

    ax.spines[['right', 'top']].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'd.svg')
    plt.close()


if __name__ == "__main__":
    main()
