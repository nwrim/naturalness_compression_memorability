"""
Generates panels B and D of Figure 2 using example images with high and low naturalness.
"""

import os
import sys
sys.path.append(os.path.join('..', 'scripts'))

from skimage.io import imread
from skimage.feature import canny
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from image_processing import raw_avg
from compressibility import calculate_gradient_magnitude, calculate_beta
from plotting import save_reversed_weighted_image, threshold_image

plt.rcParams['svg.fonttype'] = 'none'

OUT_DIR = os.path.join('outputs', 'fig2')
EXAMPLE_IMAGES = {
    'low': 'uchicago.png',
    'high': 'suncheon.png'
}
COLOR_DICT = {
    'low': "#fab255",
    'high': "#43b284"
}
BINS = np.array(list(range(0, 101, 10)) + [500])

def main():
    fig, ax = plt.subplots(figsize=(5, 3))
    for cat, image_name in EXAMPLE_IMAGES.items():
        img = imread(os.path.join('example_image', image_name))
        grayscale_img = raw_avg(img)
        gradient_magnitude = calculate_gradient_magnitude(grayscale_img)
        edges = canny(grayscale_img, sigma=3, low_threshold=0, high_threshold=0, mode='reflect')
        gradient_magnitude = gradient_magnitude * edges
        save_reversed_weighted_image(gradient_magnitude, os.path.join(OUT_DIR, f'b_{cat}_0.png'))

        for i, thresholds in enumerate([(100,), (50, 60), (0, 10)]):
            thresholded_gradient_magnitude = threshold_image(gradient_magnitude, thresholds)
            save_reversed_weighted_image(thresholded_gradient_magnitude, os.path.join(OUT_DIR, f'b_{cat}_{i + 1}.png'))

        edge_count_by_magnitude_bin, _ = np.histogram(gradient_magnitude[edges], bins=BINS)

        prop_edge_count_by_magnitude_bin = edge_count_by_magnitude_bin / edge_count_by_magnitude_bin.sum()
        reversed_bins = np.max(BINS[:-1]) - BINS[:-1]

        scatter_idxs = [[1, 2, 3, 4, 6, 7, 8, 9], [0], [5], [10]]
        scatter_shapes = ['o', 'p', 'v', 'P', 'D']

        for idx, marker in zip(scatter_idxs, scatter_shapes):
            sns.scatterplot(x=reversed_bins[idx], y=prop_edge_count_by_magnitude_bin[idx], ax=ax, 
                            color=COLOR_DICT[cat], marker=marker, s=200, edgecolor='black')
        sns.regplot(x=reversed_bins, y=prop_edge_count_by_magnitude_bin, ax=ax,
                    ci=None, color=COLOR_DICT[cat], scatter=False, line_kws={'linewidth': 4})
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xticklabels([100, 80, 60, 40, 20, 0])

        # sanity check
        beta = calculate_beta(np.array(prop_edge_count_by_magnitude_bin), reversed_bins)

        # print other values that are needed for the figure
        print(cat)
        print('% of edges: ', np.round(prop_edge_count_by_magnitude_bin[[10, 5, 0]] * 100, 2))
        print('beta :', np.round(beta, 4))

    # Hide the right and top spines
    ax.spines[['right', 'top']].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'd.svg'))
    plt.close()

if __name__ == "__main__":
    main()