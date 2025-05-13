"""
Generates visualizations for Figure 1 using example images with high and low naturalness.
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
from compressibility import calculate_gradient_magnitude
from plotting import save_reversed_weighted_image

plt.rcParams['svg.fonttype'] = 'none'

OUT_DIR = os.path.join('outputs', 'fig1')
EXAMPLE_IMAGES = {
    'low': 'uchicago.png',
    'high': 'suncheon.png'
}

def save_heatmap(data, out_path, cmap='YlGnBu', vmin=0, vmax=5):
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(data.reshape(-1, 1), cmap=cmap, ax=ax, cbar=False, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path + '.svg')
    plt.close()

def main():
    rng = np.random.default_rng(2025)
    for cat, image_name in EXAMPLE_IMAGES.items():
        # Visualize edge gradient image
        img = imread(os.path.join('example_image', image_name))
        grayscale_img = raw_avg(img)
        gradient_magnitude = calculate_gradient_magnitude(grayscale_img)
        edges = canny(grayscale_img, sigma=3, low_threshold=0, high_threshold=0, mode='reflect')
        gradient_magnitude *= edges
        save_reversed_weighted_image(gradient_magnitude, os.path.join(OUT_DIR, f'{cat}_0.png'))

        # Row-wise gradient average, reduce (for visibility) and shuffle
        info = np.mean(gradient_magnitude, axis=1) # (540, )
        info = np.mean(info.reshape(-1, 18), axis=1) 
        rng.shuffle(info)
        save_heatmap(info, os.path.join(OUT_DIR, f'{cat}_1'))

        # arbitrarily threshold the values to remove low values
        info[info <= 4] = np.nan
        save_heatmap(info, os.path.join(OUT_DIR, f'{cat}_2'))

        # arbitrarily threshold the gradient magnitude to remove low values
        thresholded_gradient_magnitude = gradient_magnitude.copy()
        thresholded_gradient_magnitude[thresholded_gradient_magnitude < 70] = 0
        save_reversed_weighted_image(thresholded_gradient_magnitude, os.path.join(OUT_DIR, f'{cat}_3.png'))

    # Colorbar for reference
    example = np.linspace(0, 5, 10).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(example, cmap='YlGnBu', cbar=True, vmin=0, vmax=5)
    plt.savefig(os.path.join(OUT_DIR, 'cbar.svg'))
    plt.close()

if __name__ == "__main__":
    main()