"""
Generates panels A and C of Figure 2 using example images with high and low naturalness.
"""

import os
import sys
sys.path.append(os.path.join('..', 'scripts'))

import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import seaborn as sns
from image_processing import raw_avg
from compressibility import calculate_dct_by_tile, create_binary_matrices_by_frequency, calculate_sum_abs_coeff_by_freq, calculate_beta
from plotting import save_reversed_weighted_image, reconstruct_image_from_one_dct_antidiagonal

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

def main():
    binary_frequency_matrices = create_binary_matrices_by_frequency()

    fig, ax = plt.subplots(figsize=(5, 3))
    for cat, image_name in EXAMPLE_IMAGES.items():
        img = imread(os.path.join('example_image', image_name))
        grayscale_img = raw_avg(img)
        imsave(os.path.join(OUT_DIR, f'a_{cat}_0.png'), grayscale_img.astype(np.uint8))

        dct_by_tile = calculate_dct_by_tile(grayscale_img)

        # reconstruct the image only using each antidiagonal
        for l, k in enumerate([1, 3, 5]):
            # initialize empty array to store the DCT result
            reconstructed_img = reconstruct_image_from_one_dct_antidiagonal(dct_by_tile, binary_frequency_matrices[k], grayscale_img.shape)
            save_reversed_weighted_image(reconstructed_img, os.path.join(OUT_DIR, f'a_{cat}_{l + 1}.png'))

        sum_abs_coeff_by_freq = calculate_sum_abs_coeff_by_freq(dct_by_tile)[1:8]
        prop_abs_coeff_by_freq = sum_abs_coeff_by_freq / sum_abs_coeff_by_freq.sum()

        scatter_xs = [[-5, -3, -1, 0], [-6], [-4], [-2]]
        scatter_y_idxs = [[1, 3, 5, 6], 0, 2, 4] 
        scatter_shapes = ['o', 's', '^', 'D']
        for x, y_idx, marker in zip(scatter_xs, scatter_y_idxs, scatter_shapes):
            sns.scatterplot(x=x, y=prop_abs_coeff_by_freq[y_idx], ax=ax, 
                            color=COLOR_DICT[cat], marker=marker, s=200, edgecolor='black')
        sns.regplot(x=np.arange(-6, 1, 1), y=prop_abs_coeff_by_freq, ax=ax, 
                    ci=None, color=COLOR_DICT[cat], scatter=False, line_kws={'linewidth': 4})
        
        # sanity check
        beta = calculate_beta(np.array(prop_abs_coeff_by_freq), np.arange(-6, 1, 1))
        
        # print other values that are needed for the figure
        print(cat)
        print('% of energy: ', np.round(prop_abs_coeff_by_freq[[0, 2, 4]] * 100, 2))
        print('beta :', np.round(beta, 2))

    # Hide the right and top spines
    ax.spines[['right', 'top']].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'c.svg'))
    plt.close()

if __name__ == "__main__":
    main()
