"""
Generates panels A and C of the Figure 2.
"""

import sys
from pathlib import Path

sys.path.append('../../src')

import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import seaborn as sns

from grayscale import raw_avg
from compressibility_metrics import (
    calculate_dct_by_tile,
    create_binary_matrices_by_frequency,
    calculate_sum_abs_coeff_by_freq,
    calculate_slope,
)
from scipy.fftpack import idctn


def reconstruct_image_from_masked_dct(dct_by_tile, mask, img_shape):
    reconstructed_img = np.zeros_like(dct_by_tile)
    for i in np.r_[:img_shape[0]:8]:
        for j in np.r_[:img_shape[1]:8]:
            tile = dct_by_tile[i:(i+8), j:(j+8)].copy()
            tile_mask = mask[:tile.shape[0], :tile.shape[1]]
            tile *= tile_mask
            reconstructed_img[i:(i+8), j:(j+8)] = idctn(tile, norm='ortho')
    return reconstructed_img

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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    binary_frequency_matrices = create_binary_matrices_by_frequency()

    fig, ax = plt.subplots(figsize=(5, 3))
    for cat, image_name in EXAMPLE_IMAGES.items():
        img = imread(IMAGE_DIR / image_name)
        grayscale_img = raw_avg(img)
        imsave(OUT_DIR / f'a_{cat}_0.png', grayscale_img.astype(np.uint8))

        dct_by_tile = calculate_dct_by_tile(grayscale_img)

        # reconstruct the image excluding each antidiagonal
        for l, k in enumerate([1, 2, 3]):
            reconstructed_img = reconstruct_image_from_masked_dct(
                dct_by_tile, 1 - binary_frequency_matrices[k], grayscale_img.shape)
            imsave(OUT_DIR / f'a_{cat}_{l + 1}.png', np.clip(reconstructed_img, 0, 255).astype(np.uint8))

        sum_abs_coeff_by_freq = calculate_sum_abs_coeff_by_freq(dct_by_tile)[1:]
        # express as percent of total energy so the y-axis matches the beta scale
        pct_abs_coeff_by_freq = sum_abs_coeff_by_freq / sum_abs_coeff_by_freq.sum() * 100

        # x runs low (left) to high (right) frequency, centered so the main antidiagonal
        # (u+v=7) sits at 0: it is the 7th AC frequency (index 6 of the [1:] array), so
        # subtract 6 -> range -6..7. The antidiagonals shown in panel A (frequencies
        # 1, 2, 3 -> indices 0, 1, 2) get distinct markers.
        freq_x = np.arange(len(pct_abs_coeff_by_freq)) - 6
        special_markers = {0: 's', 1: '^', 2: 'D'}
        other_idx = [i for i in range(len(pct_abs_coeff_by_freq)) if i not in special_markers]
        sns.scatterplot(x=freq_x[other_idx], y=pct_abs_coeff_by_freq[other_idx], ax=ax,
                        color=COLOR_DICT[cat], marker='o', s=200, edgecolor='black', zorder=2)
        for y_idx, marker in special_markers.items():
            sns.scatterplot(x=[freq_x[y_idx]], y=[pct_abs_coeff_by_freq[y_idx]], ax=ax,
                            color=COLOR_DICT[cat], marker=marker, s=200, edgecolor='black', zorder=2)
        sns.regplot(x=freq_x, y=pct_abs_coeff_by_freq, ax=ax,
                    ci=None, color=COLOR_DICT[cat], scatter=False, line_kws={'linewidth': 4, 'zorder': 1})

        # print values reported alongside the figure
        beta = calculate_slope(np.array(pct_abs_coeff_by_freq), freq_x)
        print(cat)
        print('% of energy: ', np.round(pct_abs_coeff_by_freq[[0, 1, 2]], 2))
        print('beta :', np.round(beta, 2))

    ax.spines[['right', 'top']].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'c.svg')
    plt.close()


if __name__ == "__main__":
    main()
