"""
Generates visualizations for Figure S1.
"""

import os
import sys
sys.path.append(os.path.join('..', 'scripts'))

import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.draw import rectangle_perimeter
from image_processing import raw_avg
from compressibility import calculate_dct_by_tile
from scipy.fftpack import idctn

plt.rcParams['svg.fonttype'] = 'none'

OUTPUT_DIR = os.path.join('outputs', 'figS1')
EXAMPLE_TILE_LOCATION = (320, 696)

def save_image_with_tile_overlay(grayscale_img, tile_location, out_path):
    """
    Save grayscale image with an overlaid red box indicating a selected 8x8 tile.
    """
    overlay = np.stack([grayscale_img] * 3, axis=-1)
    rr, cc = rectangle_perimeter(tile_location, extent=(8, 8), shape=grayscale_img.shape)
    overlay[rr, cc, :] = [0, 255, 0]  # Green color
    imsave(out_path, overlay.astype(np.uint8))

def plot_tile_and_dct(tile, dct_tile, out_path):
    """
    Plot side-by-side view of tile pixel intensities and DCT coefficients.
    """
    dct_tile[0, 0] = 0  # Zero out DC componen
    dct_tile = np.abs(dct_tile)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    sns.heatmap(tile, cmap='gray', ax=ax[0], square=True, cbar=False, vmin=0, vmax=255)
    sns.heatmap(dct_tile, cmap='Greens', ax=ax[1], square=True, cbar=False)

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def visualize_dct_basis_functions(out_path):
    """
    Create a visualization of each DCT basis function in an 8x8 grid.
    """
    heatmap = np.zeros((8 * 8 + 7, 8 * 8 + 7))  # Include lines between tiles
    for i in range(8):
        for j in range(8):
            if i == 0 and j == 0:
                continue  # Skip DC component
            basis = np.zeros((8, 8))
            basis[i, j] = 255
            spatial_tile = idctn(basis, norm='ortho')
            heatmap[i * 9:i * 9 + 8, j * 9:j * 9 + 8] = spatial_tile

    heatmap[8::9, :] = 255  # Horizontal grid lines
    heatmap[:, 8::9] = 255  # Vertical grid lines

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(heatmap, cmap='gray_r', ax=ax, square=True, cbar=False, vmin=0, vmax=255)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    img = imread(os.path.join('example_image', 'suncheon.png'))
    grayscale_img = raw_avg(img)

    # Grayscale Image with red tile overlay
    save_image_with_tile_overlay(grayscale_img, EXAMPLE_TILE_LOCATION,
                                 os.path.join(OUTPUT_DIR, '0.png'))

    # raw pixel intensities vs DCT coefficients
    dct_img = calculate_dct_by_tile(grayscale_img)
    tile = grayscale_img[EXAMPLE_TILE_LOCATION[0]:EXAMPLE_TILE_LOCATION[0]+8,
                         EXAMPLE_TILE_LOCATION[1]:EXAMPLE_TILE_LOCATION[1]+8].copy()
    dct_tile = dct_img[EXAMPLE_TILE_LOCATION[0]:EXAMPLE_TILE_LOCATION[0]+8,
                       EXAMPLE_TILE_LOCATION[1]:EXAMPLE_TILE_LOCATION[1]+8].copy()
    
    plot_tile_and_dct(tile, dct_tile, os.path.join(OUTPUT_DIR, '1.svg'))

    # DCT basis functions
    visualize_dct_basis_functions(os.path.join(OUTPUT_DIR, '2.svg'))

if __name__ == "__main__":
    main()
