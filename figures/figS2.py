"""
Generates visualizations for Figure S2.
"""

import os
import sys
sys.path.append(os.path.join('..', 'scripts'))

from skimage.io import imread, imsave
import scipy.ndimage as ndi
from skimage.feature import canny
import numpy as np
import matplotlib.pyplot as plt
from image_processing import raw_avg
from compressibility import calculate_gradient_magnitude
from plotting import save_reversed_weighted_image

plt.rcParams['svg.fonttype'] = 'none'

OUTPUT_DIR = os.path.join('outputs', 'figS2')

def main():
    img = imread(os.path.join('example_image', 'uchicago.png'))
    grayscale_img = raw_avg(img)
    imsave(os.path.join(OUTPUT_DIR, '0.png'), grayscale_img.astype(np.uint8))

    smoothed = ndi.gaussian_filter(grayscale_img, sigma=3, mode='reflect')
    imsave(os.path.join(OUTPUT_DIR, '1.png'), smoothed.astype(np.uint8))

    gradient_magnitude = calculate_gradient_magnitude(grayscale_img)
    save_reversed_weighted_image(gradient_magnitude, os.path.join(OUTPUT_DIR, '2.png'))

    canny_output = canny(grayscale_img, sigma=3,
                        low_threshold=0,
                        high_threshold=0, mode='reflect')
    gradient_magnitude *= canny_output
    save_reversed_weighted_image(gradient_magnitude, os.path.join(OUTPUT_DIR, '3.png'))

if __name__ == "__main__":
    main()