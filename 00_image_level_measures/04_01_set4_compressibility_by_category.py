"""
Computes JPEG-based and Canny-based compressibility for a single category in the SUN397 dataset.

Requires:
- SUN397 images organized by category in (see README for download link):
    ../data/stimuli/SUN397/

Usage:
    python 04_01_set4_compressibility_by_category.py <idx>
    (where <idx> is the zero-based index of the category to process)

Outputs:
- CSV file with compressibility scores for the specified category:
    ../data/image_level_measure/set4_by_category/set4_{category_name}_compressibility.csv
"""

import os
import sys
sys.path.append(os.path.join('..', 'scripts'))

import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from image_processing import list_all_files
from compressibility import jpeg_based_compressibility, canny_based_compressibility

def main(idx):
    # get the image directory based on the index
    with open(os.path.join('..', 'data', 'stimuli', 'sun397_categories.txt'), 'r') as f:
        categories = f.read().splitlines()
    category = categories[idx]
    image_dir = os.path.join('..', 'data', 'stimuli', 'SUN397') + category

    # list all files in the directory
    image_names = list_all_files(image_dir)

    jpeg_based_compressibilities = []
    canny_based_compressibilities = []
    for image_name in tqdm(image_names):
            image_path = os.path.join(image_dir, image_name)
            image = imread(image_path)
            try:
                jpeg_comp = jpeg_based_compressibility(image)
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
                jpeg_comp = np.nan
            try:
                canny_comp = canny_based_compressibility(image)
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
                canny_comp = np.nan
            jpeg_based_compressibilities.append(jpeg_comp)
            canny_based_compressibilities.append(canny_comp)
    # create a DataFrame with the results
    df = pd.DataFrame({
        'image_name': image_names,
        'jpeg_based_compressibility': jpeg_based_compressibilities,
        'canny_based_compressibility': canny_based_compressibilities
    })
    df.sort_values(by='image_name', inplace=True)
    df.reset_index(drop=True, inplace=True)
    # save the DataFrame to a CSV file
    renamed_category = '_'.join(category.split('/')[2:])
    df.to_csv(os.path.join('..', 'data', 'image_level_measure', 'set4_by_category', f'set4_{renamed_category}_compressibility.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('idx', type=int, help='Index of the category to process')
    args = parser.parse_args()
    idx = args.idx
    main(idx)