"""
Computes JPEG-based and Canny-based compressibility for all images across three image sets.

Requires:
- Stimuli from OSF to be downloaded and extracted into
    `../data/stimuli/set{image_set_index}/` (see README for link to OSF).

Outputs:
- CSV files containing image names and their compressibility scores for each image set.
  Saved to:
    ../data/image_level_measure/set{image_set_index}_compressibility.csv
"""

import os
import sys
sys.path.append(os.path.join('..', 'scripts'))

import pandas as pd
from tqdm import tqdm
from skimage.io import imread
from image_processing import list_all_files
from compressibility import jpeg_based_compressibility, canny_based_compressibility

def main():
    for image_set_index in [1, 2, 3]:
        image_dir = os.path.join('..', 'data', 'stimuli', f'set{image_set_index}')
        image_names = list_all_files(image_dir)
        
        jpeg_based_compressibilities = []
        canny_based_compressibilities = []
        for image_name in tqdm(image_names):
            image_path = os.path.join(image_dir, image_name)
            image = imread(image_path)
            jpeg_comp = jpeg_based_compressibility(image)
            jpeg_based_compressibilities.append(jpeg_comp)
            canny_comp = canny_based_compressibility(image)
            canny_based_compressibilities.append(canny_comp)

        df = pd.DataFrame({
            'image_name': image_names,
            'jpeg_based_compressibility': jpeg_based_compressibilities,
            'canny_based_compressibility': canny_based_compressibilities
        })
        df.sort_values(by='image_name', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(os.path.join('..', 'data', 'image_level_measure', f'set{image_set_index}_compressibility.csv'), index=False)

if __name__ == '__main__':
    main()