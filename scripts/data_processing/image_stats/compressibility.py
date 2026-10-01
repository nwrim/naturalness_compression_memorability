"""
Computes JPEG-based and Canny-based compressibility for an image set.
"""

import pandas as pd
from tqdm import tqdm
from skimage.io import imread
import argparse

import sys
sys.path.append('../../../src')

from paths import STIMULI_PATHS, IMAGE_MEASURES_PATH, STIMULI_PATH
from misc import list_image_files
from compressibility_metrics import jpeg_based_compressibility, canny_based_compressibility

def main(image_set, chunk):
    IMAGE_MEASURES_PATH.mkdir(parents=True, exist_ok=True)

    image_dir = STIMULI_PATHS[image_set]
    if chunk is None:
        # process the whole image set
        image_names = list_image_files(image_dir)
        output_path = IMAGE_MEASURES_PATH / f'{image_set}_compressibility.csv'
    else:
        # process only the images assigned to this chunk
        chunk_df = pd.read_csv(STIMULI_PATH / 'chunks' / f'{image_set}_chunk.csv')
        image_names = chunk_df.loc[chunk_df['chunk'] == chunk, 'image_name'].tolist()
        output_path = IMAGE_MEASURES_PATH / 'local' / f'{image_set}_compressibility_chunk{chunk}.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)

    jpeg_based_compressibilities = []
    canny_based_compressibilities = []
    for image_name in tqdm(image_names):
        image_path = image_dir / image_name
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
    df.to_csv(output_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute compressibility metrics for an image set.')
    parser.add_argument('--image_set', type=str, choices=list(STIMULI_PATHS.keys()), required=True, help='Which image set to process.')
    parser.add_argument('--chunk', type=int, default=None, help='If set, process only this chunk (read from data/stimuli/chunks/{image_set}_chunk.csv); otherwise process the whole image set.')
    args = parser.parse_args()

    main(args.image_set, args.chunk)
