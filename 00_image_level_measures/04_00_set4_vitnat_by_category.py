"""
Applies the ViTNat model to compute naturalness predictions for a single category in the SUN397 dataset.

Requires:
- SUN397 images organized by category in (see README for download link):
    ../data/stimuli/SUN397/
- Pretrained model `nwrim/ViTNat` from HuggingFace.

Usage:
    python 04_00_set4_vitnat_by_category.py <idx>
    (where <idx> is the zero-based index of the category to process)

Outputs:
- CSV file with ViTNat predictions for the specified category:
    ../data/image_level_measure/set4_by_category/set4_{category_name}_vitnat.csv
"""

import os
import sys
sys.path.append(os.path.join('..', 'scripts'))

import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from image_processing import list_all_files

def main(idx):
    # load the pre-trained model
    processor = ViTImageProcessor.from_pretrained('nwrim/ViTNat')
    model = ViTForImageClassification.from_pretrained('nwrim/ViTNat', num_labels=1)
    model.eval()

    # get the image directory based on the index
    with open(os.path.join('..', 'data', 'stimuli', 'sun397_categories.txt'), 'r') as f:
        categories = f.read().splitlines()
    category = categories[idx]
    image_dir = os.path.join('..', 'data', 'stimuli', 'SUN397') + category

    # list all files in the directory
    image_names = list_all_files(image_dir)

    predictions = []
    for image_name in tqdm(image_names):
        image_path = os.path.join(image_dir, image_name)
        try:
            image = Image.open(image_path)
            image = image.convert('RGB')
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                pred = outputs.logits.item()
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            pred = np.nan
        predictions.append(pred)
    # create a DataFrame with the results
    df = pd.DataFrame({
        'image_name': image_names,
        'vitnat': predictions,
    })
    df.sort_values(by='image_name', inplace=True)
    df.reset_index(drop=True, inplace=True)
    # save the DataFrame to a CSV file
    renamed_category = '_'.join(category.split('/')[2:])
    df.to_csv(os.path.join('..', 'data', 'image_level_measure', 'set4_by_category', f'set4_{renamed_category}_vitnat.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('idx', type=int, help='Index of the category to process')
    args = parser.parse_args()
    idx = args.idx
    main(idx)