"""
Applies the ViTNat model to compute naturalness predictions for out of sample image datasets.

Requires:
- Pretrained model `nwrim/ViTNat` from HuggingFace.
- Stimuli to be downloaded and extracted into
    ../data/stimuli/{set1, set2, schertz2018, coburn2019}/

Outputs:
- CSV files with ViTNat predictions for each image:
    ../data/image_level_measure/{dataset_name}_vitnat.csv
"""

import os
import sys
sys.path.append(os.path.join('..', 'scripts'))

import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from image_processing import list_all_files

def main():
    # load the pre-trained model
    processor = ViTImageProcessor.from_pretrained('nwrim/ViTNat')
    model = ViTForImageClassification.from_pretrained('nwrim/ViTNat', num_labels=1)
    model.eval()

    DATASETS = [
    ('set1', os.path.join('..', 'data', 'stimuli', 'set1')),
    ('set2', os.path.join('..', 'data', 'stimuli', 'set2')),
    ('schertz2018', os.path.join('..', 'data', 'stimuli', 'schertz2018')),
    ('coburn2019', os.path.join('..', 'data', 'stimuli', 'coburn2019')),
]

    for filename, image_dir in DATASETS:
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
        df.to_csv(os.path.join('..', 'data', 'image_level_measure', f'{filename}_vitnat.csv'), index=False)

if __name__ == '__main__':
    main()