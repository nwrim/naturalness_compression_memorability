"""
Applies the ResMem model to compute predicted memorability scores for images in sets 1, 2, and 3.

Requires:
- Stimuli to be downloaded and extracted into
    ../data/stimuli/{set1, set2, set3}/
- Pretrained ResMem model and associated image transformer (https://github.com/Brain-Bridge-Lab/resmem).

Outputs:
- CSV files with ResMem memorability predictions for each image:
    ../data/image_level_measure/set{image_set_index}_resmem.csv
"""

import os
import sys
sys.path.append(os.path.join('..', 'scripts'))

import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from resmem import ResMem, transformer
from image_processing import list_all_files

def main():
    # load the pre-trained model
    model = ResMem(pretrained=True)
    model.eval()

    for image_set_index in [1, 2, 3]:
        image_dir = os.path.join('..', 'data', 'stimuli', f'set{image_set_index}')

        # list all files in the directory
        image_names = list_all_files(image_dir)
        predictions = []
        for image_name in tqdm(image_names):
            image_path = os.path.join(image_dir, image_name)
            try:
                image = Image.open(image_path)
                image = image.convert('RGB')
                image_x = transformer(image)
                with torch.no_grad():
                    pred = model(image_x.view(-1, 3, 227, 227)).item()
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
                pred = np.nan
            predictions.append(pred)
        # create a DataFrame with the results
        df = pd.DataFrame({
            'image_name': image_names,
            'resmem': predictions,
        })
        df.sort_values(by='image_name', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(os.path.join('..', 'data', 'image_level_measure', f'set{image_set_index}_resmem.csv'), index=False)

if __name__ == '__main__':
    main()