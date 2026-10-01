"""
Computes ViTNat naturalness predictions for an image set.
"""

import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
import argparse

import sys
sys.path.append('../../../src')

from paths import STIMULI_PATHS, IMAGE_MEASURES_PATH, STIMULI_PATH
from misc import list_image_files

def main(image_set, chunk):
    IMAGE_MEASURES_PATH.mkdir(parents=True, exist_ok=True)

    processor = ViTImageProcessor.from_pretrained('nwrim/ViTNat')
    model = ViTForImageClassification.from_pretrained('nwrim/ViTNat', num_labels=1)
    model.eval()

    image_dir = STIMULI_PATHS[image_set]
    if chunk is None:
        image_names = list_image_files(image_dir)
        output_path = IMAGE_MEASURES_PATH / f'{image_set}_vitnat.csv'
    else:
        chunk_df = pd.read_csv(STIMULI_PATH / 'chunks' / f'{image_set}_chunk.csv')
        image_names = chunk_df.loc[chunk_df['chunk'] == chunk, 'image_name'].tolist()
        output_path = IMAGE_MEASURES_PATH / 'local' / f'{image_set}_vitnat_chunk{chunk}.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions = []
    for image_name in tqdm(image_names):
        image_path = image_dir / image_name
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            pred = outputs.logits.item()
        predictions.append(pred)

    df = pd.DataFrame({
        'image_name': image_names,
        'vitnat': predictions,
    })
    df.sort_values(by='image_name', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(output_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute ViTNat predictions for an image set.')
    parser.add_argument('--image_set', type=str, choices=list(STIMULI_PATHS.keys()), required=True, help='Which image set to process.')
    parser.add_argument('--chunk', type=int, default=None, help='If set, process only this chunk (read from data/stimuli/chunks/{image_set}_chunk.csv); otherwise process the whole image set.')
    args = parser.parse_args()

    main(args.image_set, args.chunk)
