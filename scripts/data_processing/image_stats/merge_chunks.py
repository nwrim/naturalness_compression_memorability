"""
Merges per-chunk CSVs produced by chunked processing scripts into a single output file.
"""

import pandas as pd
import argparse

import sys
sys.path.append('../../../src')

from paths import STIMULI_PATHS, IMAGE_MEASURES_PATH, STIMULI_PATH

METRICS = ['compressibility', 'vitnat']

def main(image_set, metric):
    local_dir = IMAGE_MEASURES_PATH / 'local'

    chunk_df = pd.read_csv(STIMULI_PATH / 'chunks' / f'{image_set}_chunk.csv')
    n_chunks = chunk_df['chunk'].nunique()

    chunk_files = sorted(local_dir.glob(f'{image_set}_{metric}_chunk*.csv'))
    if len(chunk_files) != n_chunks:
        raise FileNotFoundError(
            f'Expected {n_chunks} chunk files for {image_set}/{metric}, found {len(chunk_files)}.'
        )

    df = pd.concat([pd.read_csv(f) for f in chunk_files], ignore_index=True)
    df.sort_values(by='image_name', inplace=True)
    df.reset_index(drop=True, inplace=True)

    output_path = IMAGE_MEASURES_PATH / f'{image_set}_{metric}.csv'
    df.to_csv(output_path, index=False)

    # verify the saved file matches before deleting chunks
    saved_rows = len(pd.read_csv(output_path))
    if saved_rows != len(df):
        raise RuntimeError(f'Row count mismatch after save ({saved_rows} != {len(df)}); chunk files preserved.')

    for f in chunk_files:
        f.unlink()
    print(f'Saved {output_path} ({len(df)} rows from {len(chunk_files)} chunks); chunk files deleted.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge chunk CSVs into a single output file.')
    parser.add_argument('--image_set', type=str, choices=list(STIMULI_PATHS.keys()), required=True, help='Which image set to merge.')
    parser.add_argument('--metric', type=str, choices=METRICS, required=True, help='Which metric to merge.')
    args = parser.parse_args()

    main(args.image_set, args.metric)
