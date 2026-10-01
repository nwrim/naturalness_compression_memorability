"""
Reports the proportion of images with a naturalness rating of 5 or higher, for
Image Sets 1, 2, and 3.
"""

import pandas as pd

import sys
sys.path.append('../../src')

from paths import IMAGE_MEASURES_PATH

THRESHOLD = 5

def main():
    for image_set in ['set1', 'set2', 'set3']:
        df = pd.read_csv(IMAGE_MEASURES_PATH / f'{image_set}_naturalness.csv')
        n_total = len(df)
        n_high = (df['naturalness'] >= THRESHOLD).sum()
        print(f'{image_set}: {n_high / n_total * 100:.2f}% ({n_high} out of {n_total})')

if __name__ == '__main__':
    main()
