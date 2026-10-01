"""
Measure the out-of-sample performance of ViTNat predictions against
human-rated naturalness using Pearson correlation for each dataset.
"""

import sys
sys.path.append('../../src')

import pandas as pd
from scipy.stats import pearsonr

from paths import IMAGE_MEASURES_PATH

BEHAVIORAL_DATASETS = ['set1', 'set2']

def main():
    for image_set in BEHAVIORAL_DATASETS:
        nat_df = pd.read_csv(IMAGE_MEASURES_PATH / f'{image_set}_naturalness.csv')[['image_name', 'naturalness']]
        vitnat_df = pd.read_csv(IMAGE_MEASURES_PATH / f'{image_set}_vitnat.csv')[['image_name', 'vitnat']]
        df = nat_df.merge(vitnat_df, on='image_name', validate='1:1')
        r = pearsonr(df['naturalness'].values, df['vitnat'].values).statistic
        print(f'{image_set}: r = {r:.2f}')

    nat_df = pd.read_csv(IMAGE_MEASURES_PATH / 'schertz_naturalness.csv')[['image_name', 'naturalness']]
    vitnat_df = pd.read_csv(IMAGE_MEASURES_PATH / 'schertz_vitnat.csv')[['image_name', 'vitnat']]
    schertz = nat_df.merge(vitnat_df, on='image_name', validate='1:1')
    r = pearsonr(schertz['naturalness'].values, schertz['vitnat'].values).statistic
    print(f'schertz: r = {r:.2f}')

    nat_df = pd.read_csv(IMAGE_MEASURES_PATH / 'coburn_naturalness.csv')[['image_name', 'naturalness', 'category']]
    vitnat_df = pd.read_csv(IMAGE_MEASURES_PATH / 'coburn_vitnat.csv')[['image_name', 'vitnat']]
    coburn = nat_df.merge(vitnat_df, on='image_name', validate='1:1')
    for category in ['exterior', 'interior']:
        cat_df = coburn[coburn['category'] == category]
        r = pearsonr(cat_df['naturalness'].values, cat_df['vitnat'].values).statistic
        print(f'coburn ({category}): r = {r:.2f}')

if __name__ == '__main__':
    main()
