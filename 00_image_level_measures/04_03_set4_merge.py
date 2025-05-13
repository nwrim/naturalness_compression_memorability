"""
Merges per-category prediction files (ViTNat, ResMem, compressibility) for SUN397 into single dataset-wide CSVs.

Requires:
- Per-category prediction files located in (output of 04_00_set4_vitnat_by_category.py, 04_01_set4_compressibility_by_category.py, 04_02_set4_resmem_by_category.py):
    ../data/image_level_measure/set4_by_category/set4_{category}_{metric}.csv

Outputs:
- Merged CSV files saved to:
    ../data/image_level_measure/set4_{metric}.csv
  where {metric} is one of: vitnat, resmem, compressibility

Performs sanity checks to ensure no NaNs are present in output columns.
"""

import os
import pandas as pd

def main():
    with open(os.path.join('..', 'data', 'stimuli', 'sun397_categories.txt'), 'r') as f:
        categories = f.read().splitlines()
    renamed_categories = ['_'.join(cat.split('/')[2:]) for cat in categories]

    for metric in ['compressibility', 'resmem', 'vitnat']:
        dfs = []
        for renamed_category in renamed_categories:
            df = pd.read_csv(os.path.join('..', 'data', 'image_level_measure', 'set4_by_category', f'set4_{renamed_category}_{metric}.csv'))
            df['category'] = renamed_category
            dfs.append(df)
        concat_df = pd.concat(dfs)
        # sanity check - make sure there are no na values in the metric column
        if metric == 'compressibility':
            assert concat_df['jpeg_based_compressibility'].isna().sum() == 0
            assert concat_df['canny_based_compressibility'].isna().sum() == 0
        else:
            assert concat_df[metric].isna().sum() == 0
        concat_df.to_csv(os.path.join('..', 'data', 'image_level_measure', f'set4_{metric}.csv'), index=False)

if __name__ == '__main__':
    main()