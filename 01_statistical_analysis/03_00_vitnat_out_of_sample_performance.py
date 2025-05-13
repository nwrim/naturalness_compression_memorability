"""
Evaluates the out-of-sample performance of the ViTNat model by computing Pearson correlations
between predicted naturalness scores and human Likert ratings.

Datasets:
- Image Set 1
- Image Set 2
- Schertz et al. (2018)
- Coburn et al. (2019) (evaluated separately for interior and exterior categories)

Outputs:
- Prints Pearson correlation coefficients to stdout for each dataset.
"""

import os
import pandas as pd
from scipy.stats import pearsonr

def main():
    filenames = ['set1', 'set2', 'schertz2018', 'coburn2019']

    for filename in filenames:
        # Load the data
        vitnat_df = pd.read_csv(os.path.join('..', 'data', 'image_level_measure', f'{filename}_vitnat.csv'))
        naturalness_df = pd.read_csv(os.path.join('..', 'data', 'image_level_measure', f'{filename}_naturalness.csv'))

        # Merge the dataframes on 'image_name'
        merged_df = pd.merge(vitnat_df, naturalness_df, on='image_name')
        assert merged_df.shape[0] == vitnat_df.shape[0] == naturalness_df.shape[0], "Mismatch in number of rows after merge"

        if filename != 'coburn2019':
            corr, _ = pearsonr(merged_df['vitnat'], merged_df['naturalness'])
            print(f"Pearson correlation for {filename}: {corr}")
        else:
            ext_df = merged_df[merged_df['category'] == 'exterior']
            ext_corr, _ = pearsonr(ext_df['vitnat'], ext_df['naturalness'])
            print(f"Pearson correlation for {filename} (exterior): {ext_corr}")

            int_df = merged_df[merged_df['category'] == 'interior']
            int_corr, _ = pearsonr(int_df['vitnat'], int_df['naturalness'])
            print(f"Pearson correlation for {filename} (interior): {int_corr}")

if __name__ == '__main__':
    main()