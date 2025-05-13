"""
Evaluates the out-of-sample performance of the ResMem model by computing Pearson correlations
between predicted memorability scores and behavioral memorability data (CRR and hit rate).

Datasets:
- Image Sets 1, 2, and 3

Outputs:
- Prints Pearson correlation coefficients between ResMem predictions and:
    - Corrected recognition rate (CRR)
    - Hit rate (HR)
"""

import os
import pandas as pd
from scipy.stats import pearsonr

def main():
    for image_set_index in [1, 2, 3]:
        # Load the data
        resmem_df = pd.read_csv(os.path.join('..', 'data', 'image_level_measure', f'set{image_set_index}_resmem.csv'))
        memorability_df = pd.read_csv(os.path.join('..', 'data', 'image_level_measure', f'set{image_set_index}_memorability.csv'))

        # Merge the dataframes on 'image_name'
        merged_df = pd.merge(resmem_df, memorability_df, on='image_name')
        assert merged_df.shape[0] == resmem_df.shape[0] == memorability_df.shape[0], "Mismatch in number of rows after merge"

        for metric in ['crr', 'hr']:
            corr, _ = pearsonr(merged_df['resmem'], merged_df[f'target_{metric}'])
            print(f"Pearson correlation with {metric} for set {image_set_index}: {corr}")

if __name__ == '__main__':
    main()