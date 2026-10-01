"""
Computes the proportion of the memorability ceiling explained by each predictor.
"""

import sys
sys.path.append('../../src')

import arviz as az
import numpy as np

from paths import IDATA_PATH, RELIABILITY_PATH
from linear_regressions import FILENAME_ALIAS

REGRESSIONS = [
    ('naturalness', 'crr'),
    ('jpeg_based_compressibility', 'crr'),
    ('canny_based_compressibility', 'crr'),
]

def main():
    for image_set in ['set1', 'set2', 'set3']:
        print(f'Image Set: {image_set}')

        # load the reliability ceiling for the image set
        split_half_corr_path = RELIABILITY_PATH / f'{image_set}_memorability_split_half_correlations.npy'
        correlations = np.load(split_half_corr_path)
        reliabilities = correlations * 2 / (1 + correlations)
        sqrt_reliabilities = np.sqrt(reliabilities)
        ceiling_mean = sqrt_reliabilities.mean()
        print(f'  Ceiling: {ceiling_mean:.2f}')

        for predictor, outcome in REGRESSIONS:
            idata_path = IDATA_PATH / f'{image_set}_lr_o_{FILENAME_ALIAS[outcome]}_p_{FILENAME_ALIAS[predictor]}.nc'
            samples = az.from_netcdf(idata_path)['posterior'].to_dataset()['beta_0'].values
            beta = np.abs(samples.mean())
            print(f'  {predictor} -> {outcome}: {beta / ceiling_mean * 100:.0f}% of ceiling ({beta:.2f})')
        print()

if __name__ == '__main__':
    main()