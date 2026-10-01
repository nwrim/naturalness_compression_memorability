"""
Fits Bayesian linear regression models between image-level variables for an image set.

Naturalness is the human Likert rating ('naturalness') for sets 1-3 and the ViTNat
prediction ('vitnat') for all other sets.

Models:
- jpeg_based_compressibility ~ naturalness
- canny_based_compressibility ~ naturalness
- crr ~ naturalness
- crr ~ jpeg_based_compressibility
- crr ~ canny_based_compressibility
"""

import numpy as np
import pymc as pm
import argparse

import sys
sys.path.append('../../src')

from paths import IDATA_PATH, STIMULI_PATHS
from misc import load_data, standardize_columns, naturalness_column
from models import linear_regression

# Short tokens used in output filenames (the long column names would collide
# with the underscore-delimited _p_/_o_ filename scheme).
FILENAME_ALIAS = {
    'naturalness': 'nat',
    'vitnat': 'vitnat',
    'jpeg_based_compressibility': 'jpeg',
    'canny_based_compressibility': 'canny',
    'crr': 'crr',
}

N_DRAWS = 20000
N_TUNE = 5000
N_CHAINS = 4
SEED = 0

def main(image_set):
    rng = np.random.default_rng(SEED)

    naturalness_col = naturalness_column(image_set)

    measures = [naturalness_col, 'jpeg_based_compressibility', 'canny_based_compressibility', 'crr']
    predictors_outcomes = [
        ([naturalness_col], 'jpeg_based_compressibility'),
        ([naturalness_col], 'canny_based_compressibility'),
        ([naturalness_col], 'crr'),
        (['jpeg_based_compressibility'], 'crr'),
        (['canny_based_compressibility'], 'crr'),
    ]

    IDATA_PATH.mkdir(parents=True, exist_ok=True)

    df = load_data(image_set)

    scaled, _ = standardize_columns(df, measures)

    for predictors, outcome in predictors_outcomes:
        predictors_data = [scaled[p] for p in predictors]
        outcome_data = scaled[outcome]

        model = linear_regression(predictors_data, outcome_data)
        with model:
            idata = pm.sample(draws=N_DRAWS, tune=N_TUNE, chains=N_CHAINS, random_seed=rng)

        predictor_tokens = '_'.join(FILENAME_ALIAS[p] for p in predictors)
        outcome_token = FILENAME_ALIAS[outcome]
        idata.to_netcdf(IDATA_PATH / f'{image_set}_lr_o_{outcome_token}_p_{predictor_tokens}.nc')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit Bayesian linear regressions for an image set.')
    parser.add_argument('--image_set', type=str, choices=list(STIMULI_PATHS.keys()), required=True, help='Which image set to process.')
    args = parser.parse_args()

    main(args.image_set)
