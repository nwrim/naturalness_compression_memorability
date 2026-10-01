"""
Fits Bayesian linear regressions with a group-level random intercept between
image-level variables.

Random intercepts are defined only for set3 (grouped by category) and memcat
(grouped by subcategory). Naturalness is the human Likert rating ('naturalness')
for set3 and the ViTNat prediction ('vitnat') for memcat.

Models:
- jpeg_based_compressibility ~ naturalness
- canny_based_compressibility ~ naturalness
- crr ~ naturalness
- crr ~ jpeg_based_compressibility
- crr ~ canny_based_compressibility
"""

import numpy as np
import pandas as pd
import pymc as pm
import argparse

import sys
sys.path.append('../../src')

from paths import IDATA_PATH, STIMULI_PATH
from misc import load_data, standardize_columns, naturalness_column
from models import linear_regression_w_random_intercept
from linear_regressions import FILENAME_ALIAS

# Image sets with a random intercept, mapped to their grouping column
RANDOM_INTERCEPT_COL = {'set3': 'category', 'memcat': 'subcategory'}

N_DRAWS = 20000
N_TUNE = 5000
N_CHAINS = 4
TARGET_ACCEPT = 0.99
SEED = 0

def main(image_set):
    rng = np.random.default_rng(SEED)

    random_intercept_col = RANDOM_INTERCEPT_COL[image_set]
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
    if image_set == 'set3':
        category_df = pd.read_csv(STIMULI_PATH / f'{image_set}_category.csv')[['image_name', 'category']]
        df = df.merge(category_df, on='image_name', validate='1:1')

    scaled, _ = standardize_columns(df, measures)
    group = df[random_intercept_col].values

    for predictors, outcome in predictors_outcomes:
        predictors_data = [scaled[p] for p in predictors]
        outcome_data = scaled[outcome]

        model = linear_regression_w_random_intercept(predictors_data, outcome_data, group)
        with model:
            idata = pm.sample(draws=N_DRAWS, tune=N_TUNE, chains=N_CHAINS,
                              target_accept=TARGET_ACCEPT, random_seed=rng)

        predictor_tokens = '_'.join(FILENAME_ALIAS[p] for p in predictors)
        outcome_token = FILENAME_ALIAS[outcome]
        idata.to_netcdf(IDATA_PATH / f'{image_set}_lrri_o_{outcome_token}_p_{predictor_tokens}.nc')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit Bayesian linear regressions with a random intercept for an image set.')
    parser.add_argument('--image_set', type=str, choices=list(RANDOM_INTERCEPT_COL.keys()), required=True, help='Which image set to process (only set3 and memcat have a random intercept).')
    args = parser.parse_args()

    main(args.image_set)
