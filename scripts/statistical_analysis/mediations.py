"""
Fits Bayesian mediation models testing whether compressibility mediates the
relationship between naturalness and memorability.

Naturalness is the human Likert rating ('naturalness') for sets 1-3 and the
ViTNat prediction ('vitnat') for all other sets.

Models:
- naturalness -> jpeg_based_compressibility -> crr
- naturalness -> canny_based_compressibility -> crr
"""

import numpy as np
import pymc as pm
import argparse

import sys
sys.path.append('../../src')

from paths import IDATA_PATH, STIMULI_PATHS
from misc import load_data, standardize_columns, naturalness_column
from models import mediation_model
from linear_regressions import FILENAME_ALIAS

N_DRAWS = 20000
N_TUNE = 5000
N_CHAINS = 4
SEED = 0

def main(image_set):
    rng = np.random.default_rng(SEED)

    naturalness_col = naturalness_column(image_set)

    measures = [naturalness_col, 'jpeg_based_compressibility', 'canny_based_compressibility', 'crr']
    predictor_mediator_outcomes = [
        (naturalness_col, 'jpeg_based_compressibility', 'crr'),
        (naturalness_col, 'canny_based_compressibility', 'crr'),
    ]

    IDATA_PATH.mkdir(parents=True, exist_ok=True)

    df = load_data(image_set)

    scaled, _ = standardize_columns(df, measures)

    for predictor, mediator, outcome in predictor_mediator_outcomes:
        model = mediation_model(scaled[predictor], scaled[mediator], scaled[outcome])
        with model:
            idata = pm.sample(draws=N_DRAWS, tune=N_TUNE, chains=N_CHAINS, random_seed=rng)

        p, m, o = FILENAME_ALIAS[predictor], FILENAME_ALIAS[mediator], FILENAME_ALIAS[outcome]
        idata.to_netcdf(IDATA_PATH / f'{image_set}_med_p_{p}_m_{m}_o_{o}.nc')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit Bayesian mediation models for an image set.')
    parser.add_argument('--image_set', type=str, choices=list(STIMULI_PATHS.keys()), required=True, help='Which image set to process.')
    args = parser.parse_args()

    main(args.image_set)
