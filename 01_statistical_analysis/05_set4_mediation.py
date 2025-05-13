"""
Fits Bayesian mediation models for Image Set 4 (SUN397), testing whether compressibility
mediates the relationship between ViTNat predictions and ResMem-predicted memorability.

Requires:
- Scaled predictors and outcomes via StandardScaler objects from Image Set 3 (StandardScaler/set3_{var}.gz)

Models:
- vitnat -> jpeg -> resmem
- vitnat -> canny -> resmem

Outputs:
- Saves ArviZ InferenceData objects to:
    idata/set4_m_p_{predictor}_m_{mediator}_o_{outcome}.nc
"""

import os
import sys
sys.path.append(os.path.join('..', 'scripts'))

import joblib
import pandas as pd
from models import load_data, mediation_model_with_measurement_error

def main():
    SEED = 0
    measures = ['naturalness', 'hr', 'jpeg', 'canny']
    measure_mapping = {
        'vitnat': 'naturalness',
        'resmem': 'hr',
        'jpeg': 'jpeg',
        'canny': 'canny',
    }
    predictor_mediator_outcomes = [
        ('vitnat', 'jpeg', 'resmem'), # vitnat -> jpeg-based compressibility -> resmem
        ('vitnat', 'canny', 'resmem'), # vitnat -> canny-based compressibility -> resmem
    ]

    # load the scalars for the predictors and outcomes
    scalers = {}
    for measure in measures:
        scalers[measure] = joblib.load(os.path.join('StandardScaler', f'set3_{measure}.gz'))
        # need to add all the set3 stuff including the predicted/actual naturalness
    
    # load the data from image set 2 and set 3 for estimating the measurement error
    set2_naturalness_df = pd.read_csv(os.path.join('..', 'data', 'image_level_measure', 'set2_naturalness.csv'))
    set2_naturalness = scalers['naturalness'].transform(set2_naturalness_df['naturalness'].values.reshape(-1, 1)).flatten()
    set2_vitnat_df = pd.read_csv(os.path.join('..', 'data', 'image_level_measure', 'set2_vitnat.csv'))
    set2_vitnat = scalers['naturalness'].transform(set2_vitnat_df['vitnat'].values.reshape(-1, 1)).flatten()
    set_3_naturalness_df = pd.read_csv(os.path.join('..', 'data', 'image_level_measure', 'set3_naturalness.csv'))
    set_3_naturalness = scalers['naturalness'].transform(set_3_naturalness_df['naturalness'].values.reshape(-1, 1)).flatten()

    # Load the data for the current image set index
    df = load_data(4)
    for predictor, mediator, outcome in predictor_mediator_outcomes:
        predictor_data = scalers[measure_mapping[predictor]].transform(df[predictor].values.reshape(-1, 1)).flatten()
        mediator_data = scalers[measure_mapping[mediator]].transform(df[mediator].values.reshape(-1, 1)).flatten()
        outcome_data = scalers[measure_mapping[outcome]].transform(df[outcome].values.reshape(-1, 1)).flatten()

        idata = mediation_model_with_measurement_error(
            predictor_data, mediator_data, outcome_data, seed=SEED, 
            x_examp_true=set2_naturalness, x_examp_noisy=set2_vitnat, 
            x_scale_examp=set_3_naturalness
        )

        # Save the result
        idata.to_netcdf(os.path.join('idata', f'set4_m_p_{predictor}_m_{mediator}_o_{outcome}.nc'))

if __name__ == '__main__':
    main()
