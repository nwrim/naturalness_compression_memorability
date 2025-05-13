"""
Fits linear regression models for Image Set 4 (SUN397)

Requires:
- Scaled predictors and outcomes via StandardScaler objects from Image Set 3 (StandardScaler/set3_{var}.gz)

Models:
- jpeg ~ vitnat
- canny ~ vitnat
- resmem ~ vitnat
- resmem ~ jpeg
- resmem ~ vitnat + jpeg
- resmem ~ canny
- resmem ~ vitnat + canny

If ViTNat is used as a predictor, models are are corrected for measurement error (estimated from set2 human ratings and predictions).

Outputs:
- Saves ArviZ InferenceData objects to:
    idata/set4_lr_p_{predictors}_o_{outcome}.nc
"""

import os
import sys
sys.path.append(os.path.join('..', 'scripts'))

import joblib
import pandas as pd
from models import load_data, linear_regression, linear_regression_with_measurement_error

def main():
    SEED = 0
    measures = ['naturalness', 'hr', 'jpeg', 'canny']
    measure_mapping = {
        'vitnat': 'naturalness',
        'resmem': 'hr',
        'jpeg': 'jpeg',
        'canny': 'canny',
    }
    predictors_outcomes = [
        (['vitnat'], 'jpeg'), # jpeg-based compressibility ~ vitnat
        (['vitnat'], 'canny'), # canny-based compressibility ~ vitnat
        (['vitnat'], 'resmem'), # resmem ~ vitnat
        (['jpeg'], 'resmem'), # resmem ~ jpeg-based compressibility
        (['vitnat', 'jpeg'], 'resmem'), # resmem ~ vitnat + jpeg-based compressibility
        (['canny'], 'resmem'), # resmem ~ canny-based compressibility
        (['vitnat', 'canny'], 'resmem'), # resmem ~ vitnat + jpeg-based compressibility
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
    for predictors, outcome in predictors_outcomes:
        predictors_data = []
        for predictor in predictors:
            # Scale the predictor data
            scaled_data = scalers[measure_mapping[predictor]].transform(df[predictor].values.reshape(-1, 1)).flatten()
            predictors_data.append(scaled_data)
        outcome_data = scalers[measure_mapping[outcome]].transform(df[outcome].values.reshape(-1, 1)).flatten()
        if 'vitnat' in predictors:
            idata = linear_regression_with_measurement_error(predictors_data, outcome_data, seed=SEED, 
                                                             x_examp_true=set2_naturalness, x_examp_noisy=set2_vitnat, 
                                                             x_scale_examp=set_3_naturalness)
        else:
            idata = linear_regression(predictors_data, outcome_data, seed=SEED)

        # Save the result
        idata.to_netcdf(os.path.join('idata', f'set4_lr_p_{"_".join(predictors)}_o_{outcome}.nc'))

if __name__ == '__main__':
    main()