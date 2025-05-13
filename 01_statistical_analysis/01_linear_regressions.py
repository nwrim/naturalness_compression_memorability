"""
Fits Bayesian linear regression models between image-level variables for image sets 1-3.

Requires:
- StandardScaler objects from `StandardScaler/set{image_set_index}_{var}.gz`

Models:
- jpeg ~ naturalness
- canny ~ naturalness
- crr ~ naturalness
- crr ~ jpeg
- crr ~ naturalness + jpeg
- crr ~ canny
- crr ~ naturalness + canny

Outputs:
- Saves ArviZ InferenceData objects to:
    idata/set{image_set_index}_lr_p_{predictor}_o_{outcome}.nc
"""

import os
import sys
sys.path.append(os.path.join('..', 'scripts'))

import joblib
from models import load_data, linear_regression

def main():
    SEED = 0
    measures = ['naturalness', 'crr', 'jpeg', 'canny']
    predictors_outcomes = [
        (['naturalness'], 'jpeg'), # jpeg-based compressibility ~ naturalness
        (['naturalness'], 'canny'), # canny-based compressibility ~ naturalness
        (['naturalness'], 'crr'), # corrected recognition rate ~ naturalness
        (['jpeg'], 'crr'), # corrected recognition rate ~ jpeg-based compressibility
        (['naturalness', 'jpeg'], 'crr'), # corrected recognition rate ~ naturalness + jpeg-based compressibility
        (['canny'], 'crr'), # corrected recognition rate ~ canny-based compressibility
        (['naturalness', 'canny'], 'crr'), # corrected recognition rate ~ naturalness + jpeg-based compressibility
    ]
    
    for image_set_index in [1, 2, 3]:
        # load the scalars for the predictors and outcomes
        scalers = {}
        for measure in measures:
            scalers[measure] = joblib.load(os.path.join('StandardScaler', f'set{image_set_index}_{measure}.gz'))

        # Load the data for the current image set index
        df = load_data(image_set_index)
        for predictors, outcome in predictors_outcomes:
            predictors_data = []
            for predictor in predictors:
                # Scale the predictor data
                scaled_data = scalers[predictor].transform(df[predictor].values.reshape(-1, 1)).flatten()
                predictors_data.append(scaled_data)
            
            outcome_data = scalers[outcome].transform(df[outcome].values.reshape(-1, 1)).flatten()
            idata = linear_regression(predictors_data, outcome_data, seed=SEED)

            # Save the result
            idata.to_netcdf(os.path.join('idata', f'set{image_set_index}_lr_p_{"_".join(predictors)}_o_{outcome}.nc'))

if __name__ == '__main__':
    main()