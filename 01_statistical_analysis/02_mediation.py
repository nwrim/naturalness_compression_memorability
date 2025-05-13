"""
Fits Bayesian mediation models testing whether compressibility mediates the relationship between naturalness and memorability.

Requires:
- StandardScaler objects in StandardScaler/set{image_set_index}_{var}.gz

Models (per image set):
- naturalness -> jpeg -> crr
- naturalness -> canny -> crr

Outputs:
- Saves ArviZ InferenceData objects to:
    idata/set{image_set_index}_m_p_{predictor}_m_{mediator}_o_{outcome}.nc
"""

import os
import sys
sys.path.append(os.path.join('..', 'scripts'))

import joblib
from models import load_data, mediation_model

def main():
    SEED = 0
    measures = ['naturalness', 'crr', 'jpeg', 'canny']
    predictor_mediator_outcomes = [
        ('naturalness', 'jpeg', 'crr'), # naturalness -> jpeg-based compressibility -> corrected recognition rate
        ('naturalness', 'canny', 'crr'), # naturalness -> canny-based compressibility -> corrected recognition rate
    ]
    
    for image_set_index in [1, 2, 3]:
        # load the scalars for the predictors and outcomes
        scalers = {}
        for measure in measures:
            scalers[measure] = joblib.load(os.path.join('StandardScaler', f'set{image_set_index}_{measure}.gz'))

        # Load the data for the current image set index
        df = load_data(image_set_index)
        for predictor, mediator, outcome in predictor_mediator_outcomes:
            predictor_data = scalers[predictor].transform(df[predictor].values.reshape(-1, 1)).flatten()
            mediator_data = scalers[mediator].transform(df[mediator].values.reshape(-1, 1)).flatten()
            outcome_data = scalers[outcome].transform(df[outcome].values.reshape(-1, 1)).flatten()
            idata = mediation_model(predictor_data, mediator_data, outcome_data, seed=SEED)

            # Save the result
            idata.to_netcdf(os.path.join('idata', f'set{image_set_index}_m_p_{predictor}_m_{mediator}_o_{outcome}.nc'))

if __name__ == '__main__':
    main()