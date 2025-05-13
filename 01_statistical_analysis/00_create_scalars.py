"""
Fits and saves sklearn.preprocessing.StandardScaler objects for image-level variables.

Outputs:
- Saves fitted StandardScaler objects to:
    StandardScaler/set{image_set_index}_{variable}.gz
"""

import os
import sys
sys.path.append(os.path.join('..', 'scripts'))

from models import load_data
from sklearn.preprocessing import StandardScaler
import joblib

def main():
    measures = ['naturalness', 'crr', 'jpeg', 'canny']

    for image_set_index in [1, 2, 3]:
        # add the hit rate for image set 3 (this is because resmem estimates the hit rate, so we need to get the estimate for the hit rate for comparision)
        if image_set_index == 3:
            measures += ['hr'] 
        df = load_data(image_set_index)
        for v in measures:
            data = df[v].values.reshape(-1, 1) # Reshape to 2D for StandardScaler
            scaler = StandardScaler()
            scaler.fit(data)
            joblib.dump(scaler, os.path.join('StandardScaler', f'set{image_set_index}_{v}.gz'))

if __name__ == '__main__':
    main()