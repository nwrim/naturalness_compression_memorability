"""
Generates Supplementary Figure S4: relationship between compressibility (JPEG and Canny) and memorability (corrected recognition rate).
"""

import os
import sys
sys.path.append(os.path.join('..', 'scripts'))

import joblib
import arviz as az
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from models import load_data
from plotting import plot_linear_relationship

plt.rcParams['svg.fonttype'] = 'none'

def main():
    XTICKS_DICT =  {
        'jpeg': np.array([0.01, -0.01, -0.03, -0.05]),
        'canny': np.array([0.004, 0.002, 0.000])
    }
    YTICKS = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    RELATIONSHIPS = [('jpeg', 'crr'), ('canny', 'crr')]
    measures = ['jpeg', 'canny', 'crr']

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(6.75, 5))

    for col_idx, image_set_index in enumerate([1, 2, 3]):
        # load the scalars for the predictors and outcomes
        scalers = {}
        for measure in measures:
            scalers[measure] = joblib.load(os.path.join('..', '01_statistical_analysis', 'StandardScaler', f'set{image_set_index}_{measure}.gz'))

        # Load the data for the current image set index
        df = load_data(image_set_index)

        for row_idx, (x, y) in enumerate(RELATIONSHIPS):
            idata = az.from_netcdf(os.path.join('..', '01_statistical_analysis', 'idata', f'set{image_set_index}_lr_p_{x}_o_{y}.nc'))

            # generate the y model
            x_scaled = scalers[x].transform(df[x].values.reshape(-1, 1)).flatten()
            x_linspace = np.linspace(np.min(x_scaled), np.max(x_scaled), 1000)
            idata.posterior["y_model"] = idata.posterior["alpha"] + idata.posterior["beta_0"] * xr.DataArray(x_linspace)

            plot_linear_relationship(ax[row_idx][col_idx], df[x].values, df[y].values, idata,
                            x_scaler=scalers[x], y_scaler=scalers[y], xticks=XTICKS_DICT[x], yticks=YTICKS)

    plt.tight_layout(pad=0)
    plt.savefig(os.path.join('outputs', 'figS4.svg'))
    plt.close()

if __name__ == '__main__':
    main()