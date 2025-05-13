"""
Generates Figure 5, but with scatter plots instead of heatmaps.
"""

import os
import sys
sys.path.append(os.path.join('..', 'scripts'))

import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import xarray as xr
import arviz as az
from models import load_data
from plotting import plot_linear_relationship

plt.rcParams['svg.fonttype'] = 'none'

XTICKS = np.array([1, 4, 7])
YTICKS_DICT =  {
    'jpeg': np.array([0.03, -0.01, -0.05, -0.09]),
    'canny': np.array([0.004, 0.002, 0.000, -0.002]),
    'resmem': np.array([0.5, 0.7, 0.9])
}
MEASURE_MAPPING = {
    'vitnat': 'naturalness',
    'jpeg': 'jpeg',
    'canny': 'canny',
    'resmem': 'hr'
}
RELATIONSHIPS = [('vitnat', 'jpeg'), ('vitnat', 'canny'), ('vitnat', 'resmem')]
CMAP = clr.LinearSegmentedColormap.from_list('lavender', [(1, 1, 1), '#B57EDC'], N=256)

def main():
    scalers = {}
    for measure in ['vitnat', 'jpeg', 'canny', 'resmem']:
        scalers[measure] = joblib.load(os.path.join('..', '01_statistical_analysis', 'StandardScaler', f'set3_{MEASURE_MAPPING[measure]}.gz'))
    df = load_data(4)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(6, 2))
    for col_idx, (x, y) in enumerate(RELATIONSHIPS):
        idata = az.from_netcdf(os.path.join('..', '01_statistical_analysis', 'idata', f'set4_lr_p_{x}_o_{y}.nc'))

        # generate the y model
        x_scaled = scalers[x].transform(df[x].values.reshape(-1, 1)).flatten()
        x_linspace = np.linspace(np.min(x_scaled), np.max(x_scaled), 1000)
        idata.posterior["y_model"] = idata.posterior["alpha"] + idata.posterior["beta_0_corrected"] * xr.DataArray(x_linspace)

        plot_linear_relationship(ax[col_idx], df[x].values, df[y].values, idata,
                                            x_scaler=scalers[x], y_scaler=scalers[y],
                                            xticks=XTICKS, yticks=YTICKS_DICT[y], scatter_alpha=0.01)
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join('outputs', 'fig5_scatter.png'))
    plt.close()

if __name__ == '__main__':
    main()