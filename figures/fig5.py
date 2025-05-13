"""
Generates Figure 5: heatmap-based visualization of the relationship between naturalness predictions (ViTNat)
and compressibility (JPEG, Canny) or memorability (ResMem) in Image Set 4.
"""

import sys
sys.path.append("../scripts/")

import os
import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import arviz as az
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
from models import load_data
from plotting import plot_linear_relationship_on_heatmap

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
        plot_linear_relationship_on_heatmap(ax[col_idx], df[x].values, df[y].values, idata,
                                            cmap=CMAP, num_bins=40, x_scaler=scalers[x], y_scaler=scalers[y],
                                            xticks=XTICKS, yticks=YTICKS_DICT[y],
                                            norm=LogNorm(), cbar_kws={'ticks': LogLocator(base=10, numticks=5), 
                                            'location':"top", 'shrink': 0.8})
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join('outputs', 'fig5.svg'))
    plt.close()

if __name__ == '__main__':
    main()