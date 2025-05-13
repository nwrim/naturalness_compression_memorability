"""
Generates Supplementary Figure S8: linear relationships between compressibility (JPEG and Canny)
and ResMem-predicted memorability for Image Set 4 using heatmaps.
"""

import os
import sys
sys.path.append(os.path.join('..', 'scripts'))

import joblib
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
from models import load_data
from plotting import plot_linear_relationship_on_heatmap

plt.rcParams['svg.fonttype'] = 'none'

def main():
    CMAP = clr.LinearSegmentedColormap.from_list('lavender', [(1, 1, 1), '#B57EDC'], N=256)
    NUM_BINS = 40
    NORM = LogNorm()
    CBAR_KWS={'ticks': LogLocator(base=10, numticks=5), 
              'location':"top", 'shrink': 0.85}  
    XTICKS_DICT = {
        'jpeg': np.array([0.03, -0.01, -0.05, -0.09]),
        'canny': np.array([0.004, 0.002, 0.000, -0.002])
    }
    YTICKS = np.array([0.4, 0.6, 0.8])
    MEASURE_MAPPING = {
        'jpeg': 'jpeg',
        'canny': 'canny',
        'resmem': 'hr'
    }
    y = 'resmem'

    df = load_data(4)

    fig, ax = plt.subplots(ncols=2, figsize=(4, 2.5))
    for idx, x in enumerate(XTICKS_DICT.keys()):
        scalers = {}
        for measure in ['jpeg', 'canny', 'resmem']:
            scalers[measure] = joblib.load(os.path.join('..', '01_statistical_analysis', 'StandardScaler', f'set3_{MEASURE_MAPPING[measure]}.gz'))
        idata = az.from_netcdf(os.path.join('..', '01_statistical_analysis', 'idata', f'set4_lr_p_{x}_o_{y}.nc'))
        plot_linear_relationship_on_heatmap(ax[idx], df[x].values, df['resmem'].values, idata, beta_term='beta_0',
                                            cmap=CMAP, num_bins=NUM_BINS, x_scaler=scalers[x], y_scaler=scalers[y],
                                            xticks=XTICKS_DICT[x], yticks=YTICKS,
                                            norm=NORM, cbar_kws=CBAR_KWS)
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join('outputs', 'figS8.svg'))
    plt.close()

if __name__ == "__main__":
    main()