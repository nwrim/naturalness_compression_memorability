"""
Generates Figure 4B: linear regression plots showing the relationship between naturalness
and memorability (corrected recognition rate) across Image Sets 1-3.
"""

import sys
from pathlib import Path

sys.path.append('../../src')

import numpy as np
import arviz as az
import matplotlib.pyplot as plt

from paths import IDATA_PATH
from misc import load_data, standardize_columns
from plotting import plot_linear_relationship, axis_limits

plt.rcParams['svg.fonttype'] = 'none'

XTICKS = np.array([1, 4, 7])
YTICKS = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
MEASURES = ['naturalness', 'crr']
DATASETS = ['set1', 'set2', 'set3']

def main():
    dfs = {image_set: load_data(image_set) for image_set in DATASETS}

    xlim = axis_limits(np.concatenate([dfs[s]['naturalness'].values for s in DATASETS] + [XTICKS]))
    ylim = axis_limits(np.concatenate([dfs[s]['crr'].values for s in DATASETS] + [YTICKS]))

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(6.75, 3))
    for col_idx, image_set in enumerate(DATASETS):
        df = dfs[image_set]
        # Recompute the scalers (the fit pipeline standardizes the same load_data output)
        _, scalers = standardize_columns(df, MEASURES)
        idata = az.from_netcdf(IDATA_PATH / f'{image_set}_lr_o_crr_p_nat.nc')
        plot_linear_relationship(ax[col_idx], df['naturalness'].values, df['crr'].values, idata,
                                 x_scaler=scalers['naturalness'], y_scaler=scalers['crr'],
                                 xticks=XTICKS, yticks=YTICKS, xlim=xlim, ylim=ylim)

    plt.tight_layout()
    out_dir = Path('outputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / 'fig4.svg')
    plt.close()

if __name__ == '__main__':
    main()
