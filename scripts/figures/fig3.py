"""
Generates Figure 3: linear regression plots showing the relationship between naturalness
and compressibility (JPEG-based and Canny-based) across Image Sets 1-3.
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
YTICKS_DICT = {
    'jpeg_based_compressibility': np.array([-0.5, -1.0, -1.5, -2.0]),
    'canny_based_compressibility': np.array([0.4, 0.2, 0.0]),
}
DATASETS = ['set1', 'set2', 'set3']
# (predictor column, outcome column, outcome filename token)
RELATIONSHIPS = [
    ('naturalness', 'jpeg_based_compressibility', 'jpeg'),
    ('naturalness', 'canny_based_compressibility', 'canny'),
]
MEASURES = ['naturalness', 'jpeg_based_compressibility', 'canny_based_compressibility']

def main():
    dfs = {image_set: load_data(image_set) for image_set in DATASETS}

    # Common axis limits in original units (ticks folded in so they always render): x
    # (naturalness) shared across all panels; y shared within each row (per outcome).
    xlim = axis_limits(np.concatenate([dfs[s]['naturalness'].values for s in DATASETS] + [XTICKS]))
    ylim_by_outcome = {
        y: axis_limits(np.concatenate([dfs[s][y].values for s in DATASETS] + [YTICKS_DICT[y]]))
        for _, y, _ in RELATIONSHIPS
    }

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(6.75, 5))
    for col_idx, image_set in enumerate(DATASETS):
        df = dfs[image_set]
        # Recompute the scalers (the fit pipeline standardizes the same load_data output)
        _, scalers = standardize_columns(df, MEASURES)
        for row_idx, (x, y, outcome_token) in enumerate(RELATIONSHIPS):
            idata = az.from_netcdf(IDATA_PATH / f'{image_set}_lr_o_{outcome_token}_p_nat.nc')
            plot_linear_relationship(ax[row_idx][col_idx], df[x].values, df[y].values, idata,
                                     x_scaler=scalers[x], y_scaler=scalers[y],
                                     xticks=XTICKS, yticks=YTICKS_DICT[y],
                                     xlim=xlim, ylim=ylim_by_outcome[y])

    plt.tight_layout()
    out_dir = Path('outputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / 'fig3.svg')
    plt.close()

if __name__ == '__main__':
    main()
