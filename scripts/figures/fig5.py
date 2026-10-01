"""
Generates Figure 5: relationship between predicted naturalness (ViTNat) and compressibility
(JPEG, Canny) or memorability (corrected recognition rate) across the three datasets
(isola, memcat, lamem).

Rows are the analyses (vitnat -> jpeg / canny / crr), columns are datasets (isola, memcat,
lamem). The smaller isola set is shown as a scatter (fig3/4 style); the larger memcat and
lamem sets are shown as 2D-density heatmaps.
"""

import sys
from pathlib import Path

sys.path.append('../../src')

import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator

from paths import IDATA_PATH
from misc import load_data, standardize_columns
from plotting import plot_linear_relationship, axis_limits

plt.rcParams['svg.fonttype'] = 'none'

DATASETS = ['isola', 'memcat', 'lamem']
HEATMAP_DATASETS = {'memcat', 'lamem'}
# (outcome column, outcome filename token); predictor is always vitnat
RELATIONSHIPS = [
    ('jpeg_based_compressibility', 'jpeg'),
    ('canny_based_compressibility', 'canny'),
    ('crr', 'crr'),
]
MEASURES = ['vitnat', 'jpeg_based_compressibility', 'canny_based_compressibility', 'crr']

XTICKS = np.array([1, 4, 7])
YTICKS = {
    'jpeg_based_compressibility': np.array([0.0, -1.0, -2.0]),
    'canny_based_compressibility': np.array([0.4, 0.2, 0.0, -0.2, -0.4]),
    'crr': np.array([0.2, 0.4, 0.6, 0.8, 1.0]),
}

NUM_BINS = 40
CMAP = clr.LinearSegmentedColormap.from_list('purple', [(1, 1, 1), '#a383c6'], N=256)
CBAR_KWS = {'ticks': LogLocator(base=10, numticks=5)}

def main():
    dfs = {image_set: load_data(image_set) for image_set in DATASETS}

    # Axis limits in original units. x (vitnat) shares one window across all panels; y is
    # shared within each row (per outcome) across datasets. The ticks are folded in so they
    # always fall within the limits (otherwise ticks beyond the data range get clipped).
    xlim = axis_limits(np.concatenate([dfs[s]['vitnat'].values for s in DATASETS] + [XTICKS]))
    ylim_by_outcome = {
        outcome: axis_limits(np.concatenate([dfs[s][outcome].values for s in DATASETS] + [YTICKS[outcome]]))
        for outcome, _ in RELATIONSHIPS
    }

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(6.75, 7))
    top_meshes = {}  # col_idx -> density mesh of the top-row heatmap panel
    for col_idx, image_set in enumerate(DATASETS):
        df = dfs[image_set]
        # Recompute the scalers (the fit pipeline standardizes the same load_data output)
        _, scalers = standardize_columns(df, MEASURES)
        for row_idx, (outcome, outcome_token) in enumerate(RELATIONSHIPS):
            idata = az.from_netcdf(IDATA_PATH / f'{image_set}_lr_o_{outcome_token}_p_vitnat.nc')
            yticks = YTICKS[outcome]
            ylim = ylim_by_outcome[outcome]
            if image_set in HEATMAP_DATASETS:
                mesh = plot_linear_relationship(
                    ax[row_idx][col_idx], df['vitnat'].values, df[outcome].values, idata,
                    x_scaler=scalers['vitnat'], y_scaler=scalers[outcome],
                    xticks=XTICKS, yticks=yticks, xlim=xlim, ylim=ylim,
                    density=True, num_bins=NUM_BINS, cmap=CMAP, norm=LogNorm(), line_width=0.5)
                if row_idx == 0:
                    top_meshes[col_idx] = mesh
            else:
                plot_linear_relationship(
                    ax[row_idx][col_idx], df['vitnat'].values, df[outcome].values, idata,
                    x_scaler=scalers['vitnat'], y_scaler=scalers[outcome],
                    xticks=XTICKS, yticks=yticks, xlim=xlim, ylim=ylim)

    # leave headroom at the top so the colorbars above the top row aren't clipped
    plt.tight_layout(rect=(0, 0, 1, 0.94))

    # add the colorbars only after layout, in their own figure-level axes positioned above the
    # top-row heatmap panels, so they don't affect (shrink) any panel
    for col_idx, mesh in top_meshes.items():
        pos = ax[0][col_idx].get_position()
        cax = fig.add_axes([pos.x0, pos.y1 + 0.015, pos.width, 0.015])
        fig.colorbar(mesh, cax=cax, orientation='horizontal', **CBAR_KWS)
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')

    out_dir = Path('outputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / 'fig5.svg')
    plt.close()

if __name__ == '__main__':
    main()
