"""
Generates Supplementary Figure S3: relationship between JPEG-based and Canny-based compressibility.
"""

import os
import sys
sys.path.append(os.path.join('..', 'scripts'))

import matplotlib.pyplot as plt
import matplotlib.colors as clr
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
from models import load_data
from plotting import plot_linear_relationship_lr, plot_linear_relationship_on_heatmap_lr

plt.rcParams['svg.fonttype'] = 'none'

def main():
    CMAP = clr.LinearSegmentedColormap.from_list('lavender', [(1, 1, 1), '#B57EDC'], N=256)
    NUM_BINS = 40
    NORM = LogNorm()
    CBAR_KWS={'ticks': LogLocator(base=10, numticks=5), 
                      'location':"top", 'shrink': 0.85}
    XTICKS = [0.03, -0.01, -0.05, -0.09]
    YTICKS = [0.004, 0.002, 0.000, -0.002]

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
    for idx, exp_idx in enumerate(range(1, 4)):
        df = load_data(exp_idx)

        plot_linear_relationship_lr(ax[idx // 2][idx % 2], 
                                    df['jpeg'].values, df['canny'].values)
        ax[idx // 2][idx % 2].spines['top'].set_visible(False)
        ax[idx // 2][idx % 2].spines['right'].set_visible(False)
    
    
    df = load_data(4)
    plot_linear_relationship_on_heatmap_lr(ax[1][1], df['jpeg'].values, df['canny'].values,
                                        cmap=CMAP, num_bins=NUM_BINS, 
                                        xticks=XTICKS, yticks=YTICKS,
                                        norm=NORM, cbar_kws=CBAR_KWS)

    plt.tight_layout(pad=0)
    plt.savefig(os.path.join('outputs', 'figS3.svg'))
    plt.close()

if __name__ == "__main__":
    main()