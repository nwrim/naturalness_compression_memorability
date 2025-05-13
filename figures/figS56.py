"""
Generates Supplementary Figures S5 and S6.

- Figure S5: Relationship between ViTNat predictions and human-rated naturalness for Image Sets 1 and 2.
- Figure S6: Same relationship shown for the Schertz et al. (2018) and Coburn et al. (2019) datasets.
    - Coburn is split by 'exterior' and 'interior' image categories.
"""

import os
import sys
sys.path.append(os.path.join('..', 'scripts'))

import pandas as pd
import matplotlib.pyplot as plt
from plotting import plot_linear_relationship_lr

plt.rcParams['svg.fonttype'] = 'none'

SCATTER_ALPHA_S6 = 0.3
TICKS = [1, 4, 7]

def load_data(dataset_name):
    """
    Load the dataset based on the dataset name.
    """
    vitnat_df = pd.read_csv(os.path.join('..', 'data', 'image_level_measure', f'{dataset_name}_vitnat.csv'))
    nat_df = pd.read_csv(os.path.join('..', 'data', 'image_level_measure', f'{dataset_name}_naturalness.csv'))
    df = vitnat_df.merge(nat_df, on='image_name')
    assert len(df) == len(vitnat_df) == len(nat_df), "Dataframes do not match in length after merge."
    return df

def main():
    fig, ax = plt.subplots(ncols=2, figsize=(4.25, 2.5))
    for idx, dataset_name in enumerate(['set1', 'set2']):
        df = load_data(dataset_name)        
        plot_linear_relationship_lr(ax[idx], df['naturalness'].values, df['vitnat'].values, xticks=TICKS, yticks=TICKS)
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join('outputs', 'figS5.svg'))
    plt.close()

    fig, ax = plt.subplots(ncols=3, figsize=(6.25,3))
    for idx, dataset_name in enumerate(['schertz2018', 'coburn2019']):
        df = load_data(dataset_name)
        if dataset_name == 'coburn2019':
            for cat_idx, cat in enumerate(['exterior', 'interior']):
                cat_df = df[df['category'] == cat]
                plot_linear_relationship_lr(ax[idx + cat_idx], 
                                            cat_df['naturalness'].values, cat_df['vitnat'].values, 
                                            scatter_alpha=SCATTER_ALPHA_S6, xticks=TICKS)
        else:
            plot_linear_relationship_lr(ax[idx], df['naturalness'], df['vitnat'], 
                            scatter_alpha=SCATTER_ALPHA_S6, xticks=TICKS, yticks=TICKS)
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join('outputs', 'figS6.svg'))
    plt.close()

if __name__ == '__main__':
    main()