"""
Generates Supplementary Figure S7: relationship between ResMem-predicted memorability scores and
actual human memorability scores (hit rate and corrected recognition rate).
"""

import os
import sys
sys.path.append(os.path.join('..', 'scripts'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotting import plot_linear_relationship_lr

plt.rcParams['svg.fonttype'] = 'none'

def load_data(image_set_index):
    """
    Load the dataset based on the dataset name.
    """
    resmem_df = pd.read_csv(os.path.join('..', 'data', 'image_level_measure', f'set{image_set_index}_resmem.csv'))
    mem_df = pd.read_csv(os.path.join('..', 'data', 'image_level_measure', f'set{image_set_index}_memorability.csv'))
    df = resmem_df.merge(mem_df, on='image_name')
    df.rename(columns={'target_hr': 'hr', 'target_crr': 'crr'}, inplace=True)
    assert len(df) == len(resmem_df) == len(mem_df), "Dataframes do not match in length after merge."
    return df

def main():
    XTICKS_DICT =  {
        'hr': np.array([0.5, 0.75, 1.0]),
        'crr': np.array([0.25, 0.5, 0.75])
    }
    YTICKS = np.array([0.4, 0.6, 0.8])
    RELATIONSHIPS = [('hr', 'resmem'), ('crr', 'resmem')]

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(6.75, 5))

    for col_idx, image_set_index in enumerate([1, 2, 3]):
        # Load the data for the current image set index
        df = load_data(image_set_index)

        for row_idx, (x, y) in enumerate(RELATIONSHIPS):
            plot_linear_relationship_lr(ax[row_idx][col_idx], 
                                        df[x].values, df[y].values, xticks=XTICKS_DICT[x], yticks=YTICKS)

    plt.tight_layout(pad=0)
    plt.savefig(os.path.join('outputs', 'figS7.svg'))
    plt.close()

if __name__ == '__main__':
    main()