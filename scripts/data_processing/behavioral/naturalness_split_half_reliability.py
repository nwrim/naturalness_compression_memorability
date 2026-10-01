"""
Estimates split-half reliability of image-level naturalness in each image set.
"""

import numpy as np
from numpy.random import default_rng
from tqdm import tqdm
from scipy.stats import pearsonr

import sys
sys.path.append('../../../src')

from paths import BEH_BIDS_PATH, RELIABILITY_PATH
from beh_data import load_participants_tsv, process_likert, aggregate_likert_results

# Configuration for exclusion by image set
IMAGE_SET_ATTN_CONFIG = {
    1: None,
    2: {
        'attn_total_trials': 30,
        'attn_fail_threshold': 6,   # 20%
        'likert_total_trials': 106,
        'likert_follow_threshold': 85  # 80%
    },
    3: {
        'attn_total_trials': 10,
        'attn_fail_threshold': 3,   # 30%
        'likert_total_trials': 100,
        'likert_follow_threshold': 70  # 70%
    }
}

SEED = 0
N_PERMUTATIONS = 1000

def main():
    rng = default_rng(seed=SEED)

    RELIABILITY_PATH.mkdir(parents=True, exist_ok=True)

    for image_set_index in [1, 2, 3]:
        root_path =  BEH_BIDS_PATH / f'set{image_set_index}_naturalness'

        participant_ids = load_participants_tsv(root_path)['participant_id']

        valid_dfs = []
        for pid in participant_ids:
            df, status = process_likert(pid, root_path, 'naturalness', IMAGE_SET_ATTN_CONFIG[image_set_index])
            if status == 'valid':
                valid_dfs.append(df)
            elif status in ('fail_attn_check', 'fail_follow_check'):
                continue
            else:
                raise ValueError(f'Unexpected status for {pid}: {status}')

        n = len(valid_dfs)
        half = n // 2
        
        correlations = []
        reliabilities = []
        for _ in tqdm(range(N_PERMUTATIONS)):
            perm = rng.permutation(n)
            idx0 = perm[:half]
            idx1 = perm[half:]

            result0 = aggregate_likert_results([valid_dfs[i] for i in idx0], 'naturalness')
            result1 = aggregate_likert_results([valid_dfs[i] for i in idx1], 'naturalness')

            merged = result0.merge(result1, on='image_name', suffixes=('_0', '_1'))
            assert merged.shape[0] == result0.shape[0] == result1.shape[0]
            r, _ = pearsonr(merged['naturalness_0'], merged['naturalness_1'])
            correlations.append(r)

            # spearman brown prediction formula
            reliability = 2 * r / (1 + r)

            reliabilities.append(reliability)
        
        correlations = np.array(correlations)
        reliabilities = np.array(reliabilities)

        np.save(RELIABILITY_PATH / f'set{image_set_index}_naturalness_split_half_correlations.npy', correlations)

        mean_correlation = np.mean(correlations)
        mean_reliability = np.mean(reliabilities)

        print(f'Image Set Index: {image_set_index}')
        print(f'Mean correlation: {mean_correlation}')
        print(f'Mean reliability: {mean_reliability}')
        print()

if __name__ == "__main__":
    main()
