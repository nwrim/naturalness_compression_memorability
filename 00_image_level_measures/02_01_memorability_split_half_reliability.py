"""
Estimates split-half reliability of image-level memorability scores in each image set.

Requires:
- Memorability experiment data in:
    ../data/experiment/set{image_set_index}_memorability/
  (see README for OSF links)

Procedure:
- Randomly splits valid participants into two halves (1000 permutations).
- Computes image-level memorability scores for each half.
- Calculates Pearson correlation between halves.
- Applies the Spearman-Brown prediction formula to estimate reliability.

Outputs:
- Saves split-half correlations (one value per permutation) to:
    ../data/image_level_measure/memorability_split_half/set{image_set_index}_memorability_split_half_correlations.npy
- Prints mean correlation and reliability to console.
"""

import os
import sys
sys.path.append(os.path.join('..', 'scripts'))

import numpy as np
from numpy.random import default_rng
from tqdm import tqdm
from scipy.stats import pearsonr
from experiment_data_processing import load_participants_tsv, process_crt, convert_crt_image_level_results_to_df, initialize_image_level_results

def main():
    SEED = 0
    N_PERMUTATIONS = 1000
    VALID_RESPONSE = ['R', 'r', '82', '114']
    rng = default_rng(seed=SEED)

    for image_set_index in [1, 2, 3]:
        root_path = os.path.join('..', 'data', 'experiment', f'set{image_set_index}_memorability')

        # get the list of valid participants
        participant_ids = load_participants_tsv(root_path)['participant_id']

        valid_participant_ids = []
        for pid in participant_ids:
            try:
                _, status = process_crt(pid, root_path, VALID_RESPONSE, image_level_results={}, update_image_level_results=False)
                if status == 'valid':
                    valid_participant_ids.append(pid)
            except Exception as e:
                print(f'Error processing {pid}: {e}')
        valid_participant_ids = np.array(valid_participant_ids)

        correlations = []
        reliabilities = []
        for _ in tqdm(range(N_PERMUTATIONS)):
            split0 = rng.choice(valid_participant_ids, len(valid_participant_ids) // 2, replace=False)
            split1 = [p for p in valid_participant_ids if p not in split0]

            image_level_results_split0 = initialize_image_level_results()
            image_level_results_split1 = initialize_image_level_results()

            results = []
            for split, image_level_results in zip([split0, split1], [image_level_results_split0, image_level_results_split1]):
                for pid in split:
                    try:
                        image_level_results, _ = process_crt(pid, root_path, VALID_RESPONSE, image_level_results)
                    except Exception as e:
                        print(f'Error processing {pid}: {e}')
                result = convert_crt_image_level_results_to_df(image_level_results)
                results.append(result)
            
            merged = results[0].merge(results[1], on='image_name', suffixes=('_0', '_1'))
            assert merged.shape[0] == results[0].shape[0] == results[1].shape[0]

            r, _ = pearsonr(merged['target_crr_0'], merged['target_crr_1'])
            correlations.append(r)

            # spearman brown prediction formula
            reliability = 2 * r / (1 + r)

            reliabilities.append(reliability)

        correlations = np.array(correlations)
        reliabilities = np.array(reliabilities)

        np.save(os.path.join('..', 'data', 'image_level_measure', 'memorability_split_half', f'set{image_set_index}_memorability_split_half_correlations.npy'), correlations)

        mean_correlation = np.mean(correlations)
        mean_reliability = np.mean(reliabilities)

        print(f'Image Set Index: {image_set_index}')
        print(f'Mean correlation: {mean_correlation}')
        print(f'Mean reliability: {mean_reliability}')
        print()

if __name__ == '__main__':
    main()