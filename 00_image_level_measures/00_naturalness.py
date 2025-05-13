"""
Aggregates Likert-scale naturalness ratings for all images across three image sets.

Requires:
- Naturalness rating experimental data downloaded and extracted into 
  `../data/experiment/set{image_set_index}_naturalness/` (see README for link to OSF).

Outputs:
- CSV files containing aggregated naturalness ratings per image for each image set.
  Saved to:
    ../data/image_level_measure/set{image_set_index}_naturalness.csv
"""

import os
import sys
sys.path.append(os.path.join('..', 'scripts'))

from experiment_data_processing import load_participants_tsv, process_likert, aggregate_likert_results
from tqdm import tqdm

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

def main():
    for image_set_index in [1, 2, 3]:
        root_path = os.path.join('..', 'data', 'experiment', f'set{image_set_index}_naturalness')
        output_path = os.path.join('..', 'data', 'image_level_measure', f'set{image_set_index}_naturalness.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        participant_ids = load_participants_tsv(root_path)['participant_id']

        total = len(participant_ids)
        valid = 0
        fail_attn = 0
        fail_follow = 0

        valid_dfs = []

        for pid in tqdm(participant_ids):
            try:
                df, status = process_likert(pid, root_path, 'naturalness', 
                                            IMAGE_SET_ATTN_CONFIG[image_set_index])
                if status == 'valid':
                    valid_dfs.append(df)
                    valid += 1
                elif status == 'fail_attn_check':
                    fail_attn += 1
                elif status == 'fail_follow_check':
                    fail_follow += 1
                else:
                    print(f'Excluded {pid}: {status}')
            except Exception as e:
                print(f'Error processing {pid}: {e}')

        print(f'Image Set Index: {image_set_index}')
        print(f'Total participants: {total}')
        if image_set_index in [2, 3]:
            print(f'Valid data: {valid}')
            print(f'Excluded for failing attention check: {fail_attn}')
            print(f'Excluded for following attention instructions: {fail_follow}')
        print()

        result = aggregate_likert_results(valid_dfs, 'naturalness')
        result.to_csv(output_path, index=False)

if __name__ == '__main__':
    main()