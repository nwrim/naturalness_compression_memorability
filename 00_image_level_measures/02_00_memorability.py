"""
Aggregates image-level memorability scores based on participant responses from a continuous recognition task (CRT).

Requires:
- Memorability experiment data downloaded and extracted into:
    ../data/experiment/set{image_set_index}_memorability/
  (see README for OSF links)

Outputs:
- CSV files containing hit rates (memorability scores) for each image:
    ../data/image_level_measure/set{image_set_index}_memorability.csv
"""

import os
import sys
sys.path.append(os.path.join('..', 'scripts'))

from experiment_data_processing import load_participants_tsv, process_crt, convert_crt_image_level_results_to_df, initialize_image_level_results
from tqdm import tqdm

def main():
    # Constants for the experiment
    VALID_RESPONSE = ['R', 'r', '82', '114']
    
    for image_set_index in [1, 2, 3]:
        image_level_results = initialize_image_level_results()

        root_path = os.path.join('..', 'data', 'experiment', f'set{image_set_index}_memorability')
        output_path = os.path.join('..', 'data', 'image_level_measure', f'set{image_set_index}_memorability.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        participant_ids = load_participants_tsv(root_path)['participant_id']

        total = len(participant_ids)
        valid = 0
        fail_far = 0
        fail_miss = 0

        for pid in tqdm(participant_ids):
            try:
                image_level_results, status = process_crt(pid, root_path, VALID_RESPONSE, image_level_results)
                if status == 'valid':
                    valid += 1
                elif status == 'fail_filler_far':
                    fail_far += 1
                elif status == 'fail_vigilance_miss':
                    fail_miss += 1
                else:
                    print(f'Excluded {pid}: {status}')
            except Exception as e:
                print(f'Error processing {pid}: {e}')

            
        print(f'Image Set Index: {image_set_index}')
        print(f'Total participants: {total}')
        print(f'Valid data: {valid}')
        print(f'Excluded for having too high far on fillers: {fail_far}')
        print(f'Excluded for missing too much vigliance repeat: {fail_miss}')
        print()

        result = convert_crt_image_level_results_to_df(image_level_results)
        result.to_csv(output_path, index=False)

if __name__ == '__main__':
    main()