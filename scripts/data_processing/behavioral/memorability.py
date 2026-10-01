"""
Aggregates image-level memorability scores based on participant responses from a continuous recognition task (CRT).
"""

from tqdm import tqdm

import sys
sys.path.append('../../../src')

from paths import BEH_BIDS_PATH, IMAGE_MEASURES_PATH
from beh_data import load_participants_tsv, process_crt, convert_crt_image_level_results_to_df, initialize_image_level_results

def main():
    # Constants for the experiment
    VALID_RESPONSE = ['R', 'r', '82', '114']
    
    for image_set_index in [1, 2, 3]:
        image_level_results = initialize_image_level_results()

        root_path = BEH_BIDS_PATH / f'set{image_set_index}_memorability'
        output_path = IMAGE_MEASURES_PATH / f'set{image_set_index}_memorability.csv'
        IMAGE_MEASURES_PATH.mkdir(parents=True, exist_ok=True)

        participant_ids = load_participants_tsv(root_path)['participant_id']

        total = len(participant_ids)
        valid = 0
        fail_far = 0
        fail_miss = 0

        for pid in tqdm(participant_ids):
            image_level_results, status = process_crt(pid, root_path, VALID_RESPONSE, image_level_results)
            if status == 'valid':
                valid += 1
            elif status == 'fail_filler_far':
                fail_far += 1
            elif status == 'fail_vigilance_miss':
                fail_miss += 1
            else:
                raise ValueError(f'Unexpected status for {pid}: {status}')

        print(f'Image Set Index: {image_set_index}')
        print(f'Total participants: {total}')
        print(f'Valid data: {valid}')
        print(f'Excluded for high false alarm rate on fillers: {fail_far}')
        print(f'Excluded for high miss rate on vigilance repeats: {fail_miss}')
        print()

        result = convert_crt_image_level_results_to_df(image_level_results)
        result.to_csv(output_path, index=False)

if __name__ == '__main__':
    main()
