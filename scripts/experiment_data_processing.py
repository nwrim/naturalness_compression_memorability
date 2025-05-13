import os
import pandas as pd
import numpy as np
from collections import defaultdict

def load_participants_tsv(root_path):
    """
    Load participant metadata from a BIDS-compliant `participants.tsv` file.

    Parameters
    ----------
    root_path : str
        Path to the root directory containing `participants.tsv`.

    Returns
    -------
    pd.DataFrame
        DataFrame with participant information, with `participant_id` column parsed as string.
    """

    participants_file = os.path.join(root_path, 'participants.tsv')
    df = pd.read_csv(participants_file, sep='\t', dtype={'participant_id': str})
    return df

def process_likert(subject_id, root_path, task, attn_config=None,
                   data_type='beh', suffix='beh'):
    """
    Process a participant's Likert and attention check responses.

    Parameters
    ----------
    subject_id : str
        Subject ID to load.
    root_path : str
        Path to the root directory containing participant subfolders.
    task : str
        Task name.
    attn_config : dict, optional
        Dictionary with keys:
        - 'attn_total_trials': expected number of attention check trials.
        - 'attn_fail_threshold': maximum allowable errors on attention checks.
        - 'likert_total_trials': expected number of Likert trials.
        - 'likert_follow_threshold': maximum allowable Likert trials that match attention answer.
    data_type : str
        Subfolder within subject directory containing the data file.
    suffix : str
        File suffix (e.g., "beh").

    Returns
    -------
    tuple
        (DataFrame of valid Likert trials, status string). If invalid, returns (None, reason).
    """

    file_path = os.path.join(root_path, f'sub-{subject_id}', data_type,
                             f'sub-{subject_id}_task-{task}_{suffix}.tsv')
    df = pd.read_csv(file_path, sep='\t')

    if attn_config is None:
        return df, 'valid'

    attn = df[df['trial_type'] == 'ATTENTION']
    if len(attn) != attn_config['attn_total_trials']:
        return None, f'bad_attn_trial_count ({len(attn)} != {attn_config["attn_total_trials"]})'
    if (attn['response'] != attn['attention_answer']).sum() >= attn_config['attn_fail_threshold']:
        return None, 'fail_attn_check'

    likert = df[df['trial_type'] == 'LIKERT']
    if len(likert) != attn_config['likert_total_trials']:
        return None, f'bad_likert_trial_count ({len(likert)} != {attn_config["likert_total_trials"]})'
    if (likert['response'] == likert['attention_answer']).sum() >= attn_config['likert_follow_threshold']:
        return None, 'fail_follow_check'

    return likert, 'valid'

def aggregate_likert_results(dfs, rating_name='response'):
    """
    Aggregate Likert responses across participants.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        List of DataFrames containing Likert responses with 'stim_file' and 'response' columns.
    rating_name : str, optional
        Name of the column to use for mean rating in the output.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['image_name', rating_name, 'sd', 'count'].
    """

    concat_df = pd.concat(dfs, ignore_index=True)
    grouped = concat_df.groupby('stim_file')['response'].agg(
        mean='mean',
        sd='std',
        count='count'
    ).reset_index()
    grouped.rename(columns={'stim_file': 'image_name',
                            'mean':rating_name},  inplace=True)
    return grouped[['image_name', rating_name, 'sd', 'count']]

def initialize_image_level_results():
    """
    Initialize image-level results dictionary with counters.

    Returns
    -------
    defaultdict
        Dictionary mapping image names to dictionaries tracking hit/miss/false alarm/correct rejection counts and participant counts.
    """

    return defaultdict(lambda: {
        'target_hit': 0, 'target_miss': 0, 'target_fa': 0, 'target_cr': 0,
        'filler_hit': 0, 'filler_miss': 0, 'filler_fa': 0, 'filler_cr': 0,
        'n_participants_target': 0, 'n_participants_filler': 0
    })

def process_crt(subject_id, root_path, valid_response, image_level_results, 
                update_image_level_results=True,
                consolidate_fixation=True, 
                far_threshold=0.7, vigilance_miss_threshold=0.7,
                task='continuous_recognition',  data_type='beh', suffix='beh'):
    """
    Process a participant's continuous recognition task (CRT) data.

    Parameters
    ----------
    subject_id : str
        Subject ID.
    root_path : str
        Path to the directory containing subject data.
    valid_response : list of str
        Keypresses considered valid responses.
    image_level_results : dict
        Existing dictionary of image-level results to update.
    update_image_level_results : bool
        Whether to update image-level statistics (image_level_results).
    consolidate_fixation : bool
        Whether to treat keypresses during fixation as part of the corresponding image trial.
    far_threshold : float
        Maximum allowable false alarm rate on filler trials.
    vigilance_miss_threshold : float
        Maximum allowable miss rate on vigilance trials.
    task : str
        Task name.
    data_type : str
        Subfolder containing the data file.
    suffix : str
        File suffix.

    Returns
    -------
    tuple
        (Updated image_level_results, status string).
    """

    # load the data
    file_path = os.path.join(root_path, f'sub-{subject_id}', data_type,
                            f'sub-{subject_id}_task-{task}_{suffix}.tsv')
    df = pd.read_csv(file_path, sep='\t', dtype={'response': str})
    
    # check the fixation trials being interleaved
    assert df['trial_type'][1::2].eq('FIXATION').all()

    # getting the image, trial type, and response
    imseq = df['stim_file'].values[::2]
    imtypeseq = df['trial_type'].values[::2]
    keypressseq_raw = np.array([resp in valid_response for resp in df['response']])

    # consolidate the fixation presses as presses on the image
    onimagepressseq = keypressseq_raw[::2].copy()
    offimagepressseq = keypressseq_raw[1::2].copy()
    keypressseq = onimagepressseq | offimagepressseq if consolidate_fixation else onimagepressseq

    # if false alarm rate on filler images is >= far_threshold, we don't use this participant
    filler_false_alarm = keypressseq[imtypeseq == 'FILLER'].sum() / (imtypeseq == 'FILLER').sum()
    if filler_false_alarm >= far_threshold:
        return image_level_results, 'fail_filler_far'

    # if vigilance repeat Miss is >= vigilance_mis_threshold, we don't use this participant
    vigilance_miss_rate = (~keypressseq[imtypeseq == 'VIGILANCE']).sum() / (imtypeseq == 'VIGILANCE').sum()
    if vigilance_miss_rate >= vigilance_miss_threshold:
        return image_level_results, 'fail_vigilance_miss'

    # update the image level results
    if update_image_level_results:
        image_level_results = update_crt_image_level_results(image_level_results, imseq, imtypeseq, keypressseq)
    return image_level_results, 'valid'

def update_crt_image_level_results(image_level_results, imseq, imtypeseq, keypressseq):
    """
    Update image-level memory performance counts based on participant trial data.

    Parameters
    ----------
    image_level_results : dict
        Dictionary mapping image names to result counters.
    imseq : array-like
        Sequence of image filenames.
    imtypeseq : array-like
        Trial types corresponding to each image (e.g., TARGET, REPEAT).
    keypressseq : array-like
        Boolean array indicating whether a keypress was made on each trial.

    Returns
    -------
    dict
        Updated image-level results.
    """

    # looping through the trials
    for im, imtype, keypress in zip(imseq, imtypeseq, keypressseq):
        if imtype == 'TARGET':
            # increase the number of participants who saw this image as target
            image_level_results[im]['n_participants_target'] += 1
            # if they saw it for the first time and pressed, it is a false alarm
            if keypress:
                image_level_results[im]['target_fa'] += 1
            # if they saw it for the first time and didn't press, it is a correct rejection
            else:
                image_level_results[im]['target_cr'] += 1
        elif imtype == 'REPEAT':
            # if they saw it for the second time and pressed, it is a hit
            if keypress:
                image_level_results[im]['target_hit'] += 1
            # if they saw it for the second time and didn't press, it is a miss
            else:
                image_level_results[im]['target_miss'] += 1
        elif imtype == 'FILLER':
            # increase the number of participants who saw this image as filler
            image_level_results[im]['n_participants_filler'] += 1
            # if they saw it for the first time and pressed, it is a false alarm
            if keypress:
                image_level_results[im]['filler_fa'] += 1
            # if they saw it for the first time and didn't press, it is a correct rejection
            else:
                image_level_results[im]['filler_cr'] += 1
        elif imtype == 'VIGILANCE':
            # if they saw it for the second time and pressed, it is a hit
            if keypress:
                image_level_results[im]['filler_hit'] += 1
            # if they saw it for the second time and didn't press, it is a miss
            else:
                image_level_results[im]['filler_miss'] += 1
        else:
            raise ValueError('Unknown trial type')
    return image_level_results

def convert_crt_image_level_results_to_df(image_level_results):
    """
    Convert image-level memory performance results to a summary DataFrame.

    Parameters
    ----------
    image_level_results : dict
        Dictionary with image names as keys and memory performance counters as values.

    Returns
    -------
    pd.DataFrame
        DataFrame with per-image statistics: hit rate, false alarm rate, corrected recognition rate, and raw counts.
    """

    df = pd.DataFrame.from_dict(image_level_results, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'image_name'}, inplace=True)
    df = df.sort_values('image_name')

    # basic data checks
    assert (df['n_participants_target'].values == (df['target_cr'].values + df['target_fa'].values)).all()
    assert (df['n_participants_target'].values == (df['target_hit'].values + df['target_miss'].values)).all()
    assert (df['n_participants_filler'].values == (df['filler_cr'].values + df['filler_fa'].values)).all()

    # calculate the metrics
    df['target_hr'] = df['target_hit'] / (df['target_hit'] + df['target_miss'])
    df['target_far'] = df['target_fa'] / (df['target_fa'] + df['target_cr'])
    df['target_crr'] = df['target_hr'] - df['target_far']

    # reorder df
    df = df[['image_name', 'n_participants_target', 'target_crr', 'target_hr', 'target_far', 'target_hit', 'target_miss', 'target_fa', 'target_cr', 'n_participants_filler', 'filler_hit', 'filler_miss', 'filler_fa', 'filler_cr']]
    return df