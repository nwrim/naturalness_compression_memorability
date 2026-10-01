from pathlib import Path

# Derived from this file's own location (src/paths.py), not the process's
# current working directory, so it resolves correctly regardless of which
# directory a script is run from.
BASE_PATH = Path(__file__).resolve().parent.parent

# raw data
STIMULI_PATH = BASE_PATH / 'data' / 'stimuli'
STIMULI_PATHS = {
    'set1': STIMULI_PATH / 'set1',
    'set2': STIMULI_PATH / 'set2',
    'set3': STIMULI_PATH / 'set3',
    'isola': STIMULI_PATH / 'isola',
    'memcat': STIMULI_PATH / 'MemCat' / 'MemCat_images',
    'lamem': STIMULI_PATH / 'lamem' / 'images',
}
BEH_BIDS_PATH = BASE_PATH / 'data' / 'behavioral'

# processed data
IMAGE_MEASURES_PATH = BASE_PATH / 'data' / 'image_measures'
RELIABILITY_PATH = BASE_PATH / 'data' / 'reliability'
IDATA_PATH = BASE_PATH / 'data' / 'idata'