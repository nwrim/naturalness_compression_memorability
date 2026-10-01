from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

from paths import STIMULI_PATHS, IMAGE_MEASURES_PATH

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff', '.webp'}

# Images that are degenerate for image-level analysis and excluded from every
# model of their set, regardless of which measures a given analysis uses.
# lamem's 00017116.jpg is a constant (pure-black) image, so its compressibility
# is undefined; excluding it here (rather than relying on an incidental NaN) keeps
# every lamem model on one sample, including compressibility-free analyses and
# ones that swap in a variant metric that happens to be defined for a constant image.
DEGENERATE_IMAGES = {
    'lamem': {'00017116.jpg'},
}

# Image sets with human naturalness ratings; all other sets use the ViTNat prediction instead.
BEHAVIORAL_SETS = {'set1', 'set2', 'set3'}

def naturalness_column(image_set):
    """Return the column holding naturalness for image_set: the human Likert rating
    ('naturalness') for the behavioral sets, the ViTNat prediction ('vitnat') otherwise."""
    return 'naturalness' if image_set in BEHAVIORAL_SETS else 'vitnat'

def drop_degenerate_images(df, image_set):
    """Drop the set's degenerate images (see DEGENERATE_IMAGES) from df."""
    exclude = DEGENERATE_IMAGES.get(image_set, set())
    return df[~df['image_name'].isin(exclude)].reset_index(drop=True)

def drop_nan_and_report(df, image_set, label='measures'):
    """Drop rows with any missing value, printing a per-column count of what was
    missing. Reports relative to df's current rows, so call it after any merge so
    the counts reflect the analysis sample."""
    nan_rows = df[df.isna().any(axis=1)]
    if len(nan_rows) > 0:
        per_measure = nan_rows.isna().sum()
        print(f"{image_set}: dropping {len(nan_rows)} image(s) with missing {label}:")
        for measure, count in per_measure[per_measure > 0].items():
            print(f"  {measure}: {count}")
        df = df.dropna().reset_index(drop=True)
    return df

def list_image_files(directory):
    """
    Recursively list all image files under a directory, as paths relative to it.

    Works for both flat and nested datasets: for a flat directory the relative
    path is just the filename, while for a nested directory (e.g. a dataset's
    category subfolders) it includes the intermediate folders. In both cases
    `Path(directory) / relative_path` reconstructs the full path.

    Only files whose extension is in `IMAGE_EXTENSIONS` are returned.

    Parameters
    ----------
    directory : str or pathlib.Path
        Path to the directory to list image files from.

    Returns
    -------
    list of str
        Relative POSIX-style paths of every image file found at any depth.
    """

    directory = Path(directory)
    return [p.relative_to(directory).as_posix()
            for p in directory.rglob('*')
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]

def load_data(image_set):
    """
    Load and merge the image-level measures of an image set into one dataframe.

    Merges naturalness (the human rating for the behavioral sets, the ViTNat
    prediction otherwise; see `naturalness_column`), JPEG- and Canny-based
    compressibility, and memorability (crr) from `IMAGE_MEASURES_PATH` on
    `image_name`. Degenerate images (see `DEGENERATE_IMAGES`) and images with any
    missing measure are dropped, so every model of the set shares one sample.

    Parameters
    ----------
    image_set : str
        Name of the image set (a key of `STIMULI_PATHS`).

    Returns
    -------
    pandas.DataFrame
        One row per image, with columns 'image_name', the naturalness column
        ('naturalness' or 'vitnat'), 'jpeg_based_compressibility',
        'canny_based_compressibility', 'crr', and, for memcat only, 'subcategory'.
    """

    assert image_set in STIMULI_PATHS, f"Unknown image set: {image_set}"

    dfs = []
    # load naturalness
    naturalness_col = naturalness_column(image_set)
    naturalness_df = pd.read_csv(IMAGE_MEASURES_PATH / f'{image_set}_{naturalness_col}.csv')[['image_name', naturalness_col]]
    dfs.append(naturalness_df)
    # load compressibility
    compressibility_df = pd.read_csv(IMAGE_MEASURES_PATH / f'{image_set}_compressibility.csv')[['image_name', 'jpeg_based_compressibility', 'canny_based_compressibility']]
    dfs.append(compressibility_df)
    # load memorability
    memorability_df = pd.read_csv(IMAGE_MEASURES_PATH / f'{image_set}_memorability.csv')
    if image_set == 'memcat':
        memorability_df = memorability_df[['image_name', 'subcategory', 'crr']]
    else:
        memorability_df = memorability_df[['image_name', 'crr']]
    dfs.append(memorability_df)

    # merge all measures into a single dataframe
    final_df = dfs[0]
    for df in dfs[1:]:
        final_df = final_df.merge(df, on='image_name', validate='1:1')
    # check that dataframe has same number of rows
    for df in dfs:
        assert len(df) == len(final_df), "Dataframes have different number of rows after merging"

    # Drop degenerate images (e.g. constant images with undefined compressibility)
    # so every model shares one sample; see drop_degenerate_images.
    final_df = drop_degenerate_images(final_df, image_set)

    # Drop images with any missing measure (e.g. compressibility is undefined
    # for a uniform image) and report the per-measure count.
    final_df = drop_nan_and_report(final_df, image_set)

    return final_df

def standardize_columns(df, columns):
    """
    Standardize the given columns of a dataframe with a per-column StandardScaler.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe holding the columns to standardize.
    columns : list of str
        The columns to standardize.

    Returns
    -------
    scaled : dict of str to np.array
        The standardized values, keyed by column name.
    scalers : dict of str to sklearn.preprocessing.StandardScaler
        The fitted scalers, keyed by column name (e.g. for inverse_transform).
    """
    scaled = {}
    scalers = {}
    for column in columns:
        scaler = StandardScaler()
        scaled[column] = scaler.fit_transform(df[column].values.reshape(-1, 1)).flatten()
        scalers[column] = scaler
    return scaled, scalers
