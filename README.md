This repository contains code and data associated with the paper: **_Natural scenes are more compressible and less memorable than human-made scenes_**

> **Note:** This repository holds the code and data for a revised version of the manuscript, currently under review. The analyses and figures therefore do not fully match the preprint cited below ([Citation](#citation)). For the code and data matching the preprint, see the [`preprint`](https://github.com/nwrim/naturalness_compression_memorability/tree/preprint) tag.

# Scripts

Shared modules live in `src/`. Scripts use `sys.path.append` to make everything just work out of the box as long as you run a script from within its own directory, but you could install it too.

## Getting image-level measures

The scripts under `scripts/data_processing/` compute the per-image measures already included in `data/image_measures/` (see [Image-level measures](#image-level-measures)). You only need to run these if you want to regenerate the measures from raw data, which you'll need to download separately first (see [Stimuli](#stimuli) and [Behavioral data](#behavioral-data)).

* `behavioral/naturalness.py` — aggregates Likert-scale naturalness ratings from `data/behavioral/` into `set{1,2,3}_naturalness.csv`.
* `behavioral/memorability.py` — aggregates continuous recognition task responses from `data/behavioral/` into `set{1,2,3}_memorability.csv` (corrected recognition rate).
* `behavioral/naturalness_split_half_reliability.py` and `behavioral/memorability_split_half_reliability.py` — estimate split-half reliability of the naturalness and memorability measures above.
* `image_stats/compressibility.py` — computes JPEG- and Canny-based compressibility for an image set from `data/stimuli/`. Run as `python compressibility.py --image_set <set>`.
* `image_stats/vitnat.py` — runs the pretrained ViTNat model to predict naturalness for an image set. Run as `python vitnat.py --image_set <set>`. Needs its own environment (see [Dependencies](#dependencies)).
* `image_stats/merge_chunks.py` — `compressibility.py` and `vitnat.py` both accept a `--chunk` option to split a large image set into pieces (e.g. for parallel jobs on a cluster) instead of processing it all in one run; this script merges those per-chunk outputs back into a single CSV. Not needed for unchunked run.

## Fitting statistical models

The scripts under `scripts/statistical_analysis/` fit the Bayesian models (via PyMC) behind the paper's main results, reading from `data/image_measures/` and writing ArviZ InferenceData (`.nc` files) to `data/idata/`.

* `linear_regressions.py` — fits the main linear regressions (naturalness -> compressibility, naturalness/compressibility -> memorability) for an image set. Run as `python linear_regressions.py --image_set <set>`.
* `mediations.py` — fits mediation models testing whether compressibility mediates the naturalness -> memorability relationship. Run as `python mediations.py --image_set <set>`.
* `linear_regressions_random_intercept.py` — reruns the main linear regressions with a group-level random intercept, to control for image category. Only fit for `set3` (grouped by category) and `memcat` (grouped by subcategory). Run as `python linear_regressions_random_intercept.py --image_set <set3|memcat>`.
* `memorability_ceiling_proportion.py` — prints the proportion of the memorability reliability ceiling (from `data/reliability/`) explained by each predictor's regression beta, for Image Sets 1-3.
* `print_estimates.py` — prints the manuscript-ready posterior estimates (β, 96% HPDI, tail probability) from the fitted linear regression, mediation, and random-intercept models above.

## Generating figures

The scripts under `scripts/figures/` generate the paper's main figures as SVGs (from `data/idata/` and `data/image_measures/`). These are the data underlying each figure, not the final version.

* `fig2ac.py` and `fig2bd.py` — panels A/C and B/D of Figure 2, illustrating JPEG- and Canny-based compressibility on two example images from Image Set 2 (bundled in `scripts/figures/` as `0554.jpg`, `0998.jpg`).
* `fig3.py` — Figure 3: naturalness vs. compressibility (JPEG- and Canny-based) for Image Sets 1-3.
* `fig4.py` — Figure 4B: naturalness vs. memorability (crr) for Image Sets 1-3.
* `fig5.py` — Figure 5: ViTNat-predicted naturalness vs. compressibility/memorability for the `isola`/`memcat`/`lamem` datasets.

## Other reports

Standalone reports under `scripts/other_reports/` that don't feed into the main modeling pipeline:

* `naturalness_5_or_higher.py` — prints the percentage of images rated naturalness ≥ 5, for Image Sets 1-3.
* `vitnat_naturalness_correlation.py` — prints the Pearson correlation between ViTNat predictions and human-rated naturalness, measuring ViTNat's out-of-sample performance on Image Sets 1-2 and the external Schertz et al. (2018) and Coburn et al. (2019) validation datasets.

# Data

## Image-level measures
Per-image measures (naturalness, compressibility, memorability, and ViTNat predictions) for all datasets are included in the repository under `data/image_measures`, one CSV per dataset per measure (e.g. `set1_naturalness.csv`, `memcat_memorability.csv`).

## Reliability
Split-half reliability estimates (1000 permutations each) for the naturalness and memorability measures of Image Sets 1-3 are included in the repository under `data/reliability`, as `{image_set}_{naturalness,memorability}_split_half_correlations.npy`. See `behavioral/naturalness_split_half_reliability.py` and `behavioral/memorability_split_half_reliability.py` if you want to regenerate them.

## Stimuli
The image stimuli are hosted externally via OSF. Image measures for these sets are already included in the repository, so you only need to download the stimuli if you want to rerun the image measure scripts.

Download and extract the `.tar.gz` archives into `data/stimuli/`:

* Image Set 1: [https://osf.io/rhy8d](https://osf.io/rhy8d)
* Image Set 2: [https://osf.io/dyjk9](https://osf.io/dyjk9)
* Image Set 3: [https://osf.io/5bwgu](https://osf.io/5bwgu)

## Behavioral data
The raw, trial-level behavioral data (in BIDS format) is hosted externally via OSF. The aggregated per-image measures are already included in the repository, so you only need to download the raw data if you want to rerun the aggregation scripts.

Download and extract the `.tar.gz` archives into `data/behavioral/`:

* Naturalness rating
    - Image Set 1: [https://osf.io/9sx3c](https://osf.io/9sx3c)
    - Image Set 2: [https://osf.io/xsbqy](https://osf.io/xsbqy)
    - Image Set 3: [https://osf.io/s9ay6](https://osf.io/s9ay6)
* Continuous Recognition Task (Memorability)
    - Image Set 1: [https://osf.io/ntrkp](https://osf.io/ntrkp)
    - Image Set 2: [https://osf.io/nzc65](https://osf.io/nzc65)
    - Image Set 3: [https://osf.io/k8snr](https://osf.io/k8snr)

## External datasets
These datasets are hosted by their original authors and are not redistributed here:

* Isola memorability dataset: [https://web.mit.edu/phillipi/Public/WhatMakesAnImageMemorable/](https://web.mit.edu/phillipi/Public/WhatMakesAnImageMemorable/). Extract it into `data/stimuli/isola/`. The images are distributed as MATLAB `.mat` files and must be resaved as standard image files before use.
* MemCat: [https://gestaltrevision.be/projects/memcat/](https://gestaltrevision.be/projects/memcat/). Extract it into `data/stimuli/MemCat/`, so that images end up at `data/stimuli/MemCat/MemCat_images/`.
* LaMem: [http://memorability.csail.mit.edu/download.html](http://memorability.csail.mit.edu/download.html). Extract it into `data/stimuli/lamem/`, so that images end up at `data/stimuli/lamem/images/`.

Memorability measures for these datasets come from the same source as the stimuli themselves. We provide them reformatted to match our own image measures, in `data/image_measures`. A few notes:

* The Isola memorability dataset only provides raw hit/false alarm/miss/correct-rejection counts, so CRR is calculated from these.
* For MemCat, we use the `memorability_w_fa_correction` column, renamed to `crr`.
* For LaMem, we use only the data from split 1.

Two further external sets are used only to test ViTNat's out-of-sample performance. Their naturalness ratings and ViTNat predictions are included in `data/image_measures`:

* Schertz et al. (2018): Can be downloaded from [this repository](https://github.com/kschertz/TKF_Park_Images) ([Link to the paper](https://doi.org/10.1016/j.cognition.2018.01.011))
* Coburn et al. (2019): Please contact the authors for this image set ([Link to the paper](https://doi.org/10.1016/j.jenvp.2019.02.007))

## Model outputs
The Bayesian model fits produced by `scripts/statistical_analysis/` (ArviZ InferenceData `.nc` files) are hosted externally via OSF: [https://osf.io/6cqta/files/xfe8v](https://osf.io/6cqta/files/xfe8v). Download and extract into `data/idata/`.

The summary estimates printed from these fits are already included, as `print_estimates_output.txt` in `scripts/statistical_analysis/`.

# Dependencies

Python 3.12.13. See `requirements.txt`. This covers everything except `image_stats/vitnat.py`, which was run in a separate environment. See [https://github.com/nwrim/ViTNat](https://github.com/nwrim/ViTNat).

# Citation

If you use this code or data in a scientific publication, we would appreciate citations to the following preprint:

Rim, N., Veillette, J., Lee, S., Kardan, O., Krishnan, S., Bainbridge, W. A., & Berman, M. (2025, May 15). Natural scenes are more compressible and less memorable than human-made scenes. https://doi.org/10.31234/osf.io/xw3ek_v1

Bibtex entry:

```bibtex
@misc{rim_veillette_lee_kardan_krishnan_bainbridge_berman_2025,
 title={Natural scenes are more compressible and less memorable than human-made scenes},
 url={osf.io/preprints/psyarxiv/xw3ek_v1},
 DOI={10.31234/osf.io/xw3ek_v1},
 publisher={PsyArXiv},
 author={Rim, Nakwon and Veillette, John and Lee, Sunny and Kardan, Omid and Krishnan, Sanjay and Bainbridge, Wilma A and Berman, Marc G},
 year={2025},
 month={May}
}
```

# Disclaimer

Most of the docstrings for the functions in `src/` were written with AI assistance.