# Natural scenes are more compressible and less memorable than human-made scenes

This repository contains code and data associated with the paper: **_Natural scenes are more compressible and less memorable than human-made scenes_**

# 🧮 Scripts for Image-Level Measures (`00_image_level_measures/`)

Scripts for computing image-level measures, including behavioral ratings and image-based metrics. See docstrings and inline comments within each script for details. Many custom functions used in these scripts are in `scripts/experiment_data_processing.py`, `scripts/image_processing.py`, and `scripts/compressibility.py`.

* Analysis for Image Set 1-3
    - `00_naturalness.py`: Aggregates Likert-scale naturalness ratings across three image sets.
        * The console output is save to `00_naturalness_output.txt`
    - `01_compressibility.py`: Computes JPEG-based and Canny-based compressibility across three image sets.
    - Memorability related scripts
        * `02_00_memorability.py`: Calculates image-level memorability scores from participant responses in a continuous recognition task across three image sets.
            - The console output is save to `02_01_memorability_split_half_reliability_output.txt`
        * `02_01_memorability_reliability.py`: Estimates the split-half reliability of image-level memorability scores for each image set.
* Neural network performance checking scripts
    - `03_00_vitnat_out_of_sample_prediction.py`: Applies the ViTNat model (see [this repository](https://github.com/nwrim/ViTNat)) to compute out-of-sample naturalness predictions for:
        * Image Sets 1–2
        * External sets: Schertz et al. (2018), Coburn et al. (2019)
    - `03_01_set123_resmem.py`: Applies the ResMem model (see [this repository](https://github.com/Brain-Bridge-Lab/resmem)) to compute memorability predictions for Image Sets 1–3.
* Analysis for Image Set 4 (SUN397 database)
    - `04_00_set4_vitnat_by_category.py`: Applies the ViTNat model (see [this repository](https://github.com/nwrim/ViTNat)) to compute naturalness predictions for images in a single category of the SUN397 dataset.
    - `04_01_set4_compressibility_by_category.py`: Computes JPEG-based and Canny-based compressibility scores for images in a single category of the SUN397 dataset.
    - `04_02_set4_resmem_by_category.py`: Applies the ResMem model (see [this repository](https://github.com/Brain-Bridge-Lab/resmem)) to compute memorability predictions for images in a single SUN397 category.
    - `04_03_set4_merge.py`: Merges per-category SUN397 prediction files into full dataset CSVs:
        * Naturalness (ViTNat)
        * Compressibility
        * Memorability (ResMem)

# 📈 Scripts for Statistical Analysis (`01_statistical_analysis/`)

Scripts for statistical analyses reported in the manuscript. See docstrings and inline comments within each script for details. Many custom functions used in these scripts are in `scripts/models.py`. The ArviZ InferenceData objects from the fitted models are hosted externally in [https://osf.io/snye9](https://osf.io/snye9)

* `00_create_scalars.py`: Fits and saves `sklearn.preprocessing.StandardScaler` objects for image-level variables.
* `01_linear_regressions.py`: Fits Bayesian linear regression models between image-level variables.
* `02_mediation.py`: Fits Bayesian mediation models to test whether compressibility mediates the relationship between naturalness and memorability.
* `03_00_vitnat_out_of_sample_performance.py`: Computes Pearson correlations between ViTNat predictions and human naturalness ratings (Likert) across four out-of-sample datasets. 
    - The console outputs are printed in `03_00_vitnat_out_of_sample_performance_output.txt`
* `03_01_resmem_out_of_sample_performance.py`: Computes Pearson correlations between ResMem predictions and human memorability data (corrected recognition rate and hit rate) for image sets 1–3. 
    - The console outputs are printed in `03_01_resmem_out_of_sample_performance_output.txt`
* `04_set4_linear_regression.py`: Fits Bayesian linear regression models on Image Set 4. When ViTNat is used as a predictor, the model accounts for measurement error using human data from Image Sets 2 and 3.
* `05_set4_mediation.py`: Fits Bayesian mediation models for Image Set 4 to test whether compressibility mediates the effect of ViTNat on ResMem-predicted memorability. Corrects for measurement error using human ratings and predictions from Sets 2–3.
* `supp_00_corr_btwn_compressibility.py`: Computes the Pearson correlation between JPEG-based and Canny-based compressibility scores for image sets 1–4.
    - The console outputs are printed in `supp_00_corr_btwn_compressibility_output.txt`

# 📦 Data (`data/`) 

Data used in this study. Some files are included in the repository; others are hosted externally and must be downloaded.

## 🧮 Image-level measure (`data/image_level_measure/`)

* **Naturalness rating**: The aggregated naturalness ratings for the images are in:
    - Image Set 1: `data/image_level_measure/set1_naturalness.csv` (output of `00_image_level_measures/00_naturalness.py`)
    - Image Set 2: `data/image_level_measure/set2_naturalness.csv` (output of `00_image_level_measures/00_naturalness.py`)
    - Image Set 3: `data/image_level_measure/set3_naturalness.csv` (output of `00_image_level_measures/00_naturalness.py`)
* **Compressibility**: JPEG-based and Canny-based compressibility scores for the images are in:
    - Image Set 1: `data/image_level_measure/set1_compressibility.csv` (output of `00_image_level_measures/01_compressibility.py`)
    - Image Set 2: `data/image_level_measure/set2_compressibility.csv` (output of `00_image_level_measures/01_compressibility.py`)
    - Image Set 3: `data/image_level_measure/set3_compressibility.csv` (output of `00_image_level_measures/01_compressibility.py`)
    - Image Set 4: `data/image_level_measure/set4_compressibility.csv` (output of `00_image_level_measures/04_03_set4_merge.py`)
        > ⚠️ Note: While `00_image_level_measures/04_01_set4_compressibility_by_category.py` performs the actual calculation, the per-category output files (`data/image_level_measure/set4_by_category/set4_{category_name}_compressibility.csv`) are not included in the repository due to redundancy. All compressibility scores are merged into the single CSV above.
* **Memorability**: Image-level memorability scores (hit rates) from the continuous recognition task are in:
    - Image Set 1: `data/image_level_measure/set1_memorability.csv` (output of `00_image_level_measures/02_00_memorability.py`)
    - Image Set 2: `data/image_level_measure/set2_memorability.csv` (output of `00_image_level_measures/02_00_memorability.py`)
    - Image Set 3: `data/image_level_measure/set3_memorability.csv` (output of `00_image_level_measures/02_00_memorability.py`)
* **ViTNat**: Predicted naturalness using the ViTNat model for the images is in:
    - Image Set 1: `data/image_level_measure/set1_vitnat.csv` (output of `00_image_level_measures/03_00_vitnat_out_of_sample_prediction.py`)
    - Image Set 2: `data/image_level_measure/set2_vitnat.csv` (output of `00_image_level_measures/03_00_vitnat_out_of_sample_prediction.py`)
    - Schertz et al. (2018): `data/image_level_measure/schertz2018_vitnat.csv` (output of `00_image_level_measures/03_00_vitnat_out_of_sample_prediction.py`)
    - Coburn et al. (2019): `data/image_level_measure/coburn2019_vitnat.csv` (output of `00_image_level_measures/03_00_vitnat_out_of_sample_prediction.py`)
    - Image Set 4: `data/image_level_measure/set4_vitnat.csv` (output of `00_image_level_measures/04_03_set4_merge.py`)
        > ⚠️ Note: While `00_image_level_measures/04_00_set4_vitnat_by_category.py` performs the actual predictions, the per-category output files (`data/image_level_measure/set4_by_category/set4_{category_name}_vitnat.csv`) are not included in the repository due to redundancy. All naturalness predictions are merged into the single CSV above.
* **ResMem**: Predicted memorability scores using the ResMem model for the images is in:
    - `data/image_level_measure/set1_resmem.csv` (output of `00_image_level_measures/03_01_set123_resmem.py`)
    - `data/image_level_measure/set2_resmem.csv` (output of `00_image_level_measures/03_01_set123_resmem.py`)
    - `data/image_level_measure/set3_resmem.csv` (output of `00_image_level_measures/03_01_set123_resmem.py`)
    - `data/image_level_measure/set4_resmem.csv` (output of `00_image_level_measures/04_03_set4_merge.py`)
        > ⚠️ Note: While `00_image_level_measures/04_02_set4_resmem_by_category.py` performs the actual predictions, the per-category output files (`data/image_level_measure/set4_by_category/set4_{category_name}_resmem.csv`) are not included in the repository due to redundancy. All memorability predictions are merged into the single CSV above.

## 🧪 Experimental data (`data/experiment/`)

The raw experimental data is hosted externally via OSF. Download and extract each `.tar.gz` archive into `data/experiment/` to run the aggregation scripts.

* Naturalness rating
    - Image Set 1: [https://osf.io/9sx3c](https://osf.io/9sx3c)
    - Image Set 2: [https://osf.io/xsbqy](https://osf.io/xsbqy)
    - Image Set 3: [https://osf.io/s9ay6](https://osf.io/s9ay6)
* Continuous Recognition Task (Memorability)
    - Image Set 1: [https://osf.io/ntrkp](https://osf.io/ntrkp)
    - Image Set 2: [https://osf.io/nzc65](https://osf.io/nzc65)
    - Image Set 3: [https://osf.io/k8snr](https://osf.io/k8snr)

## 🖼️ Stimuli (`data/stimuli`)

The image stimuli are hosted externally. Download and extract the `.tar.gz` archives into `data/stimuli/` to run scripts that compute compressibility or apply ViTNat/ResMem.

* Image Set 1: [https://osf.io/rhy8d](https://osf.io/rhy8d)
* Image Set 2: [https://osf.io/dyjk9](https://osf.io/dyjk9)
* Image Set 3: [https://osf.io/5bwgu](https://osf.io/5bwgu)

Image Set 4, or the SUN397 database can be downloaded from:

* Image Set 4 (SUN397 database): [SUN397.tar.gz](http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz) ([project website](https://vision.princeton.edu/projects/2010/SUN/)).

The two external sets used to test ViTNat:

* Schertz et al. (2018): Can be downloaded from [this repository](https://github.com/kschertz/TKF_Park_Images) ([Link to the paper](https://doi.org/10.1016/j.cognition.2018.01.011))
* Coburn et al. (2019): Please contact the authors for this image set ([Link to the paper](https://doi.org/10.1016/j.jenvp.2019.02.007))

# 📊 Figure Generation Scripts (`figures/`)

Scripts for reproducing main and supplementary figures. See docstrings and inline comments within each script for details. Many custom functions used in these scripts are in `scripts/plotting.py`.

* `fig1.py`: Generates components of Figure 1. Outputs are saved in `figures/outputs/fig1/`.
* `fig2ac.py`: Generates panels A and C of Figure 2. Outputs are saved in `figures/outputs/fig2/`.
    - The console output is save to `figures/fig2ac_output.txt`
* `fig2bd.py`: Generates panels B and D of Figure 2. Outputs are saved in `figures/outputs/fig2/`.
    - The console output is save to `figures/fig2bd_output.txt`
* `fig3.py`, `fig4.py`, `fig5.py`: Generate Figures 3–5. Outputs are saved in `figures/outputs/fig{3-5}.svg`
* `figS1.py`: Generates components of Figure S1. Outputs are saved in `figures/outputs/figS1/`.
* `figS2.py`: Generates components of Figure S2. Outputs are saved in `figures/outputs/figS2/`.
* `figS3.py`, `figS4.py`, `figS7.py`, `figS8.py`: Generate Supplementary Figures S3, S4, S7, and S8. Outputs are saved in `figures/outputs/figS{3, 4, 7, 8}.svg`
* `figS56.py`: Generates Supplementary Figures S5 and S6.

# 📦 Dependencies
To install all required packages, use the provided `requirements.txt` file:

```sh
pip install -r requirements.txt
```

Alternatively, you can install the core packages manually. This project was developed with the following versions:

* `python==3.11.5`
* `numpy==2.2.5`
* `pandas==2.2.3`
* `tqdm==4.67.1`
* `scipy==1.15.2`
* `scikit-image==0.25.2`
* `pymc==5.22.0`
* `scikit-learn==1.6.1`
* `seaborn==0.13.2`

For scripts that require neural network models, please refer to the dependencies listed in their respective repositories:

* [ViTNat](https://github.com/nwrim/ViTNat)
* [ResMem](https://github.com/Brain-Bridge-Lab/resmem)
