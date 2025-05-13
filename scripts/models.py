import pandas as pd
import os
import pymc as pm
import numpy as np

def load_data(image_set_index):
    """
    Load and merge image-level measures for a given image set index.

    For image sets 1-3, returns human-rated naturalness and memorability (CRR and HR).
    For image set 4, returns predicted naturalness (ViTNat) and memorability (ResMem).
    All sets include JPEG- and Canny-based compressibility.

        Parameters
    ----------
    image_set_index : int
        The index of the image set to load data for (must be 1, 2, 3, or 4).

    Returns
    -------
    A DataFrame with one row per image and the following columns:
        - For sets 1–3: image_name, naturalness, crr, hr, jpeg, canny
        - For set 4: image_name, category, vitnat, resmem, jpeg, canny
    """

    assert image_set_index in [1, 2, 3, 4], "image_set_index must be 1, 2, 3, or 4"

    if image_set_index in [1, 2, 3]:
        # Load data from CSV files for the specified image set index
        naturalness_df = pd.read_csv(os.path.join('..', 'data', 'image_level_measure', f'set{image_set_index}_naturalness.csv'))
        memorability_df = pd.read_csv(os.path.join('..', 'data', 'image_level_measure', f'set{image_set_index}_memorability.csv'))
        # The 'target_crr' column in memorability_df is renamed to 'crr' and 'target_hr' to 'hr'
        memorability_df.rename(columns={'target_crr': 'crr', 'target_hr': 'hr'}, inplace=True)
        compressibility_df = pd.read_csv(os.path.join('..', 'data', 'image_level_measure', f'set{image_set_index}_compressibility.csv'))
        # The 'jpeg_based_compressibility' column renamed to 'jpeg' and 'canny_based_compressibility' to 'canny'
        compressibility_df.rename(columns={'jpeg_based_compressibility': 'jpeg', 'canny_based_compressibility': 'canny'}, inplace=True)

        # merge the DataFrames on 'image_name'
        df = pd.merge(naturalness_df, memorability_df, on='image_name').merge(compressibility_df, on='image_name')

        # Sanity check: ensure that all DataFrames have the same length after merging
        assert len(df) == len(naturalness_df) == len(memorability_df) == len(compressibility_df), "DataFrames have different lengths after merging"

        return df[['image_name', 'naturalness', 'crr', 'hr', 'jpeg', 'canny']]
    else:
        # Load data from CSV files for the specified image set index
        vitnat_df = pd.read_csv(os.path.join('..', 'data', 'image_level_measure', f'set{image_set_index}_vitnat.csv'))
        resmem_df = pd.read_csv(os.path.join('..', 'data', 'image_level_measure', f'set{image_set_index}_resmem.csv'))
        compressibility_df = pd.read_csv(os.path.join('..', 'data', 'image_level_measure', f'set{image_set_index}_compressibility.csv'))
        # The 'jpeg_based_compressibility' column renamed to 'jpeg' and 'canny_based_compressibility' to 'canny'
        compressibility_df.rename(columns={'jpeg_based_compressibility': 'jpeg', 'canny_based_compressibility': 'canny'}, inplace=True)

        # merge the DataFrames on 'image_name'
        df = pd.merge(vitnat_df, resmem_df, on=['image_name', 'category']).merge(compressibility_df, on=['image_name', 'category'])

        # Sanity check: ensure that all DataFrames have the same length after merging
        assert len(df) == len(vitnat_df) == len(resmem_df) == len(compressibility_df), "DataFrames have different lengths after merging"

        return df

def linear_regression(predictors, outcome, seed, draws=20000, tune=5000, chains=4):
    """
    Fit a Bayesian linear regression model using PyMC.

    Uses standard normal priors (μ = 0, σ = 1) for the intercept and regression coefficients, 
    and an exponential prior (λ = 1) for the error standard deviation.

    Parameters
    ----------
    predictors : list of np.array
        The predictor variables.
    outcome : np.array
        The outcome variable.
    seed : int
        The random seed for the sampler.
    draws : int, optional
        The number of samples to draw. Defaults to 20000. See `pymc.sample`.
    tune : int, optional
        Number of iterations to tune, defaults to 5000. See `pymc.sample`.
    chains : int, optional
        The number of chains to sample, defaults to 4. See `pymc.sample`.

    Returns
    -------
    idata : arviz.InferenceData
        The InferenceData object containing the posterior samples from the model.

    """
    rng = np.random.default_rng(seed)

    with pm.Model() as lr_model:
        # priors
        a = pm.Normal("alpha", 0.0, 1.0) # intercept
        betas = [pm.Normal(f"beta_{i}", 0.0, 1.0) for i in range(len(predictors))]
        sigma = pm.Exponential("sigma", 1.0)

        # model
        mu = a + sum([b * predictor for b, predictor in zip(betas, predictors)])
        pm.Normal('dv', mu=mu, sigma=sigma, observed=outcome)

        idata = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=rng)
    return idata

def mediation_model(predictor, mediator, outcome, seed, draws=20000, tune=5000, chains=4):
    """
    Fit a Bayesian mediation model with a single predictor, mediator, and outcome variable.
    
    Uses standard normal priors (μ = 0, σ = 1) for the intercept and regression coefficients, 
    and an exponential prior (λ = 1) for the error standard deviation.

    Parameters
    ----------
    predictor : np.array
        The predictor variable.
    mediator : np.array
        The mediator variable.
    outcome : np.array
        The outcome variable.
    seed : int
        The random seed for the sampler.
    draws : int, optional
        The number of samples to draw. Defaults to 20000. See `pymc.sample`.
    tune : int, optional
        Number of iterations to tune, defaults to 5000. See `pymc.sample`.
    chains : int, optional
        The number of chains to sample, defaults to 4. See `pymc.sample`.
    
    Returns
    -------
    idata : arviz.InferenceData
        The InferenceData object containing the samples from the model.

    """
    rng = np.random.default_rng(seed)

    with pm.Model() as med_model:
        # intercept priors
        alpha_m = pm.Normal("alpha_m", mu=0, sigma=1)
        alpha_y = pm.Normal("alpha_y", mu=0, sigma=1)

        # slope priors
        a = pm.Normal("a", mu=0, sigma=1)
        b = pm.Normal("b", mu=0, sigma=1)
        cprime = pm.Normal("cprime", mu=0, sigma=1)

        # noise priors
        sigma_m = pm.Exponential("sigma_m", 1)
        sigma_y = pm.Exponential("sigma_y", 1)

        # model
        # likelihood
        pm.Normal('mediator', mu=alpha_m + a * predictor, sigma=sigma_m, observed=mediator)
        pm.Normal('dv', mu=alpha_y + b * mediator + cprime * predictor, sigma=sigma_y, observed=outcome)

        # calculate quantities of interest
        pm.Deterministic("indirect_effect", a * b)
        pm.Deterministic("total_effect", a * b + cprime)

        idata = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=rng)
    return idata

def linear_regression_with_measurement_error(predictors, outcome, seed, x_examp_true,
    x_examp_noisy, x_scale_examp, noisy_predictor_idx=0, draws=20000, tune=5000, chains=4):
    """
    Fit a Bayesian linear regression model, correcting for measurement error in one predictor.
    Measurement error in the outcome (y) is absorbed into the residual variance and is not explicitly corrected.

    Bias in slope estimate due to measurement error is corrected via 
    "regression calibration" similar to [1]. 
    
    Notes
    -----------
    For interpretable parameter estimates, the variables should be rescaled 
    *by the same factor,* such that true variable (not noisy/observed variable) has 
    approximately unit variance on the target dataset.

    References
    ----------
    [1] Spiegelman, D., McDermott, A., & Rosner, B. (1997). 
        Regression calibration method for correcting measurement-error bias 
        in nutritional epidemiology. 
        The American journal of clinical nutrition, 65(4), 1179S-1186S.
        https://doi.org/10.1093/ajcn/65.4.1179S

    Parameters
    -----------
    predictors: np.array
        The observed predictor variables. 
        One of the predictors is assumed to be measured with error and will be corrected using regression calibration.
    outcome : np.array
        The observed outcome variable.
    seed : int
        The random seed for the sampler.
    x_examp_true : np.array
        The "true" x values from a reference dataset, on which we'll estimate
        the measurement error.
    x_examp_noisy : np.array
        The noisy x values from the same reference dataset. Should be of same shape
        as `x_examp_true`.
    x_scale_examp : np.array, optional
        "true" x data from which to estimate the variance of x.
        This can be the same as x_examp_true (default) or another dataset.
        Recommended to pick a dataset whose 'true' predictor values have similar
        variance to that of the target dataset.
    noisy_predictor_idx : int, optional
        Index of the noisy predictor. Defaults to 0 (i.e., the first predictor). 
    draws : int, optional
        The number of samples to draw. Defaults to 20000. See `pymc.sample`.
    tune : int, optional
        Number of iterations to tune, defaults to 5000. See `pymc.sample`.
    chains : int, optional
        The number of chains to sample, defaults to 4. See `pymc.sample`

    Returns
    ----------
    idata : arviz.InferenceData
        The InferenceData object containing the posterior samples from the model.
    """
    rng = np.random.default_rng(seed)

    with pm.Model() as lr_model:
        # fit the regression as if the noisy variable is the true variable
        # with correction applied separately to estimate the corrected slope for the noisy predictor.
        # priors
        a = pm.Normal("alpha", 0.0, 1.0) # intercept
        betas = [pm.Normal(f"beta_{i}", 0.0, 1.0) for i in range(len(predictors))]
        sigma = pm.Exponential("sigma", 1.0)

        # model
        mu = a + sum([b * predictor for b, predictor in zip(betas, predictors)])
        pm.Normal('dv', mu=mu, sigma=sigma, observed=outcome)

        # correcting for measurement error
        # in the example dataset, where we know the noisy and true value,
        # we estimate the magnitude of measurement error (x_error) in the predictor 
        # based on the difference between noisy and true values
        x_error = pm.Exponential('x_error', 1.)
        pm.Normal('x_examp_noisy', x_examp_true, x_error, observed = x_examp_noisy)

        # we then estimate the scale of the target data from the example dataset used above or a different dataset
        if x_scale_examp is None:
            x_scale_examp = x_examp_true
        x_scale = pm.Exponential('x_scale', 1.)
        pm.Normal('x_examp_true', 0, x_scale, observed = x_scale_examp)

        # Estimate the deflation factor λ and apply to correct the slope for the noisy predictor.
        lam = x_scale**2 / (x_scale**2 + x_error**2)
        pm.Deterministic(f'beta_{noisy_predictor_idx}_corrected', betas[noisy_predictor_idx] / lam)

        idata = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=rng)
    return idata

def mediation_model_with_measurement_error(predictor, mediator, outcome, seed, x_examp_true, 
    x_examp_noisy, x_scale_examp, draws=20000, tune=5000, chains=4):
    """
    Fit a Bayesian mediation model with single predictor, mediator, and outcome variable, correcting for measurement error in the predictor.
    
    The mediator is assumed to be measured without error. Measurement error in the outcome (y) is absorbed into the residual variance and is not explicitly corrected.
     
    Bias in parameter estimates due to measurement error is corrected via 
    "regression calibration" similar to [1]. Corrected direct/indirect 
    effect measurements are computed using corrected regression parameters.
    
    Notes
    -----------
    For interpretable parameter estimates, the variables should be rescaled 
    *by the same factor,* such that true variable (not noisy/observed variable) has 
    approximately unit variance on the target dataset.

    References
    ----------
    [1] Spiegelman, D., McDermott, A., & Rosner, B. (1997). 
        Regression calibration method for correcting measurement-error bias 
        in nutritional epidemiology. 
        The American journal of clinical nutrition, 65(4), 1179S-1186S.
        https://doi.org/10.1093/ajcn/65.4.1179S

    Parameters
    -----------
    predictor : np.array
        The predictor variable. Assumed to be measured with error and will be corrected using regression calibration.
    mediator : np.array
        The mediator variable (assumed to be error-free).
    outcome : np.array
        The outcome variable.
    seed : int
        The random seed for the sampler.
    x_examp_true : np.array
        The "true" x values from a reference dataset, on which we'll estimate the measurement error.
    x_examp_noisy : np.array
        The noisy x values from the same reference dataset. Should be of same shape
        as `x_examp_true`.
    x_scale_examp : np.array, optional
        "true" x data from which to estimate the variance of x.
        This can be the same as x_examp_true (default) or another dataset.
        Recommended to pick a dataset that matches the variance of the target
        dataset as closely as possible.
    draws : int, optional
        The number of samples to draw. Defaults to 20000. See `pymc.sample`.
    tune : int, optional
        Number of iterations to tune, defaults to 5000. See `pymc.sample`.
    chains : int, optional
        The number of chains to sample, defaults to 4. See `pymc.sample`

    Returns
    ----------
    idata : arviz.InferenceData
        The InferenceData object containing the samples from the model.
    """
    rng = np.random.default_rng(seed)

    with pm.Model() as med_model:
        # fit the mediation model as if the noisy variable is the true variable
        # intercept priors
        alpha_m = pm.Normal('alpha_m', mu = 0, sigma = 1)
        alpha_y = pm.Normal('alpha_y', mu = 0, sigma = 1)
        # slope priors
        a = pm.Normal('a', mu = 0, sigma = 1)
        b = pm.Normal('b', mu = 0, sigma = 1)
        cprime = pm.Normal('cprime', mu = 0, sigma = 1)
        # noise priors
        sigma_m = pm.Exponential('sigma_m', 1)
        sigma_y = pm.Exponential('sigma_y', 1)
        # likelihood
        pm.Normal('m', mu = alpha_m + a*predictor, sigma = sigma_m, observed = mediator)
        pm.Normal('dv', mu = alpha_y + b*mediator + cprime*predictor, sigma = sigma_y, observed = outcome)

        indirect = pm.Deterministic('indirect_effect', a*b)
        pm.Deterministic('total_effect', cprime + indirect)

        # correcting for measurement error
        # in the example dataset, where we know the noisy and true value,
        # we estimate the magnitude of measurement error (x_error) in the predictor 
        # based on the difference between noisy and true values
        x_error = pm.Exponential('x_error', 1.)
        pm.Normal('x_examp_noisy', x_examp_true, x_error, observed = x_examp_noisy)

        # we then estimate the scale of the target data from the example dataset used above or a different dataset
        if x_scale_examp is None:
            x_scale_examp = x_examp_true
        x_scale = pm.Exponential('x_scale', 1.)
        pm.Normal('x_examp_true', 0, x_scale, observed = x_scale_examp)
        # Estimate the deflation factor λ
        lam = x_scale**2 / (x_scale**2 + x_error**2) 

        ## now correct a and c for measurement error
        a_corr = pm.Deterministic('a_corrected', a / lam)
        cprime_corrected = pm.Deterministic('cprime_corrected', cprime / lam)
        # and calculate quantities of interest with corrected values
        indirect_corrected = pm.Deterministic('indirect_effect_corrected', a_corr*b)
        pm.Deterministic('total_effect_corrected', cprime_corrected + indirect_corrected)
        
        idata = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=rng)
    return idata