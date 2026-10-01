import pandas as pd
import pymc as pm

def linear_regression(predictors, outcome):
    """
    Build a Bayesian linear regression model using PyMC.

    Uses standard normal priors (μ = 0, σ = 1) for the intercept and regression coefficients,
    and an exponential prior (λ = 1) for the error standard deviation.

    Parameters
    ----------
    predictors : list of np.array
        The predictor variables.
    outcome : np.array
        The outcome variable.

    Returns
    -------
    model : pymc.Model
        The model.

    """
    with pm.Model() as model:
        # priors
        a = pm.Normal("alpha", 0.0, 1.0)
        betas = [pm.Normal(f"beta_{i}", 0.0, 1.0) for i in range(len(predictors))]
        sigma = pm.Exponential("sigma", 1.0)

        # model
        mu = a + sum(b * predictor for b, predictor in zip(betas, predictors))
        pm.Normal('dv', mu=mu, sigma=sigma, observed=outcome)
    return model

def linear_regression_w_random_intercept(predictors, outcome, random_intercept_var):
    """
    Build a Bayesian linear regression model with a random intercept.

    Uses standard normal priors (μ = 0, σ = 1) for the intercept and regression coefficients,
    and an exponential prior (λ = 1) for the residual standard deviation, and a
    hierarchical random intercept indexed by `random_intercept_var`.

    Parameters
    ----------
    predictors : list of np.array
        Predictor variables, each of shape (n_samples,).
    outcome : np.array
        Outcome variable, shape (n_samples,).
    random_intercept_var : array-like
        Group labels for the random intercept (e.g. category names),
        one per sample.

    Returns
    -------
    model : pymc.Model
        The model.
    """
    # Convert grouping variable to integer codes, while preserving labels
    group_idx, group_labels = pd.factorize(random_intercept_var, sort=True)
    coords = {"group": group_labels}

    with pm.Model(coords=coords) as model:
        # Data containers
        group_idx_data = pm.Data("group_idx", group_idx)
        predictor_data = [
            pm.Data(f"predictor_{i}", predictor)
            for i, predictor in enumerate(predictors)
        ]

        # priors
        # global intercept
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)

        # Random intercept SD
        sigma_group = pm.Exponential("sigma_group", 1.0)

        # Non-centered parameterization for random intercepts
        z_group = pm.Normal("z_group", mu=0.0, sigma=1.0, dims="group")
        alpha_group = pm.Deterministic(
            "alpha_group",
            z_group * sigma_group,
            dims="group"
        )

        # Fixed slopes
        betas = [pm.Normal(f"beta_{i}", 0.0, 1.0) for i in range(len(predictors))]

        # Residual SD
        sigma = pm.Exponential("sigma", 1.0)

        # model
        mu = alpha + alpha_group[group_idx_data]
        mu += sum(b * x for b, x in zip(betas, predictor_data))
        pm.Normal("dv", mu=mu, sigma=sigma, observed=outcome)
    return model

def mediation_model(predictor, mediator, outcome, controls=None):
    """
    Build a Bayesian mediation model with a single predictor, mediator, and outcome variable.

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
    controls : list of np.array, optional
        Control variables, added to both the mediator and outcome equations
        with separate coefficients. Defaults to None (no controls).

    Returns
    -------
    model : pymc.Model
        The model.

    """
    if controls is None:
        controls = []

    with pm.Model() as model:
        # intercept priors
        alpha_m = pm.Normal("alpha_m", mu=0, sigma=1)
        alpha_y = pm.Normal("alpha_y", mu=0, sigma=1)

        # slope priors
        a = pm.Normal("a", mu=0, sigma=1)
        b = pm.Normal("b", mu=0, sigma=1)
        cprime = pm.Normal("cprime", mu=0, sigma=1)

        # control slope priors (separate coefficients in each equation)
        gammas_m = [pm.Normal(f"gamma_m_{i}", mu=0, sigma=1) for i in range(len(controls))]
        gammas_y = [pm.Normal(f"gamma_y_{i}", mu=0, sigma=1) for i in range(len(controls))]

        # noise priors
        sigma_m = pm.Exponential("sigma_m", 1)
        sigma_y = pm.Exponential("sigma_y", 1)

        # model
        # likelihood
        mu_m = alpha_m + a * predictor + sum(g * c for g, c in zip(gammas_m, controls))
        mu_y = alpha_y + b * mediator + cprime * predictor + sum(g * c for g, c in zip(gammas_y, controls))
        pm.Normal('mediator', mu=mu_m, sigma=sigma_m, observed=mediator)
        pm.Normal('dv', mu=mu_y, sigma=sigma_y, observed=outcome)

        # calculate quantities of interest
        pm.Deterministic("indirect_effect", a * b)
        pm.Deterministic("total_effect", a * b + cprime)
    return model
