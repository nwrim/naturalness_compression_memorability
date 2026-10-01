import numpy as np
import xarray as xr
from arviz_stats import hdi
from skimage.io import imsave

def save_inverted_image(image, out_path):
    imsave(out_path, 255 - np.clip(image, 0, 255).astype(np.uint8))

def axis_limits(values, margin=0.05):
    """
    Min/max of `values` (original units) padded by `margin` of the range, for axis limits.

    Fold any tick positions into `values` so the returned limits always contain them.

    Parameters
    ----------
    values : np.ndarray
        Values (and optionally tick positions) the limits should contain.
    margin : float, default 0.05
        Fraction of the range added as padding on each side.

    Returns
    -------
    tuple of float
        The (low, high) limits.
    """
    lo, hi = np.min(values), np.max(values)
    pad = (hi - lo) * margin
    return (lo - pad, hi + pad)

def plot_linear_relationship(ax, x, y, idata, beta_term='beta_0', hdi_prob=0.96,
                             x_scaler=None, y_scaler=None,
                             xticks=None, yticks=None, xlim=None, ylim=None,
                             density=False, num_bins=40, cmap=None, norm=None,
                             scatter_color='#a383c6', scatter_alpha=0.2, scatter_size=30,
                             line_color='#61110c', line_width=1.5,
                             hdi_color='#ca4d4f', hdi_alpha=0.7,
                             set_aspect_ratio_equal=True):
    """
    Visualize a Bayesian linear regression over its data.

    Draws the data either as a scatter (default) or, for large datasets where a scatter would
    overplot, as a 2D-density heatmap (`density=True`, via ax.hist2d). Overlays the posterior
    mean regression line and a highest density interval (HDI) band, reconstructed from the
    posterior as `alpha + beta * x`; any other predictors (e.g. controls) are held at their
    standardized mean of 0.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis object to draw the plot on.
    x : np.ndarray
        Predictor variable (in original units; standardized with `x_scaler` if given).
    y : np.ndarray
        Outcome variable (in original units; standardized with `y_scaler` if given).
    idata : arviz.InferenceData
        Posterior samples containing 'alpha' and the coefficient named by `beta_term`.
    beta_term : str, default 'beta_0'
        Name of the slope coefficient for `x` in `idata.posterior`.
    hdi_prob : float, default 0.96
        Credible interval width for the HDI band.
    x_scaler, y_scaler : sklearn.preprocessing.StandardScaler, optional
        Scalers used to map x and y (and the ticks) into the standardized model space.
    xticks, yticks : np.ndarray, optional
        Tick positions in original units, mapped into standardized space for display.
    xlim, ylim : tuple, optional
        Axis limits in original units, mapped into standardized space (e.g. to give every
        panel a common window).
    density : bool, default False
        If True, draw the data as a 2D-density heatmap (ax.hist2d) instead of a scatter.
    num_bins : int, default 40
        Number of bins along each axis for the density heatmap (used when `density=True`).
    cmap : matplotlib Colormap, optional
        Colormap for the density heatmap (used when `density=True`).
    norm : matplotlib.colors.Normalize, optional
        Color normalization for the density heatmap, e.g. LogNorm (used when `density=True`).
    scatter_color, scatter_alpha, scatter_size : aesthetic controls for the scatter.
    line_color, line_width : aesthetic controls for the regression line.
    hdi_color, hdi_alpha : aesthetic controls for the HDI band.
    set_aspect_ratio_equal : bool, default True
        If True, force the panel (axes box) to be square via set_box_aspect(1).

    Returns
    -------
    matplotlib.collections.QuadMesh or None
        The density mesh when `density=True` (for attaching a colorbar), else None.
    """
    if x_scaler is not None:
        x = x_scaler.transform(x.reshape(-1, 1)).flatten()
    if y_scaler is not None:
        y = y_scaler.transform(y.reshape(-1, 1)).flatten()

    if density:
        *_, mesh = ax.hist2d(x, y, bins=num_bins, cmap=cmap, norm=norm)
    else:
        ax.scatter(x, y, alpha=scatter_alpha, color=scatter_color, s=scatter_size)
        mesh = None

    # reconstruct the regression line over the predictor range (controls held at mean 0)
    post = idata['posterior'].to_dataset()
    x_linspace = np.linspace(np.min(x), np.max(x), 1000)
    y_model = post['alpha'] + post[beta_term] * xr.DataArray(x_linspace, dims='point')
    ax.plot(x_linspace, y_model.mean(dim=('chain', 'draw')), color=line_color, lw=line_width)

    # HDI band, computed (vectorized over the predictor range) from the posterior
    band = hdi(y_model, prob=hdi_prob, dim=['chain', 'draw'])
    ax.fill_between(x_linspace, band.sel(ci_bound='lower'), band.sel(ci_bound='upper'),
                    color=hdi_color, alpha=hdi_alpha)

    # ticks are given in original units and mapped into the standardized space
    if xticks is not None and x_scaler is not None:
        ax.set_xticks(x_scaler.transform(xticks.reshape(-1, 1)).flatten())
        ax.set_xticklabels(xticks)
    if yticks is not None and y_scaler is not None:
        ax.set_yticks(y_scaler.transform(yticks.reshape(-1, 1)).flatten())
        ax.set_yticklabels(yticks)

    if xlim is not None:
        if x_scaler is not None:
            xlim = x_scaler.transform(np.reshape(xlim, (-1, 1))).flatten()
        ax.set_xlim(xlim)
    if ylim is not None:
        if y_scaler is not None:
            ylim = y_scaler.transform(np.reshape(ylim, (-1, 1))).flatten()
        ax.set_ylim(ylim)

    if set_aspect_ratio_equal:
        # square the axes box directly (robust to later limit changes, e.g. shared axes)
        ax.set_box_aspect(1)

    return mesh
