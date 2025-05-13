from skimage.io import imsave
from scipy.fftpack import idctn
import numpy as np
import xarray as xr
import arviz as az
import seaborn as sns
from sklearn.linear_model import LinearRegression

def save_reversed_weighted_image(image, out_path, weight=5):
    """
    Save an inverted and weighted image to disk.

    The image is multiplied by a weight, clipped to [0, 255], inverted (255 - value),
    cast to uint8, and saved.

    Parameters
    ----------
    image : np.ndarray
        The image to be saved.
    out_path : str
        File path where the image will be saved.
    weight : float, optional
        Scaling factor applied to the image before inversion. Default is 5.
    """

    imsave(out_path, 255 - np.clip(image * weight, 0, 255).astype(np.uint8))

def reconstruct_image_from_one_dct_antidiagonal(dct_by_tile, mask, img_shape):
    """
    Reconstruct an image by applying an inverse DCT using only coefficients along a specified DCT antidiagonal.

    Parameters
    ----------
    dct_by_tile : np.ndarray
        DCT coefficients for each 8x8 tile in the image.
    mask : np.ndarray
        Binary mask applied to each 8x8 DCT tile to select specific frequencies.
    img_shape : tuple
        Shape of the original image.

    Returns
    -------
    reconstructed_img : np.ndarray
        Image reconstructed from inverse DCT applied to masked tiles.
    """

    reconstructed_img = np.zeros_like(dct_by_tile)
    for i in np.r_[:img_shape[0]:8]:
        for j in np.r_[:img_shape[1]:8]:
            tile = dct_by_tile[i:(i+8), j:(j+8)].copy()
            mask = mask[:tile.shape[0], :tile.shape[1]]
            tile *= mask
            reconstructed_img[i:(i+8), j:(j+8)] = idctn(tile, norm='ortho')
    return reconstructed_img

def threshold_image(image, thresholds):
    """
    Threshold an image by zeroing values outside a specified range.

    If one threshold is provided, all pixel values below it are set to 0.
    If two thresholds are provided, values below the first or above the second are set to 0.

    Parameters
    ----------
    image : np.ndarray
        The input image.
    thresholds : list or tuple
        One or two threshold values. Values outside the range are zeroed.

    Returns
    -------
    thresholded_image : np.ndarray
        The thresholded image.
    """

    thresholded_image = image.copy()
    thresholded_image[thresholded_image < thresholds[0]] = 0
    if len(thresholds) > 1:
        thresholded_image[thresholded_image >= thresholds[1]] = 0
    return thresholded_image

def plot_linear_relationship(ax, x, y, idata, hdi_prob=0.96,
                             x_scaler=None, y_scaler=None, rescale_ticks=None,
                             xticks=None, yticks=None, 
                             scatter_color='#B57EDC', scatter_alpha=0.2, scatter_size=30, 
                             line_color='#674846', hdi_color='#E52B50', hdi_alpha=0.7,
                             set_aspect_ratio_equal=True):
    """
    Visualize a Bayesian linear regression using posterior samples.

    Plots a scatter of predictor vs. outcome values, the posterior mean regression line,
    and a highest density interval (HDI) band from the posterior samples.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis object to draw the plot on.
    x : array-like
        Predictor variable.
    y : array-like
        Outcome variable.
    idata : arviz.InferenceData
        Posterior samples containing 'alpha', 'beta', and 'y_model'.
    x_scaler : StandardScaler, optional
        Scaler used to transform x (for z-scoring).
    y_scaler : StandardScaler, optional
        Scaler used to transform y.
    rescale_ticks : bool, optional
        If rescale_ticks = True, you assume the input xticks and yticks are in original units and need scaling.
    x_scaler, y_scaler : sklearn.preprocessing.StandardScaler, optional
        Scalers used to transform x and y to standardized space.
    scatter_color, scatter_alpha, scatter_size : aesthetic controls for scatterplot.
    line_color, hdi_color, hdi_alpha : aesthetic controls for regression line and credible band.
    set_aspect_ratio_equal : bool, default True
        If True, enforces a 1:1 axis scale.
    """

    if x_scaler is not None:
        x = x_scaler.transform(x.reshape(-1, 1)).flatten()
    if y_scaler is not None:
        y = y_scaler.transform(y.reshape(-1, 1)).flatten()
    if rescale_ticks is None and (x_scaler is not None or y_scaler is not None):
        rescale_ticks = True
    elif rescale_ticks is None:
        rescale_ticks = False


    # plot the scatter plot
    ax.scatter(x, y, alpha=scatter_alpha, color=scatter_color, s=scatter_size)

    # plot the y model
    x_linspace = np.linspace(np.min(x), np.max(x), 1000)
    ax.plot(x_linspace, idata.posterior["y_model"].mean(dim=("chain", "draw")), color=line_color)

    # plot the hdi
    az.plot_hdi(x_linspace, idata.posterior["y_model"], ax=ax, color=hdi_color, hdi_prob=hdi_prob,
                fill_kwargs={'alpha': hdi_alpha})

    if rescale_ticks:
        if xticks is not None:
            ax.set_xticks(x_scaler.transform(xticks.reshape(-1, 1)).flatten())
            ax.set_xticklabels(xticks)
        if yticks is not None:
            ax.set_yticks(y_scaler.transform(yticks.reshape(-1, 1)).flatten())
            ax.set_yticklabels(yticks)
    else:
        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)

    if set_aspect_ratio_equal:
        ax.set_aspect(np.diff(ax.get_xlim()) / np.diff(ax.get_ylim()))

def plot_linear_relationship_lr(ax, x, y,
                                xticks=None, yticks=None, 
                                scatter_color='#B57EDC', scatter_alpha=0.2, scatter_size=30,
                                line_color='#674846', set_aspect_ratio_equal=True):
    """
    Plot a OLS linear regression with a scatterplot using seaborn.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on.
    x : array-like
        Predictor variable.
    y : array-like
        Outcome variable.
    xticks, yticks : array-like, optional
        Custom tick positions.
    scatter_color, scatter_alpha, scatter_size : aesthetics for scatterplot.
    line_color : str
        Color of regression line.
    set_aspect_ratio_equal : bool, default True
        If True, enforces a 1:1 axis scale.
    """

    sns.regplot(x=x, y=y,
                ax=ax, color=scatter_color,
                scatter_kws={'alpha': scatter_alpha},
                line_kws={'color': line_color},
                ci=None)
    
    ax.set_xlabel('')
    ax.set_ylabel('')

    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if set_aspect_ratio_equal:
        ax.set_aspect(np.diff(ax.get_xlim()) / np.diff(ax.get_ylim()))    

def plot_linear_relationship_on_heatmap(ax, x, y, idata, cmap, num_bins, hdi_prob=0.96, 
                                        x_scaler=None, y_scaler=None, rescale_ticks=None,
                                        beta_term='beta_0_corrected',
                                        line_color='#674846', lw=0.5,
                                        xticks=None, yticks=None,
                                        norm=None, cbar_kws=None,
                                        hdi_color='#E52B50', hdi_alpha=0.7,
                                        set_aspect_ratio_equal=True):
    """
    Overlay a Bayesian linear regression on a 2D heatmap of x-y density.

    Creates a 2D histogram of x and y, plots it as a heatmap, and overlays the posterior
    mean regression line and HDI band from the posterior samples.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis object to draw the plot on.
    x, y : np.ndarray
        Predictor and outcome variables.
    idata : arviz.InferenceData
        Posterior samples containing 'alpha' and a beta coefficient (specified in `beta_term`).
    cmap : matplotlib Colormap
        Colormap to use for the heatmap.
    num_bins : int
        Number of bins for histogram in each axis.
    hdi_prob : float, default 0.96
        Credible interval width for HDI.
    beta_term : str, default "beta_0_corrected"
        The variable name in `idata.posterior` representing the slope coefficient.
    x_scaler, y_scaler : sklearn.preprocessing.StandardScaler, optional
        Scalers used to transform x and y to standardized space.
    rescale_ticks : bool, optional
        If True, input ticks are inverse-transformed to original scale.
    xticks, yticks : array-like, optional
        Custom tick positions.
    line_color, hdi_color, hdi_alpha : aesthetic controls for regression line and HDI band.
    norm : matplotlib.colors.Normalize, optional
        Normalization method for heatmap color scale.
    cbar_kws : dict, optional
        Keyword arguments passed to the colorbar.
    set_aspect_ratio_equal : bool, default True
        If True, enforces a 1:1 axis scale.
    """

    if x_scaler is not None:
        x = x_scaler.transform(x.reshape(-1, 1)).flatten()
    if y_scaler is not None:
        y = y_scaler.transform(y.reshape(-1, 1)).flatten()
    if rescale_ticks is None and (x_scaler is not None or y_scaler is not None):
        rescale_ticks = True
    elif rescale_ticks is None:
        rescale_ticks = False

    hist, xedges, yedges = np.histogram2d(x, y, bins=num_bins)

    sns.heatmap(hist.T, ax=ax, cmap=cmap, norm=norm, square=True, 
                cbar_kws=cbar_kws)
    ax.invert_yaxis()
    
    # warp x and y to the bin edges
    xbin_width = xedges[1] - xedges[0]
    ybin_width = yedges[1] - yedges[0]
    x_linspace = np.linspace(0, num_bins, 1000)
    x_linspace_warped = xedges[0] + xbin_width * x_linspace

    # reconstruct the y model using the warped x_linspace
    y_model = (idata.posterior["alpha"] + idata.posterior[beta_term] * xr.DataArray(x_linspace_warped) - yedges[0]) / ybin_width

    ax.plot(x_linspace, y_model.mean(dim=("chain", "draw")), 
            color=line_color, lw=lw)
    az.plot_hdi(x_linspace, y_model, hdi_prob,
                ax=ax, color=hdi_color, fill_kwargs={'alpha': hdi_alpha})
    
    if rescale_ticks:
        if xticks is not None:
            xticks_scaled = x_scaler.transform(xticks.reshape(-1, 1)).flatten()
            ax.set_xticks((xticks_scaled - xedges[0]) / xbin_width)
            ax.set_xticklabels(xticks, rotation=0)
        if yticks is not None:
            yticks_scaled = y_scaler.transform(yticks.reshape(-1, 1)).flatten()
            ax.set_yticks((yticks_scaled - yedges[0]) / ybin_width)
            ax.set_yticklabels(yticks, rotation=0)
    else:
        if xticks is not None:
            ax.set_xticks((xticks - xedges[0]) / xbin_width)
            ax.set_xticklabels(xticks, rotation=0)
        if yticks is not None:
            ax.set_yticks((yticks - yedges[0]) / ybin_width)
            ax.set_yticklabels(yticks, rotation=0)

    if set_aspect_ratio_equal:
        ax.set_aspect(np.diff(ax.get_xlim()) / np.diff(ax.get_ylim()))
    
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_visible(True)

def plot_linear_relationship_on_heatmap_lr(ax, x, y, cmap, num_bins,
                                           line_color='#674846',
                                           xticks=None, yticks=None,
                                           norm=None, cbar_kws=None,
                                           set_aspect_ratio_equal=True):
    """
    Overlay a OLS linear regression on a 2D heatmap of x-y density.

    Creates a 2D histogram of x and y, plots it as a heatmap, and overlays a linear regression 
    line fitted using scikit-learn.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis object to draw the plot on.
    x, y : np.ndarray
        Predictor and outcome variables.
    cmap : matplotlib Colormap
        Colormap for the heatmap.
    num_bins : int
        Number of bins along each axis for the histogram.
    line_color : str, default '#674846'
        Color of the regression line.
    xticks, yticks : array-like, optional
        Custom ticks for x and y axes.
    norm : matplotlib.colors.Normalize, optional
        Color normalization for the heatmap.
    cbar_kws : dict, optional
        Additional keyword arguments passed to `sns.heatmap`.
    set_aspect_ratio_equal : bool, default True
        If True, enforces a 1:1 axis scale.
    """
    
    hist, xedges, yedges = np.histogram2d(x, y, bins=num_bins)

    sns.heatmap(hist.T, ax=ax, cmap=cmap, norm=norm, square=True, 
                cbar_kws=cbar_kws)
    ax.invert_yaxis()
    
    # warp x and y to the bin edges
    xbin_width = xedges[1] - xedges[0]
    ybin_width = yedges[1] - yedges[0]
    x_linspace = np.linspace(0, num_bins, 1000)
    x_linspace_warped = xedges[0] + xbin_width * x_linspace

    # fit a linear regression
    reg = LinearRegression().fit(x.reshape(-1, 1), y.reshape(-1, 1))
    y_pred = reg.predict(x_linspace_warped.reshape(-1, 1))

    ax.plot(x_linspace, (y_pred - yedges[0]) / ybin_width, color=line_color)

    if xticks is not None:
        ax.set_xticks((xticks - xedges[0]) / xbin_width)
        ax.set_xticklabels(xticks, rotation=0)
    if yticks is not None:
        ax.set_yticks((yticks - yedges[0]) / ybin_width)
        ax.set_yticklabels(yticks, rotation=0)

    if set_aspect_ratio_equal:
        ax.set_aspect(np.diff(ax.get_xlim()) / np.diff(ax.get_ylim()))
    
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_visible(True)