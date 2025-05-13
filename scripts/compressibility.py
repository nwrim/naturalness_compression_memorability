import numpy as np
from scipy.stats import linregress
from scipy.fftpack import dctn
import scipy.ndimage as ndi
from skimage.feature import canny
from image_processing import raw_avg

def calculate_beta(values, X):
    """
    Calculate the beta coefficient (slope) of a linear regression model with a single independent variable.    using scipy.stats.linregress.

    The dependent variable is normalized before fitting, so the beta is computed on the proportion of values.

    Parameters
    ----------
    values : numpy.ndarray
        The dependent variable (will be normalized before regression).
    X : numpy.ndarray
        The independent variable.

    Returns
    -------
    beta : float
      The slope (beta coefficient) from the linear regression.
    """

    # Normalize the dependent variable
    values = values / values.sum()
    
    # Perform linear regression using scipy
    slope, _, _, _, _ = linregress(X, values)

    return slope

def create_binary_matrices_by_frequency():
    """
    Create binary 8x8 masks selecting each antidiagonal.

    Each matrix isolates one antidiagonal (i.e., all coefficients with the same u+v),
    where u and v are horizontal and vertical index. This yields 15 total masks.

    Returns
    -------    
    binary_matrices: dict
       dictionary mapping frequency index (0–14) to a binary 8x8 mask.

    Examples
    --------
    >>> binary_frequency_matrices = create_binary_matrices_by_frequency()
    >>> binary_frequency_matrices[0]
    array([[1., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.]])
    >>> binary_frequency_matrices[4]
    array([[0., 0., 0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 1., 0., 0., 0., 0.],
           [0., 0., 1., 0., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.]])    
    """

    binary_frequency_matrices = {}
    for k in range(15):
        # create 8x8 matrix with 1s in the diagonal and 0 elsewhere
        mat = np.eye(8, k=7-k)
        # flip the matrix horizontally
        mat = np.flip(mat, axis=1)
        binary_frequency_matrices[k] = mat
    return binary_frequency_matrices

def calculate_dct_by_tile(img):
    """
    Split the image into 8x8 tiles and apply 2D DCT to each tile.

    Parameters
    ----------
    img: numpy.ndarray
      Grayscale image.

    Returns
    -------
    dct_by_tile: numpy.ndarray
      Array of same shape as input containing DCT coefficients for each 8x8 tile.
    """

    imsize = img.shape
    # initialize empty array to store the DCT result
    dct_by_tile = np.zeros(imsize)
    # split into 8x8 tiles (following the JPEG standard) and apply 2D DCT to each tile
    for i in np.r_[:imsize[0]:8]:
        for j in np.r_[:imsize[1]:8]:
            dct_by_tile[i:(i+8),j:(j+8)] = dctn(img[i:(i+8),j:(j+8)], norm='ortho')
    return dct_by_tile

def calculate_sum_abs_coeff_by_freq(dct_by_tile):
    """
    Compute the sum of absolute DCT coefficients across all image tiles, grouped by frequency index.

    Parameters
    ----------
    dct_by_tile: numpy.ndarray
      the DCT result for each tile patched back together (DCT coefficients from `calculate_dct_by_tile`.)

    Returns
    -------
    sum_abs_coeff_by_freq: numpy.ndarray
      1D array of length 15 with total absolute DCT coefficient values for each frequency index.
    """

    binary_frequency_matrices = create_binary_matrices_by_frequency()

    imsize = dct_by_tile.shape
    # create an array to store the result
    # There are 15 possible frequencies (sum of horizontal and vertical frequency) of the cosine function for an 8x8 tile. 
    # Each column corresponds to one frequency
    sum_abs_coeff_by_freq = np.zeros(15)
    # Across all 8x8 tiles, calculate the sum of the absolute DCT coefficients for each frequency.
    for i in np.r_[:imsize[0]:8]:
        for j in np.r_[:imsize[1]:8]:
            sum_abs_coeffs = []
            for k in range(15):
                # get the dct result for the 8x8 tile
                single_tile_dct = dct_by_tile[i:(i+8),j:(j+8)].copy()
                # get the binary matrix for the frequency
                binary_frequency_matrix = binary_frequency_matrices[k][:single_tile_dct.shape[0], :single_tile_dct.shape[1]]
                # apply the binary matrix to the DCT result
                single_tile_dct_subset = single_tile_dct * binary_frequency_matrix
                # calculate the sum of the absolute values of the DCT coefficients
                sum_abs_coeff = np.sum(np.sum(np.abs(single_tile_dct_subset)))
                sum_abs_coeffs.append(sum_abs_coeff)
            # add the sum of the absolute values for each frequency
            sum_abs_coeff_by_freq += np.array(sum_abs_coeffs)
    return sum_abs_coeff_by_freq

def calculate_dct_sum_abs_coeff_by_freq(img, func):
    """
    Convert an image to grayscale, and calculate the sum of the absolute DCT coefficients across all tiles for each frequency of the cosine function.
    
    Wrapper around `calculate_dct_by_tile` and `calculate_sum_abs_coeff_by_freq`.

    Parameters
    ----------
    img : numpy.ndarray
      the input image
    func: callable
      Function for converting RGB to grayscale.
    
    Returns
    -------
    sum_abs_coeff_by_freq: numpy.ndarray
      the sum of the absolute DCT coefficients across all tiles for each frequency of the cosine function based on the grayscale image (length 15).
    """

    # read the image and convert it to grayscale
    grayscale = func(img)
    
    # split into 8x8 tiles (following the JPEG standard) and apply 2D DCT to each tile
    dct_by_tile = calculate_dct_by_tile(grayscale)

    # calculate the sum of the absolute values of coefficients across all tiles for each frequency of the cosine function
    sum_abs_coeff_by_freq = calculate_sum_abs_coeff_by_freq(dct_by_tile)
    return sum_abs_coeff_by_freq

def jpeg_based_compressibility(img, func=raw_avg, n_freq=7):
    """
    Calculate the JPEG-based compressibility of an image.

    This is done by getting the beta coefficient (slope) of a linear regression model with independent variable as the 
    underlying frequency of the cosine function, and the dependent variable as the proportion of the sum of the absolute
    values of coefficients across all tiles for that underlying frequency.

    Also see `calculate_dct_sum_abs_coeff_by_freq` and `calculate_beta`.

    Parameters
    ----------
    img : numpy.ndarray
      the input image
    func: callable, optional
      Function for converting RGB to grayscale (default: `raw_avg`).
    n_freq : int, optional
      Number of frequency levels to use (starting after DC). Default is 7.
    
    Returns
    -------
    beta: float
      the JPEG-based compressibility of the input image.
    """

    # calculate the sum of the absolute values of coefficients across all tiles for each frequency of the cosine function based on the grayscale image
    sum_abs_coeff_by_freq = calculate_dct_sum_abs_coeff_by_freq(img, func)
    
    # skip the first value (frequency 0) as this is the DC component
    sum_abs_coeff_by_freq = sum_abs_coeff_by_freq[1:1+n_freq]

    # calculate the beta coefficient of the linear regression
    beta = calculate_beta(sum_abs_coeff_by_freq, np.arange(-6, -6 + n_freq, 1))
    return beta

def calculate_gradient_magnitude(img):
    """
    Calculate the estimated gradient magnitude at each pixel of a smoothed grayscale image.
    Note that the image is smoothed by applying a Gaussian filter with sigma=3 before calculating the gradient.
    
    Parameters
    ----------
    img: numpy.ndarray
      Grayscale image.
    
    Returns
    -------
    magnitude: numpy.ndarray
      the gradient magnitude at each pixel of the image
    """

    # smooth the image by applying Gaussian filter to the grayscale image
    smoothed = ndi.gaussian_filter(img, sigma=3, mode='reflect')
    # calculate the horizontal gradient
    sobel_h = ndi.sobel(smoothed, 0, mode='reflect')
    # calculate the vertical gradient
    sobel_v = ndi.sobel(smoothed, 1, mode='reflect')
    # calculate the estimated gradient magnitude
    magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
    return magnitude

def calculate_edge_count_by_magnitude_bin(img, func, bins=None):
    """
    Count number of edge pixels within gradient magnitude bins.

    Edges are detected by applying Canny edge detection, with both threshold value set to 0.
    
    Note that the image is smoothed by applying a Gaussian filter with sigma=3 before calculating the gradient.
    
    Parameters
    ----------
    img : numpy.ndarray
      the input image
    func: callable
      Function for converting RGB to grayscale.
    bins: numpy.ndarray
      the bins for the histogram of the magnitude
      If None, it defaults to [0, 10, ..., 100, np.max(magnitude)+1] when np.max(magnitude) > 100, 
      or [0, 10, ..., 100, 500] otherwise (500 is an arbitrarly large number).
    
    Returns
    -------
    bins : numpy.ndarray
      Bin edges used for histogram.
    edge_count_by_magnitude_bin: numpy.ndarray
      the number of edges (in pixels) in each magnitude bin
    """

    # read the image and convert it to grayscale
    grayscale = func(img)
  
    # calculate the gradient magnitude
    magnitude = calculate_gradient_magnitude(grayscale)

    # define the bins if not provided
    if bins is None:
        if np.max(magnitude) > 100:
            bins = np.array(list(range(0, 101, 10)) + [np.max(magnitude) + 1])
        else:
            bins = np.array(list(range(0, 101, 10)) + [500]) # 500 is some arbitrary large number
    
    # apply Canny edge detection without thresholding
    # This is to get the edge pixel after non-maximum suppression
    canny_output = canny(grayscale, sigma=3,
                         low_threshold=0,
                         high_threshold=0, mode='reflect')
    
    # subset the magnitude based on the Canny output
    # note that the magnitude is flattened (i.e., 1D array)
    magnitude = magnitude[canny_output]
    
    # calculate the histogram of the magnitude
    edge_count_by_magnitude_bin, _ = np.histogram(magnitude, bins=bins)

    return bins, edge_count_by_magnitude_bin

def canny_based_compressibility(img, func=raw_avg, bins=None, reverse=True):
    """
    Calculate the Canny-based compressibility of an image.

    This is done by getting the beta coefficient (slope) of a linear regression model with independent variable as the 
    left endpoints of the magnitude bins, and the dependent variable as the proportion of edges in each magnitude bin.

    Also see the `calculate_edge_count_by_magnitude_bin` and `calculate_beta`.

    Parameters
    ----------
    img : numpy.ndarray
      the input image
    func: callable, optional
      the function to convert the image to grayscale (default: `raw_avg`).
    bins: array-like, optional
      the bins for the histogram of the magnitude. See `calculate_edge_count_by_magnitude_bin` for default behavior.
    reverse : bool, optional
      If True, reverses X-axis to align beta direction with JPEG-based compressibility (deafult: True).

    Returns
    -------
    beta: float
      the Canny-based compressibility of the input image.
    """

    # calculate the proportion of edges in each magnitude bin
    bins, edge_count_by_magnitude_bin = calculate_edge_count_by_magnitude_bin(img, func, bins)
    
    # calculate the beta coefficient of the linear regression
    if reverse:
        # if reverse is True, the bins (X) are reversed
        # this is to align with the direction of the beta with JPEG-based compressibility, such that a higher beta indicates greater compressibility.
        beta = calculate_beta(edge_count_by_magnitude_bin, np.max(bins[:-1]) - bins[:-1])
    else:
        beta = calculate_beta(edge_count_by_magnitude_bin, bins[:-1])
    return beta
