import os
import numpy as np
from skimage.color import rgba2rgb

def list_all_files(dir):
    """
    List all files (excluding directories) in the specified folder.

    Parameters
    ----------
    dir : str
        Path to the directory to list files from.

    Returns
    -------
    list of str
        List of filenames (not full paths) corresponding to regular files in the directory.

    Notes
    -----
    Adapted from: https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    """

    # Filter and return only regular files (exclude directories)
    return [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

def prepare_image_for_grayscale(img):
    """
    Internal helper to validate and prepare image for grayscale conversion.

    Parameters
    ----------
    img : numpy.ndarray
        Input image array.

    Returns
    -------
    numpy.ndarray
        Prepared image. If grayscale or single-channel, returns original.
        If RGB, ensures proper shape. Raises error if shape is unsupported.
    """
    # Remove singleton dimensions if present (e.g., shape (H, W, 1, 1) → (H, W))
    if img.ndim > 3:
        img = img.squeeze()
    # Convert RGBA to RGB using alpha blending if necessary
    if img.ndim == 3 and img.shape[2] == 4:
        img = rgba2rgb(img)
    # If image is already grayscale or single-channel, return it after clipping
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        return img.clip(0, 255)
    # raise error if image is not 2D grayscale or 3D RGB
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Input must be a 2D grayscale or 3D RGB image with 3 channels.")
    return img

def raw_avg(img):
    """
    Compute the average of the three color channels in an RGB image.

    This function converts a color image to a grayscale-like representation
    by averaging the RGB values across the last axis (channel dimension).
    The result is clipped to the range [0, 255].

    If the input image is already grayscale or has a single channel,
    it will return the same image with values clipped to [0, 255].

    Parameters
    ----------
    img : numpy.ndarray
        An image represented as a NumPy array of shape (H, W, 3) for RGB input,
        or (H, W) / (H, W, 1) for grayscale.

    Returns
    -------
    numpy.ndarray
        A 2D array of shape (H, W) containing the channel-averaged grayscale values, clipped to [0, 255].
    """

    img = prepare_image_for_grayscale(img)
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        return img
    # Compute mean across RGB channels and clip
    return np.mean(img, axis=2).clip(0, 255)

def bt601(img):
    """
    Compute the brightness component of an RGB image using ITU-R BT.601 weights.

    This function approximates grayscale intensity based on human visual sensitivity
    by applying the BT.601 standard weights:
    - Red:   0.299
    - Green: 0.587
    - Blue:  0.114

    These weights are defined by the ITU-R BT.601 recommendation.

    If the input image is already grayscale or has a single channel,
    it will return the same image with values clipped to [0, 255].

    Parameters
    ----------
    img : numpy.ndarray
        An image represented as a NumPy array of shape (H, W, 3) for RGB input,
        or (H, W) / (H, W, 1) for grayscale.

    Returns
    -------
    numpy.ndarray
        A 2D array of shape (H, W) containing the per-pixel luminance values based on BT.601,
        clipped to the range [0, 255].
    """

    img = prepare_image_for_grayscale(img)
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        return img
    # Extract R, G, B channels
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    # Apply BT.601 luminance weights and clip
    return (r * 0.299 + g * 0.587 + b * 0.114).clip(0, 255)

def bt709(img):
    """
    Compute the brightness component of an RGB image using ITU-R BT.709 weights.

    This function approximates grayscale intensity based on human visual sensitivity
    by applying the BT.709 standard weights:
    - Red:   0.2126
    - Green: 0.7152
    - Blue:  0.0722

    These weights are defined by the ITU-R BT.709 recommendation.

    If the input image is already grayscale or has a single channel,
    it will return the same image with values clipped to [0, 255].

    Parameters
    ----------
    img : numpy.ndarray
        An image represented as a NumPy array of shape (H, W, 3) for RGB input,
        or (H, W) / (H, W, 1) for grayscale.

    Returns
    -------
    numpy.ndarray
        A 2D array of shape (H, W) containing the per-pixel luminance values based on BT.709,
        clipped to the range [0, 255].
    """

    img = prepare_image_for_grayscale(img)
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        return img
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return (r * 0.2126 + g * 0.7152 + b * 0.0722).clip(0, 255)