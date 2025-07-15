"""
AFM Image Flattening and Background Leveling Tools
===================================================

This module provides background leveling and flattening routines for Atomic
Force Microscopy (AFM) images and video stacks. It supports a range of
polynomial- and median-based leveling strategies, enabling the correction
of background planes, row/column-wise drift, and systematic noise from AFM
topographic data.

The functions here were ported from the original MATLAB NanoLocz Library, and
maintain compatibility with high-speed AFM, localization AFM, and static
imaging data.

Supported Leveling Methods
--------------------------
- 'plane'       : Polynomial line + plane subtraction in X and Y.
- 'line'        : Row-wise and column-wise polynomial leveling.
- 'med_line'    : Row-wise median line flattening.
- 'med_line_y'  : Column-wise median flattening.
- 'smed_line'   : Smoothed median line subtraction.
- 'mean_plane'  : Global mean subtraction.
- 'log_y'       : Logarithmic curve subtraction along the Y-axis.

Typical usage involves calling the `level()` function with an image (2D) or
image stack (3D) and specifying the desired method and polynomial orders.

Examples
--------
>>> from pnanolocz_lib.filters.level import level
>>> leveled = level(img, polyx=2, polyy=2, method="plane")

>>> from pnanolocz_lib.filters.level import level_plane
>>> flattened = level_plane(img, mask=None, polyx=2, polyy=2)

Author
------
George Heath, University of Leeds (2025)
D. E. Rollins, University of Leeds (2025)

This module is part of the pNanoLocz-Lib Python library for AFM analysis.
"""

import numpy as np
from typing import Optional, Literal
from scipy.optimize import curve_fit

# Constants
SMOOTHING_WINDOW = 10
LOG_FIT_BOUNDS = ([0.1, 0.01, 0.1], [1000, 20, 100])


def level_plane(
    img: np.ndarray, mask: Optional[np.ndarray], polyx: int, polyy: int
) -> np.ndarray:
    """
    Plane leveling fitting by subtracting polynomial curves in X and Y.

    Parameters
    ----------
    img : ndarray
        2D AFM image to be leveled.
    mask : ndarray or None
        Binary mask or weighting matrix, same shape as img.
        If None, all pixels are considered valid.
    polyx : int
        Polynomial order for X-direction leveling.
    polyy : int
        Polynomial order for Y-direction leveling.

    Returns
    -------
    leveled : ndarray
        The leveled image.
    """
    if mask is None:
        mask = ~np.isnan(img)

    leveled = img.copy()

    if np.sum(mask) <= 5:
        return leveled  # Not enough valid points

    # Fit polynomial in X direction (rows)
    xp = np.nanmean(leveled * mask, axis=0)
    valid_x = ~np.isnan(xp)
    xl = np.arange(len(xp))[valid_x]
    xf = xp[valid_x]
    if polyx > 0 and len(xl) > polyx:
        p_x = np.polyfit(xl, xf, polyx)
        correction_x = np.polyval(p_x, np.arange(img.shape[1]))
        leveled = leveled - correction_x[None, :]
    else:
        return leveled

    # Fit polynomial in Y direction (columns)
    yp = np.nanmean(leveled * mask, axis=1)
    valid_y = ~np.isnan(yp)
    yl = np.arange(len(yp))[valid_y]
    yf = yp[valid_y]
    if polyy > 0 and len(yl) > polyy:
        p_y = np.polyfit(yl, yf, polyy)
        correction_y = np.polyval(p_y, np.arange(img.shape[0]))
        leveled = leveled - correction_y[:, None]

    return leveled


def level_line(
    img: np.ndarray, mask: Optional[np.ndarray], polyx: int, polyy: int
) -> np.ndarray:
    """
    Polynomial line leveling, correcting each row and column separately.

    Parameters
    ----------
    img : ndarray
        2D AFM image.
    mask : ndarray or None
        Binary mask or weighting matrix.
    polyx : int
        Polynomial order for X direction.
    polyy : int
        Polynomial order for Y direction.

    Returns
    -------
    leveled : ndarray
        The leveled image.
    """
    if mask is None:
        mask = ~np.isnan(img)

    leveled = img.copy()

    if polyx > 0:
        y2 = np.zeros_like(img)
        failed_rows = []
        for i in range(img.shape[0]):
            pos = mask[i, :] > 0
            if np.sum(pos) > polyx + 8:
                x1 = np.arange(img.shape[1])[pos]
                y1 = img[i, pos]
                p = np.polyfit(x1, y1, polyx)
                y2[i] = np.polyval(p, np.arange(img.shape[1]))
                leveled[i, :] = img[i, :] - y2[i]
            else:
                failed_rows.append(i)
        for i in failed_rows:
            try:
                leveled[i, :] = img[i, :] - np.median(y2, axis=0)
            except Exception:
                pass

    if polyy > 0:
        for i in range(img.shape[1]):
            col_mask = mask[:, i]
            yp = img[:, i] * col_mask
            valid = ~np.isnan(yp)
            yl = np.arange(img.shape[0])[valid]
            yf = yp[valid]
            if len(yl) >= polyy:
                p = np.polyfit(yl, yf, polyy)
                fy = np.polyval(p, np.arange(img.shape[0]))
                leveled[:, i] -= fy

    return leveled


def level_med_line(
    img: np.ndarray, mask: Optional[np.ndarray], polyx: int, polyy: int
) -> np.ndarray:
    """
    Row-wise median line leveling.

    Parameters
    ----------
    img : ndarray
        2D AFM image.
    mask : ndarray or None
        Binary mask or weighting matrix.
    polyx : int
        Used as a scale factor if > 0, otherwise 1.
    polyy : int
        Unused.

    Returns
    -------
    leveled : ndarray
        The leveled image.
    """
    if mask is None:
        mask = ~np.isnan(img)

    leveled = img.copy()
    bg = np.nanmedian(img[mask > 0])

    for i in range(img.shape[0]):
        pos = ~np.isnan(img[i, :])
        if np.sum(pos) > 10:
            y1 = np.median(img[i, pos])
            scale = polyx if polyx > 0 else 1
            leveled[i, :] = img[i, :] - scale * y1 + bg

    return leveled


def level_med_line_y(
    img: np.ndarray, mask: Optional[np.ndarray], polyx: int, polyy: int
) -> np.ndarray:
    """
    Column-wise median line leveling.

    Parameters
    ----------
    img : ndarray
        2D AFM image.
    mask : ndarray or None
        Binary mask or weighting matrix.
    polyx : int
        Unused.
    polyy : int
        Unused.

    Returns
    -------
    leveled : ndarray
        The leveled image.
    """
    if mask is None:
        mask = ~np.isnan(img)

    leveled = img.copy()
    bg = np.nanmedian(img[mask > 0])

    for i in range(img.shape[1]):
        pos = ~np.isnan(img[:, i])
        if np.sum(pos) > 10:
            y1 = np.median(img[pos, i])
            leveled[:, i] = img[:, i] - y1 + bg

    return leveled


def level_smed_line(
    img: np.ndarray, mask: Optional[np.ndarray], polyx: int, polyy: int
) -> np.ndarray:
    """
    Smoothed median line subtraction.

    Parameters
    ----------
    img : ndarray
        2D AFM image.
    mask : ndarray or None
        Binary mask or weighting matrix.
    polyx : int
        Unused.
    polyy : int
        Unused.

    Returns
    -------
    leveled : ndarray
        The leveled image.
    """
    if mask is None:
        mask = ~np.isnan(img)

    leveled = img.copy()
    background = np.nanmedian(img[mask > 0])
    y1 = np.zeros(img.shape[0])

    for i in range(img.shape[0]):
        pos = ~np.isnan(img[i, :])
        if np.sum(pos) > 10:
            y1[i] = np.median(img[i, pos]) + background
        else:
            y1[i] = background

    background2 = np.convolve(
        y1, np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW, mode="same"
    )

    leveled = img - (y1[:, None] - background2[:, None])

    return leveled


def level_mean_plane(
    img: np.ndarray, mask: Optional[np.ndarray], polyx: int, polyy: int
) -> np.ndarray:
    """
    Mean plane subtraction.

    Parameters
    ----------
    img : ndarray
        2D AFM image.
    mask : ndarray or None
        Binary mask or weighting matrix.
    polyx : int
        Unused.
    polyy : int
        Unused.

    Returns
    -------
    leveled : ndarray
        The leveled image.
    """
    if mask is None:
        mask = ~np.isnan(img)

    mean_val = np.nanmean(img[mask > 0])
    return img - mean_val


def level_log_y(
    img: np.ndarray, mask: Optional[np.ndarray], polyx: int, polyy: int
) -> np.ndarray:
    """
    Logarithmic curve subtraction along the Y-axis.

    Parameters
    ----------
    img : ndarray
        2D AFM image.
    mask : ndarray or None
        Binary mask or weighting matrix (unused here).
    polyx : int
        Unused.
    polyy : int
        Scale factor for the X-axis in log fit.

    Returns
    -------
    leveled : ndarray
        The leveled image.
    """
    y = np.mean(img, axis=1)
    correction = _log_y_correction(y, polyy)
    return img - correction[::-1][:, None]


def _log_y_correction(y: np.ndarray, scale: float) -> np.ndarray:
    """
    Fit and return a logarithmic correction curve.

    Parameters
    ----------
    y : ndarray
        1D array of mean pixel values along Y-axis.
    scale : float
        Scale factor for X-axis in log fitting.

    Returns
    -------
    correction : ndarray
        The log correction curve.
    """
    y = y - np.min(y)
    x = np.linspace(0, 10, len(y)) / scale
    pos = x < 5
    x_fit = x[pos]
    y_fit = y[pos]

    def _log_model(x, a, b, c):
        return a * np.log(c * x + b)

    try:
        popt, _ = curve_fit(
            _log_model, x_fit, y_fit, p0=[5, 1, 2], bounds=LOG_FIT_BOUNDS
        )
        return _log_model(x, *popt)
    except Exception:
        return np.zeros_like(y)


def level(
    img: np.ndarray,
    polyx: int,
    polyy: int,
    method: Literal[
        "plane",
        "line",
        "med_line",
        "med_line_y",
        "smed_line",
        "mean_plane",
        "log_y",
    ],
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Level or flatten AFM images or stacks using various polynomial and
    median-based methods.

    Parameters
    ----------
    img : ndarray
        Input AFM image or image stack. Shape can be (H, W) or (N, H, W),
        where N is the number of frames in a stack.
    polyx : int
        Polynomial order for X-direction leveling. Set to 0 to skip.
    polyy : int
        Polynomial order for Y-direction leveling. Set to 0 to skip.
    method : {'plane', 'line', 'med_line', 'med_line_y', 'smed_line',
            'mean_plane', 'log_y'}
        Leveling or flattening method to apply.
    mask : ndarray, optional
        Binary mask or weighting matrix with same shape as `img`.
        If None, all pixels are considered valid.

    Returns
    -------
    leveled : ndarray
        The leveled image or image stack, same shape as input.

    Raises
    ------
    ValueError
        If the provided method is not supported or mask shape does not match
          img shape.

    Notes
    -----
    This function dispatches to specialized leveling methods that replicate the
    MATLAB NanoLocz library functionality. For 3D stacks, each frame is
    processed independently.

    The interpretation of `polyx` and `polyy` depends on the method:
    - For 'plane' and 'line', they are polynomial fit orders.
    - For 'log_y', `polyy` is a scaling factor for the log fit.
    - For median-based methods, `polyx` may be used as a scaling factor.
    """
    # Ensure input is ndarray
    img = np.asarray(img)
    is_stack = img.ndim == 3
    frames = img if is_stack else img[None, ...]

    if mask is None:
        mask = ~np.isnan(frames)
    else:
        mask = np.asarray(mask)
        if mask.shape != frames.shape:
            raise ValueError("mask must have the same shape as img")

    leveled_frames = []

    for k in range(frames.shape[0]):
        f = frames[k]
        m = mask[k]

        if method == "plane":
            leveled = level_plane(f, m, polyx, polyy)
        elif method == "line":
            leveled = level_line(f, m, polyx, polyy)
        elif method == "med_line":
            leveled = level_med_line(f, m, polyx, polyy)
        elif method == "med_line_y":
            leveled = level_med_line_y(f, m, polyx, polyy)
        elif method == "smed_line":
            leveled = level_smed_line(f, m, polyx, polyy)
        elif method == "mean_plane":
            leveled = level_mean_plane(f, m, polyx, polyy)
        elif method == "log_y":
            leveled = level_log_y(f, m, polyx, polyy)
        else:
            raise ValueError(f"Unknown leveling method: {method}")

        leveled_frames.append(leveled)

    leveled_array = np.stack(leveled_frames)
    return leveled_array if is_stack else leveled_array[0]
