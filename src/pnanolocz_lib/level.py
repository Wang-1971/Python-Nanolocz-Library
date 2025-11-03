"""
AFM image flattening and background leveling tools.

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

Typical usage involves calling the `apply_level()` function with an image (2D)
or image stack (3D) and specifying the desired method and polynomial orders.

The `get_background()` function generates the array of fitted lines without
subtracting this from the image (i.e to visualise the background).

Examples
--------
>>> from pnanolocz_lib.filters.level import level
>>> leveled = level(img, polyx=2, polyy=2, method="plane")

>>> from pnanolocz_lib.filters.level import level_plane
>>> flattened = level_plane(img, mask=None, polyx=2, polyy=2)

Authors
-------
George Heath, University of Leeds (2025)
Maya Tekchandani, University of Leeds (2025)
Daniel. E. Rollins, University of Leeds (2025)

This module is part of the pNanoLocz-Lib Python library for AFM analysis.
"""

import warnings
from typing import Literal, Optional

import numpy as np
from numpy.polynomial.polyutils import RankWarning  # type: ignore
from scipy.optimize import curve_fit

# Constants
SMOOTHING_WINDOW = 10
LOG_FIT_BOUNDS = ([0.1, 0.01, 0.1], [1000, 20, 100])


def level_plane(
    img: np.ndarray, mask: Optional[np.ndarray], polyx: int, polyy: int
) -> np.ndarray:
    """
    Plane leveling fitting by subtracting polynomial curves in X and Y.

    This attempts to replicate MATLAB's centered polynomial approach.

    Parameters
    ----------
    img : ndarray
        2D AFM image to be leveled.
    mask : ndarray or None
        Boolean mask of same shape as img (True = valid). If None,
        all pixels are valid.
    polyx : int
        Polynomial order for X-direction leveling.
    polyy : int
        Polynomial order for Y-direction leveling.

    Returns
    -------
    leveled_img : ndarray
        The leveled image.
    """
    # If no mask provided, treat all pixels as valid
    if mask is None:
        mask = ~np.isnan(img)
    # Must have at least 6 valid pixels to fit anything
    if np.sum(mask) <= 5:
        return np.asarray(img.copy())

    # ————— X DIRECTION —————
    # Compute column-wise mean over valid pixels
    masked_for_columns = np.where(mask, img, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        column_means = np.nanmean(masked_for_columns, axis=0)

    valid_columns = ~np.isnan(column_means)
    column_indices = np.flatnonzero(valid_columns)
    if column_indices.size <= polyx:
        # Not enough points to fit X polynomial
        return np.asarray(img.copy())

    # Center & scale column indices
    # replicate MATLAB centering
    # mu = [mean(column_indices), std(column_indices)]
    col_centroid = column_indices.mean()
    col_scale = column_indices.std(ddof=1)
    standardized_columns = (column_indices - col_centroid) / col_scale

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RankWarning)
        # polyfit(..., polyx) with centering ⇒ same as MATLAB’s p, ~, mu
        x_coeffs = np.polyfit(
            standardized_columns,
            column_means[valid_columns],
            polyx,
        )

    # Evaluate polynomial at every column
    all_cols = np.arange(img.shape[1])
    standardized_all_cols = (all_cols - col_centroid) / col_scale
    x_plane = np.polyval(x_coeffs, standardized_all_cols)[None, :]

    # Subtract X-plane
    leveled_img = img - x_plane

    # ————— Y DIRECTION —————
    # Compute row-wise mean over valid pixels after X subtraction
    masked_for_rows = np.where(mask, leveled_img, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        row_means = np.nanmean(masked_for_rows, axis=1)

    valid_rows = ~np.isnan(row_means)
    row_indices = np.flatnonzero(valid_rows)
    if row_indices.size <= polyy:
        # Not enough points to fit Y polynomial
        return leveled_img

    # Center & scale row indices
    # replicate MATLAB centering
    # mu = [mean(column_indices), std(column_indices)]
    row_centroid = row_indices.mean()
    row_scale = row_indices.std(ddof=1)
    standardized_rows = (row_indices - row_centroid) / row_scale

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RankWarning)
        y_coeffs = np.polyfit(
            standardized_rows,
            row_means[valid_rows],
            polyy,
        )

    # Evaluate polynomial at every row
    all_rows = np.arange(img.shape[0])
    standardized_all_rows = (all_rows - row_centroid) / row_scale
    y_plane = np.polyval(y_coeffs, standardized_all_rows)[:, None]

    # Subtract Y-plane
    return leveled_img - y_plane


def level_line(
    img: np.ndarray, mask: Optional[np.ndarray], polyx: int, polyy: int
) -> np.ndarray:
    """
    Polynomial line leveling, correcting each row and column separately.

    This uses centered/scaled index fitting.

    Parameters
    ----------
    img : ndarray
        2D AFM image.
    mask : ndarray or None
        Boolean mask of same shape as img (True = valid).
    polyx : int
        Polynomial order for per-row fitting.
    polyy : int
        Polynomial order for per-column fitting.

    Returns
    -------
    leveled_img : ndarray
        The leveled image.
    """
    if mask is None:
        mask = ~np.isnan(img)
    leveled_img = img.copy()

    # ————— Per-row polynomial leveling —————
    if polyx > 0:
        row_fits = np.zeros_like(img)
        fallback_rows: list[int] = []

        for row_idx in range(img.shape[0]):
            valid_cols = mask[row_idx, :]
            if valid_cols.sum() > polyx + 8:
                col_indices = np.flatnonzero(valid_cols)
                row_values = img[row_idx, col_indices]

                # Center & scale col_indices
                centroid_col = col_indices.mean()
                scale_col = col_indices.std(ddof=1)
                standardized_cols = (col_indices - centroid_col) / scale_col

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RankWarning)
                    row_coeffs = np.polyfit(
                        standardized_cols, row_values, polyx
                    )  # noqa

                all_cols = np.arange(img.shape[1])
                standardized_all_cols = (all_cols - centroid_col) / scale_col
                fitted_row = np.polyval(row_coeffs, standardized_all_cols)

                row_fits[row_idx, :] = fitted_row
                leveled_img[row_idx, :] = img[row_idx, :] - fitted_row
            else:
                fallback_rows.append(row_idx)

        # For rows without enough points, subtract the median
        # of all fitted rows
        median_row_fit = np.median(row_fits, axis=0)
        for row_idx in fallback_rows:
            leveled_img[row_idx, :] = img[row_idx, :] - median_row_fit

    # ————— Per-column polynomial leveling —————
    if polyy > 0:
        for col_idx in range(img.shape[1]):
            valid_rows = mask[:, col_idx]
            if valid_rows.sum() >= polyy:
                row_indices = np.flatnonzero(valid_rows)
                col_values = img[row_indices, col_idx]

                # Center & scale row_indices
                centroid_row = row_indices.mean()
                scale_row = row_indices.std(ddof=1)
                standardized_rows = (row_indices - centroid_row) / scale_row

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RankWarning)
                    col_coeffs = np.polyfit(
                        standardized_rows, col_values, polyy
                    )  # noqa

                all_rows = np.arange(img.shape[0])
                standardized_all_rows = (all_rows - centroid_row) / scale_row
                fitted_col = np.polyval(col_coeffs, standardized_all_rows)

                leveled_img[:, col_idx] -= fitted_col

    return np.asarray(leveled_img)


def level_med_line(
    img: np.ndarray,
    mask: Optional[np.ndarray],
    polyx: int,
    polyy: int,  # unused (MATLAB semantics)
) -> np.ndarray:
    """
    Row-wise median line leveling for AFM images.

    Parameters
    ----------
    img : ndarray
        2D AFM image.
    mask : ndarray or None
        Binary mask (1 = include pixel). If None, all non-NaN pixels are used.
    polyx : int
        Scaling factor for median subtraction.
        If polyx == 0, scale factor is 1 (MATLAB behaviour).
    polyy : int
        Unused. Kept for interface compatibility.

    Returns
    -------
    leveled : ndarray
        The leveled image.
    """
    if mask is None:
        mask = ~np.isnan(img)

    # Masked image values
    masked_img = np.where(mask > 0, img, np.nan)

    # Global background (median of masked values)
    bg = np.nanmedian(masked_img)

    leveled = img.copy()

    # Effective scale: polyx or 1 if polyx == 0
    scale = polyx if polyx > 0 else 1

    for i in range(img.shape[0]):
        row = masked_img[i, :]
        pos = ~np.isnan(row)

        if np.sum(pos) > 1:
            row_median = np.nanmedian(row)
            leveled[i, :] = img[i, :] - scale * row_median + bg

    return np.asarray(leveled)


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

    return np.asarray(leveled)


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

    return np.asarray(leveled)


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
    return np.asarray(img - mean_val)


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
    return np.asarray(img - correction[::-1][:, None])


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

    def _log_model(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * np.log(c * x + b)

    try:
        popt, _ = curve_fit(
            _log_model, x_fit, y_fit, p0=[5, 1, 2], bounds=LOG_FIT_BOUNDS
        )
        return _log_model(x, *popt)
    except Exception:
        return np.zeros_like(y)


def apply_level(
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
    Apply a function to level or flatten AFM images or stacks.

    This applies the various polynomial and median-based methods found in this module.

    Parameters
    ----------
    img : ndarray
        Input AFM image or image stack. Shape can be (H, W) or (N, H, W),
        where N is the number of frames in a stack.
    polyx : int
        Polynomial order for X-direction leveling. Set to 0 to skip.
    polyy : int
        Polynomial order for Y-direction leveling. Set to 0 to skip.
    method : str
        Leveling method to apply.
    mask : ndarray, optional
        Binary mask or weighting matrix with same shape as `img`.
        If None, all pixels are considered valid.

    Returns
    -------
    leveled : ndarray
        The leveled image or image stack, same shape as input.
    """
    img = np.asarray(img)
    is_stack = img.ndim == 3

    # Convert to (N, H, W) for consistent processing
    if is_stack:
        frames = img
    else:
        frames = img[np.newaxis, ...]  # shape (1, H, W)

    if mask is not None:
        mask = np.asarray(mask)
        if mask.ndim == 2:
            mask = mask[np.newaxis, ...]  # shape (1, H, W)
        elif mask.shape != frames.shape:
            raise ValueError("mask must have the same shape as img")

    leveled_frames = []

    for idx in range(frames.shape[0]):
        frame = frames[idx]
        frame_mask = mask[idx] if mask is not None else None

        if method == "plane":
            leveled = level_plane(frame, frame_mask, polyx, polyy)
        elif method == "line":
            leveled = level_line(frame, frame_mask, polyx, polyy)
        elif method == "med_line":
            leveled = level_med_line(frame, frame_mask, polyx, polyy)
        elif method == "med_line_y":
            leveled = level_med_line_y(frame, frame_mask, polyx, polyy)
        elif method == "smed_line":
            leveled = level_smed_line(frame, frame_mask, polyx, polyy)
        elif method == "mean_plane":
            leveled = level_mean_plane(frame, frame_mask, polyx, polyy)
        elif method == "log_y":
            leveled = level_log_y(frame, frame_mask, polyx, polyy)
        else:
            raise ValueError(f"Unknown leveling method: {method}")

        leveled_frames.append(leveled)

    result = np.stack(leveled_frames, axis=0)

    return result if is_stack else result[0]


def get_background(
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
    Compute a background surface/lines that would be subtracted by `apply_level(...)`.

    This does not apply the operation to the data.

    Parameters
    ----------
    img : (N, H, W) array
        Input AFM image or image stack. Shape can be (H, W) or (N, H, W),
        where N is the number of frames in a stack.
    polyx : int
        Polynomial order for X-direction leveling. Set to 0 to skip.
    polyy : int
        Polynomial order for Y-direction leveling. Set to 0 to skip.
    method : str
        One of 'plane', 'line', 'med_line', etc.
    mask : (H, W) bool array, optional
        Valid-pixel mask. If None, all pixels are valid.

    Returns
    -------
    background : (N, H, W) array
        The fitted background surface or line-by-line fits.
    """
    img = np.asarray(img)
    is_stack = img.ndim == 3

    # Convert to (N, H, W) for consistent processing
    if is_stack:
        frames = img
    else:
        frames = img[np.newaxis, ...]  # shape (1, H, W)

    if mask is not None:
        mask = np.asarray(mask)
        if mask.ndim == 2:
            mask = mask[np.newaxis, ...]  # shape (1, H, W)
        elif mask.shape != frames.shape:
            raise ValueError("mask must have the same shape as img")

    background_frames = []

    for idx in range(frames.shape[0]):
        frame = frames[idx]
        frame_mask = mask[idx] if mask is not None else None
        if method == "plane":
            bg = frame - level_plane(frame, frame_mask, polyx, polyy)
        elif method == "line":
            bg = frame - level_line(frame, frame_mask, polyx, polyy)
        elif method == "med_line":
            bg = frame - level_med_line(frame, frame_mask, polyx, polyy)
        elif method == "med_line_y":
            bg = frame - level_med_line_y(frame, frame_mask, polyx, polyy)
        elif method == "smed_line":
            bg = frame - level_smed_line(frame, frame_mask, polyx, polyy)
        elif method == "mean_plane":
            bg = frame - level_mean_plane(frame, frame_mask, polyx, polyy)
        elif method == "log_y":
            bg = frame - level_log_y(frame, frame_mask, polyx, polyy)
        else:
            raise ValueError(f"Unknown leveling method: {method}")

        background_frames.append(bg)

    result = np.stack(background_frames, axis=0)

    return result if is_stack else result[0]


__all__ = [
    "apply_level",
    "level_plane",
    "level_line",
    "level_med_line",
    "level_med_line_y",
    "level_smed_line",
    "level_mean_plane",
    "level_log_y",
]
