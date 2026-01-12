"""
Level and flatten AFM images and image stacks using MATLAB-aligned background
correction methods.

This module provides background leveling and flattening routines for Atomic
Force Microscopy (AFM) images and image stacks. The implemented methods correct
for background planes, line-by-line drift, median offsets, and systematic
row- or column-wise artefacts commonly observed in AFM topographic data.

All public leveling functions accept an *exclusion mask* (same convention as
``pnanolocz_lib.thresholder``): ``True`` = excluded, ``False`` = valid. Excluded pixels are
omitted from fitting and summary statistics using MATLAB-style NaN-outside
semantics (i.e., excluded pixels behave like NaN during fitting) but are
preserved in the output array.

The implementation is a Python port of the MATLAB NanoLocz Library:
    https://github.com/George-R-Heath/NanoLocz-Matlab-Library
Original MATLAB code by George Heath, University of Leeds.


MATLAB alignment
----------------
This Python version aims for algorithmic and numerical alignment with the MATLAB
reference implementation. Due to differences in underlying numerical libraries
(NumPy/SciPy vs MATLAB), polynomial conditioning, floating-point behaviour, and
edge-case handling, results may not be bit-for-bit identical. Where relevant,
functions document any intentional deviations adopted to match the reference
NanoLocz outputs (e.g., stage gating or fallback behaviour).

Available leveling methods
--------------------------
There methods can be used directly and applied to 2D arrays or are selected via the ``method``
argument in :func:`apply_level`:

- ``plane``       : Subtract a polynomial plane via masked column/row means.
- ``line``        : Subtract row-wise polynomial trends and optionally subtract
                    column-wise polynomial trends (applied only when ``polyy > 0``
                    for parity with the reference outputs used here).
- ``med_line``    : Subtract a per-row median baseline and re-center to a global
                    masked median.
- ``med_line_y``  : Subtract a per-column median baseline and re-center to a
                    global masked median.
- ``smed_line``   : Subtract a smoothed per-row median baseline (moving median).
- ``mean_plane``  : Subtract the masked mean value.
- ``log_y``       : Subtract a fitted logarithmic trend along the Y-axis.

Dispatcher and usage
--------------------
The primary entry point is the ``apply_level`` function, which dispatches to
the appropriate leveling routine based on the requested method and applies
it frame-by-frame if a 3D stack is provided. All methods accept an optional
exclusion mask, where excluded pixels are excluded from fitting operations but
preserved in the output.

The companion function ``get_background`` computes the fitted background
surface or lines *without subtracting them*, enabling visualisation or
diagnostic inspection of the estimated background.

Stacks
------
Functions operate on single images with shape ``(H, W)`` and stacks with shape
``(N, H, W)``, processing stacks frame-by-frame. If a 2D mask is provided for a
single image it is used directly; for stacks, masks must match the stack shape
(or be promoted appropriately by the dispatcher).

Examples
--------
>>> from pnanolocz_lib.level import level
>>> leveled_stack = apply_level(stack, polyx=2, polyy=2, method="plane")

>>> from pnanolocz_lib.level import level_plane
>>> flattened = level_plane(img, mask=None, polyx=2, polyy=2)

Authors
-------
George Heath, University of Leeds (2025)
Maya Tekchandani, University of Leeds (2025)
Daniel. E. Rollins, University of Leeds (2025)

This module is part of the ``pNanoLocz-Lib`` Python library for AFM analysis.
"""

import warnings
from typing import Any, Literal, Optional

import numpy as np
from numpy.polynomial.polyutils import RankWarning  # type: ignore[attr-defined]
from scipy.optimize import curve_fit

# Constants
SMOOTHING_WINDOW = 10
LOG_FIT_BOUNDS = ([0.1, 0.01, 0.1], [1000, 20, 100])


def _validity_mask(
    arr: np.ndarray,
    mask_excl: Optional[np.ndarray],
    *,
    name: str = "mask",
) -> np.ndarray:
    """
    Convert an exclusion mask into a finite-aware validity mask.

    Parameters
    ----------
    arr : ndarray
        2D image frame used to determine finite pixels.
    mask_excl : ndarray of bool, optional
        Exclusion mask with the same shape as `arr`.
        True = excluded pixel, False = valid pixel.
        If None, validity is determined only by finiteness of `arr`.
    name : str, default "mask"
        Name used in error messages when validating the mask shape.

    Returns
    -------
    m_valid : ndarray of bool
        Validity mask where True indicates a pixel is valid for fitting and
        False indicates it is excluded or non-finite.

    Notes
    -----
    - Non-finite pixels in `arr` are always marked invalid, regardless of mask.
    - This function enforces the module-wide contract that public masks are
      exclusion masks (True = excluded), while internal computations typically
      operate on validity masks (True = valid).
    """
    finite = np.isfinite(arr)

    if mask_excl is None:
        return finite

    m_excl = np.asarray(mask_excl, dtype=bool)
    if m_excl.shape != arr.shape:
        raise ValueError(
            f"{name} shape {m_excl.shape} must match img shape {arr.shape}"
        )

    return (~m_excl) & finite


def level_plane(
    img: np.ndarray[Any, np.dtype[np.float64]],
    mask: Optional[np.ndarray[Any, np.dtype[np.bool_]]],
    polyx: int,
    polyy: int,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Subtract a fitted polynomial plane using masked row and column means.

    This implements NanoLocz-style plane leveling by fitting and subtracting a
    polynomial along X (columns) using column-wise masked means, then fitting and
    subtracting a polynomial along Y (rows) from the partially leveled image.

    Parameters
    ----------
    img : ndarray
        2D AFM image to be leveled.
    mask : ndarray of bool, optional
        Exclusion mask with the same shape as `img`.
        True = excluded pixel (ignored in fitting; treated as NaN),
        False = valid / included pixel.
        If None, all finite pixels are treated as valid.
    polyx : int
        Polynomial degree for the X-direction (column-wise) fit.
    polyy : int
        Polynomial degree for the Y-direction (row-wise) fit.

    Returns
    -------
    leveled : ndarray
        Leveled image with the same shape as `img`.

    Notes
    -----
    - Excluded pixels are omitted from summary statistics using NaN-outside
    semantics (`np.where(valid, value, np.nan)` + `nanmean`).
    - Indices are treated as 1-based for parity with MATLAB's `polyfit`.
    - Centering/scaling uses population standard deviation (ddof=0) to mimic
    MATLAB's `polyfit(..., mu)` convention.
    - If there are too few valid samples to support a fit, the input is returned
    unchanged (or partially leveled if X succeeds but Y cannot be fit).
    """
    arr = np.asarray(img, dtype=np.float64)  # Convert input image to float64

    # --- Build MATLAB-style validity mask (True = valid pixel) ---
    m = _validity_mask(
        arr, mask, name="mask"
    )  # m : boolean validity mask (True = valid pixel).

    # Global gate: must have >5 valid pixels overall (matches MATLAB intent)
    if m.sum() <= 5:
        return arr.copy()

    # ========== X DIRECTION ==========
    # Column-wise masked mean with NaN-outside semantics
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        column_means = np.nanmean(np.where(m, arr, np.nan), axis=0)

    valid_columns = ~np.isnan(column_means)
    if valid_columns.sum() <= polyx:
        # Not enough points to fit X polynomial
        return arr.copy()

    # 1-based indices (MATLAB uses 1..W)
    column_indices = (np.nonzero(valid_columns)[0] + 1).astype(np.float64)

    # Center and scale like MATLAB's polyfit mu (population std, ddof=0)
    col_centroid = column_indices.mean()
    col_scale = column_indices.std(ddof=0)
    if col_scale == 0:  # very rare, but prevents divide-by-zero
        return arr.copy()

    standardized_columns = (column_indices - col_centroid) / col_scale

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RankWarning)
        x_coeffs = np.polyfit(standardized_columns, column_means[valid_columns], polyx)

    # Evaluate polynomial at every column (1..W) using the same mu
    all_cols = (np.arange(arr.shape[1]) + 1).astype(np.float64)
    standardized_all_cols = (all_cols - col_centroid) / col_scale
    x_plane = np.polyval(x_coeffs, standardized_all_cols)[None, :]

    leveled = arr - x_plane

    # ========== Y DIRECTION ==========
    # Row-wise masked mean after X subtraction (NaN-outside semantics)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        row_means = np.nanmean(np.where(m, leveled, np.nan), axis=1)

    valid_rows = ~np.isnan(row_means)
    if valid_rows.sum() <= polyy:
        return np.asarray(leveled)

    # 1-based indices (MATLAB uses 1..H)
    row_indices = (np.nonzero(valid_rows)[0] + 1).astype(np.float64)

    # Center and scale like MATLAB's polyfit mu
    row_centroid = row_indices.mean()
    row_scale = row_indices.std(ddof=0)
    if row_scale == 0:
        return np.asarray(leveled)

    standardized_rows = (row_indices - row_centroid) / row_scale

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RankWarning)
        y_coeffs = np.polyfit(standardized_rows, row_means[valid_rows], polyy)

    # Evaluate polynomial at every row (1..H) with the same mu
    all_rows = (np.arange(arr.shape[0]) + 1).astype(np.float64)
    standardized_all_rows = (all_rows - row_centroid) / row_scale
    y_plane = np.polyval(y_coeffs, standardized_all_rows)[:, None]

    return np.asarray(leveled - y_plane)


def level_line(
    img: np.ndarray[Any, np.dtype[np.float64]],
    mask: Optional[np.ndarray[Any, np.dtype[np.bool_]]],
    polyx: int,
    polyy: int,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Subtract line-wise polynomial trends along X and optionally along Y.

    This implements NanoLocz-style 'line' leveling in two stages:
    (1) row-wise polynomial subtraction along X, followed by
    (2) optional column-wise polynomial subtraction along Y.

    Parameters
    ----------
    img : ndarray
        2D AFM image to be leveled.
    mask : ndarray of bool, optional
        Exclusion mask with the same shape as `img`.
        True = excluded pixel (ignored in fitting; treated as NaN),
        False = valid / included pixel.
        If None, all finite pixels are treated as valid.
    polyx : int
        Polynomial degree for the row-wise (X-direction) fit. Set <= 0 to skip.
    polyy : int
        Polynomial degree for the column-wise (Y-direction) fit. If `polyy > 0`,
        a per-column polynomial is fit and subtracted.

    Returns
    -------
    leveled : ndarray
        Leveled image with the same shape as `img`.

    Notes
    -----
    - Stage 1 (rows): each row is fit using only valid pixels. Rows with too few
    valid pixels fall back to subtracting the median fitted curve computed over
    successfully fit rows (mirrors the MATLAB fallback idea).
    - Stage 2 (columns): this implementation applies the Y-stage only when
    `polyy > 0`.
    - Polynomial fitting uses 1-based indices and population std (ddof=0) for
    MATLAB-like centering/scaling.
    """
    arr = np.asarray(img, dtype=np.float64)  # Convert input image to float64

    # --- Build MATLAB-style validity mask (True = valid pixel) ---
    m = _validity_mask(
        arr, mask, name="mask"
    )  # boolean validity mask (True = valid pixel).

    out = arr.copy()

    # ---------------- Row X-stage ----------------
    if polyx > 0:
        row_fits = np.full_like(arr, np.nan)
        fitted_rows, fallback_rows = [], []

        img_width = arr.shape[1]
        for i in range(arr.shape[0]):
            pos = m[i, :]  # True = valid pixels in row i
            if pos.sum() > (
                polyx + 8
            ):  # pos: per-row validity mask (True = valid pixel)
                # 1-based indices for parity
                x_idx = (np.nonzero(pos)[0] + 1).astype(
                    np.float64
                )  # x_idx: 1-based indices of valid pixels in row
                y_vals = arr[i, pos].astype(
                    np.float64
                )  # y_vals: observed values at valid pixels in row i

                mu = x_idx.mean()  # mu: centroid of x_idx for MATLAB-style centering
                sd = (
                    x_idx.std(ddof=0) or 1.0
                )  # sd: population std for MATLAB-style scaling (guarded)
                xs = (x_idx - mu) / sd  # xs: standardized x indices used for fitting

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RankWarning)
                    p_coeff = np.polyfit(xs, y_vals, polyx)

                all_cols = (np.arange(img_width) + 1).astype(np.float64)
                xs2 = (all_cols - mu) / sd
                fit = np.polyval(p_coeff, xs2)

                row_fits[i, :] = fit
                out[i, :] = arr[i, :] - fit
                fitted_rows.append(i)
            else:
                fallback_rows.append(i)

        if fitted_rows and fallback_rows:
            # median(y2) across fitted rows (ignore NaNs)
            median_curve = np.nanmedian(row_fits[fitted_rows, :], axis=0)
            for i in fallback_rows:
                out[i, :] = arr[i, :] - median_curve

    # ---------------- Column Y-stage ----------------
    if polyy > 0:
        img_height = arr.shape[0]
        for j in range(arr.shape[1]):
            yp = np.where(m[:, j], out[:, j], np.nan)

            valid_rows = ~np.isnan(yp)
            yl = (np.nonzero(valid_rows)[0] + 1).astype(
                np.float64
            )  # yl: 1-based indices of valid samples in column

            if yl.size < (polyy + 1):
                out[:, j] = arr[:, j]
                continue

            y_vals = yp[valid_rows].astype(
                np.float64
            )  # y_vals: observed values at valid rows for this column

            mu = yl.mean()  # mu: centroid of yl for MATLAB-style centering
            sd = (
                yl.std(ddof=0) or 1.0
            )  # sd: population std for MATLAB-style scaling (guarded)
            ys = (yl - mu) / sd  # ys: standardized y indices used for fitting

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RankWarning)
                p_coeff = np.polyfit(ys, y_vals, polyy)

            all_rows = (np.arange(img_height) + 1).astype(np.float64)
            ys2 = (all_rows - mu) / sd  # ys2: standardized 1..H using same mu/sd
            fit = np.polyval(
                p_coeff, ys2
            )  # fit: fitted baseline across the full column

            out[:, j] = out[:, j] - fit

    return np.asarray(out)


def level_med_line(
    img: np.ndarray[Any, np.dtype[np.float64]],
    mask: Optional[np.ndarray[Any, np.dtype[np.bool_]]],
    polyx: float,
    polyy: int,  # unused
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Subtract a per-row median baseline and re-center to a global masked median.

    For each row with sufficient valid pixels, this subtracts the row median
    (computed from valid pixels only) and adds back a global background level
    given by the median of the masked image. This mirrors NanoLocz 'med_line'.

    Parameters
    ----------
    img : ndarray
        2D AFM image to be leveled.
    mask : ndarray of bool, optional
        Exclusion mask with the same shape as `img`.
        True = excluded pixel (ignored; treated as NaN),
        False = valid / included pixel.
        If None, all finite pixels are treated as valid.
     polyx : float
        NanoLocz/MATLAB behaviour: `polyx` acts as a gain on the row-median
        baseline *only if `polyx > 0`*. Otherwise, a gain of 1 is used.
        (This is why MATLAB sometimes passes 0.6 here, i.e. in ``level_auto``.)
    polyy : int
        Unused (kept for API parity).

    Returns
    -------
    leveled : ndarray
        Leveled image with the same shape as `img`.

    Notes
    -----
    - Rows with <= 10 valid pixels are left unchanged (matching the MATLAB guard).
    - The global background is computed as `median(where(valid, img, NaN))`.
    """
    arr = np.asarray(img, dtype=np.float64)

    # --- Build MATLAB-style validity mask (True = valid pixel) ---
    m = _validity_mask(
        arr, mask, name="mask"
    )  # boolean validity mask (True = valid pixel).

    # MATLAB: bg = median(imgt .* r, 'all', 'omitnan')
    # Here, `m` defines valid pixels; excluded pixels behave like NaN.
    masked = np.where(m, arr, np.nan)
    bg = float(np.nanmedian(masked))

    out = arr.copy()

    # MATLAB behaviour:
    #   if polyx > 0: subtract polyx * row_med
    #   else:         subtract 1.0 * row_med
    strength = float(polyx) if float(polyx) > 0 else 1.0

    for i in range(arr.shape[0]):
        # MATLAB: pos = ~isnan(imgt(i,:,k))
        pos = m[i, :]
        if pos.sum() > 10:
            # MATLAB uses median() (no omitnan needed because pos excludes invalids
            row_med = float(np.median(arr[i, pos]))
            out[i, :] = arr[i, :] - (strength * row_med) + bg
        else:
            out[i, :] = arr[i, :]  # unchanged if too few points

    return np.asarray(out)


def level_med_line_y(
    img: np.ndarray[Any, np.dtype[np.float64]],
    mask: Optional[np.ndarray[Any, np.dtype[np.bool_]]],
    polyx: int,  # unused
    polyy: int,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Subtract a per-column median baseline and re-center to a global masked median.

    For each column with sufficient valid pixels, this subtracts the column median
    (computed from valid pixels only) and adds back a global background level
    given by the median of the masked image. This mirrors NanoLocz 'med_line_y'.

    Parameters
    ----------
    img : ndarray
        2D AFM image to be leveled.
    mask : ndarray of bool, optional
        Exclusion mask with the same shape as `img`.
        True = excluded pixel (ignored; treated as NaN),
        False = valid / included pixel.
        If None, all finite pixels are treated as valid.
    polyx : int
        Unused (kept for API parity).
    polyy : int
        Unused (kept for API parity).

    Returns
    -------
    leveled : ndarray
        Leveled image with the same shape as `img`.

    Notes
    -----
    - Columns with <= 10 valid pixels are left unchanged (matching the MATLAB guard).
    - The global background is computed as `median(where(valid, img, NaN))`.
    - Unlike ``level_med_line`` this does not have a gain parameter taken form `polyy`,
    this mirrors the MATLAB version.
    """
    arr = np.asarray(img, dtype=np.float64)

    # --- Build MATLAB-style validity mask (True = valid pixel) ---
    m = _validity_mask(
        arr, mask, name="mask"
    )  # boolean validity mask (True = valid pixel).

    bg = np.nanmedian(np.where(m, arr, np.nan))

    out = arr.copy()
    for j in range(arr.shape[1]):
        pos = m[:, j]
        if pos.sum() > 10:
            col_med = np.nanmedian(arr[pos, j])
            out[:, j] = arr[:, j] - col_med + bg
        else:
            out[:, j] = arr[:, j]

    return np.asarray(out)


def level_smed_line(
    img: np.ndarray[Any, np.dtype[np.float64]],
    mask: Optional[np.ndarray[Any, np.dtype[np.bool_]]],
    polyx: int,  # unused
    polyy: int,  # unused
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Subtract a smoothed per-row median baseline using a moving median filter.

    This mirrors NanoLocz 'smed_line' by computing a per-row baseline from the row
    median (valid pixels only), smoothing that baseline with a moving median, and
    subtracting the (baseline - smoothed_baseline) from the image.

    Parameters
    ----------
    img : ndarray
        2D AFM image.
    mask : ndarray of bool, optional
        Exclusion mask with the same shape as `img`.
        True = excluded pixel (ignored; treated as NaN),
        False = valid / included pixel.
        If None, all finite pixels are treated as valid.
    polyx : int
        Unused (kept for API parity).
    polyy : int
        Unused (kept for API parity).

    Returns
    -------
    leveled : ndarray
        Leveled image with the same shape as `img`.

    Notes
    -----
    - The global background is computed as `median(where(valid, img, NaN))`.
    - Rows with <= 10 valid pixels use the global background as their median.
    - The smoothing window length is `SMOOTHING_WINDOW` and is intended to match
    MATLAB's `movmedian(..., 10)` behaviour as closely as practical.
    """
    arr = np.asarray(img, dtype=np.float64)

    # --- Build MATLAB-style validity mask (True = valid pixel) ---
    m = _validity_mask(
        arr, mask, name="mask"
    )  # boolean validity mask (True = valid pixel).

    # Global background with NaN-outside
    bg = np.nanmedian(np.where(m, arr, np.nan))

    # Row medians (use mask row), then add bg
    img_height, img_width = arr.shape
    y1 = np.empty(img_height, dtype=np.float64)
    for i in range(img_height):
        pos = m[i, :]
        if pos.sum() > 10:
            y1[i] = np.nanmedian(arr[i, pos]) + bg
        else:
            y1[i] = bg

    # movmedian over rows (centered, shrink at edges)
    def _movmedian(x: np.ndarray, w: int) -> np.ndarray:
        n = x.size
        out = np.empty_like(x)
        half = w // 2
        for i in range(n):
            start = max(0, i - half)
            end = min(n, start + w)
            out[i] = np.median(x[start:end])
        return out

    bg2 = _movmedian(y1, SMOOTHING_WINDOW)

    # Apply smoothed baseline: r = img - (y1 - bg2)
    out = arr - (y1[:, None] - bg2[:, None])
    return np.asarray(out)


def level_mean_plane(
    img: np.ndarray,
    mask: Optional[np.ndarray],
    polyx: int,  # unused
    polyy: int,  # unused
) -> np.ndarray:
    """
    Subtract the masked mean value from an image.

    This computes the mean of valid pixels only (finite and not excluded) and
    subtracts it from the full image, mirroring NanoLocz 'mean_plane'.

    Parameters
    ----------
    img : ndarray
        2D AFM image.
    mask : ndarray of bool, optional
        Exclusion mask with the same shape as `img`.
        True = excluded pixel (ignored; treated as NaN),
        False = valid / included pixel.
        If None, all finite pixels are treated as valid.
    polyx : int
        Unused (kept for API parity).
    polyy : int
        Unused (kept for API parity).

    Returns
    -------
    leveled : ndarray
        Mean-subtracted image with the same shape as `img`.

    Notes
    -----
    - The mean is computed as `nanmean(where(valid, img, NaN))`.
    - If no valid pixels exist, the input is returned unchanged.
    """
    arr = np.asarray(img, dtype=np.float64)
    finite = np.isfinite(arr)

    if mask is None:
        m_valid = finite
    else:
        mask_excl = np.asarray(mask, dtype=bool)
        if mask_excl.shape != arr.shape:
            raise ValueError(
                f"mask shape {mask_excl.shape} must match img shape {arr.shape}"
            )

        # exclusion -> validity
        m_valid = (~mask_excl) & finite

    if not np.any(m_valid):
        return arr.copy()

    masked = np.where(m_valid, arr, np.nan)
    mean_val = np.nanmean(masked)

    return np.asarray(arr - mean_val)


def level_log_y(
    img: np.ndarray[Any, np.dtype[np.float64]],
    mask: Optional[np.ndarray[Any, np.dtype[np.bool_]]],
    polyx: int,
    polyy: int,
    *,
    orientation: str = "auto",  # "auto" | "normal" | "reverse"
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Subtract a fitted logarithmic background trend along the Y-axis.

    This estimates a 1D correction curve from the row-wise mean of the image and
    subtracts it from each row. The correction is obtained by fitting a logarithmic
    model on a restricted X-range (mirroring the NanoLocz approach).

    Parameters
    ----------
    img : ndarray
        2D AFM image.
    mask : ndarray of bool, optional
        Unused. Present for API consistency with other leveling functions.
    polyx : int
        Unused.
    polyy : int
        Scale factor controlling the X-axis normalization used in the log fit.
    orientation : {"auto","normal","reverse"}, default "auto"
        Controls whether the correction is applied in normal or flipped order.
        "auto" evaluates both and chooses the one that best flattens row means.

    Returns
    -------
    leveled : ndarray
        Corrected image with the same shape as `img`.

    Notes
    -----
    - This function currently ignores `mask` (NanoLocz MATLAB code also does not
    apply masking in the shown log-y snippet).
    - The `"auto"` orientation mode is a Python convenience for robustness; MATLAB
    typically applies a fixed orientation (often involving `flip`).
    """
    y = np.mean(img, axis=1)
    correction = _log_y_correction(y, polyy)

    def _apply(
        img_: np.ndarray[Any, np.dtype[np.float64]],
        corr: np.ndarray[Any, np.dtype[np.float64]],
        rev: bool,
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        if rev:
            corr = corr[::-1]
        return np.asarray(img_ - corr[:, None])

    if orientation == "normal":
        return np.asarray(_apply(img, correction, rev=False))
    if orientation == "reverse":
        return np.asarray(_apply(img, correction, rev=True))

    # "auto": choose orientation that best flattens row means
    cand1 = _apply(img, correction, rev=False)
    cand2 = _apply(img, correction, rev=True)
    rng1 = np.ptp(cand1.mean(axis=1))
    rng2 = np.ptp(cand2.mean(axis=1))
    return np.asarray(cand1 if rng1 <= rng2 else cand2)


def _log_y_correction(
    y: np.ndarray[Any, np.dtype[Any]], scale: float
) -> np.ndarray[Any, np.dtype[Any]]:
    """
    Fit a logarithmic correction curve to a 1D row-mean profile.

    Parameters
    ----------
    y : ndarray
        1D signal (typically the row-wise mean of an image).
    scale : float
        Scale factor applied in the X-axis normalization used for fitting.

    Returns
    -------
    correction : ndarray
        1D correction curve of the same length as `y`. If fitting fails, a
        zero-array is returned.
    """
    y = y - np.min(y)
    x = np.linspace(0, 10, len(y)) / scale
    pos = x < 5
    x_fit = x[pos]
    y_fit = y[pos]

    def _log_model(
        x: np.ndarray[Any, np.dtype[Any]], a: float, b: float, c: float
    ) -> np.ndarray[Any, np.dtype[Any]]:
        return np.asarray(a * np.log(c * x + b))

    try:
        popt, _ = curve_fit(
            _log_model, x_fit, y_fit, p0=[5, 1, 2], bounds=LOG_FIT_BOUNDS
        )
        return _log_model(x, *popt)
    except Exception:
        return np.zeros_like(y)


def apply_level(
    img: np.ndarray[Any, np.dtype[np.float64]],
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
    mask: Optional[np.ndarray[Any, np.dtype[np.bool_]]] = None,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Apply a leveling method to an AFM image or stack.

    This is the public dispatcher for the leveling routines in this module. It
    normalizes inputs to a frame-first representation ``(N, H, W)``, applies the
    requested method frame-by-frame, and returns an array with the same shape as
    the input.

    Parameters
    ----------
    img : ndarray
        Input AFM image or image stack. Supported shapes are ``(H, W)`` for a
        single frame or ``(N, H, W)`` for a stack, where ``N`` is the number of
        frames.
    polyx : int
        X-stage polynomial degree (interpretation depends on `method`).
        Use ``0`` to disable polynomial fitting along X where applicable.
    polyy : int
        Y-stage polynomial degree (interpretation depends on `method`).
        Use ``0`` to disable polynomial fitting along Y where applicable.
    method : {"plane", "line", "med_line", "med_line_y", "smed_line", "mean_plane",
        "log_y"}
        Leveling method to apply.
    mask : ndarray of bool, optional
        Exclusion mask with the same shape as `img` (after any single-image
        promotion to ``(1, H, W)``).
        ``True`` marks excluded pixels and ``False`` marks valid pixels.
        If None, validity is determined only by finiteness of `img`.

    Returns
    -------
    leveled : ndarray
        Leveled output with the same shape as `img`.

    Raises
    ------
    ValueError
        If `mask` is provided and its shape does not match the promoted `img`
        shape (``(1, H, W)`` for a single image or ``(N, H, W)`` for a stack),
        or if `method` is not recognized.

    Notes
    -----
    - Mask semantics are **exclusion-based** (``True`` = excluded). Individual
    methods convert this to an internal validity mask and use NaN-outside style
    operations (e.g. ``where(valid, value, nan)``) during fitting.
    - Method-specific MATLAB parity notes (e.g., stage gating such as `polyy > 0`
    in ``"line"``) are documented in the corresponding function docstrings.
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
            mask = mask[np.newaxis, ...]  # promote to (1, H, W) for single image
        # Always validate shape after any promotion
        if mask.shape != frames.shape:
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

    return np.asarray(result) if is_stack else np.asarray(result[0])


def get_background(
    img: np.ndarray[Any, np.dtype[np.float64]],
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
    mask: Optional[np.ndarray[Any, np.dtype[np.bool_]]] = None,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Compute the background estimated by a leveling method without modifying the input.

    This function returns the background surface/lines that would be subtracted by
    :func:`apply_level` for the given `method`, computed frame-by-frame. The result
    is defined as ``background = input - leveled`` using the same method-specific
    logic and masking semantics.

    Parameters
    ----------
    img : ndarray
        Input AFM image or image stack. Supported shapes are ``(H, W)`` for a
        single frame or ``(N, H, W)`` for a stack.
    polyx : int
        X-stage polynomial degree (interpretation depends on `method`).
    polyy : int
        Y-stage polynomial degree (interpretation depends on `method`).
    method : {"plane", "line", "med_line", "med_line_y", "smed_line", "mean_plane",
       "log_y"}
        Leveling method whose estimated background should be returned.
    mask : ndarray of bool, optional
        Exclusion mask with the same shape as `img` (after any single-image
        promotion to ``(1, H, W)``).
        ``True`` marks excluded pixels and ``False`` marks valid pixels.
        If None, validity is determined only by finiteness of `img`.

    Returns
    -------
    background : ndarray
        Estimated background with the same shape as `img`.

    Raises
    ------
    ValueError
        If `mask` is provided and its shape does not match the promoted `img`
        shape (``(1, H, W)`` for a single image or ``(N, H, W)`` for a stack),
        or if `method` is not recognized.

    Notes
    -----
    - This function computes ``background`` as ``frame - leveled_frame`` using the
    same implementation as :func:`apply_level`. As a consequence, any intentional
    MATLAB-parity behaviours in the underlying method (e.g., skipping stages under
    certain parameter values) are inherited here.
    - Excluded pixels are preserved in the output arrays; masking primarily affects
    which pixels contribute to fitted estimates.
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
            mask = mask[np.newaxis, ...]  # promote to (1, H, W) for single image
        # Always validate shape after any promotion
        if mask.shape != frames.shape:
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

    return np.asarray(result) if is_stack else np.asarray(result[0])


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
