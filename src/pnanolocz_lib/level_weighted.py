"""
Weighted-region AFM image flattening and background leveling tools.

This module provides a Python port of the MATLAB ``level_weighted.m``
function used in the NanoLocz workflow. It implements region-wise weighted
polynomial and median-based background estimation for Atomic Force Microscopy
(AFM) images, enabling correction of multi-region drift, structured background,
and non-uniform masking effects.

The original Nanolocz-lib script was adapted from FindSteps.m and PolyfitLineMasked.m
scripts from the SPIW project (<https://sourceforge.net/projects/spiw/>) and combined
with NanoLocz leveling methods.

Supported Leveling Methods
--------------------------
- 'plane'       : Region-weighted polynomial plane subtraction in X and Y.
- 'line'        : Region-weighted row/column polynomial leveling.
- 'med_line'    : Region-weighted row-wise median line flattening.
- 'med_line_y'  : Region-weighted column-wise median line flattening.
- 'smed_line'   : Region-weighted smoothed median line subtraction.

Typical usage involves calling the :func:`apply_level_weighted` dispatcher with
an AFM image (2D) or a stack (3D) and choosing one of the methods above.

Examples
--------
>>> from pnanolocz_lib.filters.level_weighted import apply_level_weighted
>>> leveled = apply_level_weighted(img, polyx=2, polyy=1, method='plane', mask=mask)

Authors
-------
George Heath, University of Leeds (2025)
Daniel E. Rollins, University of Leeds (2025)
"""

from __future__ import annotations

import warnings
from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.polynomial.polyutils import RankWarning  # type: ignore[attr-defined]
from scipy import ndimage

# ---------------------
# Low-level helpers
# ---------------------


def _center_scale_indices(
    indices: np.ndarray[Any, np.dtype[np.float64]],
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], float, float]:
    """Center and scale a 1-D index array.

    Parameters
    ----------
    indices
        1-D integer index positions (e.g. column or row indices).

    Returns
    -------
    std_indices : np.ndarray
        Centered and scaled indices (float).
    centroid : float
        Mean of the original indices.
    scale : float
        Population standard deviation of the original indices (`ddof=0`); guaranteed
        non-zero: defaults to 1.0 for empty input, a single value,
        or any degenerate/constant data where the computed std is 0.
    """
    if indices.size == 0:
        return indices.astype(float), 0.0, 1.0

    centroid = float(indices.mean())
    scale = float(indices.std(ddof=0)) if indices.size > 1 else 1.0
    if scale == 0:
        scale = 1.0
    std_indices = (indices - centroid) / scale
    return std_indices, centroid, scale


def _polyfit_centered(
    x: np.ndarray, y: np.ndarray, order: int
) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Fit polynomial to y vs x after centering and scaling x.
    Equivalent to MATLAB polyfit with mu output.
    Returns:
      coeffs: polynomial coefficients (highest power first)
      (centroid, scale): centering and scaling applied to x
    """
    if x.size == 0 or y.size == 0 or x.size <= order:
        return np.zeros(order + 1, dtype=float), (0.0, 1.0)

    centroid = float(np.nanmean(x))
    scale = float(np.nanstd(x, ddof=0)) if x.size > 1 else 1.0
    if scale == 0:
        scale = 1.0

    std_x = (x - centroid) / scale

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RankWarning)
        coeffs = np.polyfit(std_x, y, order)
    return coeffs, (centroid, scale)


def _polyval_centered(
    coeffs: np.ndarray[Any, np.dtype[np.float64]],
    centering: Tuple[float, float],
    points: np.ndarray[Any, np.dtype[np.int64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Evaluate a polynomial fitted on centered-and-scaled x.

    Equivalent to MATLAB function: polyval.

    Parameters
    ----------
    coeffs : np.ndarray
        Polynomial coefficients from :func:`_polyfit_centered`.
    centering : tuple
        ``(centroid, scale)`` used to standardise x during fitting.
    points : np.ndarray
        Points (original coordinate space) at which to evaluate.
    """
    centroid, scale = centering
    if scale == 0:
        scale = 1.0
    std_points = (points - centroid) / scale
    return np.polyval(coeffs, std_points)


def _find_regions(
    mask: np.ndarray[Any, np.dtype[np.bool_]], min_area: int
) -> List[np.ndarray[Any, np.dtype[np.int64]]]:
    """
    Replicate MATLAB bwconncomp(mask,8) + MATLAB min_area filtering.

    Parameters
    ----------
    mask : ndarray(bool)
        True = foreground (same semantic as MATLAB `imgt ~= 0`).
    min_area : int
        (Ignored input value) We compute min_area exactly as MATLAB does:
        max(1, floor(0.01 * H * W)). This keeps Python and MATLAB aligned.

    Returns
    -------
    regions : list of 1-D np.ndarray of dtype int
        Each element is a flat index array (row-major / numpy.ravel order)
        describing the pixels in that region. This matches the layout used
        elsewhere in the python port (so `region_masked.flat[flat_idx] = ...`
        behaves correctly).
    """

    # 8-connectivity structure (exactly like MATLAB bwconncomp(mask,8))
    structure = np.ones((3, 3), dtype=int)
    labeled, num_features = ndimage.label(mask, structure=structure)

    # compute MATLAB-style minimum area
    h, w = mask.shape
    if min_area is None:
        min_area = max(1, int(np.floor(0.01 * (h * w))))

    # areas: sum of True values for each label (labels 1..num_features)
    # ndimage.sum returns results in label order
    if num_features == 0:
        return []

    areas = ndimage.sum(mask, labeled, index=np.arange(1, num_features + 1))

    # keep labels that satisfy MATLAB's >= min_area
    keep_labels = [
        lab for lab, area in zip(range(1, num_features + 1), areas) if area >= min_area
    ]

    # produce flat indices for each kept label (row-major to match numpy.flat)
    regions: List[np.ndarray] = []
    for lab in keep_labels:
        rows_idx, cols_idx = np.where(labeled == lab)
        # build flat indices in numpy order (row-major)
        flat_idx = np.ravel_multi_index((rows_idx, cols_idx), mask.shape, order="C")
        regions.append(flat_idx.astype(np.int64))

    return regions


# ---------------------
# Per-method implementations
# ---------------------


def level_weighted_plane(
    img: np.ndarray[Any, np.dtype[np.float64]],
    regions: List[np.ndarray[Any, np.dtype[np.int64]]],
    polyx: int,
    polyy: int,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Region-weighted polynomial plane subtraction along X and Y.

    The function computes per-region polynomial fits of the mean profile in the
    X- and Y-directions and forms a weighted average of those per-region fits
    (weights proportional to region pixel counts). The combined plane is
    subtracted from ``img``.

    Parameters
    ----------
    img : np.ndarray
        2-D AFM image.
    regions : list of np.ndarray
        List of flat index arrays describing foreground regions.
    polyx, polyy : int
        Polynomial orders for the X (columns) and Y (rows) directions.

    Returns
    -------
    np.ndarray
        The leveled image (float64).
    """
    rows, cols = img.shape
    img_f = np.asarray(img, dtype=float)

    n_regions = len(regions)

    if n_regions == 0:
        return img_f.copy()

    region_pixel_counts = np.zeros(n_regions, dtype=float)

    x_poly_list: List[np.ndarray[Any, np.dtype[np.float64]]] = []
    x_centroid_list: List[float] = []
    x_scale_list: List[float] = []

    y_poly_list: List[np.ndarray[Any, np.dtype[np.float64]]] = []
    y_centroid_list: List[float] = []
    y_scale_list: List[float] = []

    for i, region_indices in enumerate(regions):
        # Nanolocz- build regionMatrix (here region_masked)
        region_masked = np.full(img_f.shape, np.nan, dtype=float)
        region_masked.flat[region_indices] = img_f.flat[region_indices]
        region_pixel_counts[i] = (
            region_indices.size
        )  # w(i) in MATLAB Nanolocz George says w is weighting

        # X-direction: mean of each column within region
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_by_col = np.nanmean(
                region_masked, axis=0
            )  # Nanolocz-mean_by_col is xp
        valid_cols = ~np.isnan(mean_by_col)  # Nanolocz xf
        col_values = mean_by_col[valid_cols]
        col_positions = np.flatnonzero(valid_cols)

        if col_positions.size > polyx:
            coeffs_x, (cent_x, scale_x) = _polyfit_centered(
                col_positions.astype(float), col_values.astype(float), polyx
            )
        else:
            coeffs_x = np.zeros(polyx + 1, dtype=float)
            cent_x, scale_x = 0.0, 1.0

        x_poly_list.append(coeffs_x)
        x_centroid_list.append(cent_x)
        x_scale_list.append(scale_x)

        # Y-direction: mean of each row within region
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_by_row = np.nanmean(region_masked, axis=1)
        valid_rows = ~np.isnan(mean_by_row)
        row_values = mean_by_row[valid_rows]
        row_positions = np.flatnonzero(valid_rows)

        if row_positions.size > polyy:
            coeffs_y, (cent_y, scale_y) = _polyfit_centered(
                row_positions.astype(float), row_values.astype(float), polyy
            )
        else:
            coeffs_y = np.zeros(polyy + 1, dtype=float)
            cent_y, scale_y = 0.0, 1.0

        y_poly_list.append(coeffs_y)
        y_centroid_list.append(cent_y)
        y_scale_list.append(scale_y)

    # region weights (normalized; exclude tiny regions by thresholding)
    # Weights are W in Nanlocz-lib
    weights = region_pixel_counts / (
        region_pixel_counts.sum() if region_pixel_counts.sum() > 0 else 1.0
    )
    # Exclude regions with less that 2% area
    weights = np.where(weights > 0.02, weights, 0.0)

    # Pad coefficient arrays to the same length then take weighted sum
    max_len_x = max((p.size for p in x_poly_list), default=0)
    max_len_y = max((p.size for p in y_poly_list), default=0)
    x_poly_arr = np.stack(
        [np.pad(p, (0, max_len_x - p.size), mode="constant") for p in x_poly_list],
        axis=1,
    )
    y_poly_arr = np.stack(
        [np.pad(p, (0, max_len_y - p.size), mode="constant") for p in y_poly_list],
        axis=1,
    )

    weighted_x_coeffs = (x_poly_arr * weights[None, :]).sum(axis=1)  # x_poly_arr is W
    weighted_y_coeffs = (y_poly_arr * weights[None, :]).sum(axis=1)
    weighted_x_centroid = (np.array(x_centroid_list) * weights).sum()
    weighted_x_scale = (np.array(x_scale_list) * weights).sum()
    weighted_y_centroid = (np.array(y_centroid_list) * weights).sum()
    weighted_y_scale = (np.array(y_scale_list) * weights).sum()

    all_cols = np.arange(cols)
    all_rows = np.arange(rows)
    background_x = _polyval_centered(
        weighted_x_coeffs, (weighted_x_centroid, weighted_x_scale), all_cols
    )[None, :]
    background_y = _polyval_centered(
        weighted_y_coeffs, (weighted_y_centroid, weighted_y_scale), all_rows
    )[:, None]

    background_plane = background_x + background_y
    return img_f - background_plane


def level_weighted_line(img, regions, polyx, polyy):
    img_f = np.asarray(img, dtype=float)
    rows, cols = img_f.shape
    n_regions = len(regions)
    r = img_f.copy()

    # ----- X direction (rows) -----
    if polyx > 0:
        w_rows = np.zeros((rows, n_regions), dtype=float)
        px_coeffs = np.zeros((rows, polyx + 1, n_regions), dtype=float)
        mux_centroid = np.zeros((rows, n_regions), dtype=float)
        mux_scale = np.ones((rows, n_regions), dtype=float)

        for i, region_idx in enumerate(regions):
            region_masked = np.full((rows, cols), np.nan)
            region_masked.flat[region_idx] = img_f.flat[region_idx]
            for rr in range(rows):
                pos = ~np.isnan(region_masked[rr, :])
                w_rows[rr, i] = pos.sum()
                if pos.sum() > polyx + 1:
                    # >>> 1-based indices <<<
                    xl = (np.flatnonzero(pos) + 1).astype(float)
                    xf = img_f[rr, pos].astype(float)
                    coeffs, (cent, sc) = _polyfit_centered(xl, xf, polyx)
                    px_coeffs[rr, :, i] = coeffs
                    mux_centroid[rr, i] = cent
                    mux_scale[rr, i] = sc if sc != 0 else 1.0
                else:
                    px_coeffs[rr, :, i] = 0.0
                    mux_centroid[rr, i] = 0.0
                    mux_scale[rr, i] = 1.0

        denom = w_rows.sum(axis=1, keepdims=True)
        denom = np.where(denom == 0, 1.0, denom)
        W = np.divide(w_rows, denom, out=np.zeros_like(w_rows), where=denom != 0)
        W = W * (W > 0.02)  # threshold like MATLAB; do not renormalize

        px_w = (px_coeffs * W[:, None, :]).sum(axis=2)
        px_w[:, -1] = 0.0  # zero constant term
        mu_w_centroid = (mux_centroid * W).sum(axis=1)
        mu_w_scale = (mux_scale * W).sum(axis=1)

        xgrid_1b = np.arange(1, cols + 1, dtype=float)
        lines_x = np.zeros_like(img_f, dtype=float)
        for rr in range(rows):
            mu_row = (
                float(mu_w_centroid[rr]),
                float(mu_w_scale[rr]) if mu_w_scale[rr] != 0 else 1.0,
            )
            lines_x[rr, :] = _polyval_centered(px_w[rr, :], mu_row, xgrid_1b)
        r = r - lines_x

    # ----- Y direction (cols) -----
    if polyy > 0:
        w_cols = np.zeros((cols, n_regions), dtype=float)
        py_coeffs = np.zeros((cols, polyy + 1, n_regions), dtype=float)
        muy_centroid = np.zeros((cols, n_regions), dtype=float)
        muy_scale = np.ones((cols, n_regions), dtype=float)

        for i, region_idx in enumerate(regions):
            region_masked = np.full((rows, cols), np.nan)
            region_masked.flat[region_idx] = img_f.flat[region_idx]
            for cc in range(cols):
                pos = ~np.isnan(region_masked[:, cc])
                w_cols[cc, i] = pos.sum()
                if pos.sum() > polyy + 1:
                    # >>> 1-based indices <<<
                    yl = (np.flatnonzero(pos) + 1).astype(float)
                    yf = img_f[pos, cc].astype(float)
                    coeffs, (cent, sc) = _polyfit_centered(yl, yf, polyy)
                    py_coeffs[cc, :, i] = coeffs
                    muy_centroid[cc, i] = cent
                    muy_scale[cc, i] = sc if sc != 0 else 1.0
                else:
                    py_coeffs[cc, :, i] = 0.0
                    muy_centroid[cc, i] = 0.0
                    muy_scale[cc, i] = 1.0

        denom = w_cols.sum(axis=1, keepdims=True)
        denom = np.where(denom == 0, 1.0, denom)
        W = np.divide(w_cols, denom, out=np.zeros_like(w_cols), where=denom != 0)
        W = W * (W > 0.02)

        py_w = (py_coeffs * W[:, None, :]).sum(axis=2)
        py_w[:, -1] = 0.0
        mu_w_centroid = (muy_centroid * W).sum(axis=1)
        mu_w_scale = (muy_scale * W).sum(axis=1)

        ygrid_1b = np.arange(1, rows + 1, dtype=float)
        lines_y = np.zeros_like(img_f, dtype=float)
        for cc in range(cols):
            mu_col = (
                float(mu_w_centroid[cc]),
                float(mu_w_scale[cc]) if mu_w_scale[cc] != 0 else 1.0,
            )
            lines_y[:, cc] = _polyval_centered(py_w[cc, :], mu_col, ygrid_1b)
        r = r - lines_y

    return r


def level_weighted_med_line(
    img: np.ndarray[Any, np.dtype[np.float64]],
    regions: List[np.ndarray[Any, np.dtype[np.int64]]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    MATLAB 'med_line': region-weighted row-wise median subtraction.
    For each row:
      - Compute median of valid pixels per region.
      - Weight by region size.
      - Subtract weighted median profile from image.
    """
    img_f = np.asarray(img, dtype=np.float64)
    rows, cols = img_f.shape
    n_regions = len(regions)
    if n_regions == 0:
        return img_f.copy()

    # Initialize arrays
    w = np.zeros((rows, n_regions), dtype=float)
    y1 = np.zeros((rows, n_regions), dtype=float)
    bg = np.zeros(n_regions, dtype=float)

    # Compute per-region medians and weights
    for i, region_idx in enumerate(regions):
        region_masked = np.full((rows, cols), np.nan)
        region_masked.flat[region_idx] = img_f.flat[region_idx]
        bg[i] = np.nanmedian(region_masked)
        for rr in range(rows):
            pos = ~np.isnan(region_masked[rr, :])
            w[rr, i] = pos.sum()
            if w[rr, i] > 2:
                y1[rr, i] = np.nanmedian(img_f[rr, pos]) - bg[i]
            else:
                y1[rr, i] = -bg[i]

    # Compute weighted median profile
    W = np.divide(
        w,
        w.sum(axis=1, keepdims=True),
        out=np.zeros_like(w),
        where=w.sum(axis=1, keepdims=True) != 0,
    )
    yf = np.sum(W * y1, axis=1)
    pos = w.sum(axis=1) == 0
    r = img_f.copy()
    r[~pos, :] = img_f[~pos, :] - yf[~pos, None]
    return r


def level_weighted_med_line_y(
    img: np.ndarray[Any, np.dtype[np.float64]],
    regions: List[np.ndarray[Any, np.dtype[np.int64]]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    MATLAB 'med_line_y': region-weighted column-wise median subtraction.
    """
    img_f = np.asarray(img, dtype=float)
    rows, cols = img_f.shape
    n_regions = len(regions)
    if n_regions == 0:
        return img_f.copy()

    w = np.zeros((cols, n_regions), dtype=float)
    y1 = np.zeros((cols, n_regions), dtype=float)
    bg = np.zeros(n_regions, dtype=float)

    for i, region_idx in enumerate(regions):
        region_masked = np.full((rows, cols), np.nan)
        region_masked.flat[region_idx] = img_f.flat[region_idx]
        bg[i] = np.nanmedian(region_masked)
        for cc in range(cols):
            pos = ~np.isnan(region_masked[:, cc])
            w[cc, i] = pos.sum()
            if w[cc, i] > 2:
                y1[cc, i] = np.nanmedian(img_f[pos, cc]) - bg[i]
            else:
                y1[cc, i] = -bg[i]

    W = np.divide(
        w,
        w.sum(axis=1, keepdims=True),
        out=np.zeros_like(w),
        where=w.sum(axis=1, keepdims=True) != 0,
    )

    yf = np.sum(W * y1, axis=1)
    pos = w.sum(axis=1) == 0
    r = img_f.copy()
    r[:, ~pos] = img_f[:, ~pos] - yf[~pos][None, :]
    return r


# --- helper: MATLAB-like movmedian (include NaNs), centered, even window ---
def _movmedian_centered_includenan(x: np.ndarray, w: int) -> np.ndarray:
    """
    MATLAB movmedian default: include NaNs.
    Even w: window centered about current & previous (left=w//2, right=w-w//2).
    Shrink symmetrically at edges.
    """
    n = x.size
    out = np.empty(n, dtype=float)
    left = w // 2
    right = w - left
    for i in range(n):
        start = max(0, i - left)
        end = min(n, i + right)  # end is exclusive
        win = x[start:end]
        # include NaNs => if any NaN in window, median becomes NaN (like MATLAB default)
        out[i] = np.median(win)
    return out


def level_weighted_smed_line(
    img: np.ndarray[Any, np.dtype[np.float64]],
    regions: List[np.ndarray[Any, np.dtype[np.int64]]],
    smoothing_window: int = 10,
) -> np.ndarray[Any, np.dtype[np.float64]]:

    img_f = np.asarray(img, dtype=float)
    rows, cols = img_f.shape
    n_regions = len(regions)
    if n_regions == 0:
        return img_f.copy()

    # per-region row medians with >2 guard, and region backgrounds
    w = np.zeros((rows, n_regions), dtype=float)
    y1 = np.zeros((rows, n_regions), dtype=float)
    bg = np.zeros(n_regions, dtype=float)

    for i, region_idx in enumerate(regions):
        region_masked = np.full((rows, cols), np.nan)
        region_masked.flat[region_idx] = img_f.flat[region_idx]
        bg[i] = np.nanmedian(region_masked)  # region-wide background

        for rr in range(rows):
            pos = ~np.isnan(region_masked[rr, :])
            w[rr, i] = pos.sum()
            if w[rr, i] > 2:
                # raw row median in that region (no -bg here for smed_line)
                y1[rr, i] = np.nanmedian(img_f[rr, pos])
            else:
                y1[rr, i] = -bg[i]

    # row-normalized weights (like MATLAB; rows with sum=0 produce NaNs in W and yf)
    den = w.sum(axis=1, keepdims=True)
    W = w / den  # may create NaNs when den==0 (expected)
    yf = (W * y1).sum(axis=1)  # row baseline

    # rows with zero total coverage
    zero_rows = den[:, 0] == 0

    # MATLAB movmedian default (include NaNs) with even window; then NaNs -> 0
    yf_sm = _movmedian_centered_includenan(yf, smoothing_window)
    yf_sm = np.where(np.isfinite(yf_sm), yf_sm, 0.0)

    # subtract the SMOOTHED BASELINE ITSELF on rows with coverage
    r = img_f.copy()
    r[~zero_rows, :] = img_f[~zero_rows, :] - yf_sm[~zero_rows, None]
    return r


def apply_level_weighted(
    img: np.ndarray[Any, np.dtype[np.float64]],
    polyx: int,
    polyy: int,
    method: str,
    mask: Optional[np.ndarray[Any, np.dtype[np.bool_]]] = None,
    smoothing_window: int = 10,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Apply a weighted-region leveling method to a 2D AFM image or stack.

    Dispatcher for the level_weighted functions.

    Parameters
    ----------
    img : np.ndarray
        2-D image (H * W) or 3-D stack (N * H * W).
    polyx, polyy : int
        Polynomial orders for X (columns) and Y (rows) fits when relevant.
    method : str
        One of ``'plane'``, ``'line'``, ``'med_line'``, ``'med_line_y'``,
        or ``'smed_line'``.
    mask : Optional[np.ndarray]
        Mask with same shape as ``img`` (or H * W for single image). Non-zero
        values are treated as foreground. If ``None``, the entire image is used.
    smoothing_window : int
        Window for ``smed_line`` smoothing.

    Returns
    -------
    np.ndarray
        Leveled image with same shape as ``img`` (or stack).
    """
    arr = np.asarray(img)
    is_stack = arr.ndim == 3

    frames = arr if is_stack else arr[np.newaxis, ...]

    if mask is not None:
        mask_arr = np.asarray(mask)
        if mask_arr.ndim == 2:
            mask_arr = mask_arr[np.newaxis, ...]  # handle single image
        if mask_arr.shape != frames.shape:
            raise ValueError("mask must have the same shape as img or stack")

        # Force mask to boolean (True = included, False = excluded)
        mask_arr = mask_arr.astype(bool)
    else:
        mask_arr = None

    leveled_frames: List[np.ndarray[Any, np.dtype[np.float64]]] = []
    for frame_idx in range(frames.shape[0]):
        frame = frames[frame_idx]
        # Convert mask to boolean "include/exclude"
        frame_mask = (
            mask_arr[frame_idx]
            if mask_arr is not None
            else np.ones_like(frame, dtype=bool)
        )
        mask_bool = frame_mask.astype(bool)

        n_rows, n_cols = frame.shape
        min_area = max(1, int(0.01 * n_rows * n_cols))
        regions = _find_regions(mask_bool, min_area)

        method = method.lower()
        if method == "plane":
            leveled = level_weighted_plane(frame, regions, polyx, polyy)
        elif method == "line":
            leveled = level_weighted_line(frame, regions, polyx, polyy)
        elif method == "med_line":
            leveled = level_weighted_med_line(frame, regions)
        elif method == "med_line_y":
            leveled = level_weighted_med_line_y(frame, regions)
        elif method == "smed_line":
            leveled = level_weighted_smed_line(frame, regions, smoothing_window)
        else:
            raise ValueError(f"Unknown leveling method: {method}")

        leveled_frames.append(leveled)

    stacked = np.stack(leveled_frames, axis=0)
    return np.asarray(stacked if is_stack else stacked[0])


__all__ = [
    "apply_level_weighted",
    "level_weighted_plane",
    "level_weighted_line",
    "level_weighted_med_line",
    "level_weighted_med_line_y",
    "level_weighted_smed_line",
]
