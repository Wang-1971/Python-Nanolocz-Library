"""
Weighted-region AFM image flattening and background leveling tools.

This module provides a Python port of the MATLAB ``level_weighted.m``
function used in the NanoLocz workflow. It implements region-wise weighted
polynomial and median-based background estimation for Atomic Force Microscopy
(AFM) images, enabling correction of multi-region drift, structured background,
and non-uniform masking effects.

All public leveling functions accept an *exclusion mask* (same convention as
``pnanolocz_lib.thresholder``): ``True`` = excluded, ``False`` = valid. Excluded
pixels are omitted from region formation and fitting using MATLAB-style NaN-outside
semantics (i.e., excluded pixels behave like NaN during fitting) but are
preserved in the output array.

The implementation is a Python port of the MATLAB NanoLocz Library:
    https://github.com/George-R-Heath/NanoLocz-Matlab-Library
Original MATLAB code by George Heath, University of Leeds.

The original Nanolocz-lib script was adapted from FindSteps.m and PolyfitLineMasked.m
scripts from the SPIW project (<https://sourceforge.net/projects/spiw/>) and combined
with NanoLocz leveling methods.

MATLAB alignment
----------------
This Python version aims for algorithmic and numerical alignment with the MATLAB
reference implementation. Due to differences in underlying numerical libraries
(NumPy/SciPy vs MATLAB), polynomial conditioning, floating-point behaviour, and
edge-case handling, results may not be bit-for-bit identical. Where relevant,
functions document any intentional deviations adopted to match the reference
NanoLocz outputs.

Supported Leveling Methods
--------------------------
- 'plane'       : Region-weighted polynomial plane subtraction in X and Y.
- 'line'        : Region-weighted row/column polynomial leveling.
- 'med_line'    : Region-weighted row-wise median line flattening.
- 'med_line_y'  : Region-weighted column-wise median line flattening.
- 'smed_line'   : Region-weighted smoothed median line subtraction.

Dispatcher and usage
--------------------
The primary entry point is the ``apply_level_weighted`` function, which dispatches to
the appropriate weighted leveling routine based on the requested method and applies
it frame-by-frame if a 3D stack is provided. All methods accept an optional
exclusion mask, where excluded pixels are excluded from fitting operations but
preserved in the output.

Stacks
------
Functions operate on single images with shape ``(H, W)`` and stacks with shape
``(N, H, W)``, processing stacks frame-by-frame. If a 2D mask is provided for a
single image it is used directly; for stacks, masks must match the stack shape
(or be promoted appropriately by the dispatcher).

Examples
--------
>>> from pnanolocz_lib.level_weighted import apply_level_weighted
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


def _validity_mask(
    arr: np.ndarray[Any, np.dtype[np.float64]],
    mask_excl: np.ndarray[Any, np.dtype[np.bool_]] | None,
    *,
    name: str = "mask",
) -> np.ndarray[Any, np.dtype[np.bool_]]:
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
        return np.asarray(finite, dtype=np.bool_)

    m_excl = np.asarray(mask_excl, dtype=np.bool_)
    if m_excl.shape != arr.shape:
        raise ValueError(
            f"{name} shape {m_excl.shape} must match img shape {arr.shape}"
        )

    return np.asarray((~m_excl) & finite, dtype=np.bool_)


def _polyfit_centered(
    x: np.ndarray[Any, np.dtype[np.float64]],
    y: np.ndarray[Any, np.dtype[np.float64]],
    order: int,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], tuple[float, float]]:
    """
    Fit polynomial to y vs x after centering and scaling x.

    Equivalent to MATLAB polyfit with mu output.

    Returns
    -------
      coeffs: polynomial coefficients (highest power first)
      (centroid, scale): centering and scaling applied to x
    """
    if x.size == 0 or y.size == 0 or x.size <= order:
        return np.zeros(order + 1, dtype=float), (0.0, 1.0)

    centroid = float(np.nanmean(x))
    scale = float(np.nanstd(x, ddof=1)) if x.size > 1 else 1.0
    if scale == 0:
        scale = 1.0

    std_x = (x - centroid) / scale

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RankWarning)
        coeffs = np.polyfit(std_x, y, order)

    return np.asarray(coeffs, dtype=np.float64), (centroid, scale)


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
    return np.asarray(np.polyval(coeffs, std_points), dtype=np.float64)


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
        Compute min_area exactly as MATLAB does:
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
        lab
        for lab, area in zip(range(1, num_features + 1), areas, strict=False)
        if area >= min_area
    ]

    # produce flat indices for each kept label (row-major to match numpy.flat)
    regions: List[np.ndarray[Any, np.dtype[np.int64]]] = []
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
    Subtract a region-weighted polynomial plane fitted along rows and columns.

    This method reproduces the NanoLocz MATLAB ``level_weighted(...,'plane')``
    behavior by fitting polynomials to per-region mean intensity profiles in
    the X-direction (column means) and Y-direction (row means), then forming a
    weighted average of the per-region polynomial models. Weights are
    proportional to region pixel count and are zeroed for regions contributing
    less than 2% of the total included area.

    Parameters
    ----------
    img : ndarray
        2D AFM image with shape ``(H, W)``. Values must be finite to contribute
        to fitting; non-finite values are treated as excluded.
    regions : list of ndarray
        Foreground regions as flat indices (NumPy row-major / ``order='C'``).
        Regions are typically computed from the validity mask (included pixels)
        using 8-connectivity, mirroring MATLAB ``bwconncomp(mask, 8)``.
    polyx : int
        Polynomial order used for the X-direction fit (column profile).
    polyy : int
        Polynomial order used for the Y-direction fit (row profile).

    Returns
    -------
    leveled : ndarray
        Leveled image with the same shape as ``img`` and dtype ``float64``.

    Notes
    -----
    - Region weights are computed as ``w_i / sum(w)`` and then thresholded with
      ``W_i = 0`` for ``W_i <= 0.02`` (MATLAB behavior). Weights are not
      renormalized after thresholding.
    - Polynomial fits use centering/scaling (MATLAB ``polyfit`` ``mu`` output)
      so that evaluation can reproduce MATLAB-style numerical conditioning.
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


def level_weighted_line(
    img: np.ndarray[Any, np.dtype[np.float64]],
    regions: List[np.ndarray[Any, np.dtype[np.int64]]],
    polyx: int,
    polyy: int,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Subtract a region-weighted polynomial line background along rows and columns.

    This method reproduces the NanoLocz MATLAB ``level_weighted(...,'line')``
    behavior by fitting polynomials within each region separately for each row
    (X-direction) and/or each column (Y-direction), then combining per-region
    fits using per-row (or per-column) weights proportional to the number of
    region pixels present in that row/column. Regions contributing less than
    2% of the row/column support are excluded from the weighted sum.

    Parameters
    ----------
    img : ndarray
        2D AFM image with shape ``(H, W)``.
    regions : list of ndarray
        Foreground regions as flat indices (NumPy row-major / ``order='C'``).
    polyx : int
        Polynomial order for row-wise fits (X-direction). If ``polyx <= 0``,
        no row-wise correction is applied.
    polyy : int
        Polynomial order for column-wise fits (Y-direction). If ``polyy <= 0``,
        no column-wise correction is applied.

    Returns
    -------
    leveled : ndarray
        Leveled image with the same shape as ``img`` and dtype ``float64``.

    Notes
    -----
    - Fitting uses 1-based coordinates for evaluation grids (``1..W`` and
      ``1..H``) to match MATLAB indexing conventions used in the reference
      implementation.
    - After computing weighted coefficients per row/column, the constant term
      is set to zero (MATLAB behavior: ``px_w(:,end)=0`` / ``py_w(:,end)=0``).
    - Rows/columns whose synthesized background contains NaNs are set to 0
      before subtraction (MATLAB update noted: "set line fit NaNs = 0").
    """
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
    Subtract a region-weighted row-wise median baseline.

    This method mirrors the NanoLocz MATLAB ``level_weighted(...,'med_line')``
    behavior. For each region a region-wide background level is computed
    (median over region pixels). For each row and region, a row median is
    computed over the region pixels in that row; the region background is then
    removed to form a per-region row offset. Offsets are combined across
    regions using per-row weights proportional to the number of region pixels
    contributing in that row.

    Parameters
    ----------
    img : ndarray
        2D AFM image with shape ``(H, W)``.
    regions : list of ndarray
        Foreground regions as flat indices (NumPy row-major / ``order='C'``).

    Returns
    -------
    leveled : ndarray
        Leveled image with the same shape as ``img`` and dtype ``float64``.

    Notes
    -----
    - For a given row and region, if fewer than 3 region pixels are present,
      the per-region row offset falls back to ``-bg_region`` (MATLAB behavior).
    - Rows with no coverage from any region are left unchanged.
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
    Subtract a region-weighted column-wise median baseline.

    This method mirrors the NanoLocz MATLAB ``level_weighted(...,'med_line_y')``
    behavior. For each region a region-wide background level is computed
    (median over region pixels). For each column and region, a column median
    is computed over the region pixels in that column; the region background is
    then removed to form a per-region column offset. Offsets are combined
    across regions using per-column weights proportional to the number of
    region pixels contributing in that column.

    Parameters
    ----------
    img : ndarray
        2D AFM image with shape ``(H, W)``.
    regions : list of ndarray
        Foreground regions as flat indices (NumPy row-major / ``order='C'``).

    Returns
    -------
    leveled : ndarray
        Leveled image with the same shape as ``img`` and dtype ``float64``.

    Notes
    -----
    - For a given column and region, if fewer than 3 region pixels are present,
      the per-region column offset falls back to ``-bg_region`` (MATLAB behavior).
    - Columns with no coverage from any region are left unchanged.
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
def _movmedian_centered_includenan(
    x: np.ndarray[Any, np.dtype[np.float64]],
    w: int,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Compute a centered moving median with NaN inclusion and symmetric edge shrinking.

    MATLAB movmedian default: include NaNs. Even w: window centered about current &
    previous (left=w//2, right=w-w//2). Shrink symmetrically at edges.
    """
    n = x.size
    out = np.empty(n, dtype=np.float64)
    left = w // 2
    right = w - left
    for i in range(n):
        start = max(0, i - left)
        end = min(n, i + right)  # end is exclusive
        win = x[start:end]
        # include NaNs => if any NaN in window, median becomes NaN (like MATLAB default)
        out[i] = np.median(win)
    return np.asarray(out, dtype=np.float64)


def level_weighted_smed_line(
    img: np.ndarray[Any, np.dtype[np.float64]],
    regions: List[np.ndarray[Any, np.dtype[np.int64]]],
    smoothing_window: int = 10,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Subtract a smoothed region-weighted row-wise median baseline.

    This method mirrors the NanoLocz MATLAB ``level_weighted(...,'smed_line')``
    behavior. It first computes a region-weighted per-row baseline using row
    medians (without subtracting the per-region background in the main path),
    then applies a centered moving-median smoothing operation to that baseline
    before subtracting it from the image.

    Parameters
    ----------
    img : ndarray
        2D AFM image with shape ``(H, W)``.
    regions : list of ndarray
        Foreground regions as flat indices (NumPy row-major / ``order='C'``).
    smoothing_window : int, default 10
        Window length for the moving-median smoothing of the baseline. The
        implementation follows MATLAB ``movmedian`` defaults: include NaNs and
        use a centered window with MATLAB's even-window convention.

    Returns
    -------
    leveled : ndarray
        Leveled image with the same shape as ``img`` and dtype ``float64``.

    Notes
    -----
    - NaNs produced during smoothing are replaced with 0 prior to subtraction
      (consistent with the MATLAB-style "NaNs -> 0" handling in this workflow).
    - Rows with no region coverage are left unchanged.
    """
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
    W = np.zeros_like(w, dtype=float)
    np.divide(w, den, out=W, where=(den != 0))  # may create NaNs when den==0 (expected)
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
    Apply a weighted-region leveling method to an AFM image or stack.

    This is the primary public entry point for weighted-region leveling. The
    function converts the user-provided *exclusion mask* into an internal
    validity mask (valid pixels are those that are finite and not excluded),
    finds connected regions on the valid pixels using 8-connectivity, and then
    dispatches to the requested weighted leveling method.

    Parameters
    ----------
    img : ndarray
        Input image or stack. Accepted shapes are ``(H, W)`` for a single image
        and ``(N, H, W)`` for a frame stack. The returned array matches the
        input shape.
    polyx : int
        Polynomial order for X-direction fits (columns/row-wise) for methods
        that use polynomial fitting (e.g., ``'plane'`` and ``'line'``).
    polyy : int
        Polynomial order for Y-direction fits (rows/column-wise) for methods
        that use polynomial fitting (e.g., ``'plane'`` and ``'line'``).
    method : str
        Leveling method name. Supported values are:

        - ``'plane'``      : region-weighted polynomial plane subtraction
        - ``'line'``       : region-weighted polynomial line subtraction
        - ``'med_line'``   : region-weighted row-wise median subtraction
        - ``'med_line_y'`` : region-weighted column-wise median subtraction
        - ``'smed_line'``  : smoothed region-weighted row-wise median subtraction
    mask : ndarray of bool, optional
        *Exclusion mask* with the same shape as ``img`` (or ``(H, W)`` for a
        single-image mask applied per frame). Mask convention is:
        ``True = excluded``, ``False = valid``. Excluded pixels are omitted from
        region formation and fitting but are preserved in the output array.
        Non-finite pixels in ``img`` are always treated as excluded.
    smoothing_window : int, default 10
        Moving-median window length used only for ``method='smed_line'``.

    Returns
    -------
    leveled : ndarray
        Leveled image or stack with the same shape as ``img`` and dtype
        ``float64``.

    Raises
    ------
    ValueError
        If ``mask`` has an incompatible shape or if ``method`` is not
        recognized.

    Notes
    -----
    - Regions are computed per frame with 8-connectivity, matching MATLAB
      ``bwconncomp(mask, 8)``.
    - A MATLAB-style minimum region area is enforced via:
      ``min_area = max(1, floor(0.01 * H * W))``.
    """
    arr = np.asarray(img, dtype=np.float64)
    is_stack = arr.ndim == 3
    frames = arr if is_stack else arr[np.newaxis, ...]
    leveled_frames: List[np.ndarray[Any, np.dtype[np.float64]]] = []

    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.ndim == 2:
            mask_arr = mask_arr[np.newaxis, ...]
        if mask_arr.shape != frames.shape:
            raise ValueError("mask must have the same shape as img or stack")
    else:
        mask_arr = None

    for frame_idx in range(frames.shape[0]):
        frame = frames[frame_idx]

        # Convert EXCLUSION mask -> validity mask (True = valid/included)
        m_valid = _validity_mask(
            frame, None if mask_arr is None else mask_arr[frame_idx]
        )

        # Region finding expects True = foreground => use validity
        n_rows, n_cols = frame.shape
        min_area = max(1, int(np.floor(0.01 * n_rows * n_cols)))
        regions = _find_regions(m_valid, min_area)

        method_lc = method.lower()
        if method_lc == "plane":
            leveled = level_weighted_plane(frame, regions, polyx, polyy)
        elif method_lc == "line":
            leveled = level_weighted_line(frame, regions, polyx, polyy)
        elif method_lc == "med_line":
            leveled = level_weighted_med_line(frame, regions)
        elif method_lc == "med_line_y":
            leveled = level_weighted_med_line_y(frame, regions)
        elif method_lc == "smed_line":
            leveled = level_weighted_smed_line(frame, regions, smoothing_window)
        else:
            raise ValueError(f"Unknown leveling method: {method_lc}")

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
