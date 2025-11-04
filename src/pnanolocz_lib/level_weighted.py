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
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
from scipy import ndimage

if TYPE_CHECKING:
    from numpy import RankWarning
else:
    try:
        from numpy.polynomial.polyutils import RankWarning  # type: ignore
    except Exception:
        RankWarning = Warning


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
        Sample standard deviation (ddof=1) of the original indices; guaranteed
        non-zero (defaults to 1.0 when degenerate).
    """
    if indices.size == 0:
        return indices.astype(float), 0.0, 1.0

    centroid = float(indices.mean())
    scale = float(indices.std(ddof=1)) if indices.size > 1 else 1.0
    if scale == 0:
        scale = 1.0
    std_indices = (indices - centroid) / scale
    return std_indices, centroid, scale


def _polyfit_centered(
    x: np.ndarray[Any, np.dtype[np.float64]],
    y: np.ndarray[Any, np.dtype[np.float64]],
    order: int,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], Tuple[float, float]]:
    """Fit polynomial to ``y`` vs ``x`` after centering and scaling ``x``.

    This tries to replicate the MATLAB polyfit function.

    Parameters
    ----------
    x : np.ndarray
        1-D positions used as the independent variable.
    y : np.ndarray
        1-D values (dependent variable).
    order : int
        Polynomial order.

    Returns
    -------
    coeffs : np.ndarray
        Coefficients in decreasing power order compatible with ``np.polyval``.
    (centroid, scale) : tuple
        The centering and scaling applied to ``x`` (so evaluation may use the
        same parameters).
    """
    if x.size == 0 or y.size == 0 or x.size <= order:
        return np.zeros(order + 1, dtype=float), (0.0, 1.0)

    std_x, centroid, scale = _center_scale_indices(x)
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
    """Find connected foreground regions and return their flat indices.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask where True indicates foreground.
    min_area : int
        Minimum number of pixels for a region to be kept.

    Returns
    -------
    regions : list of np.ndarray
        Each element is a 1-D array of flat indices for that region.
    """
    structure = np.ones((3, 3), dtype=int)
    labeled, num_features = ndimage.label(mask, structure=structure)
    regions: List[np.ndarray[Any, np.dtype[np.int64]]] = []
    for lab in range(1, num_features + 1):
        flat_idx = np.flatnonzero(labeled.ravel() == lab)
        if flat_idx.size >= min_area:
            regions.append(flat_idx)
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
        region_pixel_counts[i] = region_indices.size  # w(i) in Nanolocz

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

    if weights.sum() == 0:
        weights = region_pixel_counts / (
            region_pixel_counts.sum() if region_pixel_counts.sum() > 0 else 1.0
        )

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

    weighted_x_coeffs = (x_poly_arr * weights[None, :]).sum(axis=1)
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
    Region-weighted per-row and per-column polynomial leveling.

    This function removes large-scale background trends from an image by fitting
    per-row and/or per-column polynomials to user-provided regions, then
    subtracting the region-weighted background from the image.

    For each region, a polynomial is fit independently to each row (if
    ``polyx > 0``) and/or each column (if ``polyy > 0``) using only pixels
    inside that region. The polynomial coefficients and the centering parameters
    (mean and scale) for each row/column are aggregated across regions using
    weights proportional to the per-row/column counts of valid pixels in each
    region. Extremely small weights (< 0.02) are nulled to reduce noise; rows or
    columns whose weights zero out are reweighted by raw pixel counts. The
    resulting weighted polynomial background is evaluated and subtracted from
    the image (rows first, then columns if both are enabled).

    Parameters
    ----------
    img : ndarray of float64, shape (H, W)
        Input image to level.
    regions : list of 1D ndarray of int64
        A list of flat (ravelled) indices specifying disjoint or overlapping
        regions within ``img``. Each array contains indices into
        ``np.ravel(img)`` (i.e., C-order flattening). Only pixels belonging to a
        given region are used to fit that region's per-row/column polynomials.
    polyx : int
        Polynomial degree for row-wise fitting. If ``polyx <= 0``, no row-wise
        leveling is performed.
    polyy : int
        Polynomial degree for column-wise fitting. If ``polyy <= 0``, no
        column-wise leveling is performed.

    Returns
    -------
    ndarray of float64, shape (H, W)
        The leveled image. If both ``polyx > 0`` and ``polyy > 0``, the result is
        the input with the row-wise background subtracted first, followed by the
        column-wise background subtraction.

    Notes
    -----
    - For each row/column and region, polynomial fitting is performed on the
      coordinate positions that have valid pixels within the region. A fit
      requires at least ``degree + 2`` valid points; otherwise, a neutral
      centering ``(0.0, 1.0)`` is recorded and coefficients are left at zero
      for that row/column in that region.
    - Region aggregation uses normalized pixel-count weights per row/column.
      Weights below 0.02 are set to 0 to suppress weak regions; if all weights
      become zero for a given row/column, raw pixel-count normalization is used
      as a fallback.
    - Polynomial evaluation is done with centered/scaled coordinates obtained
      from each fit (mean, scale) aggregated across regions using the same
      weights as for the coefficients.
    - This function relies on helper routines
      ``_polyfit_centered(x, y, degree) -> (coeffs, (mean, scale))`` and
      ``_polyval_centered(coeffs, (mean, scale), x) -> y``.

    Examples
    --------
    >>> H, W = 128, 256
    >>> img = np.random.randn(H, W).astype(float)
    >>> # Define two rectangular regions via flat indices
    >>> r1 = np.ravel_multi_index(
    ...     np.mgrid[10:60, 20:120].reshape(2, -1), dims=img.shape, order='C'
    ... )
    >>> r2 = np.ravel_multi_index(
    ...     np.mgrid[70:120, 100:220].reshape(2, -1), dims=img.shape, order='C'
    ... )
    >>> leveled = level_weighted_line(img, [r1, r2], polyx=2, polyy=1)
    """
    rows, cols = img.shape
    img_f = np.asarray(img, dtype=float)

    leveled_image = img_f.copy()

    # Row-wise polynomial fitting per region
    if polyx > 0 and len(regions) > 0:
        # MATLAB: px{k}(ii, i)  -> per-row polynomial coefficients per region
        # Python: we store a (rows, polyx+1) array per region, then stack to (rows,
        # polyx+1, n_regions)
        row_coeffs_regions: List[np.ndarray[Any, np.dtype[np.float64]]] = []
        # MATLAB Nanolocz-lib: mux{1}(ii, i) and mux{2}(ii, i) -> centering (mean,
        # scale) per row and region
        # Python: store as (rows, 2) per region, then combine with weights
        row_centering_regions: List[np.ndarray[Any, np.dtype[np.float64]]] = []
        # MATLAB: w(ii, i) -> per-row valid-pixel counts per region
        # (row_pixel_counts_regions)
        # Python: store as (rows,) per region; later stack to (rows, n_regions)
        row_pixel_counts_regions: List[np.ndarray[Any, np.dtype[np.float64]]] = []

        for region_indices in regions:
            region_masked = np.full(img_f.shape, np.nan, dtype=float)
            region_masked.flat[region_indices] = img_f.flat[region_indices]

            coeffs_for_rows = np.zeros(
                (rows, polyx + 1), dtype=float
            )  # coeffs_for_rows ~ MATLAB Nanolocz-lib px{k} matrices collected by k
            centering_for_rows = np.zeros((rows, 2), dtype=float)  #
            pixel_counts_per_row = np.zeros(rows, dtype=float)

            for row_idx in range(rows):
                valid_columns = ~np.isnan(region_masked[row_idx, :])
                pixel_counts_per_row[row_idx] = valid_columns.sum()

                if valid_columns.sum() > polyx + 1:
                    col_positions = np.flatnonzero(valid_columns).astype(float)
                    values = img_f[row_idx, valid_columns]
                    coeffs, centering = _polyfit_centered(col_positions, values, polyx)
                    coeffs_for_rows[row_idx, : coeffs.size] = coeffs
                    centering_for_rows[row_idx, :] = centering
                else:
                    centering_for_rows[row_idx, :] = (0.0, 1.0)

            row_coeffs_regions.append(coeffs_for_rows)
            row_centering_regions.append(centering_for_rows)
            row_pixel_counts_regions.append(pixel_counts_per_row)

        row_coeffs_stack = np.stack(
            row_coeffs_regions, axis=2
        )  # (rows, poly+1, n_regions)
        row_pixel_counts_array = np.stack(
            row_pixel_counts_regions, axis=1
        )  # (rows, n_regions)

        total_counts_per_row = row_pixel_counts_array.sum(axis=1, keepdims=True)
        row_weights = row_pixel_counts_array / np.where(
            total_counts_per_row == 0, 1.0, total_counts_per_row
        )
        row_weights = np.where(row_weights > 0.02, row_weights, 0.0)

        zero_weight_rows = row_weights.sum(axis=1) == 0
        if zero_weight_rows.any():
            row_weights[zero_weight_rows, :] = row_pixel_counts_array[
                zero_weight_rows, :
            ] / np.maximum(total_counts_per_row[zero_weight_rows], 1.0)

        row_weights_expanded = row_weights[:, None, :]
        weighted_row_coeffs = (row_coeffs_stack * row_weights_expanded).sum(axis=2)

        row_cent0 = np.stack([c[:, 0] for c in row_centering_regions], axis=1)
        row_cent1 = np.stack([c[:, 1] for c in row_centering_regions], axis=1)
        weighted_row_centroid = (row_cent0 * row_weights).sum(axis=1)
        weighted_row_scale = (row_cent1 * row_weights).sum(axis=1)

        # Evaluate row background and subtract
        row_background = np.zeros_like(img_f)
        col_positions_all = np.arange(cols)
        for r_idx in range(rows):
            row_background[r_idx, :] = _polyval_centered(
                weighted_row_coeffs[r_idx],
                (
                    weighted_row_centroid[r_idx],
                    (
                        weighted_row_scale[r_idx]
                        if weighted_row_scale[r_idx] != 0
                        else 1.0
                    ),
                ),
                col_positions_all,
            )

        leveled_image = img_f - row_background

    # Column-wise polynomial fitting per region
    if polyy > 0 and len(regions) > 0:
        # IN MATLAB Nanolocz-lib col_coeffs_regions is py
        col_coeffs_regions: List[np.ndarray[Any, np.dtype[np.float64]]] = []
        # IN MATLAB Nanolocz-lib col_centering_regions is muy
        col_centering_regions: List[np.ndarray[Any, np.dtype[np.float64]]] = []
        # IN MATLAB Nanolocz-lib col_pixel_counts_regions is w
        col_pixel_counts_regions: List[np.ndarray[Any, np.dtype[np.float64]]] = []

        for region_indices in regions:
            region_masked = np.full(img_f.shape, np.nan, dtype=float)
            region_masked.flat[region_indices] = img_f.flat[region_indices]

            coeffs_for_cols = np.zeros((cols, polyy + 1), dtype=float)
            centering_for_cols = np.zeros((cols, 2), dtype=float)
            pixel_counts_per_col = np.zeros(cols, dtype=float)

            for col_idx in range(cols):
                valid_rows = ~np.isnan(region_masked[:, col_idx])
                pixel_counts_per_col[col_idx] = valid_rows.sum()

                if valid_rows.sum() > polyy + 1:
                    row_positions = np.flatnonzero(valid_rows).astype(float)
                    values = img_f[valid_rows, col_idx]
                    coeffs, centering = _polyfit_centered(row_positions, values, polyy)
                    coeffs_for_cols[col_idx, : coeffs.size] = coeffs
                    centering_for_cols[col_idx, :] = centering
                else:
                    centering_for_cols[col_idx, :] = (0.0, 1.0)

            col_coeffs_regions.append(coeffs_for_cols)
            col_centering_regions.append(centering_for_cols)
            col_pixel_counts_regions.append(pixel_counts_per_col)

        col_coeffs_stack = np.stack(
            col_coeffs_regions, axis=2
        )  # (cols, poly+1, n_regions)
        col_pixel_counts_array = np.stack(
            col_pixel_counts_regions, axis=1
        )  # (cols, n_regions)

        total_counts_per_col = col_pixel_counts_array.sum(axis=1, keepdims=True)
        col_weights = col_pixel_counts_array / np.where(
            total_counts_per_col == 0, 1.0, total_counts_per_col
        )
        col_weights = np.where(col_weights > 0.02, col_weights, 0.0)

        zero_weight_cols = col_weights.sum(axis=1) == 0
        if zero_weight_cols.any():
            col_weights[zero_weight_cols, :] = col_pixel_counts_array[
                zero_weight_cols, :
            ] / np.maximum(total_counts_per_col[zero_weight_cols], 1.0)

        col_weights_expanded = col_weights[:, None, :]
        weighted_col_coeffs = (col_coeffs_stack * col_weights_expanded).sum(axis=2)

        # After computing weighted_row_coeffs and weighted_col_coeffs
        # Force constant term to zero like MATLAB Nanolocz:
        weighted_row_coeffs[:, -1] = 0.0
        weighted_col_coeffs[:, -1] = 0.0

        col_cent0 = np.stack([c[:, 0] for c in col_centering_regions], axis=1)
        col_cent1 = np.stack([c[:, 1] for c in col_centering_regions], axis=1)
        weighted_col_centroid = (col_cent0 * col_weights).sum(axis=1)
        weighted_col_scale = (col_cent1 * col_weights).sum(axis=1)

        # Evaluate column background and subtract
        col_background = np.zeros_like(img_f)
        row_positions_all = np.arange(rows)
        for c_idx in range(cols):
            col_background[:, c_idx] = _polyval_centered(
                weighted_col_coeffs[c_idx],
                (
                    weighted_col_centroid[c_idx],
                    (
                        weighted_col_scale[c_idx]
                        if weighted_col_scale[c_idx] != 0
                        else 1.0
                    ),
                ),
                row_positions_all,
            )

        leveled_image = leveled_image - col_background

    return np.asarray(leveled_image)


def level_weighted_med_line(
    image: np.ndarray[Any, np.dtype[np.float64]],
    regions: List[np.ndarray[Any, np.dtype[np.int64]]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Region-weighted median line subtraction along image rows.

    Computes a region-weighted median background per row and subtracts it
    from the image. Behaviour mirrors the MATLAB ``med_line`` case but uses
    descriptive variable names and NumPy-style docstrings.

    Parameters
    ----------
    image : np.ndarray
        2-D AFM image (rows * columns).
    regions : list of np.ndarray
        List of flat-index arrays describing connected foreground regions.

    Returns
    -------
    np.ndarray
        Row-leveled image (float64).
    """
    image_float = np.asarray(image, dtype=float)
    n_rows, n_cols = image_float.shape
    n_regions = len(regions)

    # Per-row counts and per-row median offsets for each region
    per_row_counts = np.zeros((n_rows, n_regions), dtype=float)
    per_row_offsets = np.zeros((n_rows, n_regions), dtype=float)
    region_baselines = np.zeros(n_regions, dtype=float)

    for r_idx, region_indices in enumerate(regions):
        region_masked = np.full(image_float.shape, np.nan, dtype=float)
        region_masked.flat[region_indices] = image_float.flat[region_indices]

        region_baselines[r_idx] = np.nanmedian(region_masked)

        for row_idx in range(n_rows):
            valid = ~np.isnan(region_masked[row_idx, :])
            per_row_counts[row_idx, r_idx] = valid.sum()
            if valid.sum() > 2:
                per_row_offsets[row_idx, r_idx] = (
                    np.nanmedian(image_float[row_idx, valid]) - region_baselines[r_idx]
                )
            else:
                per_row_offsets[row_idx, r_idx] = -region_baselines[r_idx]

    # Compute normalized weights per row
    totals = per_row_counts.sum(axis=1, keepdims=True)
    denom = np.where(totals == 0, 1.0, totals)
    weights = per_row_counts / denom
    weights = np.where(weights > 0.02, weights, 0.0)

    zero_weight_rows = weights.sum(axis=1) == 0
    if zero_weight_rows.any():
        weights[zero_weight_rows, :] = per_row_counts[zero_weight_rows, :] / np.maximum(
            denom[zero_weight_rows], 1.0
        )

    weighted_row_background = (weights * per_row_offsets).sum(axis=1)
    has_data = per_row_counts.sum(axis=1) > 0

    leveled = image_float.copy()
    leveled[has_data, :] = (
        image_float[has_data, :] - weighted_row_background[has_data, None]
    )
    return np.asarray(leveled)


def level_weighted_med_line_y(
    image: np.ndarray[Any, np.dtype[np.float64]],
    regions: List[np.ndarray[Any, np.dtype[np.int64]]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Region-weighted median line subtraction along image columns.

    Parameters
    ----------
    image : np.ndarray
        2-D AFM image (rows * columns).
    regions : list of np.ndarray
        List of flat-index arrays describing connected foreground regions.

    Returns
    -------
    np.ndarray
        Column-leveled image (float64).
    """
    image_float = np.asarray(image, dtype=float)
    n_rows, n_cols = image_float.shape
    n_regions = len(regions)

    per_col_counts = np.zeros((n_cols, n_regions), dtype=float)
    per_col_offsets = np.zeros((n_cols, n_regions), dtype=float)
    region_baselines = np.zeros(n_regions, dtype=float)

    for r_idx, region_indices in enumerate(regions):
        region_masked = np.full(image_float.shape, np.nan, dtype=float)
        region_masked.flat[region_indices] = image_float.flat[region_indices]

        region_baselines[r_idx] = np.nanmedian(region_masked)

        for col_idx in range(n_cols):
            valid = ~np.isnan(region_masked[:, col_idx])
            per_col_counts[col_idx, r_idx] = valid.sum()
            if valid.sum() > 2:
                per_col_offsets[col_idx, r_idx] = (
                    np.nanmedian(image_float[valid, col_idx]) - region_baselines[r_idx]
                )
            else:
                per_col_offsets[col_idx, r_idx] = -region_baselines[r_idx]

    totals = per_col_counts.sum(axis=1, keepdims=True)
    denom = np.where(totals == 0, 1.0, totals)
    weights = per_col_counts / denom
    weights = np.where(weights > 0.02, weights, 0.0)

    zero_weight_cols = weights.sum(axis=1) == 0
    if zero_weight_cols.any():
        weights[zero_weight_cols, :] = per_col_counts[zero_weight_cols, :] / np.maximum(
            denom[zero_weight_cols], 1.0
        )

    weighted_col_background = (weights * per_col_offsets).sum(axis=1)
    has_data = per_col_counts.sum(axis=1) > 0

    leveled = image_float.copy()
    cols_with_data = has_data
    leveled[:, cols_with_data] = (
        image_float[:, cols_with_data]
        - weighted_col_background[cols_with_data][None, :]
    )
    return np.asarray(leveled)


def level_weighted_smed_line(
    image: np.ndarray[Any, np.dtype[np.float64]],
    regions: List[np.ndarray[Any, np.dtype[np.int64]]],
    smoothing_window: int = 10,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Region-weighted smoothed median line subtraction along rows.

    Computes a weighted median profile per row and then subtracts the difference
    between that profile and a moving-median-smoothed version of it (MATLAB
    ``smed_line`` behaviour).

    Parameters
    ----------
    image : np.ndarray
        2-D AFM image (rows * columns).
    regions : list of np.ndarray
        List of flat-index arrays describing connected foreground regions.
    smoothing_window : int, optional
        Window length for moving-median smoothing (default 10).

    Returns
    -------
    np.ndarray
        Smoothed-median-leveled image.
    """
    image_float = np.asarray(image, dtype=float)
    n_rows, n_cols = image_float.shape
    n_regions = len(regions)

    median_per_row = np.zeros(n_rows, dtype=float)
    per_row_counts = np.zeros((n_rows, n_regions), dtype=float)
    region_baselines = np.zeros(n_regions, dtype=float)

    for r_idx, region_indices in enumerate(regions):
        region_masked = np.full(image_float.shape, np.nan, dtype=float)
        region_masked.flat[region_indices] = image_float.flat[region_indices]

        region_baselines[r_idx] = np.nanmedian(region_masked)

        for row_idx in range(n_rows):
            valid = ~np.isnan(region_masked[row_idx, :])
            per_row_counts[row_idx, r_idx] = valid.sum()
            if valid.sum() > 2:
                median_per_row[row_idx] = np.nanmedian(image_float[row_idx, valid])
            else:
                median_per_row[row_idx] = -region_baselines[r_idx]

    totals = per_row_counts.sum(axis=1, keepdims=True)
    denom = np.where(totals == 0, 1.0, totals)
    weights = per_row_counts / denom
    weights = np.where(weights > 0.02, weights, 0.0)

    zero_weight_rows = weights.sum(axis=1) == 0
    if zero_weight_rows.any():
        weights[zero_weight_rows, :] = per_row_counts[zero_weight_rows, :] / np.maximum(
            denom[zero_weight_rows], 1.0
        )

    weighted_background = (weights * median_per_row[:, None]).sum(axis=1)

    # moving median smoothing
    k = int(smoothing_window)
    if k <= 1:
        smoothed = weighted_background.copy()
    else:
        pad = k // 2
        padded = np.pad(weighted_background, pad, mode="edge")
        smoothed = np.empty_like(weighted_background)
        for i in range(n_rows):
            smoothed[i] = np.median(padded[i : i + k])

    return np.asarray(image_float - (weighted_background[:, None] - smoothed[:, None]))


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
            mask_arr = mask_arr[np.newaxis, ...]
        if mask_arr.shape != frames.shape:
            raise ValueError("mask must have the same shape as img or stack")
    else:
        mask_arr = None

    leveled_frames: List[np.ndarray[Any, np.dtype[np.float64]]] = []
    for frame_idx in range(frames.shape[0]):
        frame = frames[frame_idx]
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
