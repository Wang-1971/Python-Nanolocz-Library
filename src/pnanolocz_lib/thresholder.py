# mypy: disallow_untyped_calls = False
"""
Image thresholding and edge detection tools for AFM data.

This module provides a unified interface for multiple thresholding and edge-
detection methods applied to AFM images or stacks. Each method returns a mask
with `NaN` in excluded regions.

Supported Methods
-----------------
- selection     : Pass-through user-provided mask
- histogram     : Intensity-based threshold limits
- otsu          : Single-level Otsu thresholding
- 2 level otsu  : Two-level Otsu thresholding (unimplemented)
- auto edges    : Sobel gradient + morphological cleanup
- hist edges    : Histogram threshold + edge detection
- otsu edges    : Otsu threshold + edge detection
- otsu skel     : Otsu threshold + skeletonization
- hist skel     : Histogram threshold + skeletonization
- line_step     : Step detection along each row
- adaptive      : Adaptive Sobel edge detection + morphology

Usage
-----
>>> from pnanolocz_lib.filters.thresholder import thresholder, histogram, otsu
>>> mask = histogram(img, limits=(0, 1), invert=False)

>>> mask_otsu = thresholder(img, 'otsu', invert = False)

Parameters
----------
img : ndarray
    2D or 3D AFM image data.
method : str
    One of the supported methods listed above.
limits : tuple or list
    Threshold parameters specific to method. E.g., (min, max) for 'histogram'.
invert : bool
    If True, invert the resulting mask (NaN regions become valid and vice
    versa).

Returns
-------
mask : ndarray
    Boolean or NaN-masked array of same shape as img.

Authors
-------
George Heath, University of Leeds (2025)
D. E. Rollins, University of Leeds (2025)

This module is part of the pNanoLocz-Lib Python library for AFM analysis.
"""

from typing import Callable, TypeVar

import numpy as np
import ruptures as rpt
import sknw
from scipy.ndimage import gaussian_filter, sobel
from skimage.filters import threshold_otsu
from skimage.morphology import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    disk,
    remove_small_objects,
    skeletonize,
    thin,
)

# Map method names to handler functions
_METHOD_MAP = {}

F = TypeVar("F", bound=Callable[..., np.ndarray])


def _register(name: str) -> Callable[[F], F]:
    """Decorator to register a threshold method."""

    def decorator(func: F) -> F:
        _METHOD_MAP[name.lower()] = func
        return func

    return decorator


def to_nan_mask(binary_mask: np.ndarray) -> np.ndarray:
    """
    Convert a boolean mask to a float mask with NaNs in False positions.

    Parameters
    ----------
    binary_mask : np.ndarray
        Boolean array where True indicates valid regions.

    Returns
    -------
    np.ndarray
        Float array where True becomes 1.0 and False becomes NaN.
    """
    mask = binary_mask.astype(float)
    mask[~binary_mask] = np.nan
    return mask


def prune_skeleton_min_branch_length(
    skel: np.ndarray, min_branch_length: int
) -> np.ndarray:
    """
    Prune branches shorter than min_branch_length from a skeleton image.

    Parameters
    ----------
    skel : ndarray
        Binary skeleton image (bool or 0/1).
    min_branch_length : int
        Minimum branch length to keep.

    Returns
    -------
    pruned_skel : ndarray
        Binary skeleton with short branches removed.
    """
    # Build graph from skeleton
    graph = sknw.build_sknw(skel.astype(np.uint8), multi=True)

    # Remove edges shorter than min_branch_length
    for s, e, k in list(graph.edges(keys=True)):
        edge = graph[s][e][k]
        if edge["weight"] < min_branch_length:
            graph.remove_edge(s, e, k)

    # Remove isolated nodes (with no edges)
    isolated_nodes = [
        node for node, degree in dict(graph.degree()).items() if degree == 0
    ]
    graph.remove_nodes_from(isolated_nodes)

    # Reconstruct skeleton from graph edges
    pruned_skel = np.zeros_like(skel, dtype=bool)
    for _s, _e, _k, data in graph.edges(keys=True, data=True):
        pts = data["pts"]
        pruned_skel[pts[:, 0], pts[:, 1]] = True

    return pruned_skel


@_register("selection")
def selection(
    img: np.ndarray, limits: tuple[float, float] | list[float] | str | None = None
) -> np.ndarray:
    """
    Pass-through user-provided mask (interpreted as boolean).

    Parameters
    ----------
    img : np.ndarray
        Input mask image. Non-zero or True values are considered selected.
    limits : None
        Not used for this method.

    Returns
    -------
    np.ndarray
        Mask with NaNs where the input mask is False or zero.
    """
    mask = img.astype(bool)
    return to_nan_mask(mask)


@_register("histogram")
def histogram(
    img: np.ndarray, limits: tuple[float, float] | list[float] | str | None = None
) -> np.ndarray:
    """
    Threshold image based on intensity limits.

    Parameters
    ----------
    img : np.ndarray
        Input image to threshold.
    limits : tuple of float
        Intensity range (low, high) to keep.

    Returns
    -------
    np.ndarray
        Mask with True inside limits and NaN outside.

    Raises
    ------
    ValueError
        If limits are not provided.
    """
    if limits is None:
        raise ValueError("limits must be provided for histogram method")
    if isinstance(limits, (tuple, list)) and len(limits) == 2:
        low, high = limits
    else:
        raise ValueError("limits must be a tuple or list of 2 elements")

    mask = (img >= low) & (img <= high)
    return to_nan_mask(mask)


@_register("otsu")
def otsu(
    img: np.ndarray, limits: tuple[float, float] | list[float] | str | None = None
) -> np.ndarray:
    """
    Apply single-level Otsu thresholding.

    Parameters
    ----------
    img : np.ndarray
        Input image to threshold.
    limits : None
        Not used for this method.

    Returns
    -------
    np.ndarray
        Mask with True for pixels below Otsu threshold and NaN elsewhere.
    """
    thresh = threshold_otsu(img)
    mask = img <= thresh
    return to_nan_mask(mask)


@_register("auto edges")
def auto_edges(
    img: np.ndarray, limits: tuple[float, float] | list[float] | str | None = None
) -> np.ndarray:
    """
    Detect edges using Sobel gradient and morphological filtering.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    limits : None
        Not used for this method.

    Returns
    -------
    np.ndarray
        Mask with NaNs in edge regions and 1 elsewhere.
    """
    sm = gaussian_filter(img, sigma=2)
    gx = sobel(sm, axis=1)
    gy = sobel(sm, axis=0)
    grad = 2 * gx**2 + gy**2
    thresh = grad.min() + (grad.mean() - grad.min()) * 1.5
    bw = grad > thresh
    bw = remove_small_objects(bw, min_size=100)
    bw = ~remove_small_objects(~bw, min_size=50)
    bw = binary_closing(bw, footprint=disk(5))
    bw = binary_dilation(bw, footprint=disk(5))
    bw = binary_closing(bw, footprint=disk(5))
    return to_nan_mask(~bw)


@_register("hist edges")
def hist_edges(
    img: np.ndarray,
    limits: tuple[float, float] | list[float] | str | None = None,
) -> np.ndarray:
    """
    Detect edges by thresholding with histogram limits and morphological operations.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    limits : tuple of float
        Intensity range (low, high) to exclude.

    Returns
    -------
    np.ndarray
        Mask with NaNs on detected edges and 1 elsewhere.
    """
    sm = gaussian_filter(img, sigma=2)
    if isinstance(limits, (tuple, list)) and len(limits) == 2:
        low, high = limits
    else:
        raise ValueError("limits must be a tuple or list of 2 elements")

    thresh_mask = (sm >= low) & (sm <= high)
    edges = np.zeros_like(img, dtype=bool)

    edge_mask = ~thresh_mask
    edges = binary_erosion(edge_mask) ^ edge_mask
    edges = binary_dilation(edges, footprint=disk(3))

    return to_nan_mask(~edges)


@_register("otsu edges")
def otsu_edges(
    img: np.ndarray,
    limits: tuple[float, float] | list[float] | str | None = None,
) -> np.ndarray:
    """
    Detect edges after Otsu thresholding using morphological operations.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    limits : None
        Not used for this method.

    Returns
    -------
    np.ndarray
        Mask with NaNs on edges and 1 elsewhere.
    """
    sm = gaussian_filter(img, sigma=2)
    thresh = threshold_otsu(sm)
    binary = sm <= thresh

    def process_slice(slice_: np.ndarray) -> np.ndarray:
        e = binary_erosion(~slice_) ^ ~slice_
        e = remove_small_objects(e, 100)
        e = ~remove_small_objects(~e, 50)
        return np.asarray(binary_dilation(e, footprint=disk(3)))

    edges = process_slice(binary)

    return to_nan_mask(~edges)


@_register("otsu skel")
def otsu_skel(
    img: np.ndarray, limits: tuple[float, float] | list[float] | str | None = None
) -> np.ndarray:
    """
    Skeletonize regions selected by Otsu thresholding.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    limits : None
        Not used for this method.

    Returns
    -------
    np.ndarray
        Skeleton mask with NaNs outside skeleton.
    """
    from skimage.measure import label

    sm = gaussian_filter(img, sigma=2)
    thresh = threshold_otsu(sm)
    binary = ~(sm <= thresh)
    mbl = 10  # Minimum branch length

    def _process_slice(slice_: np.ndarray) -> np.ndarray:
        labeled = label(slice_)
        thin_mask = thin(labeled)
        skel = skeletonize(thin_mask)

        # Prune short branches
        pruned = prune_skeleton_min_branch_length(skel, min_branch_length=mbl)

        # Optional further cleaning similar to bwmorph spur/clean
        return np.asarray(binary_dilation(pruned))

    skeleton = _process_slice(binary)

    return to_nan_mask(~skeleton)


@_register("hist skel")
def hist_skel(
    img: np.ndarray, limits: tuple[float, float] | list[float] | str | None = None
) -> np.ndarray:
    """
    Skeletonize regions selected by histogram thresholding.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    limits : tuple of float
        Intensity range (low, high) to exclude.

    Returns
    -------
    np.ndarray
        Skeleton mask with NaNs outside skeleton.
    """
    from skimage.measure import label

    sm = gaussian_filter(img, sigma=2)
    if isinstance(limits, (tuple, list)) and len(limits) == 2:
        low, high = limits
    else:
        raise ValueError("limits must be a tuple or list of 2 elements")

    binary = ~((sm >= low) & (sm <= high))
    mbl = 10  # Minimum branch length

    def _process_slice(slice_: np.ndarray) -> np.ndarray:
        labeled = label(slice_)
        thin_mask = thin(labeled)
        skel = skeletonize(thin_mask)

        # Prune short branches
        pruned = prune_skeleton_min_branch_length(skel, min_branch_length=mbl)

        # Optional further cleaning here if desired

        return np.asarray(binary_dilation(pruned))

    skeleton = _process_slice(binary)

    return to_nan_mask(~skeleton)


@_register("line_step")
def line_step(
    img: np.ndarray, limits: tuple[float, float] | list[float] | str | None = None
) -> np.ndarray:
    """
    Detect step changes along each row using PELT change point detection.

    Parameters
    ----------
    img : np.ndarray
        Input 2D image array.
    limits : tuple of float
        Parameters for step detection. Second element is used as penalty
        threshold.

    Returns
    -------
    np.ndarray
        Mask with detected step regions marked as 1 and NaN elsewhere.
    """
    mask = np.full_like(img, np.nan, dtype=float)
    if limits is None:
        raise ValueError("limits must be provided")

    if isinstance(limits, (tuple, list)):
        threshold = limits[1]
    else:
        raise TypeError("limits must be a tuple or list of floats")

    for j in range(img.shape[0]):
        x = img[j, :]
        try:
            model = rpt.Pelt(model="l2").fit(x)
            cps = model.predict(pen=threshold)
            cps = [pt for pt in cps if 4 <= pt < len(x) - 4]
        except Exception:
            cps = []

        xp = np.full_like(x, np.nan)
        if cps:
            cps = [0] + cps
            for i in range(len(cps) - 1):
                left = cps[i]
                right = cps[i + 1]
                mean_l = x[left:right].mean()
                mean_r = x[right : min(right + 3, len(x))].mean()  # noqa
                if mean_l < mean_r:
                    xp[left:right] = 1
        else:
            xp[:] = 1
        mask[j, :] = xp

    return mask


def thresholder(
    img: np.ndarray,
    method: str,
    limits: tuple[float, float] | list[float] | str | list[float] | str | None = None,
    invert: bool = False,
) -> np.ndarray:
    """
    Apply a thresholding or edge detection method to an image or stack.

    Parameters
    ----------
    img : np.ndarray
        Input 2D or 3D AFM image data.
    method : str
        Thresholding method name. Must be one of the registered methods.
    limits : tuple, list, str, or None, optional
        Parameters specific to the method (default is None).
    invert : bool, optional
        If True, invert the mask (default is False).

    Returns
    -------
    np.ndarray
        Masked array with NaNs in excluded regions.
    """
    method = method.lower()
    if method not in _METHOD_MAP:
        raise ValueError(f"Unknown thresholding method: {method}")

    func = _METHOD_MAP[method]

    result: np.ndarray
    # Handle 3D stacks frame-by-frame
    if img.ndim == 3:
        masks: list[np.ndarray] = [func(frame, limits) for frame in img]
        result = np.stack(masks)
    else:
        result = func(img, limits)

    if invert:
        result = np.isnan(result).astype(float)
        result[result == 0] = np.nan

    return result


__all__ = [
    "thresholder",
    "selection",
    "histogram",
    "otsu",
    "auto_edges",
    "hist_edges",
    "otsu_edges",
    "otsu_skel",
    "hist_skel",
    "line_step",
]
