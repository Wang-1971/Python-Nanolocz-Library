# mypy: disallow_untyped_calls = False
"""
Thresholding and edge-detection utilities for AFM images and image stacks.

This module implements a collection of thresholding, segmentation, and
edge-detection routines used to generate boolean masks for Atomic Force
Microscopy (AFM) image processing. The functions are designed to operate on
either single 2D frames or 3D stacks (frame-first convention) and return
boolean masks suitable for downstream leveling and background correction
operations.

The implementation is a Python port of thresholding routines from the
NanoLocz MATLAB library:
    https://github.com/George-R-Heath/NanoLocz-Matlab-Library
    George Heath, University of Leeds

The Python implementation aims to be *algorithmically aligned* with the
original MATLAB code. However, due to differences between MATLAB and Python
numerical libraries (e.g. morphology operators, thresholding implementations,
floating-point behavior), results may not be bit-for-bit identical in all
cases.

Where possible, MATLAB-equivalent operations are reproduced explicitly
(e.g. `bwmorph('remove')`, `imdilate`, `bwareaopen`). In cases where no direct
Python equivalent exists, reasonable approximations are used and documented
inline.

The intent is functional and scientific equivalence rather than strict
binary equivalence.

Mask conventions
----------------
Unless otherwise stated, functions in this module return boolean masks with
the following convention:

    True  → excluded / masked pixels (MATLAB NaN regions)
    False → valid / included pixels

This convention matches the effective behavior of NaN-masked regions in MATLAB and
and the downstream leveling and weighting logic used in the NanoLocz workflow while
remaining compatible with NumPy boolean indexing.

Available methods
-----------------
The following thresholding and edge-detection methods are provided and can be
accessed either directly or via the `apply_thresholder` dispatcher:

- selection      : Pass-through of a user-supplied mask
- histogram      : Intensity range thresholding
- otsu           : Single-level Otsu thresholding (NaN-safe)
- auto edges     : Sobel-based edge detection with morphological cleanup
- hist edges     : Histogram-gated edge detection
- otsu edges     : Otsu-gated edge detection
- otsu skel      : Otsu thresholding followed by skeletonization
- hist skel      : Histogram thresholding followed by skeletonization
- line_step      : Line-wise step detection using PELT change-point detection
- adaptive       : Adaptive edge-based masking mimicking MATLAB behaviour

Dispatcher
----------
The `apply_thresholder` function provides a unified interface to all registered
methods, handling method lookup, argument normalization, stack dispatch, and
optional inversion.

Authors
-------
George Heath, University of Leeds (MATLAB reference implementation)
D. E. Rollins, University of Leeds (Python implementation)

Part of the pNanoLocz-Lib Python library for AFM analysis.
"""

from typing import Any, Callable, TypeVar

import numpy as np
import ruptures as rpt
import sknw
from scipy.ndimage import gaussian_filter, sobel, binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.morphology import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    disk,
    remove_small_objects,
    skeletonize,
    thin,
    diamond,
    erosion,
    footprint_rectangle,
    remove_small_holes,
)

# Map method names to handler functions
_METHOD_MAP = {}

F = TypeVar("F", bound=Callable[..., np.ndarray[Any, np.dtype[Any]]])


def _register(name: str) -> Callable[[F], F]:
    """Decorator to register a threshold method."""

    def decorator(func: F) -> F:
        _METHOD_MAP[name.lower()] = func
        return func

    return decorator


def to_bool_mask(mask_like: np.ndarray) -> np.ndarray:
    """
    Convert an input mask-like array to a clean boolean mask.

    Any nonzero or True values become True; zeros, False, or NaNs become False.
    Then flipped so that True indicates excluded regions (masked), False indicates valid regions.

    Parameters
    ----------
    mask_like : np.ndarray
        Input array to be interpreted as a mask.

    Returns
    -------
    np.ndarray
        Boolean mask where True = excluded, False = valid.
    """
    arr = np.asarray(mask_like)
    # True = valid / nonzero & finite, False = invalid / zero or NaN
    valid = np.asarray(arr, dtype=bool) & np.isfinite(arr)
    # Flip to convention: True = excluded
    return ~valid


def to_nan_mask(
    binary_mask: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[Any]]:
    """
    Convert a boolean mask to a float mask with NaNs in False positions.

    Kept for compatibility with older code. Prefer bool masks in new code.

    Parameters
    ----------
    binary_mask : np.ndarray
        Boolean array where True indicates valid regions.

    Returns
    -------
    mask : np.ndarray
        Float array where True becomes 1.0 and False becomes NaN.
    """
    mask = binary_mask.astype(float)
    mask[~binary_mask] = np.nan
    return mask


@_register("selection")
def selection(
    img: np.ndarray[Any, np.dtype[np.float64]],
    limits: tuple[float, float] | list[float] | str | None = None,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """
    Pass-through a user-provided mask.

    Interprets the input array as an exclusion mask without modification. This function
    assumes the input already uses the module's convention:
    True = excluded, False = valid.

    No inversion or normalization is performed.”

    Mask convention:
        True  → excluded / masked pixel (MATLAB NaN region)
        False → valid / included pixel

    Parameters
    ----------
    img : np.ndarray
        Input mask image. Non-zero or True values indicate excluded regions.
    limits : None
        Not used for this method (kept for API consistency).

    Returns
    -------
    np.ndarray
        Boolean exclusion mask with the same shape as `img`.

    Notes
    -----
    - This function performs no inversion or processing.
    - Intended for direct use with MATLAB-style masks where True corresponds
      to NaN (excluded) regions.
    """
    mask = img.astype(bool)
    return mask.astype(bool)


@_register("histogram")
def histogram(
    img: np.ndarray[Any, np.dtype[np.float64]],
    limits: tuple[float, float] | list[float] | str | None = None,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """
    Create a mask using histogram-based intensity thresholding.

    Computes a logical gate based on intensity limits. Pixels within the
    specified range are marked as True and pixels outside the range as False.

    Mask convention (pipeline-level interpretation):
        True  → excluded / masked pixel
        False → valid / included pixel

    Parameters
    ----------
    img : np.ndarray
        Input image to threshold.
    limits : tuple of float
        Intensity range (low, high).

    Returns
    -------
    np.ndarray
        Boolean mask derived from the intensity gate.

    Raises
    ------
    ValueError
        If limits are not provided or invalid.

    Notes
    -----
    - This function mirrors MATLAB histogram gating logic but does not
      explicitly invert the result.
    - Interpretation as an exclusion mask is handled consistently at the
      pipeline level.
    """
    if limits is None:
        raise ValueError("limits must be provided for histogram method")
    if isinstance(limits, (tuple, list)) and len(limits) == 2:
        low, high = limits
    else:
        raise ValueError("limits must be a tuple or list of 2 elements")

    mask = (img >= low) & (img <= high)
    return mask.astype(bool)


@_register("otsu")
def otsu(
    img: np.ndarray[Any, np.dtype[np.float64]],
    limits: tuple[float, float] | list[float] | str | None = None,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """
    Use a single-level Otsu thresholding (NaN-safe) to build a boolean mask.

    This function computes an Otsu threshold using only finite-valued pixels and returns
    a boolean mask based on intensity comparison.

    Mask convention (pipeline-level interpretation):
        True  → excluded / masked pixel (MATLAB NaN region)
        False → valid / included pixel

    Parameters
    ----------
    img : np.ndarray
        Input 2D image or single frame from a stack.
    limits : None
        Ignored for this method (kept for API consistency).

    Returns
    -------
    np.ndarray
        Boolean mask derived from Otsu thresholding.

    Notes
    -----
    - Finite pixels below or equal to the threshold are marked as True.
    - Non-finite pixels (NaN/Inf) are always marked as False.
    - This aligns algorithmically with MATLAB's `graythresh`, but exact
      thresholds may differ due to implementation details in scikit-image.
    - Interpretation as an exclusion mask is handled consistently elsewhere.
    """
    arr = np.asarray(img, dtype=np.float64)

    # Identify finite-valued pixels (NaN/Inf excluded from threshold computation)
    finite = np.isfinite(arr).astype(bool)  # <-- ensure proper bool type

    # No finite pixels -> no excluded regions
    if limits is None:
        # proceed with default behavior
        limits = ()  # or ignore it completely

    # No finite pixels -> no exclusion
    if finite.sum() == 0:
        return np.zeros_like(arr, dtype=bool)

    # Compute threshold on finite values only (MATLAB graythresh equivalent)
    thresh = threshold_otsu(arr[finite])

    # Logical gate: pixels below or equal to threshold
    inside = arr <= thresh

    # Non-finite pixels are always treated as valid (not excluded)
    inside[~finite] = False

    return inside.astype(bool)


@_register("auto edges")
def auto_edges(
    img: np.ndarray[Any, np.dtype[np.float64]],
    limits: tuple[float, float] | list[float] | str | None = None,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """
    Create a mask using sobel-based automatic edge detection with cleanup.

    This method approximates the MATLAB NanoLocz `auto edges` routine. It
    combines Gaussian smoothing, directional Sobel gradients, adaptive
    thresholding, and a sequence of morphological operations to identify
    edge-like structures in AFM images.

    Mask convention (pipeline-level interpretation):
        True  → excluded / masked pixels (MATLAB NaN regions, edges)
        False → valid / included pixels (interior)

    Parameters
    ----------
    img : np.ndarray
        Input 2D AFM image.
    limits : None
        Not used for this method (kept for API consistency).

    Returns
    -------
    np.ndarray
        Boolean exclusion mask with the same shape as `img`.

    Notes
    -----
    - `gx` and `gy` are the horizontal and vertical Sobel gradients.
    - Vertical gradients are median-corrected row-wise to reduce scan-line
      bias, mirroring MATLAB heuristics used in NanoLocz.
    - Gradient energy is computed as ``2*gx**2 + gy**2`` to preferentially
      emphasize horizontal features typical of AFM scan artifacts.
    - Morphological operations approximate MATLAB `bwmorph`, `imdilate`,
      and `imclose` calls but are not bitwise identical.
    - Final inversion ensures True corresponds to excluded (edge) regions.
    """
    # 1. Gaussian smoothing (MATLAB: imgaussfilt)
    img = np.asarray(img, dtype=np.float64)
    sm = gaussian_filter(img, sigma=2, mode="nearest")  # replicate padding

    # 2. Sobel gradients
    gx = sobel(sm, axis=1, mode="nearest")  # gx : horizontal Sobel gradient
    gy = sobel(sm, axis=0, mode="nearest")  # gy : vertical Sobel gradient

    gy = gy.astype(np.float64)
    # Remove row-wise bias in vertical gradient (MATLAB alignment heuristic)
    gy = gy - 0.5 * np.median(gy, axis=1, keepdims=True)

    # Gradient energy (heuristic weighting to favor horizontal features)
    grad = 2 * gx**2 + gy**2

    # 3. Adaptive thresholding
    thresh = grad.min() + (grad.mean() - grad.min()) * 1.5
    bw = grad > thresh  # bw : binary edge mask (forground)

    # 4. Remove small connected components (MATLAB bwareaopen equivalents)
    bw = remove_small_objects(bw, min_size=100)
    bw = ~remove_small_objects(~bw, min_size=50)

    # 5. Morphological cleanup
    se = diamond(4)  # se = structuring element
    bw = binary_closing(bw, footprint=se)
    bw = binary_dilation(bw, footprint=se)
    bw = binary_closing(bw, footprint=se)
    bw = binary_dilation(bw, footprint=se)

    # 6. Mimic MATLAB bwmorph('bridge')
    # small closing to connect nearby pixels

    # Approximate MATLAB bwmorph('bridge'):

    x3 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool)  # 3x3 X-shape
    bw = binary_closing(bw, footprint=x3)

    fp = diamond(1)  # gentler than disk(1)
    for _ in range(max(1, 3)):
        bw = binary_erosion(bw, footprint=fp)

    # 7. Return boolean mask where True = excluded
    return ~bw.astype(bool)


@_register("hist edges")
def hist_edges(
    img: np.ndarray[Any, np.dtype[np.float64]],
    limits: tuple[float, float] | list[float] | None = None,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """
    Create a mask using histogram-gated edge detection with perimeter extraction.

    This function implements the MATLAB NanoLocz `hist edges` routine for a
    single AFM image frame. It uses intensity gating followed by morphological
    perimeter extraction to identify edge-adjacent regions.

    Processing pipeline (MATLAB equivalent):
        imgaussfilt →
        (low ≤ I ≤ high) →
        invert →
        bwmorph('remove') →
        imdilate(strel('disk',3)) →
        invert

    Mask convention::
        True  → excluded / masked pixels (edges, MATLAB NaN regions)
        False → valid / included pixels (interior)

    Parameters
    ----------
    img : np.ndarray
        Input 2D AFM image.
    limits : tuple of float
        Intensity range (low, high) used for histogram gating.

    Returns
    -------
    np.ndarray
        Boolean mask with the same shape as `img`.

    Raises
    ------
    ValueError
        If `img` is not 2D.
        If `limits` is not a (low, high) tuple.

    Notes
    -----
    - Morphological operations approximate MATLAB behavior but are not
      bitwise identical due to implementation differences.
    """

    # 1. Validate inputs
    if img.ndim != 2:
        raise ValueError("hist_edges expects a 2D image")

    if not (isinstance(limits, (tuple, list)) and len(limits) == 2):
        raise ValueError("limits must be a (low, high) tuple or list")

    low, high = float(limits[0]), float(limits[1])
    img = np.asarray(img, dtype=np.float64)

    #  2. Gaussian smoothing (MATLAB: imgaussfilt)
    sm = gaussian_filter(img, sigma=2, mode="nearest")  # sm = smoothed image

    # 3. Histogram gating then invert
    # MATLAB:
    # imgt = (h<=max).*(h>=min);
    # imgt = ~imgt;
    inside = (sm >= low) & (sm <= high)  # intensity-gated interior
    inverted = ~inside  # candidate edge regions

    # 4. Perimeter extraction (MATLAB: bwmorph('remove'))
    eroded = erosion(inverted, footprint_rectangle((3, 3)))
    perimeter = inverted & ~eroded  # retain only perimeter pixels

    # 5. Thicken perimeter (MATLAB: imdilate(strel('disk',3)))
    thick_perim = binary_dilation(perimeter, footprint=disk(3))

    # 6. Invert to obtain exclusion mask
    interior = ~thick_perim

    return interior.astype(bool)


@_register("otsu edges")
def otsu_edges(
    img: np.ndarray[Any, np.dtype[np.float64]],
    limits: tuple[float, float] | list[float] | str | None = None,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """
    Create a mask using otsu-based edge detection with perimeter extraction.

    This function implements the MATLAB NanoLocz `otsu edges` routine for a
    single AFM image frame. Edge-adjacent regions are identified using Otsu
    thresholding followed by morphological perimeter extraction and cleanup.

    Processing pipeline (MATLAB equivalent):
        imgaussfilt →
        multithresh (1 level) →
        invert →
        bwmorph('remove') →
        bwareaopen / hole filling →
        imdilate(strel('disk',2)) →
        cleanup →
        invert

    Mask convention:
        True  → excluded / masked pixels (edges, MATLAB NaN regions)
        False → valid / included pixels (interior)

    Parameters
    ----------
    img : np.ndarray
        Input 2D AFM image.
    limits : None
        Ignored for this method (present for API consistency).

    Returns
    -------
    np.ndarray
        Boolean mask with the same shape as `img`.

    Raises
    ------
    ValueError
        If `img` is not 2D.

    Notes
    -----
    - This function is intentionally 2D-only.
    - Stack (3D) handling should be performed by the caller or dispatcher.
    - Morphological operations approximate MATLAB behavior but are not
      bitwise identical due to implementation differences.
    """

    # 1. Validate input
    img = np.asarray(img, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("otsu_edges expects a 2D image")

    # 2. Gaussian smoothing (MATLAB: imgaussfilt)
    sm = gaussian_filter(img, sigma=2, mode="nearest")

    # 3. Otsu threshold then invert
    # MATLAB:
    # min_max(2) = multithresh(h,1);
    # min_max(1) = -inf;
    # imgt = (h<=min_max(2)).*(h>=min_max(1));
    # imgt = ~imgt;
    thresh = threshold_otsu(sm)
    inside = sm <= thresh
    inverted = ~inside.astype(bool)

    # 4. Perimeter extraction (MATLAB: bwmorph('remove'))
    eroded = erosion(inverted, footprint_rectangle((3, 3)))
    perimeter = inverted & ~eroded

    # 5. Area cleanup (MATLAB: bwareaopen / hole fill)
    perimeter = remove_small_objects(perimeter, min_size=100, connectivity=2)
    perimeter = remove_small_holes(perimeter, area_threshold=50, connectivity=2)

    # ---- 6. Dilation (MATLAB: imdilate(strel('disk',2)))
    thick_perim = binary_dilation(perimeter, footprint=disk(2))

    # 7. Final cleanup
    thick_perim = remove_small_objects(thick_perim, min_size=100, connectivity=2)
    thick_perim = remove_small_holes(thick_perim, area_threshold=50, connectivity=2)

    # 8. Invert → exclusion mask
    interior = ~thick_perim

    return interior.astype(bool)


def prune_skeleton_min_branch_length(
    skel: np.ndarray, min_branch_length: int = 10
) -> np.ndarray:
    """
    Prune short branches from a skeletonized binary image.

    This function converts a skeleton image into a graph representation,
    removes branches shorter than a specified minimum length, and reconstructs
    a pruned skeleton image from the remaining graph edges.

    Parameters
    ----------
    skel : np.ndarray
        Binary skeleton image where True values represent skeleton pixels.
    min_branch_length : int, optional
        Minimum allowed branch length (in pixels). Skeleton branches with a
        total path length below this threshold are removed.

    Returns
    -------
    np.ndarray
        Boolean image containing the pruned skeleton.

    Notes
    -----
    - This uses `sknw.build_sknw` to construct a pixel-accurate skeleton graph.
    - Edge weights correspond to branch lengths measured along the skeleton.
    - Isolated nodes created by pruning are removed to avoid orphan pixels.
    - This approximates MATLAB skeleton pruning workflows based on branch length
      filtering rather than iterative `spur` removal.
    """
    graph = sknw.build_sknw(skel.astype(np.uint8), multi=True)

    # Remove graph edges (branches) shorter than the minimum length
    # s : start node, e : end node, k : edge key (MultiGraph)
    for s, e, k in list(graph.edges(keys=True)):
        if graph[s][e][k]["weight"] < min_branch_length:
            graph.remove_edge(s, e, k)
    # Remove isolated nodes left after edge pruning
    # n : node index, d : node degree
    isolated_nodes = [n for n, d in dict(graph.degree()).items() if d == 0]
    graph.remove_nodes_from(isolated_nodes)
    # Reconstruct a boolean skeleton image from remaining graph edges
    pruned = np.zeros_like(skel, dtype=bool)
    for _, _, _, data in graph.edges(keys=True, data=True):
        pts = data["pts"]
        pruned[pts[:, 0], pts[:, 1]] = True

    return pruned


def _skeletonize_frame(
    binary_mask: np.ndarray, min_branch_length: int = 10
) -> np.ndarray:
    """
    Skeletonize and clean a binary edge mask for a single image frame.

    This helper function performs thinning and skeletonization on a binary
    edge mask, removes short skeleton branches, and applies light morphological
    cleanup to approximate MATLAB spur and clean operations.

    Parameters
    ----------
    binary_mask : np.ndarray
        Binary image where True values represent edge or foreground pixels.
    min_branch_length : int, optional
        Minimum skeleton branch length to retain during pruning.

    Returns
    -------
    np.ndarray
        Boolean image containing the cleaned skeleton.

    Notes
    -----
    - `thin` reduces thick structures to single-pixel-wide lines.
    - `skeletonize` enforces connectivity and topological consistency.
    - Branch pruning approximates MATLAB skeleton cleanup heuristics.
    - A final dilation reconnects small gaps introduced during pruning.
    """
    # Thinning on the binary mask to reduce structures to minimal width
    thin_mask = thin(binary_mask.astype(bool))

    # Skeletonize the thinned mask
    skel = skeletonize(thin_mask)
    # Remove short skeleton branches
    pruned = prune_skeleton_min_branch_length(skel, min_branch_length)

    # Cleanup similar to MATLAB spur/clean
    cleaned = binary_dilation(pruned)

    return cleaned


@_register("otsu skel")
def otsu_skel(
    img: np.ndarray[Any, np.dtype[np.float64]],
    limits: tuple[float, float] | list[float] | str | None = None,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """
    Generate a skeleton-based exclusion mask using Otsu thresholding.

    This method identifies edge regions using Otsu thresholding, extracts
    a skeleton representation of those edges, and returns a boolean mask
    indicating valid interior pixels.

    Mask convention:
        True  → valid / included pixels (interior)
        False → excluded / masked pixels (edges, MATLAB NaN regions)

    Parameters
    ----------
    img : np.ndarray
        Input 2D image.
    limits : None
        Ignored for this method (present for API consistency).

    Returns
    -------
    np.ndarray
        Boolean mask with the same shape as `img`.

    Raises
    ------

    ValueError
        If `img` is not 2D.

    Notes
    -----
    - Gaussian smoothing reduces noise prior to thresholding.
    - Edge regions are defined as pixels above the Otsu threshold.
    - Skeletonization reduces edge regions to single-pixel-wide structures.
    - This approximates MATLAB workflows combining `multithresh`,
      skeletonization, and spur cleanup.
    """
    sm = gaussian_filter(img, sigma=2, mode="nearest")
    thresh = threshold_otsu(sm)
    binary = ~(sm <= thresh)  # edges foreground

    def _skel_valid_frame(bmask: np.ndarray) -> np.ndarray[np.bool_]:
        skeleton = _skeletonize_frame(bmask)  # True on skeleton (edges)
        return (~skeleton).astype(bool)  # True = valid, False = edges

    if img.ndim == 2:
        return _skel_valid_frame(binary)
    else:
        raise ValueError("img must be 2D")


@_register("hist skel")
def hist_skel(
    img: np.ndarray[Any, np.dtype[np.float64]],
    limits: tuple[float, float] | list[float] | str | None = None,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """
    Generate a skeleton-based exclusion mask using histogram gating.

    This method identifies edge regions using intensity limits, extracts
    a skeleton representation of those edges, and returns a boolean mask
    indicating valid interior pixels.

    Mask convention:
        True  → valid / included pixels (interior)
        False → excluded / masked pixels (edges, MATLAB NaN regions)

    Parameters
    ----------
    img : np.ndarray
        Input 2D image.
    limits : tuple of float
        Intensity range (low, high) defining interior pixels.

    Returns
    -------
    np.ndarray
        Boolean mask with the same shape as `img`.

    Raises
    ------
    ValueError
        If `limits` is not a 2-element tuple or list.
        If `img` is not 2D.

    Notes
    -----
    - Pixels outside the intensity limits are treated as edge foreground.
    - Skeletonization converts thick edge regions into minimal structures.
    - Branch pruning removes short, spurious skeleton segments.
    - This mirrors MATLAB NanoLocz histogram + skeleton workflows.
    """
    if not (isinstance(limits, (tuple, list)) and len(limits) == 2):
        raise ValueError("limits must be a (low, high) tuple or list")
    low, high = limits
    sm = gaussian_filter(img, sigma=2, mode="nearest")
    binary = ~((sm >= low) & (sm <= high))  # edges foreground

    def _skel_valid_frame(bmask: np.ndarray) -> np.ndarray[np.bool_]:
        skeleton = _skeletonize_frame(bmask)
        return (~skeleton).astype(bool)

    if img.ndim == 2:
        return _skel_valid_frame(binary)
    else:
        raise ValueError("img must be 2D")


@_register("line_step")
def line_step(
    img: np.ndarray[Any, np.dtype[np.float64]],
    limits: tuple[float, float] | list[float] | str | None = None,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """
    Detect row-wise step changes and construct a validity mask using PELT.

    This function detects step changes (change points) using the PELT algorithm on an
    L2 cost model, and classifies contiguous row segments as valid or excluded
    based on local mean comparisons around each detected change point.

    Mask convention (pipeline-level interpretation):
        True  → excluded / masked pixel (MATLAB NaN region)
        False → valid / included pixel

    Parameters
    ----------
    img : np.ndarray
        Input 2D AFM image (rows, cols).
    limits : tuple of float or list of float, optional
        Sequence with at least two elements; only `limits[1]` is used as the PELT
        penalty (float). The first element is ignored for API uniformity.

    Returns
    -------
    np.ndarray
        Boolean mask with the same shape as `img`.

    Raises
    ------
    ValueError
        If `limits` is not a tuple or list with at least two elements.
        If `img` is not a 2D array.
    RuntimeError
        If change-point detection repeatedly fails for technical reasons.
        (Function will fall back to an all-valid mask on unexpected errors.)

    Notes
    -----
    - Change-point detection uses **PELT (Pruned Exact Linear Time)** with
      a squared-error (L2) cost; implementation via `ruptures.Pelt(model="l2")`.
    - Neighborhood size is fixed to **3 pixels** on each side of a change point.
    - Change points closer than **4 pixels** to the row boundaries are ignored
      to avoid degenerate neighborhoods.
    - This routine returns a validity mask (`True` = valid) to be used directly
      by downstream levelling functions that accept boolean masks.
    - The `limits` shape is retained for consistency with other mask builders
      where `(low, high)` are used; here only `limits[1]` (the second element)
      is meaningful as a numerical penalty.

    See Also
    --------
    ruptures.Pelt : Change-point detection via PELT.
    """
    try:
        if not (isinstance(limits, (tuple, list)) and len(limits) >= 2):
            raise ValueError("limits must be a tuple or list with at least 2 elements")

        rows, cols = img.shape
        penalty = float(limits[1])
        mask = np.zeros((rows, cols), dtype=bool)  # False = valid

        for j in range(rows):
            x = np.asarray(img[j, :], dtype=float)

            # Detect change points; cps : change points
            try:
                cps = rpt.Pelt(model="l2").fit(x).predict(pen=penalty)
            except Exception:
                cps = []

            # Filter CPs: at least 4 from edges
            cps = [int(cp) for cp in cps if 4 <= cp <= cols - 4]
            cps = sorted(set(cps))

            if cps:
                for i in range(len(cps) + 1):
                    if i == 0:
                        # First segment: [0 : cps[0])
                        cp = cps[0]
                        L = x[max(cp - 3, 0) : cp]
                        R = x[cp : min(cp + 3, cols)]
                        rising = L.size > 0 and R.size > 0 and np.mean(L) < np.mean(R)
                        mask[j, 0:cp] = not rising

                    elif i == len(cps):
                        # Last segment: [cps[-1] : end)
                        cp = cps[-1]
                        L = x[max(cp - 3, 0) : cp]
                        R = x[cp : min(cp + 3, cols)]
                        falling_valid = (
                            L.size > 0 and R.size > 0 and np.mean(L) > np.mean(R)
                        )
                        mask[j, cp:cols] = not falling_valid

                    else:
                        # Middle segment: [cps[i-1] : cps[i])
                        cp_prev = cps[i - 1]
                        cp_curr = cps[i]
                        if cp_prev < cp_curr:  # avoid empty slice
                            L = x[max(cp_curr - 3, cp_prev) : cp_curr]
                            R = x[cp_curr : min(cp_curr + 3, cols)]
                            rising = (
                                L.size > 0 and R.size > 0 and np.mean(L) < np.mean(R)
                            )
                            mask[j, cp_prev:cp_curr] = not rising
            else:
                mask[j, :] = False  # whole row valid

        return ~mask

    except Exception as e:
        print(f"line_step failed: {e}")
        # Fallback: return all valid (False) as a boolean mask
        return np.zeros_like(img, dtype=bool)


@_register("adaptive")
def adaptive(
    img: np.ndarray[Any, np.dtype[np.float64]],
    limits: tuple[float, float] | list[float] | None = None,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """
    Build a exclusion mask using an adaptive, edge-based pipeline (Sobel + morphology).

    This method detects edges with Sobel filtering, consolidates edge regions using
    morphological closing/dilation, fills interior holes, erodes to refine edge
    thickness, and finally applies inclusive intensity gating to return a boolean
    exclusion mask.

    Mask convention (pipeline-level interpretation):
        True  → excluded / masked pixel (MATLAB NaN region)
        False → valid / included pixel

    Processing pipeline (MATLAB equivalent)
    ---------------------------------------
    1) Gaussian smoothing (`imgaussfilt`)
    2) Edge detection (`edge('sobel')`) → Otsu threshold → binary edge map
    3) Morphological closing (`imclose(strel('disk',10))`)
    4) Remove small objects (`bwareaopen`, min_size=10)
    5) Dilate with line SEs: `imdilate(line,10,90)` then `imdilate(line,10,0)`
    6) Pad left/right columns with ones (edge guard)
    7) Fill holes (`imfill('holes')`)
    8) Erode with line SEs: `imerode(line,10,0)` then `imerode(line,10,90)`
    9) Crop padding
    10) Inclusive intensity gating: `(Dfin == 0) & (low ≤ I ≤ high)`
    11) Return **exclusion** mask (`True = excluded`)

    Parameters
    ----------
    img : np.ndarray
        Input 2D AFM image.
    limits : tuple of float or list of float
            Inclusive intensity bounds `(low, high)` used in the final gating step.

    Returns
    -------
    np.ndarray
        Boolean exclusion mask with the same shape as `img`.

    Raises
    ------
    ValueError
        If `limits` is not a (low, high) tuple or list.
        If `img` is not a 2D array.

    Notes
    -----
    - Edge detection uses Sobel magnitude with Otsu thresholding to obtain a binary
      edge map. Morphological closing/dilation/erosion refine edge connectivity and
      thickness; hole filling ensures interior regions are coherent.
    - Intensity gating is **inclusive**: pixels must satisfy `low ≤ img ≤ high`.
    - Morphological footprints (`disk(10)`, `line(10, 0/90)`) approximate MATLAB
      structuring elements; exact bitwise equality is not guaranteed due to library
      implementation differences.

    See Also
    --------
    skimage.filters.sobel : Sobel operator for edge detection.
    skimage.filters.threshold_otsu : Otsu threshold estimation.
    skimage.morphology.binary_closing, binary_dilation, binary_erosion
    numpy.pad, scipy.ndimage.binary_fill_holes
    """
    # 0. Validate limits
    if not (isinstance(limits, (tuple, list)) and len(limits) == 2):
        raise ValueError("limits must be a (low, high) tuple or list")
    low, high = float(limits[0]), float(limits[1])
    img = np.asarray(img, dtype=np.float64)

    # 1) Gaussian smoothing
    sm = gaussian_filter(
        img, sigma=0.1, mode="nearest"
    )  # sm : smoothed image (Gaussian pre-filter)

    # 2) Sobel magnitude + Otsu threshold
    sob = np.hypot(
        sobel(sm, axis=0, mode="nearest"), sobel(sm, axis=1, mode="nearest")
    )  # sob : Sobel magnitude (edge strength)
    try:
        thresh = threshold_otsu(sob)
        T = sob > thresh  # T : raw binary edge map (True=edge)
    except Exception:
        T = sob > 0  # fallback

    # 3) Closing with disk(10)
    T = binary_closing(T, footprint=disk(10))

    # 4) bwareaopen with size=10
    T = remove_small_objects(T, min_size=10)

    # 5) Dilate with line(10,90) then line(10,0)
    se_line_vert = np.ones((10, 1), dtype=bool)  # line length 10, angle 90
    se_line_horiz = np.ones((1, 10), dtype=bool)  # line length 10, angle 0
    dil = binary_dilation(T, footprint=se_line_vert)
    dil = binary_dilation(dil, footprint=se_line_horiz)

    # 6) Pad columns by 1 with ones
    padded = np.pad(dil, ((0, 0), (1, 1)), mode="constant", constant_values=True)

    # 7) Fill holes
    filled = binary_fill_holes(padded)

    # 8) Erode with horizontal then vertical line SEs
    er1 = binary_erosion(filled, footprint=se_line_horiz)
    er2 = binary_erosion(er1, footprint=se_line_vert)

    # 9) Remove padding
    Dfin = er2[
        :, 1:-1
    ]  # Dfin : final binary edge band after removing padding (True=edge)

    # 10) MATLAB intensity gating:
    #     imgt = (Dfin==0) .* (img>=low) .* (img<=high)
    interior = (~Dfin) & (img >= low) & (img <= high)

    # 11) Output exclusion mask
    return (interior).astype(bool)


def apply_thresholder(
    img: np.ndarray[Any, np.dtype[np.float64]],
    method: str,
    limits: tuple[float, float] | list[float] | str | None = None,
    invert: bool = False,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Apply a thresholding or edge detecti

    This is the unified front-end for all registered `thresholder` methods. It validates
    the requested `method`, prepares per-method `limits`, applies the method to a 2D
    frame or a 3D stack (frame-by-frame), and optionally inverts the boolean result.

    Parameters
    ----------
    img : np.ndarray
        Input 2D or 3D AFM image data.
    method : str
        Name of a registered thresholding/edge method (e.g., `"histogram"`, `"otsu"`,
        `"auto edges"`, `"selection"`, `"hist edges"`, `"adaptive"`, etc.). The set
        of available methods is defined by `_METHOD_MAP`.
    limits : tuple, list, str, or None, optional
        Parameters specific to the method (default is None).
    invert : bool, optional
        If True, invert the mask (default is False).

    Returns
    -------
    np.ndarray
        Boolean mask array with the same shape as `img`, where:
            True  → excluded / masked pixels
            False → valid / included pixels

    Raises
    ------
    ValueError
        If `method` is unknown, or if `limits` are missing/invalid for the chosen method.

    Notes
    -----
    - 3D stacks are processed frame-by-frame and re-stacked on the leading axis.
    - This dispatcher **does not modify** the numeric data in `img`; it produces a
      boolean mask suitable for masked leveling or post-processing.
    - Methods define their own internal operations (e.g., Sobel, Otsu, morphology)
      and may treat `limits` differently. This function only prepares and routes
      arguments consistently.

    Examples
    --------
    >>> # Histogram gating: keep [-1, +1] nm as valid; everything else excluded.
    >>> mask_excl = apply_thresholder(img, method="histogram", limits=(-1.0, 1.0))
    >>> # Otsu on a stack, then invert to obtain a validity mask (True = valid)
    >>> excl = apply_thresholder(stack, method="otsu")
    >>> valid = np.logical_not(excl)
    """
    method = method.lower()
    if method not in _METHOD_MAP:
        raise ValueError(f"Unknown thresholding method: {method}")

    func = _METHOD_MAP[method]

    # Prepare safe method-specific limits:
    # - Some methods ignore limits entirely.
    # - Some methods require (low, high).
    # - Otherwise, pass through whatever was provided (tuple/list/str/None).
    if method in ["otsu", "auto edges", "selection"]:
        limits_safe = None
    elif method in ["histogram", "hist edges"]:
        if limits is None:
            raise ValueError(f"Method '{method}' requires limits (tuple/list)")
        limits_safe = tuple(limits)  # ensure Python tuple
    else:
        limits_safe = limits  # fallback

    result: np.ndarray[Any, np.dtype[np.float64]]

    # Compute mask:
    # - 3D stacks are processed frame-by-frame and concatenated.
    # - Each registered method returns a boolean mask per frame.
    if img.ndim == 3:
        masks = [func(frame, limits_safe) for frame in img]  # should be bool
        result = np.stack(masks, axis=0)
    else:
        result = func(img, limits_safe)

    # Optional inversion for callers that need the complement.
    # Convention reminder: True=excluded, False=valid
    if invert:
        result = np.logical_not(result)

    return result


__all__ = [
    "apply_thresholder",
    "selection",
    "histogram",
    "otsu",
    "auto_edges",
    "hist_edges",
    "otsu_edges",
    "otsu_skel",
    "hist_skel",
    "line_step",
    "adaptive",
]
