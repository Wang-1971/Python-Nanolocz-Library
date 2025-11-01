"""
Automated Multi-Frame Leveling Routines for AFM Data
====================================================

This module implements automated, data-driven multi-frame leveling routines for
Atomic Force Microscopy (AFM) image stacks. It applies sequences of background
correction strategies—including polynomial plane fitting, line-based
correction, threshold-based masking, and iterative refinement—to each frame in
a stack.

Ported from the MATLAB NanoLocz library, these routines support a variety of
pre-defined workflows for common scenarios in high-speed and localization AFM.

Supported Routines
------------------
- plane-line
- iterative 1nm high
- iterative -1nm low
- iterative high low
- Line1 + Otsu Line2
- high-low x2 (fit)
- iterative fit holes
- iterative fit peaks
- multi-plane-edges
- multi-plane-otsu

Usage
-----
>>> from pnanolocz_lib.level_auto import apply_level_auto
>>> result = apply_level_auto(
                img_stack,
                routine="multi-plane-otsu"
            )

Parameters
----------
Refer to `apply_level_auto` docstring below for detailed parameter
descriptions.

Notes
-----
Each routine is defined in the `ROUTINES` dictionary as an ordered list
of steps.
Steps may invoke `level`, `level_weighted`, or `thresholder`, passing
parameters for polynomial orders, threshold bounds, or other options.

Authors
-------
George Heath, University of Leeds (2025)
D. E. Rollins, University of Leeds (2025)

This module is part of the pNanoLocz-Lib Python library for AFM analysis.
"""

import numpy as np
from typing import Sequence, Dict, Any

from scipy import stats

from pnanolocz_lib.level import apply_level
from pnanolocz_lib.thresholder import thresholder

# Data‑driven routine definitions
ROUTINES: Dict[str, Sequence[Dict[str, Any]]] = {
    # Plane fit routine
    "plane-line": [
        {
            "func": apply_level,
            "polyx": 1,
            "polyy": 1,
            "method": "plane",
        },
        {
            "func": apply_level,
            "polyx": 1,
            "polyy": 0,
            "method": "med_line",
        },
    ],
    # Iterative 1 nm high threshold routine
    "iterative 1nm high": [
        {
            "func": apply_level,
            "polyx": 1,
            "polyy": 1,
            "method": "plane",
        },
        {
            "func": thresholder,
            "method": "histogram",
            "args": [-np.inf, 1],
            "invert": False,
        },
        {
            "func": apply_level,
            "polyx": 1,
            "polyy": 1,
            "method": "plane",
        },
        {
            "func": thresholder,
            "method": "histogram",
            "args": [-np.inf, 1],
            "invert": False,
        },
        {
            "func": apply_level,
            "polyx": 0,
            "polyy": 0,
            "method": "med_line",
        },
        {
            "func": apply_level,
            "polyx": 1,
            "polyy": 0,
            "method": "plane",
        },
        {
            "func": thresholder,
            "method": "histogram",
            "args": [-np.inf, 1],
            "invert": False,
        },
        {
            "func": apply_level,
            "polyx": 0,
            "polyy": 0,
            "method": "med_line",
        },
        {
            "func": apply_level,
            "polyx": 2,
            "polyy": 0,
            "method": "plane",
        },
    ],
    # Iterative 1 nm low threshold routine
    "iterative -1nm low": [
        {
            "func": apply_level,
            "polyx": 1,
            "polyy": 1,
            "method": "plane",
        },
        {
            "func": thresholder,
            "method": "histogram",
            "args": [-1, np.inf],
            "invert": False,
        },
        {
            "func": apply_level,
            "polyx": 1,
            "polyy": 1,
            "method": "plane",
        },
        {
            "func": thresholder,
            "method": "histogram",
            "args": [-1, np.inf],
            "invert": False,
        },
        {
            "func": apply_level,
            "polyx": 0,
            "polyy": 0,
            "method": "med_line",
        },
        {
            "func": apply_level,
            "polyx": 1,
            "polyy": 0,
            "method": "plane",
        },
        {
            "func": thresholder,
            "method": "histogram",
            "args": [-1, np.inf],
            "invert": False,
        },
        {
            "func": apply_level,
            "polyx": 0,
            "polyy": 0,
            "method": "med_line",
        },
        {
            "func": apply_level,
            "polyx": 2,
            "polyy": 0,
            "method": "plane",
        },
    ],
    # Iterative 1 nm high and 1 nm low threshold routine
    "iterative high low": [
        {
            "func": apply_level,
            "polyx": 1,
            "polyy": 1,
            "method": "plane",
        },
        {
            "func": thresholder,
            "method": "histogram",
            "args": [-1, 1],
            "invert": False,
        },
        {
            "func": apply_level,
            "polyx": 1,
            "polyy": 1,
            "method": "plane",
        },
        {
            "func": thresholder,
            "method": "histogram",
            "args": [-1, 1],
            "invert": False,
        },
        {
            "func": apply_level,
            "polyx": 0,
            "polyy": 0,
            "method": "med_line",
        },
        {
            "func": apply_level,
            "polyx": 1,
            "polyy": 0,
            "method": "plane",
        },
        {
            "func": thresholder,
            "method": "histogram",
            "args": [-1, 1],
            "invert": False,
        },
        {
            "func": apply_level,
            "polyx": 0,
            "polyy": 0,
            "method": "med_line",
        },
        {
            "func": apply_level,
            "polyx": 2,
            "polyy": 0,
            "method": "plane",
        },
    ],
    # Line level followered by Otsu threshold and a second line level
    "Line1 + Otsu Line2": [
        {
            "func": apply_level,
            "polyx": 1,
            "polyy": 0,
            "method": "line",
        },
        {
            "func": thresholder,
            "method": "otsu",
            "args": [],
            "invert": False,
        },
        {
            "func": apply_level,
            "polyx": 2,
            "polyy": 0,
            "method": "line",
        },
    ],
    # High- low twice
    "high-low x2 (fit)": [
        {
            "func": apply_level,
            "polyx": 1,
            "polyy": 1,
            "method": "plane",
        },
        {
            "func": apply_level,
            "polyx": 0,
            "polyy": 0,
            "method": "med_line",
        },
        {
            "func": thresholder,
            "method": "histogram",
            "args": ["gauss_fit"],
            "invert": False,
        },
        {
            "func": apply_level,
            "polyx": 1,
            "polyy": 1,
            "method": "plane",
        },
        {
            "func": apply_level,
            "polyx": 0,
            "polyy": 0,
            "method": "med_line",
        },
    ],
    # Iterativly fit holes
    "iterative fit holes": [
        {
            "func": apply_level,
            "polyx": 2,
            "polyy": 2,
            "method": "plane",
        },
        {
            "func": apply_level,
            "polyx": 0,
            "polyy": 0,
            "method": "med_line",
        },
        {
            "func": thresholder,
            "method": "histogram",
            "args": ["gauss_holes"],
            "invert": False,
        },
        {
            "func": apply_level,
            "polyx": 2,
            "polyy": 2,
            "method": "plane",
        },
        {
            "func": apply_level,
            "polyx": 0,
            "polyy": 0,
            "method": "med_line",
        },
        {
            "func": thresholder,
            "method": "histogram",
            "args": ["gauss_holes"],
            "invert": False,
        },
        {
            "func": apply_level,
            "polyx": 2,
            "polyy": 2,
            "method": "plane",
        },
        {
            "func": apply_level,
            "polyx": 1,
            "polyy": 0,
            "method": "line",
        },
    ],
    # Iterativly fit peaks
    "iterative fit peaks": [
        {
            "func": apply_level,
            "polyx": 2,
            "polyy": 2,
            "method": "plane",
        },
        {
            "func": apply_level,
            "polyx": 0,
            "polyy": 0,
            "method": "med_line",
        },
        {
            "func": thresholder,
            "method": "histogram",
            "args": ["gauss_peaks"],
            "invert": False,
        },
        {
            "func": apply_level,
            "polyx": 2,
            "polyy": 2,
            "method": "plane",
        },
        {
            "func": apply_level,
            "polyx": 0,
            "polyy": 0,
            "method": "med_line",
        },
        {
            "func": thresholder,
            "method": "histogram",
            "args": ["gauss_peaks"],
            "invert": False,
        },
        {
            "func": apply_level,
            "polyx": 2,
            "polyy": 2,
            "method": "plane",
        },
        {
            "func": apply_level,
            "polyx": 1,
            "polyy": 0,
            "method": "line",
        },
    ],
}
"""
    # Multi plane edges level 0uses level_weighted
    "multi-plane-edges": [
        {
            "func": apply_level,
            "polyx": 1,
            "polyy": 1,
            "method": "plane",
        },
        {
            "func": thresholder,
            "method": "auto edges",
            "args": [0, 0],
            "invert": False,
        },
        {
            "func": level_weighted,
            "polyx": 2,
            "polyy": 2,
            "method": "plane",
        },
        {
            "func": thresholder,
            "method": "auto edges",
            "args": [-np.inf, np.inf],
            "invert": False,
        },
        {
            "func": level_weighted,
            "polyx": 2,
            "polyy": 2,
            "method": "plane",
        },
        {
            "func": level_weighted,
            "polyx": 0,
            "polyy": 0,
            "method": "med_line",
        },
        {
            "func": thresholder,
            "method": "otsu",
            "args": [0, 0],
            "invert": False,
        },
        {
            "func": apply_level,
            "polyx": 0,
            "polyy": 0,
            "method": "mean_plane",
        },
    ],
    # Multi plane otsu level- uses level weighted
    "multi-plane-otsu": [
        {
            "func": apply_level,
            "polyx": 1,
            "polyy": 1,
            "method": "plane",
        },
        {
            "func": thresholder,
            "method": "otsu edges",
            "args": [0, 0],
            "invert": False,
        },
        {
            "func": level_weighted,
            "polyx": 2,
            "polyy": 2,
            "method": "plane",
        },
        {
            "func": thresholder,
            "method": "otsu edges",
            "args": [0, 0],
            "invert": False,
        },
        {
            "func": level_weighted,
            "polyx": 2,
            "polyy": 2,
            "method": "plane",
        },
        {
            "func": thresholder,
            "method": "otsu edges",
            "args": [0, 0],
            "invert": False,
        },
        {
            "func": level_weighted,
            "polyx": 2,
            "polyy": 2,
            "method": "plane",
        },
        {
            "func": level_weighted,
            "polyx": 0,
            "polyy": 0,
            "method": "med_line",
        },
        {
            "func": thresholder,
            "method": "otsu",
            "args": [0, 0],
            "invert": False,
        },
        {
            "func": apply_level,
            "polyx": 0,
            "polyy": 0,
            "method": "mean_plane",
        },
    ],

}
"""


def _compute_gauss_limits(image: np.ndarray, kind: str) -> tuple[float, float]:
    """
    Compute intensity threshold limits from a Gaussian fit to the image data.

    Parameters
    ----------
    image : np.ndarray
        2D image array from which to compute Gaussian-based thresholds. NaN
        values are ignored.
    kind : str
        Type of Gaussian thresholding to apply. Must be one of:
        - 'gauss_fit'   : Return symmetric limits around the
        mean (mu ± 1.5 * sigma).
        - 'gauss_holes' : Return lower-bound threshold (mu - 1.5 * sigma, ∞),
        for dark features.
        - 'gauss_peaks' : Return upper-bound threshold (-∞, mu + 1.5 * sigma),
        for bright features.

    Returns
    -------
    limits : tuple of float
        The (low, high) threshold bounds based on the Gaussian fit.

    Raises
    ------
    ValueError
        If `kind` is not a recognized thresholding type.

    Notes
    -----
    The method fits a single normal distribution to the image values using
    `scipy.stats.norm.fit`, then derives bounds based on 1.5 standard
    deviations from the mean. Useful for automatic intensity-based masking.

    Examples
    --------
    >>> low, high = _compute_gauss_limits(img, 'gauss_peaks')
    >>> mask = (img >= low) & (img <= high)
    """
    # flatten and drop NaNs
    data = image.ravel()
    data = data[~np.isnan(data)]
    # fit a single gaussian (mean, std)
    mu, sigma = stats.norm.fit(data)
    delta = 1.5 * sigma

    if kind == "gauss_fit":
        return mu - delta, mu + delta
    elif kind == "gauss_holes":
        # holes = low side only
        return mu - delta, np.inf
    elif kind == "gauss_peaks":
        # peaks = high side only
        return -np.inf, mu + delta
    else:
        raise ValueError(f"Unknown fit kind {kind!r}")


def apply_level_auto(
    img_stack: np.ndarray,
    routine: str,
) -> np.ndarray:
    """
    Apply leveling "routines" across specified frames of an AFM image stack.

    Parameters
    ----------
    img_stack : ndarray
        AFM image stack. Shape can be (H, W) or (N, H, W).
    routine : str
        Name of a routine defined in ROUTINES.

    Returns
    -------
    result : ndarray
        Same shape as img_stack (2D or 3D), with selected frames leveled.

    Raises
    ------
    ValueError
        If the specified routine is not found or input shape is invalid.
    IndexError
        If any frame index is out of bounds.
    """
    img_stack = np.asarray(img_stack)

    if img_stack.ndim == 2:
        img_stack = img_stack[np.newaxis, :, :]
        was_2d = True
    elif img_stack.ndim == 3:
        was_2d = False
    else:
        raise ValueError(
            "img_stack must be either 2D or 3D with shape (H, W) or (N, H, W)"
        )

    if routine not in ROUTINES:
        raise ValueError(f"Unknown routine '{routine}'")

    result = img_stack.copy()
    steps = ROUTINES[routine]

    frames = range(img_stack.shape[0])

    for i in frames:
        img = result[i]
        mask = None

        for step in steps:
            func = step["func"]
            params = {k: v for k, v in step.items() if k != "func"}

            if func is thresholder:
                method = params["method"]
                args = params.get("args", None)
                invert = params.get("invert", False)

                # Intercept any ["gauss_*"] args
                if (
                    isinstance(args, (list, tuple))
                    and len(args) == 1
                    and args[0].startswith("gauss_")
                ):
                    low, high = _compute_gauss_limits(img, args[0])
                    mask = thresholder(img, method, (low, high), invert=invert)
                else:
                    mask = thresholder(img, method, args, invert=invert)

                # squeeze away any extra frame axis:
                if mask.ndim == 3 and mask.shape[0] == 1:
                    mask = mask[0]
            else:
                img = func(
                    img,
                    mask=mask,
                    **{
                        k: v
                        for k, v in params.items()
                        if k not in ("args", "invert")  # noqa
                    },
                )

        result[i] = img

    if was_2d:
        return result[0]
    return result


__all__ = ["apply_level_auto", "ROUTINES"]
