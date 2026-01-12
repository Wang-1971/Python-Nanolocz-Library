"""
Automated multi-frame leveling routines for AFM data.

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
Steps may invoke `level`, `level_weighted`, or `apply_thresholder`, passing
parameters for polynomial orders, threshold bounds, or other options.

Authors
-------
George Heath, University of Leeds (2025)
D. E. Rollins, University of Leeds (2025)

This module is part of the pNanoLocz-Lib Python library for AFM analysis.
"""

from typing import Any, Dict, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit

from pnanolocz_lib.level import apply_level
from pnanolocz_lib.level_weighted import apply_level_weighted
from pnanolocz_lib.thresholder import apply_thresholder

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
            "func": apply_thresholder,
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
            "func": apply_thresholder,
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
            "func": apply_thresholder,
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
            "func": apply_thresholder,
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
            "func": apply_thresholder,
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
            "func": apply_thresholder,
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
            "func": apply_thresholder,
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
            "func": apply_thresholder,
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
            "func": apply_thresholder,
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
            "func": apply_thresholder,
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
            "func": apply_thresholder,
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
            "func": apply_thresholder,
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
            "func": apply_thresholder,
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
            "func": apply_thresholder,
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
            "func": apply_thresholder,
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
            "func": apply_thresholder,
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
            "func": apply_thresholder,
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
            "func": apply_thresholder,
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
    # Multi plane edges level uses level_weighted
    "multi-plane-edges": [
        {"func": apply_level, "polyx": 1, "polyy": 1, "method": "plane"},
        {
            "func": apply_thresholder,
            "method": "auto edges",
            "args": [0, 0],
            "invert": False,
        },
        {"func": apply_level_weighted, "polyx": 2, "polyy": 2, "method": "plane"},
        {
            "func": apply_thresholder,
            "method": "auto edges",
            "args": [-np.inf, np.inf],
            "invert": False,
        },
        {"func": apply_level_weighted, "polyx": 2, "polyy": 2, "method": "plane"},
        {"func": apply_level_weighted, "polyx": 0, "polyy": 0, "method": "med_line"},
        {"func": apply_level_weighted, "polyx": 2, "polyy": 2, "method": "plane"},
        {"func": apply_level_weighted, "polyx": 0, "polyy": 0, "method": "med_line"},
        {"func": apply_thresholder, "method": "otsu", "args": [0, 0], "invert": False},
        {"func": apply_level, "polyx": 0, "polyy": 0, "method": "mean_plane"},
    ],
    # Multi plane otsu level- uses level weighted
    "multi-plane-otsu": [
        {"func": apply_level, "polyx": 1, "polyy": 1, "method": "plane"},
        {
            "func": apply_thresholder,
            "method": "otsu edges",
            "args": [0, 0],
            "invert": False,
        },
        {"func": apply_level_weighted, "polyx": 2, "polyy": 2, "method": "plane"},
        {
            "func": apply_thresholder,
            "method": "otsu edges",
            "args": [0, 0],
            "invert": False,
        },
        {"func": apply_level_weighted, "polyx": 2, "polyy": 2, "method": "plane"},
        {
            "func": apply_thresholder,
            "method": "otsu edges",
            "args": [0, 0],
            "invert": False,
        },
        {"func": apply_level_weighted, "polyx": 2, "polyy": 2, "method": "plane"},
        {"func": apply_level_weighted, "polyx": 0, "polyy": 0, "method": "med_line"},
        {
            "func": apply_thresholder,
            "method": "otsu edges",
            "args": [0, 0],
            "invert": False,
        },
        {"func": apply_level_weighted, "polyx": 2, "polyy": 2, "method": "plane"},
        {"func": apply_level_weighted, "polyx": 0, "polyy": 0, "method": "med_line"},
        {"func": apply_thresholder, "method": "otsu", "args": [0, 0], "invert": False},
        {"func": apply_level, "polyx": 0, "polyy": 0, "method": "mean_plane"},
    ],
}


# --- Anisotropy preconditioning policies --------------------------------------
# A routine may declare that, *after* a specific step (trigger), we compute
# the Y/X anisotropy ratio from the current image and possibly inject a
# med_line precondition with the given polyx (float allowed).
#
# 'gates' are checked in order; for the first gate whose `ratio > factor`,
# we apply med_line(polyx=<value>, polyy=0). For routines with a single gate,
# just provide one pair.
#
# Trigger schema (current use-case):
#   'after_step': {'func': 'apply_level', 'method': 'plane', 'polyx': 1, 'polyy': 1}
#
PRECOND_POLICIES: dict[str, dict] = {
    "multi-plane-edges": {
        "trigger": {
            "after_step": {
                "func": "apply_level",
                "method": "plane",
                "polyx": 1,
                "polyy": 1,
            }
        },
        "gates": [
            (7.0, 1.0),  # strong preconditioning: med_line(1.0)
            (5.0, 0.6),  # light preconditioning:  med_line(0.6)
        ],
        "method": "med_line",
    },
    "iterative 1nm high": {
        "trigger": {
            "after_step": {
                "func": "apply_level",
                "method": "plane",
                "polyx": 1,
                "polyy": 1,
            }
        },
        "gates": [
            (7.0, 1.0),  # strong preconditioning: med_line(1.0)
            (5.0, 0.6),  # light preconditioning:  med_line(0.6)
        ],
        "method": "med_line",
    },
    "iterative -1nm low": {
        "trigger": {
            "after_step": {
                "func": "apply_level",
                "method": "plane",
                "polyx": 1,
                "polyy": 1,
            }
        },
        "gates": [
            (7.0, 1.0),  # strong preconditioning: med_line(1.0)
            (5.0, 0.6),  # light preconditioning:  med_line(0.6)
        ],
        "method": "med_line",
    },
    "iterative high low": {
        "trigger": {
            "after_step": {
                "func": "apply_level",
                "method": "plane",
                "polyx": 1,
                "polyy": 1,
            }
        },
        "gates": [
            (7.0, 1.0),  # strong preconditioning: med_line(1.0)
            (5.0, 0.6),  # light preconditioning:  med_line(0.6)
        ],
        "method": "med_line",
    },
    "multi-plane-otsu": {
        "trigger": {
            "after_step": {
                "func": "apply_level",
                "method": "plane",
                "polyx": 1,
                "polyy": 1,
            }
        },
        "gates": [
            (5.7, 1.0),  # single gate in MATLAB
        ],
        "method": "med_line",
    },
    # Add more routines here as    # Add more routines here as needed...
}


# --- Per-step mask semantics for parity with MATLAB auto ----------------------
MEAN_PLANE_NAN_MASK = {"multi-plane-edges", "multi-plane-otsu"}  # final step
# All weighted steps use zeros-outside (boolean OK) – plain mean of W*img in level_weighted.


def _matches_trigger(func_obj, params: dict, trigger_spec: dict) -> bool:
    """
    Return True if the just-executed step matches the 'after_step' trigger.
    """
    aft = trigger_spec.get("after_step", {})
    # func check
    func_name = aft.get("func", None)
    if func_name is not None:
        # Identify func by symbol (fast) or by name fallback
        if func_name == "apply_level" and func_obj.__name__ != "apply_level":
            return False
        if (
            func_name == "apply_level_weighted"
            and func_obj.__name__ != "apply_level_weighted"
        ):
            return False
    # key-value checks inside params (e.g., method='plane', polyx=1, polyy=1)
    for k, v in aft.items():
        if k == "func":
            continue
        if params.get(k) != v:
            return False
    return True


def _compute_anisotropy_ratio(img: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute std_x, std_y, ratio from row/col means on current image (NaN-safe).
    """
    col_means = np.nanmean(img, axis=0)
    row_means = np.nanmean(img, axis=1)
    std_x = float(np.nanstd(col_means))
    std_y = float(np.nanstd(row_means))
    if std_x == 0.0:
        # Avoid inf; if both are zero, ratio==0; if only X=0 and Y>0, treat as inf.
        ratio = 0.0 if std_y == 0.0 else float("inf")
    else:
        ratio = std_y / std_x
    return std_x, std_y, ratio


def _maybe_inject_precond(
    img: np.ndarray,
    routine: str,
    func_obj: Any,
    params: dict,
    injected: bool,
    *,
    # dependency injections for testing
    apply_level_fn=None,
    debug: bool = False,
) -> Tuple[np.ndarray, bool]:
    """
    If 'routine' has a preconditioning policy and this step matches its trigger,
    compute anisotropy and, if a gate passes, inject a med_line precondition.
    Returns (possibly-modified img, injected_flag).
    """
    if injected:
        return img, True

    policy = PRECOND_POLICIES.get(routine)
    if policy is None:
        return img, False

    trigger = policy.get("trigger", {})
    if not _matches_trigger(func_obj, params, trigger):
        return img, False

    std_x, std_y, ratio = _compute_anisotropy_ratio(img)
    if debug:
        print(
            f"[auto] routine={routine} post-plane(1,1) ratio={ratio:.3f} (std_y={std_y:.3g}, std_x={std_x:.3g})"
        )

    gates = policy.get("gates", [])
    method = policy.get("method", "med_line")
    # iterate gates in order; first winner applies
    for factor, polyx_value in gates:
        if ratio > factor:
            if apply_level_fn is None:
                from pnanolocz_lib.level import (
                    apply_level as apply_level_fn,
                )  # lazy import
            img = apply_level_fn(
                img, polyx=polyx_value, polyy=0, method=method, mask=None
            )
            if debug:
                print(
                    f"[auto]  precond applied: {method}(polyx={polyx_value}) for ratio>{factor}"
                )

            return img, True

    if debug:
        print("[auto]  precond not applied (no gate passed)")
    return img, False


def _gauss1_model(x, a1, b1, c1):
    # MATLAB gauss1: a1 * exp(-((x - b1)^2) / c1^2)
    return a1 * np.exp(-((x - b1) ** 2) / (c1**2))


def _compute_gauss_limits(
    image: np.ndarray[Any, np.dtype[np.float64]], kind: str
) -> tuple[float, float]:
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
    bins = 100
    # 100-bin histogram like MATLAB's hist(double(t(:)),100)
    hy, edges = np.histogram(data, bins=bins)
    # Bin centers to match MATLAB's 'x' returned by hist
    x = 0.5 * (edges[:-1] + edges[1:])

    if not np.any(hy):
        # fallback in pathological cases
        mu = float(np.nanmean(data)) if data.size else 0.0
        c1 = float(np.nanstd(data)) * np.sqrt(2) if data.size else 1.0
        b1 = mu
    else:
        # Initial guesses matter for stable fits
        a0 = float(hy.max())
        b0 = float(np.mean(data))
        # Note: choose c0 so that sigma ≈ c1/√2 initially (not critical, just stable)
        c0 = float(np.std(data)) * np.sqrt(2) if data.size else 1.0

        # Fit gauss1 to histogram centers vs counts
        popt, _ = curve_fit(
            _gauss1_model,
            x,
            hy.astype(float),
            p0=[a0, b0, c0],
            maxfev=10000,
        )
        a1, b1, c1 = popt

    delta = 1.5 * c1
    if kind == "gauss_fit":
        low, high = b1 - delta, b1 + delta
    elif kind == "gauss_holes":
        low, high = b1 - delta, np.inf
    elif kind == "gauss_peaks":
        low, high = -np.inf, b1 + delta
    else:
        raise ValueError(f"Unknown fit kind {kind!r}")

    return float(low), float(high)


def apply_level_auto(
    img_stack: np.ndarray[Any, np.dtype[np.float64]],
    routine: str,
) -> np.ndarray[Any, np.dtype[np.float64]]:
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
        injected_precond = False

        for idx, step in enumerate(steps):
            func = step["func"]
            params = {k: v for k, v in step.items() if k != "func"}

            if func is apply_thresholder:
                method = params["method"]
                args = params.get("args", None)
                invert = params.get("invert", False)
                # Intercept Gaussian-derived bounds
                if (
                    isinstance(args, (list, tuple))
                    and len(args) == 1
                    and isinstance(args[0], str)
                    and args[0].startswith("gauss_")
                ):
                    low, high = _compute_gauss_limits(img, args[0])
                    mask = apply_thresholder(img, method, (low, high), invert=invert)
                else:
                    mask = apply_thresholder(img, method, args, invert=invert)
                if mask.ndim == 3 and mask.shape[0] == 1:
                    mask = mask[0]
                continue  # go to next step

            # Generic path for all other steps (unchanged)
            img = func(
                img,
                mask=mask,
                **{k: v for k, v in params.items() if k not in ("args", "invert")},
            )

            # Preconditioning (your existing logic)
            img, injected_precond = _maybe_inject_precond(
                img,
                routine,
                func_obj=func,
                params=params,
                injected=injected_precond,
                debug=True,
            )

            result[i] = img

    return np.asarray(result[0]) if was_2d else np.asarray(result)


__all__ = ["apply_level_auto", "ROUTINES"]
