"""
Automated leveling of AFM image stacks using MATLAB-aligned multi-step routines.

This module implements automated, data-driven background correction workflows
for Atomic Force Microscopy (AFM) images and image stacks. Each routine applies
an ordered sequence of leveling and masking operations—polynomial plane fits,
line-based drift correction, region-weighted leveling, and iterative refinement—
to improve background flattening across challenging frames.

All routines operate frame-by-frame and reuse the public function contracts of
``pnanolocz.level``, ``pnanolocz.level_weighted``, and
``pnanolocz.thresholder``. Masks follow the *exclusion mask* convention:
``True`` = excluded, ``False`` = valid. Excluded pixels are omitted from fitting
using MATLAB-style NaN-outside semantics (i.e., excluded pixels behave like NaN
during fitting) but are preserved in the output arrays.

The implementation is a Python port of the MATLAB NanoLocz Library:
    https://github.com/George-R-Heath/NanoLocz-Matlab-Library
Original MATLAB code by George Heath, University of Leeds.

MATLAB alignment
----------------
This Python version aims for algorithmic and numerical alignment with the MATLAB
reference implementation of ``level_auto.m``. Due to differences in underlying
numerical libraries (NumPy/SciPy vs MATLAB), floating-point behaviour, and
optimizer conditioning, results may not be bit-for-bit identical. Where
relevant, this module documents intentional alignment decisions such as
anisotropy-gated preconditioning and MATLAB-style Gaussian histogram fitting.

Where in the MATLAB version the Gaussian histograms are fitted using the whole
3D stack, in this Python version each 2D frame is fitted individually because
each frame is processed independently. This is a known deviation from MATLAB
and may be addressed in future versions.

Available routines
------------------
Routines are selected by name via :func:`apply_level_auto` and are defined in
the :data:`ROUTINES` mapping as an ordered list of steps. The supported routines
mirror the MATLAB NanoLocz presets:

- ``plane-line``
- ``iterative 1nm high``
- ``iterative -1nm low``
- ``iterative high low``
- ``Line1 + Otsu Line2``
- ``high-low x2 (fit)``
- ``iterative fit holes``
- ``iterative fit peaks``
- ``multi-plane-edges``
- ``multi-plane-otsu``

Routine mechanics
-----------------
Each step is one of the following:

- A leveling step via :func:`pnanolocz.level.apply_level`
  (e.g., ``plane``, ``line``, ``med_line``, ``mean_plane``).
- A region-weighted leveling step via
  :func:`pnanolocz.level_weighted.apply_level_weighted` (e.g., weighted
  ``plane`` or weighted ``med_line``).
- A masking step via :func:`pnanolocz.thresholder.apply_thresholder`
  which updates the current exclusion mask carried forward to subsequent steps.

Some routines compute histogram bounds from a Gaussian fit to the image value
distribution. These bounds are produced by fitting a MATLAB-style ``gauss1``
model to a 100-bin histogram using SciPy and then forming thresholds from the
fitted center and width.

Anisotropy preconditioning
--------------------------
To match the MATLAB implementation, selected routines may inject a one-off
``med_line`` preconditioning step after a specific trigger (typically
``plane(polyx=1, polyy=1)``). The injection is gated by an anisotropy ratio
computed from the standard deviation of row-mean and column-mean profiles.
Policies are defined in :data:`PRECOND_POLICIES`.

Stacks
------
Functions operate on single images with shape ``(H, W)`` and stacks with shape
``(N, H, W)``. A 2D input is treated as a single-frame stack internally and is
returned as 2D.

Examples
--------
>>> from pnanolocz.level_auto import apply_level_auto
>>> leveled = apply_level_auto(stack, routine="multi-plane-otsu")

>>> img_leveled = apply_level_auto(img, routine="plane-line")

Authors
-------
George Heath, University of Leeds (2025)
Daniel E. Rollins, University of Leeds (2025)

This module is part of the ``pNanoLocz-Lib`` Python library for AFM analysis.
"""

from collections.abc import Callable, Mapping
from typing import Any, Dict, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from pnanolocz.level import apply_level
from pnanolocz.level_weighted import apply_level_weighted
from pnanolocz.thresholder import apply_thresholder

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

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
    # Line level followed by Otsu threshold and a second line level
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
    # Iteratively fit holes
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
PRECOND_POLICIES: dict[str, dict[str, Any]] = {
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


def _matches_trigger(
    func_obj: Callable[..., Any],
    params: Mapping[str, Any],
    trigger_spec: Mapping[str, Any],
) -> bool:
    """
    Match a just-executed step against a preconditioning trigger specification.

    Parameters
    ----------
    func_obj : callable
        Function used by the executed step (e.g., ``apply_level``).
    params : dict
        Step parameters excluding ``func`` (e.g., ``{"method": "plane", "polyx": 1}``).
    trigger_spec : dict
        Trigger specification of the form ``{"after_step": {...}}`` where the
        inner mapping may include ``func`` and any subset of step parameters.

    Returns
    -------
    match : bool
        True if the executed step matches the trigger specification, else False.

    Notes
    -----
    - The function compares the trigger's ``func`` against ``func_obj.__name__``
      and compares any remaining key/value pairs against entries in ``params``.
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


def _compute_anisotropy_ratio(img: FloatArray) -> tuple[float, float, float]:
    """
    Compute an anisotropy ratio from the standard deviation of row/column means.

    The ratio is computed as ``std_y / std_x`` where:
    ``std_x`` is the standard deviation of the column-mean profile and
    ``std_y`` is the standard deviation of the row-mean profile.

    Parameters
    ----------
    img : ndarray
        2D image array. NaNs are ignored when computing means and standard
        deviations.

    Returns
    -------
    std_x : float
        Standard deviation of the column means (X-direction profile).
    std_y : float
        Standard deviation of the row means (Y-direction profile).
    ratio : float
        ``std_y / std_x`` with guards for ``std_x == 0``. If ``std_x == 0`` and
        ``std_y > 0``, ratio is ``inf``; if both are zero, ratio is ``0``.

    Notes
    -----
    This mirrors the MATLAB logic used to decide whether to inject a
    ``med_line`` preconditioning step.
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
    img: FloatArray,
    routine: str,
    func_obj: Callable[..., Any],
    params: Mapping[str, Any],
    injected: bool,
    *,
    apply_level_fn: Callable[..., FloatArray] | None = None,
    debug: bool = False,
) -> tuple[FloatArray, bool]:
    """
    Inject a preconditioning leveling step when a routine-specific gate fires.

    If the current routine declares a preconditioning policy and the just-run
    step matches its trigger, this function computes the anisotropy ratio and
    applies the first gate that passes by injecting a ``med_line`` step.

    Parameters
    ----------
    img : ndarray
        Current 2D image state after executing the step under consideration.
    routine : str
        Routine name used to look up a policy in ``PRECOND_POLICIES``.
    func_obj : Any
        Function object for the executed step (e.g., ``apply_level``).
    params : dict
        Parameters used for the executed step (excluding ``func``).
    injected : bool
        Whether a preconditioning step has already been injected for this frame.
        If True, no further injections occur.
    apply_level_fn : callable, optional
        Dependency injection hook for testing. If not provided, ``apply_level``
        is imported lazily.
    debug : bool, default False
        If True, print diagnostic messages about the decision and injection.

    Returns
    -------
    img_out : ndarray
        Image, possibly modified by an injected preconditioning step.
    injected_out : bool
        Updated injection flag.

    Notes
    -----
    - At most one preconditioning injection is performed per frame.
    - The injected call uses ``mask=None`` to mimic the MATLAB preconditioning
      behavior (preconditioning is based on global structure, not masking).
    """
    if injected:
        return np.asarray(img, dtype=np.float64), True

    policy = PRECOND_POLICIES.get(routine)
    if policy is None:
        return img, False

    trigger = policy.get("trigger", {})
    if not _matches_trigger(func_obj, params, trigger):
        return img, False

    std_x, std_y, ratio = _compute_anisotropy_ratio(img)
    if debug:
        print(
            f"[auto] routine={routine} post-plane(1,1) ratio={ratio:.3f} (std_y={std_y:.3g}, std_x={std_x:.3g})"  # noqa
        )

    gates = policy.get("gates", [])
    method = policy.get("method", "med_line")
    # iterate gates in order; first winner applies
    for factor, polyx_value in gates:
        if ratio > factor:
            fn: Callable[..., FloatArray]
            if apply_level_fn is None:
                from pnanolocz.level import apply_level as _apply_level

                fn = _apply_level
            else:
                fn = apply_level_fn

            img = fn(img, polyx=polyx_value, polyy=0, method=method, mask=None)

        if debug:
            print(
                f"[auto]  precond applied: {method}(polyx={polyx_value}) for ratio>{factor}"  # noqa
            )

        return img, True

    if debug:
        print("[auto]  precond not applied (no gate passed)")
    return np.asarray(img, dtype=np.float64), False


def _gauss1_model(
    x: FloatArray,
    a1: float,
    b1: float,
    c1: float,
) -> FloatArray:
    """
    Evaluate a MATLAB-style single-Gaussian model used for histogram fitting.

    Parameters
    ----------
    x : ndarray
        Histogram bin centers.
    a1, b1, c1 : float
        MATLAB ``gauss1`` parameters: ``a1 * exp(-((x - b1)^2) / c1^2)``.

    Returns
    -------
    y : ndarray
        Model values at ``x``.
    """
    # MATLAB gauss1: a1 * exp(-((x - b1)^2) / c1^2)
    return np.asarray(a1 * np.exp(-((x - b1) ** 2) / (c1**2)), dtype=np.float64)


def _compute_gauss_limits(
    image: np.ndarray[Any, np.dtype[np.float64]], kind: str
) -> tuple[float, float]:
    """
    Compute threshold bounds by fitting a MATLAB-style Gaussian to a histogram.

    This function replicates the MATLAB pattern:
    ``[hy, x] = hist(double(t(:)), 100); gfit = fit(x', hy', 'gauss1')`` and then
    forms bounds from the fitted center and width. NaNs in the image are ignored.

    Parameters
    ----------
    image : ndarray
        2D image used to derive histogram-based Gaussian limits. NaNs are ignored.
    kind : str
        Gaussian limit policy to apply. Must be one of:

        - ``'gauss_fit'``   : symmetric limits ``(b1 - 1.5*c1, b1 + 1.5*c1)``
        - ``'gauss_holes'`` : low limit for dark features ``(b1 - 1.5*c1, +inf)``
        - ``'gauss_peaks'`` : high limit for bright features ``(-inf, b1 + 1.5*c1)``

    Returns
    -------
    low, high : float
        The computed intensity bounds.

    Raises
    ------
    ValueError
        If ``kind`` is not recognized.

    Notes
    -----
    - The fit is performed using ``scipy.optimize.curve_fit`` on a 100-bin
      histogram of the finite image values, using the model
      ``a1 * exp(-((x - b1)^2) / c1^2)`` (MATLAB ``gauss1``).
    - These limits are intended for use with the histogram thresholder step.
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
    Apply an automated leveling routine to each frame of an AFM image stack.

    Parameters
    ----------
    img_stack : ndarray
        AFM image input with shape ``(H, W)`` (single image) or ``(N, H, W)``
        (stack). The output has the same shape as the input.
    routine : str
        Name of a routine defined in ``ROUTINES`` (e.g., ``'multi-plane-otsu'``).

    Returns
    -------
    result : ndarray
        Leveled image or stack with the same shape as ``img_stack``.

    Raises
    ------
    ValueError
        If ``img_stack`` is not 2D/3D or if ``routine`` is not found in
        ``ROUTINES``.

    Notes
    -----
    - Each routine is an ordered list of steps. Steps may call ``apply_level``,
      ``apply_level_weighted``, or ``apply_thresholder``.
    - Thresholding steps compute an *exclusion mask* (``True = excluded``,
      ``False = valid``) which is carried forward and passed to subsequent
      leveling steps until replaced by a later thresholding step.
    - Preconditioning (anisotropy-gated ``med_line``) is applied at most once
      per frame according to ``PRECOND_POLICIES``.
    - Gaussian-derived histogram bounds are computed by fitting a single
      Gaussian to a 100-bin histogram (MATLAB ``gauss1``-style) via
      ``_compute_gauss_limits`` when a thresholder step declares ``args=['gauss_*']``.
    - The Gaussian-derived histogram bounds are calculated for each frame rather
      than globally across the stack, diverging from MATLAB behavior.

    Version
    -------
    0.1.0
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

        for _idx, step in enumerate(steps):
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
                debug=False,
            )

            result[i] = img

    return np.asarray(result[0]) if was_2d else np.asarray(result)


apply_level_auto.__version__ = "0.1.0"


__all__ = ["apply_level_auto", "ROUTINES"]
