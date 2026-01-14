"""Tests for the level_auto module."""

from typing import Any

import numpy as np
import pytest

from pnanolocz_lib import level_auto
from pnanolocz_lib.level_auto import (
    ROUTINES,
    _compute_anisotropy_ratio,
    _compute_gauss_limits,
    _gauss1_model,
    _matches_trigger,
    _maybe_inject_precond,
    apply_level_auto,
)

# ------------------------------
# Smoke/tests of ROUTINES
# ------------------------------


def test_routines_contains_expected_names():
    """Exposes expected routine keys such as 'plane-line' (smoke check)."""
    assert "plane-line" in ROUTINES
    assert "iterative 1nm high" in ROUTINES
    assert "iterative -1nm low" in ROUTINES
    assert "iterative high low" in ROUTINES
    assert "Line1 + Otsu Line2" in ROUTINES
    assert "high-low x2 (fit)" in ROUTINES
    assert "iterative fit holes" in ROUTINES
    assert "iterative fit peaks" in ROUTINES
    assert "multi-plane-edges" in ROUTINES
    assert "multi-plane-otsu" in ROUTINES


# ------------------------------
# _compute_gauss_limits
# ------------------------------


def test_compute_gauss_limits_fit_band_is_mu_pm_1p5_sqrt2_sigma():
    """'gauss_fit' uses gauss1 width (c1), so limits ≈ μ ± 1.5*sqrt(2)*σ."""
    rng = np.random.default_rng(0)
    base = rng.normal(loc=5.0, scale=2.0, size=30_000).astype(float)
    base[::97] = np.nan  # sprinkle NaNs to exercise NaN-safe path

    lo, hi = _compute_gauss_limits(base, "gauss_fit")

    # Empirical μ, σ over finite values
    finite = base[np.isfinite(base)]
    mu = float(np.mean(finite))
    sigma = float(np.std(finite, ddof=0))

    # For gauss1, band is μ ± 1.5*sqrt(2)*σ (since c1 = sqrt(2)*σ)
    expected_delta = 1.5 * np.sqrt(2.0) * sigma

    # Allow a modest tolerance to account for histogram binning + fit noise
    assert abs((mu - expected_delta) - lo) < 0.5
    assert abs((mu + expected_delta) - hi) < 0.5

    # Optional symmetry check around the empirical center
    center = 0.5 * (lo + hi)
    assert abs(center - mu) < 0.25


def test_compute_gauss_limits_peaks_and_holes_shapes():
    """Produces half-bounded intervals for 'gauss_peaks' and 'gauss_holes'."""
    rng = np.random.default_rng(1)
    x = rng.normal(loc=0.0, scale=1.0, size=20_000)
    lo_p, hi_p = _compute_gauss_limits(x, "gauss_peaks")
    lo_h, hi_h = _compute_gauss_limits(x, "gauss_holes")
    # Peaks: (-inf, μ + 1.5σ); Holes: (μ - 1.5σ, +inf).
    assert np.isneginf(lo_p) and np.isfinite(hi_p)
    assert np.isfinite(lo_h) and np.isposinf(hi_h)


def test_compute_gauss_limits_rejects_unknown_kind():
    """Raises ValueError for unknown 'kind' strings."""
    with pytest.raises(ValueError):
        _compute_gauss_limits(np.array([0.0, 1.0]), "not-a-kind")


# ------------------------------
# _matches_trigger
# ------------------------------


def test_matches_trigger_true_for_apply_level_plane_1_1():
    """Test that _matches_trigger returns True for matching apply_level params."""

    def apply_level():  # name matters
        raise RuntimeError("not called")

    params = {"method": "plane", "polyx": 1, "polyy": 1}
    trigger = {
        "after_step": {
            "func": "apply_level",
            "method": "plane",
            "polyx": 1,
            "polyy": 1,
        }  # list of triggers
    }
    assert _matches_trigger(apply_level, params, trigger) is True


def test_matches_trigger_false_for_wrong_func_name():
    """Test that _matches_trigger returns False for non-matching func name."""

    def something_else():
        raise RuntimeError("not called")

    params = {"method": "plane", "polyx": 1, "polyy": 1}
    trigger = {
        "after_step": {"func": "apply_level", "method": "plane", "polyx": 1, "polyy": 1}
    }
    assert _matches_trigger(something_else, params, trigger) is False


def test_matches_trigger_false_for_param_mismatch():
    """Test that _matches_trigger returns False for non-matching params."""

    def apply_level():
        raise RuntimeError("not called")

    params = {"method": "line", "polyx": 1, "polyy": 1}  # method mismatch
    trigger = {
        "after_step": {
            "func": "apply_level",
            "method": "plane",
            "polyx": 1,
            "polyy": 1,
        }  # list of methods
    }
    assert _matches_trigger(apply_level, params, trigger) is False


# ------------------------------
# _compute_anisotropy_ratio
# ------------------------------


def test_compute_anisotropy_ratio_constant_image_gives_zero_ratio():
    """Test that _compute_anisotropy_ratio returns zero ratio for a constant image."""
    img = np.zeros((32, 32), dtype=np.float64)
    std_x, std_y, ratio = _compute_anisotropy_ratio(img)
    assert std_x == 0.0
    assert std_y == 0.0
    assert ratio == 0.0


def test_compute_anisotropy_ratio_std_x_zero_std_y_nonzero_gives_inf():
    """Test that _compute_anisotropy_ratio returns inf ratio if std_x=0 and std_y>0."""
    # rows vary, columns do not => col_means constant => std_x = 0, std_y > 0
    H, W = 40, 30
    img = np.tile(np.linspace(0.0, 10.0, H, dtype=np.float64)[:, None], (1, W))
    std_x, std_y, ratio = _compute_anisotropy_ratio(img)
    assert std_x == 0.0
    assert std_y > 0.0
    assert np.isinf(ratio)


def test_compute_anisotropy_ratio_handles_nans():
    """Test that _compute_anisotropy_ratio handles NaNs without error."""
    H, W = 32, 32
    img = np.tile(np.linspace(0.0, 1.0, H, dtype=np.float64)[:, None], (1, W))
    img[::7, ::5] = np.nan

    std_x, std_y, ratio = _compute_anisotropy_ratio(img)

    assert std_x >= 0.0
    assert std_y >= 0.0
    assert np.isfinite(std_x)
    assert np.isfinite(std_y)
    # ratio can be finite or inf depending on std_x guard; but must not be NaN
    assert not np.isnan(ratio)


# ------------------------------
# _gauss1_model
# ------------------------------


def test_gauss1_model_shape_and_dtype():
    """Test that _gauss1_model retnrs the correct shape and data type."""
    x = np.linspace(-3, 3, 101, dtype=np.float64)
    y = _gauss1_model(x, a1=10.0, b1=0.5, c1=2.0)
    assert y.shape == x.shape
    assert y.dtype == np.float64
    assert np.all(y >= 0.0)


# ------------------------------
# _maybe_inject_precond
# ------------------------------


def test_maybe_inject_precond_no_policy_no_injection():
    """Test that no injection occurs when routine is not in triggers."""
    img = np.zeros((16, 16), dtype=np.float64)

    def apply_level():
        raise RuntimeError("not called")

    out, injected = _maybe_inject_precond(
        img,
        routine="not-a-routine",
        func_obj=apply_level,
        params={"method": "plane", "polyx": 1, "polyy": 1},
        injected=False,
        apply_level_fn=lambda *a, **k: img + 999.0,  # would be obvious if called
        debug=False,
    )
    assert injected is False
    assert np.allclose(out, img)


def test_maybe_inject_precond_trigger_mismatch_no_injection():
    """Test that no injection occurs when trigger does not match."""
    # Routine exists, but params don't match trigger (method != plane)
    img = np.zeros((16, 16), dtype=np.float64)

    def apply_level():
        raise RuntimeError("not called")

    out, injected = _maybe_inject_precond(
        img,
        routine="multi-plane-edges",
        func_obj=apply_level,
        params={"method": "line", "polyx": 1, "polyy": 1},
        injected=False,
        apply_level_fn=lambda *a, **k: img + 999.0,
        debug=False,
    )
    assert injected is False
    assert np.allclose(out, img)


def test_maybe_inject_precond_ratio_gate_fires_and_calls_apply_level_fn():
    """Test that injection occurs when anisotropy ratio gate fires."""
    # Build an image with std_x=0, std_y>0 => ratio = inf => should pass first gate.
    H, W = 50, 20
    img = np.tile(np.linspace(0.0, 10.0, H, dtype=np.float64)[:, None], (1, W))

    calls: list[dict[str, Any]] = []

    def apply_level_fn(im, *, polyx, polyy, method, mask):
        calls.append(
            {
                "polyx": polyx,
                "polyy": polyy,
                "method": method,
                "mask_is_none": mask is None,
            }
        )
        return np.asarray(im + 1.0, dtype=np.float64)

    # func_obj name must be "apply_level" to match trigger
    def apply_level():
        raise RuntimeError("not called")

    out, injected = _maybe_inject_precond(
        img,
        routine="multi-plane-edges",
        func_obj=apply_level,
        params={"method": "plane", "polyx": 1, "polyy": 1},
        injected=False,
        apply_level_fn=apply_level_fn,
        debug=False,
    )

    assert injected is True
    assert len(calls) == 1
    assert calls[0]["method"] == "med_line"
    assert calls[0]["polyy"] == 0
    assert calls[0]["mask_is_none"] is True
    # first gate for multi-plane-edges is (7.0, 1.0) => polyx should be 1.0
    assert calls[0]["polyx"] == 1.0
    assert np.allclose(out, img + 1.0)


def test_maybe_inject_precond_only_once_if_already_injected():
    """Test that no injection occurs if already injected."""
    img = np.zeros((20, 20), dtype=np.float64)

    calls = {"n": 0}

    def apply_level_fn(im, **kwargs):
        calls["n"] += 1
        return np.asarray(im + 1.0, dtype=np.float64)

    def apply_level():
        raise RuntimeError("not called")

    out, injected = _maybe_inject_precond(
        img,
        routine="multi-plane-edges",
        func_obj=apply_level,
        params={"method": "plane", "polyx": 1, "polyy": 1},
        injected=True,  # already injected
        apply_level_fn=apply_level_fn,
        debug=False,
    )

    assert injected is True
    assert calls["n"] == 0  # must not call again
    assert np.allclose(out, img)


# ------------------------------
# apply_level_auto: gaussian args intercept
# ------------------------------


def test_apply_level_auto_intercepts_gauss_args_and_passes_tuple(monkeypatch):
    """Test that apply_level_auto intercepts 'gauss_fit' args and passes limits."""
    # Patch compute_gauss_limits so we can assert it was used
    seen = {"called": False, "kind": None}

    def fake_compute_gauss_limits(image, kind):
        seen["called"] = True
        seen["kind"] = kind
        return (-1.0, 2.0)

    monkeypatch.setattr(level_auto, "_compute_gauss_limits", fake_compute_gauss_limits)

    # Patch apply_thresholder inside la namespace
    # (identity check uses la.apply_thresholder)
    def fake_apply_thresholder(img, method, args, invert=False):
        # args must be the tuple from fake_compute_gauss_limits
        assert args == (-1.0, 2.0)
        assert method == "histogram"
        return np.zeros_like(img, dtype=bool)

    monkeypatch.setattr(level_auto, "apply_thresholder", fake_apply_thresholder)

    # Patch apply_level to avoid bringing in the real leveling stack
    def fake_apply_level(img, polyx, polyy, method, mask=None):
        return np.asarray(img, dtype=np.float64)  # identity

    monkeypatch.setattr(level_auto, "apply_level", fake_apply_level)

    # Create a tiny custom routine that *only* runs the gaussian threshold step.
    # (Important: func must be la.apply_thresholder so the "is" check triggers)
    monkeypatch.setitem(
        ROUTINES,
        "_gauss_only_test",
        [
            {
                "func": level_auto.apply_thresholder,
                "method": "histogram",
                "args": ["gauss_fit"],
                "invert": False,
            },
            {
                "func": level_auto.apply_level,
                "polyx": 0,
                "polyy": 0,
                "method": "mean_plane",
            },
        ],
    )

    img = np.random.default_rng(0).normal(size=(16, 16)).astype(np.float64)
    out = apply_level_auto(img, routine="_gauss_only_test")

    assert seen["called"] is True
    assert seen["kind"] == "gauss_fit"
    assert out.shape == img.shape
    assert np.isfinite(out).all()


# ------------------------------
# apply_level_auto: shape & errors
# ------------------------------


def test_apply_level_auto_2d_input_returns_2d():
    """Accepts 2D (H,W) input and returns a 2D array for valid routine."""
    rng = np.random.default_rng(7)
    H, W = 32, 24
    # Synthetic tilted plane + noise
    y = np.linspace(-1, 1, H)[:, None]
    x = np.linspace(-1.5, 1.5, W)[None, :]
    img2d = (0.8 * x - 0.3 * y + 0.1) + 0.05 * rng.normal(size=(H, W))
    out = apply_level_auto(img2d, routine="plane-line")  # plane then med_line
    assert out.shape == img2d.shape
    assert np.isfinite(out).all()


def test_apply_level_auto_3d_stack_shape_preserved():
    """Accepts (N,H,W) stack and preserves shape for a valid routine."""
    rng = np.random.default_rng(21)
    img = rng.normal(size=(2, 32, 32)).astype(float)
    out = apply_level_auto(img, routine="Line1 + Otsu Line2")
    assert out.shape == img.shape
    assert np.isfinite(out).all()


def test_apply_level_auto_unknown_routine_raises():
    """Raises ValueError when routine name is not defined."""
    img = np.zeros((16, 16))
    with pytest.raises(ValueError):
        apply_level_auto(img, routine="this-routine-does-not-exist")


def test_apply_level_auto_invalid_ndim_raises():
    """Raises ValueError for inputs not 2D or 3D."""
    bad = np.zeros((2, 3, 4, 5))
    with pytest.raises(ValueError):
        apply_level_auto(bad, routine="plane-line")


# ------------------------------
# apply_level_auto: effect checks
# ------------------------------


def test_plane_line_reduces_row_col_trend_on_tilted_plane():
    """Plane-line routine reduces row/column mean trends on a synthetic plane."""
    rng = np.random.default_rng(0)
    H, W = 48, 36
    y = np.linspace(-1, 1, H)[:, None]
    x = np.linspace(-2, 2, W)[None, :]
    plane = 1.2 * x - 0.7 * y + 0.25
    img = plane + 0.05 * rng.normal(size=(H, W))

    out = apply_level_auto(img, routine="plane-line")  # plane → med_line
    # Compare dynamic range of row/col means before vs. after
    row_rng_before = np.ptp(img.mean(axis=1))
    col_rng_before = np.ptp(img.mean(axis=0))
    row_rng_after = np.ptp(out.mean(axis=1))
    col_rng_after = np.ptp(out.mean(axis=0))

    assert row_rng_after <= row_rng_before * 0.6  # materially reduced
    assert col_rng_after <= col_rng_before * 0.6


def test_high_low_x2_fit_runs_and_is_finite():
    """'high-low x2 (fit)' routine runs end-to-end and yields finite output."""
    rng = np.random.default_rng(123)
    img = rng.normal(size=(32, 32))
    out = apply_level_auto(img, routine="high-low x2 (fit)")
    assert out.shape == img.shape
    assert np.isfinite(out).all()


def test_iterative_fit_holes_and_peaks_run_and_stay_finite():
    """Test that iterative fit holes and peaks routines both return finite arrays."""
    rng = np.random.default_rng(5)
    stack = rng.normal(size=(2, 24, 24))
    out_holes = apply_level_auto(stack, routine="iterative fit holes")
    out_peaks = apply_level_auto(stack, routine="iterative fit peaks")
    assert out_holes.shape == stack.shape
    assert out_peaks.shape == stack.shape
    assert np.isfinite(out_holes).all()
    assert np.isfinite(out_peaks).all()


# ---------------------------------
# apply_level_auto: real data tests
# ---------------------------------


def test_plane_line_runs_on_real_spm_resource_and_improves_trend(load_npz):
    """Test that plane-line routine runs on real AFM data and reduces trends."""
    z = load_npz("afm_0_00003_raw.npz")
    data = z["data"]
    assert data.ndim in (2, 3)

    # pick frame 5 if stack, else use image
    img = data[5] if data.ndim == 3 and data.shape[0] > 5 else data

    out = apply_level_auto(img, routine="plane-line")

    assert out.shape == img.shape
    assert np.isfinite(out).all()

    # Trend reduction check: row/col mean range should drop noticeably
    row_before = np.ptp(np.nanmean(img, axis=1))
    col_before = np.ptp(np.nanmean(img, axis=0))
    row_after = np.ptp(np.nanmean(out, axis=1))
    col_after = np.ptp(np.nanmean(out, axis=0))

    assert row_after <= row_before * 0.8
    assert col_after <= col_before * 0.8


def nrms(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the normalized root-mean-square difference between two arrays."""
    denom = np.linalg.norm(b.ravel())
    if denom == 0:
        return float(np.linalg.norm((a - b).ravel()))
    return float(np.linalg.norm((a - b).ravel()) / denom)


def test_plane_line_matches_reference_with_tolerance(load_npz):
    """Test that plane-line routine matches reference data within tolerance."""
    z_img = load_npz("afm_0_00003_raw.npz")  # raw input image
    z_peaks_ref = load_npz(
        "afm_0p0_00003_nanolocz_fitpeaks.npz"
    )  # reference image processed with Nanolocz using 'iterative fit peaks' routine

    z_mpo_ref = load_npz(
        "afm_0p0_00003_nanolocz_multiplaneotsu.npz"
    )  # reference image processed with Nanolocz using 'multi-plane-otsu' routine
    img = z_img.get("data")
    peaks_ref = z_peaks_ref.get("data")
    mpo_ref = z_mpo_ref.get("data")

    img = img.astype(float)
    peaks_ref = peaks_ref.astype(float)
    mpo_ref = mpo_ref.astype(float)

    out_peaks = apply_level_auto(img, routine="iterative fit peaks")
    out_mpo = apply_level_auto(img, routine="multi-plane-otsu")
    assert nrms(out_peaks, peaks_ref) < 0.02  # within 2% NRMS difference
    assert nrms(out_mpo, mpo_ref) < 0.02  # within 2% NRMS difference
