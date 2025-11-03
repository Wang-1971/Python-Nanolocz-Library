"""Tests for the level_auto module."""

import numpy as np
import pytest

from pnanolocz_lib.level_auto import ROUTINES, _compute_gauss_limits, apply_level_auto

# ------------------------------
# Smoke/tests of ROUTINES
# ------------------------------


def test_routines_contains_expected_names():
    """Exposes expected routine keys such as 'plane-line' (smoke check)."""
    assert "plane-line" in ROUTINES
    assert "iterative 1nm high" in ROUTINES
    assert "iterative -1nm low" in ROUTINES


# ------------------------------
# _compute_gauss_limits
# ------------------------------


def test_compute_gauss_limits_fit_band_is_mu_pm_1p5_sigma():
    """Returns limits close to μ ± 1.5σ for 'gauss_fit' with NaN-tolerant fit."""
    rng = np.random.default_rng(0)
    # Target μ ≈ 5, σ ≈ 2 with some NaNs sprinkled in
    base = rng.normal(loc=5.0, scale=2.0, size=30_000)
    base[::97] = np.nan
    lo, hi = _compute_gauss_limits(base, "gauss_fit")
    # Expected band ~ [5 - 3, 5 + 3] with small tolerance
    assert abs((5.0 - 3.0) - lo) < 0.4
    assert abs((5.0 + 3.0) - hi) < 0.4


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
