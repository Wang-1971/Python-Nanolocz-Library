"""Tests for the level module."""

import numpy as np
import pytest

from pnanolocz_lib.level import (
    apply_level,
    get_background,
    level_line,
    level_log_y,
    level_mean_plane,
    level_med_line,
    level_med_line_y,
    level_plane,
    level_smed_line,
)


# -------------------------------
# Helpers
# -------------------------------
def _with_nans(arr, frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    arr = arr.copy().astype(float)
    m, n = arr.shape
    k = int(frac * m * n)
    if k > 0:
        idx = rng.choice(m * n, size=k, replace=False)
        arr.reshape(-1)[idx] = np.nan
    return arr


# -------------------------------
# Existing tests you had (kept)
# -------------------------------


def test_level_med_line_basic():
    """Row-wise median subtraction + bg re-add with explicit mask."""
    img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    mask = np.array([[1, 1, 0], [1, 0, 1], [1, 1, 1]], dtype=int)

    # polyx = 1, polyy ignored
    leveled = level_med_line(img, mask, polyx=1, polyy=0)

    # Row medians over masked pixels:
    # row 0: median([1,2]) = 1.5
    # row 1: median([4,6]) = 5.0
    # row 2: median([7,8,9]) = 8.0
    bg = np.median([1, 2, 4, 6, 7, 8, 9])  # all masked values
    expected = img - np.array([[1.5], [5.0], [8.0]]) + bg

    np.testing.assert_allclose(leveled, expected)


def test_level_med_line_default_mask():
    """When mask=None, use non-NaN pixels and bg=nanmedian(img)."""
    img = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
    leveled = level_med_line(img, mask=None, polyx=1, polyy=0)

    bg = np.nanmedian(img)
    row_medians = np.array(
        [
            np.nanmedian(img[0, :]),
            np.nanmedian(img[1, :]),
            np.nanmedian(img[2, :]),
        ]
    )
    expected = img - row_medians[:, None] + bg

    np.testing.assert_allclose(leveled, expected)


def test_level_med_line_polyx_scale_and_zero():
    """Polyx acts as a scale; polyx==0 behaves like scale=1."""
    img = np.array([[1, 3], [2, 4]], dtype=float)
    # mask None -> all valid
    leveled_scale1 = level_med_line(img, None, polyx=0, polyy=0)
    leveled_scale2 = level_med_line(img, None, polyx=2, polyy=0)

    # For 2x2, row medians are [2,3], bg = median([1,2,3,4]) = 2.5
    expected1 = img - np.array([[2], [3]]) + 2.5
    expected2 = img - 2 * np.array([[2], [3]]) + 2.5

    np.testing.assert_allclose(leveled_scale1, expected1)
    np.testing.assert_allclose(leveled_scale2, expected2)


# -------------------------------
# level_med_line_y (columns)
# -------------------------------


def test_level_med_line_y_threshold_and_bg():
    """
    Test level_med_line_y: per-column subtraction of the column median + bg.

    Code subtracts only when valid count > 10 (edge condition).
    """
    # Construct an image where only some columns exceed the >10 threshold
    img = np.tile(np.arange(12, dtype=float), (12, 1))  # shape (12,12)
    # Add some NaNs to one column so it barely meets threshold
    img_nan = img.copy()
    img_nan[:1, 0] = np.nan  # 11 valid -> >10; should adjust
    img_nan[:3, 1] = np.nan  # 9 valid -> not adjusted

    leveled = level_med_line_y(img_nan, mask=None, polyx=0, polyy=0)

    # Column 0 should be adjusted by subtracting its median and adding bg
    bg = np.nanmedian(img_nan[~np.isnan(img_nan)])
    col0 = img_nan[:, 0]
    med0 = np.nanmedian(col0)
    expected_col0 = col0 - med0 + bg

    # Column 1 should remain unchanged
    np.testing.assert_allclose(leveled[:, 0], expected_col0)
    np.testing.assert_allclose(leveled[:, 1], img_nan[:, 1])

    # A column with all valid (>10) is also adjusted
    col2 = img_nan[:, 2]
    med2 = np.nanmedian(col2)
    expected_col2 = col2 - med2 + bg
    np.testing.assert_allclose(leveled[:, 2], expected_col2)


# -------------------------------
# level_smed_line (smoothed per-row medians)
# -------------------------------


def test_level_smed_line_basic_shape_and_nan_safety():
    """Returns same shape, handles NaNs, and reduces row-wise trends."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(40, 40))
    drift = np.linspace(-5, 5, 40)[:, None]  # row drift
    img = base + drift
    img = _with_nans(img, frac=0.05, seed=1)

    leveled = level_smed_line(img, mask=None, polyx=0, polyy=0)
    assert leveled.shape == img.shape

    # Row-wise mean absolute deviation should decrease vs original
    with np.errstate(invalid="ignore"):
        mad_before = np.nanmean(
            np.abs(img - np.nanmedian(img, axis=1)[:, None])
        )  # mad before
        mad_after = np.nanmean(
            np.abs(leveled - np.nanmedian(leveled, axis=1)[:, None])
        )  # mad after
    assert mad_after <= mad_before


# -------------------------------
# level_mean_plane
# -------------------------------


def test_level_mean_plane_centers_to_zero_mean():
    """Test that level_mean_plane centres to zero mean."""
    rng = np.random.default_rng(0)
    img = rng.normal(size=(16, 16))
    img[0, 0] = np.nan  # ensure NaN handling
    leveled = level_mean_plane(img, mask=None, polyx=0, polyy=0)
    assert abs(np.nanmean(leveled)) < 1e-12


# -------------------------------
# level_plane (global plane fit)
# -------------------------------


def test_level_plane_removes_synthetic_plane():
    """
    Build a synthetic tilted plane plus noise.

    Then verify mean-of-rows/cols is near zero after leveling.
    """
    H, W = 64, 48
    y = np.linspace(-1, 1, H)[:, None]
    x = np.linspace(-2, 2, W)[None, :]
    true_plane = 2.0 * x - 1.0 * y + 0.3
    rng = np.random.default_rng(0)
    img = true_plane + 0.05 * rng.normal(size=(H, W))

    # Enough polynomial order to capture the plane after centering
    leveled = level_plane(img, mask=None, polyx=1, polyy=1)

    # After removal, residual mean pattern along rows/cols should be small
    row_means = leveled.mean(axis=1)
    col_means = leveled.mean(axis=0)
    assert np.all(np.abs(row_means) < 0.1)
    assert np.all(np.abs(col_means) < 0.1)


# -------------------------------
# level_line (per-row/column polynomial)
# -------------------------------


def test_level_line_handles_low_valid_rows_with_median_fallback():
    """
    Test that code uses a fallback for rows with too few valid points.

    Subtracts the median of the fitted rows.
    """
    # Third row has only 3 valid values -> fallback triggers, but not all NaN
    row0 = np.linspace(0, 10, 20)  # enough points
    row1 = np.linspace(0, 10, 20)  # enough points
    row2 = np.full(20, np.nan)
    row2[-3:] = [1.0, 2.0, 3.0]  # only 3 valid (< 1+8 = 9)

    img = np.vstack([row0, row1, row2])
    mask = ~np.isnan(img)

    leveled = level_line(img, mask=mask, polyx=1, polyy=0)
    assert leveled.shape == img.shape

    # Valid positions in the third row should be finite
    valid = mask[2, :]
    assert np.all(np.isfinite(leveled[2, valid]))
    # NaNs should remain where input was NaN
    assert np.all(np.isnan(leveled[2, ~valid]))


# -------------------------------
# level_log_y
# -------------------------------


def test_level_log_y_reduces_y_trend():
    """
    Average along rows (y) is corrected using a log fit.

    Verify that the row-mean dynamic range is reduced.
    """
    H, W = 60, 40
    rng = np.random.default_rng(0)
    base = rng.normal(scale=0.2, size=(H, W))
    ytrend = (np.log(np.linspace(1, 10, H)))[..., None]
    img = base + ytrend

    leveled = level_log_y(img, mask=None, polyx=0, polyy=2)  # scale via polyy
    row_means_before = img.mean(axis=1)
    row_means_after = leveled.mean(axis=1)
    assert np.ptp(row_means_after) <= np.ptp(row_means_before)


# -------------------------------
# apply_level & get_background (single image)
# -------------------------------


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("plane", {"polyx": 1, "polyy": 1}),
        ("line", {"polyx": 1, "polyy": 1}),
        ("med_line", {"polyx": 1, "polyy": 0}),
        ("med_line_y", {"polyx": 0, "polyy": 0}),
        ("smed_line", {"polyx": 0, "polyy": 0}),
        ("mean_plane", {"polyx": 0, "polyy": 0}),
        ("log_y", {"polyx": 0, "polyy": 2}),
    ],
)
def test_apply_level_and_get_background_reconstruct(method, kwargs):
    """Check img ≈ leveled + background for all methods."""
    rng = np.random.default_rng(42)
    img = rng.normal(size=(24, 32)).astype(float)
    img = _with_nans(img, frac=0.05, seed=5)

    leveled = apply_level(img, method=method, mask=None, **kwargs)
    bg = get_background(img, method=method, mask=None, **kwargs)

    # Reconstruction (allow some floating tolerance)
    np.testing.assert_allclose(leveled + bg, img, rtol=1e-6, atol=1e-6)


def test_apply_level_raises_on_unknown_method():
    """Test that apply_level raises a ValueError on unknown method."""
    img = np.zeros((5, 5))
    with pytest.raises(ValueError):
        apply_level(img, polyx=0, polyy=0, method="not-a-method", mask=None)


def test_apply_level_mask_shape_mismatch():
    """Test that apply_level raises a ValueError if mask shape does not match image."""
    img = np.zeros((5, 5))
    mask = np.ones((5, 6), dtype=bool)
    with pytest.raises(ValueError):
        apply_level(img, polyx=0, polyy=0, method="mean_plane", mask=mask)


# -------------------------------
# apply_level & get_background (stack)
# -------------------------------


def test_apply_level_and_background_on_stack_shape_and_reconstruction():
    """Test that apply level and get_background give the same shape."""
    rng = np.random.default_rng(123)
    img = rng.normal(size=(3, 16, 12))
    mask = np.ones_like(img, dtype=bool)
    # Different methods spot-check; pick one polynomial and one median-based
    for method, kwargs in [
        ("plane", {"polyx": 1, "polyy": 1}),
        ("med_line", {"polyx": 1, "polyy": 0}),
    ]:
        leveled = apply_level(img, method=method, mask=mask, **kwargs)
        bg = get_background(img, method=method, mask=mask, **kwargs)

        assert leveled.shape == img.shape
        assert bg.shape == img.shape
        np.testing.assert_allclose(leveled + bg, img, rtol=1e-6, atol=1e-6)


# -------------------------------
# Idempotence on already-flat images
# -------------------------------


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("mean_plane", {"polyx": 0, "polyy": 0}),
        ("med_line", {"polyx": 1, "polyy": 0}),
        ("med_line_y", {"polyx": 0, "polyy": 0}),
        ("smed_line", {"polyx": 0, "polyy": 0}),
        ("plane", {"polyx": 1, "polyy": 1}),
        ("line", {"polyx": 1, "polyy": 1}),
    ],
)
def test_idempotence_on_flat_data(method, kwargs):
    """If data are already zero-mean per chosen method, changes should be tiny."""
    rng = np.random.default_rng(7)
    img = rng.normal(scale=1e-5, size=(20, 20))  # nearly flat noise
    leveled = apply_level(img, method=method, mask=None, **kwargs)
    np.testing.assert_allclose(leveled, img, rtol=1e-6, atol=5e-5)
