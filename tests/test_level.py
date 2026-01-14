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
# level_med_line (rows)
# -------------------------------


def test_level_med_line_small_image_rows_unchanged():
    """Test rows with <=10 valid pixels are left unchanged (MATLAB guard)."""
    img = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=float)

    # User mask: 1 = valid; convert to exclusion (True=excluded)
    user_mask_valid = np.array([[1, 1, 0], [1, 0, 1], [1, 1, 1]], dtype=int)
    mask_excl = user_mask_valid == 0  # bool

    leveled = level_med_line(img, mask_excl, polyx=1.0, polyy=0)

    # Because each row has <=10 valid pixels, the function leaves rows unchanged.
    np.testing.assert_allclose(leveled, img, equal_nan=True)


def test_level_med_line_large_image_median_subtraction_and_recentering():
    """
    Test rows with >10 valid pixels.

    Subtract per-row median and add global masked median.
    """
    # Build a 3x20 image with clear, easy-to-check medians
    _, cols = 3, 20
    # Row medians will be easy if rows are arithmetic sequences:
    img = np.vstack(
        [
            np.linspace(1.0, 20.0, cols),  # row 0 median = 10.5
            np.linspace(101.0, 120.0, cols),  # row 1 median = 110.5
            np.linspace(-20.0, -1.0, cols),  # row 2 median = -10.5
        ]
    )

    # Exclusion mask: False everywhere (i.e., all pixels valid)
    mask_excl = np.zeros_like(img, dtype=bool)

    # Use polyx=1.0 (strength = 1), polyy ignored
    leveled = level_med_line(img, mask_excl, polyx=1.0, polyy=0)

    # Global masked median over all valid pixels
    bg = float(np.median(img))

    # Expected per-row: img - row_median + bg
    row_meds = np.array(
        [np.median(img[0, :]), np.median(img[1, :]), np.median(img[2, :])], dtype=float
    )
    expected = img - row_meds[:, None] + bg

    np.testing.assert_allclose(leveled, expected, rtol=1e-7, atol=0, equal_nan=True)


def test_level_med_line_mixed_row_coverage():
    """Test rows that pass the guard are leveled; rows that fail remain unchanged."""
    _, cols = 3, 20
    img = np.vstack(
        [
            np.linspace(1.0, 20.0, cols),  # row 0
            np.linspace(4.0, 23.0, cols),  # row 1
            np.linspace(7.0, 26.0, cols),  # row 2
        ]
    )

    # Exclusion mask:
    # - Row 0: 12 valid pixels (False) then 8 excluded (True) -> passes guard (>10)
    # - Row 1: 10 valid pixels then 10 excluded -> fails guard (==10)
    # - Row 2: all 20 valid -> passes guard
    mask_excl = np.zeros_like(img, dtype=bool)
    mask_excl[0, 12:] = True  # exclude last 8 columns
    mask_excl[1, 10:] = True  # exclude last 10 columns
    # row 2 remains all False (all valid)

    leveled = level_med_line(img, mask_excl, polyx=1.0, polyy=0)

    # Compute bg over valid pixels only
    masked = np.where(~mask_excl, img, np.nan)
    bg = float(np.nanmedian(masked))

    # Row medians over valid pixels
    r0_med = float(np.median(img[0, ~mask_excl[0]]))  # 12 valid
    _ = float(np.median(img[1, ~mask_excl[1]]))  # 10 valid (fails guard)
    r2_med = float(np.median(img[2, ~mask_excl[2]]))  # 20 valid

    expected = img.copy()
    expected[0, :] = img[0, :] - r0_med + bg  # row 0 leveled
    expected[1, :] = img[1, :]  # row 1 unchanged (<=10 valids)
    expected[2, :] = img[2, :] - r2_med + bg  # row 2 leveled

    np.testing.assert_allclose(leveled, expected, rtol=1e-7, atol=0, equal_nan=True)


def test_level_med_line_gain_polyx_positive():
    """Test that polyx>0 scales the row median baseline before re-centering."""
    _, cols = 2, 20
    img = np.vstack(
        [
            np.linspace(0.0, 19.0, cols),  # row 0 median = 9.5
            np.linspace(10.0, 29.0, cols),  # row 1 median = 19.5
        ]
    )
    mask_excl = np.zeros_like(img, dtype=bool)  # all valid

    polyx = 0.6
    leveled = level_med_line(img, mask_excl, polyx=polyx, polyy=0)

    bg = float(np.median(img))
    row_meds = np.array([np.median(img[0, :]), np.median(img[1, :])], dtype=float)
    expected = img - (polyx * row_meds[:, None]) + bg

    np.testing.assert_allclose(leveled, expected, rtol=1e-7, atol=0)


def test_level_med_line_gain_polyx_nonpositive_defaults_to_one():
    """Test that polyx<=0 defaults gain to 1.0."""
    _, cols = 2, 20
    img = np.vstack(
        [np.arange(cols, dtype=float), np.arange(cols, dtype=float) + 100.0]
    )
    mask_excl = np.zeros_like(img, dtype=bool)

    leveled = level_med_line(img, mask_excl, polyx=0.0, polyy=0)  # nonpositive
    bg = float(np.median(img))
    row_meds = np.array([np.median(img[0, :]), np.median(img[1, :])], dtype=float)
    expected = img - row_meds[:, None] + bg

    np.testing.assert_allclose(leveled, expected, rtol=1e-7, atol=0)


def test_level_med_line_mask_convention_exclusion_true():
    """
    Library expects exclusion masks (True=excluded).

    Passing a validity mask instead produces a different result once rows meet the
    guard.
    """
    _, cols = 2, 20
    # Two rows with different medians to avoid bg == row_median cancellation
    # Row 0: 0..19  (median ≈ 9.5)
    # Row 1: 100..119 (median ≈ 109.5)
    img = np.vstack(
        [
            np.linspace(0.0, 19.0, cols),
            np.linspace(100.0, 119.0, cols),
        ]
    )

    # User-level validity mask: 1 = valid, 0 = invalid.
    # In row 0, mark first 5 pixels invalid; row 1 all valid.
    user_valid = np.ones_like(img, dtype=int)
    user_valid[0, 0:5] = 0

    # WRONG: pass validity directly (True=valid) to an API that expects
    # exclusion mask (True=excluded). This will invert semantics.
    wrong_mask = user_valid.astype(bool)  # True=valid (WRONG for API)
    leveled_wrong = level_med_line(img, wrong_mask, polyx=1.0, polyy=0)

    # RIGHT: convert to exclusion mask (True=excluded)
    mask_excl = user_valid == 0  # True=excluded (CORRECT)
    leveled_right = level_med_line(img, mask_excl, polyx=1.0, polyy=0)

    # Both rows have >10 valid pixels under the CORRECT mask:
    # - Row 0: 15 valid → row_median applied
    # - Row 1: 20 valid → row_median applied
    # The global masked median (over both rows) differs from each row median,
    # so leveled_right != img and leveled_wrong != leveled_right in general.
    assert not np.allclose(leveled_wrong, leveled_right)


def test_level_med_line_default_mask_small_image_rows_unchanged():
    """Test when mask=None and rows have <=10 valid pixels, rows remain unchanged."""
    img = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
    leveled = level_med_line(img, mask=None, polyx=1, polyy=0)

    # Guard (>10) prevents median subtraction on all rows in this tiny image.
    np.testing.assert_allclose(leveled, img, equal_nan=True)


def test_level_med_line_default_mask_wide_image_applies_row_median_and_bg():
    """Test when mask=None and rows have >10 finite pixels.

    Subtract per-row median and add global nanmedian.
    """
    _, cols = 3, 20
    img = np.vstack(
        [
            np.linspace(1.0, 20.0, cols),  # no NaNs -> valid count 20
            np.linspace(101.0, 120.0, cols),
            np.linspace(-20.0, -1.0, cols),
        ]
    )
    # Sprinkle a few NaNs but keep valid count > 10
    img[0, [3, 7]] = np.nan
    img[1, [1, 12, 18]] = np.nan
    img[2, [5]] = np.nan

    leveled = level_med_line(img, mask=None, polyx=1, polyy=0)

    bg = float(np.nanmedian(img))
    row_medians = np.array(
        [np.nanmedian(img[0, :]), np.nanmedian(img[1, :]), np.nanmedian(img[2, :])],
        dtype=float,
    )
    expected = img - row_medians[:, None] + bg

    np.testing.assert_allclose(leveled, expected, equal_nan=True)


def test_level_med_line_polyx_scale_and_zero():
    """Test that polyx acts as a scale.

    polyx==0 behaves like scale=1 (rows must pass guard).
    """
    _, cols = 2, 20
    # Simple sequences with known medians:
    # row 0 median = 9.5, row 1 median = 29.5
    img = np.vstack(
        [
            np.linspace(0.0, 19.0, cols),
            np.linspace(20.0, 39.0, cols),
        ]
    )

    # mask=None → all finite are valid; each row has 20 valid pixels (>10)
    leveled_scale1 = level_med_line(img, mask=None, polyx=0, polyy=0)  # defaults to 1.0
    leveled_scale2 = level_med_line(img, mask=None, polyx=2, polyy=0)  # scale = 2.0

    bg = float(np.nanmedian(img))  # median of all valid pixels

    row_medians = np.array(
        [
            np.nanmedian(img[0, :]),
            np.nanmedian(img[1, :]),
        ],
        dtype=float,
    )  # → [9.5, 29.5]

    expected1 = img - row_medians[:, None] + bg
    expected2 = img - 2.0 * row_medians[:, None] + bg

    np.testing.assert_allclose(leveled_scale1, expected1, equal_nan=True)
    np.testing.assert_allclose(leveled_scale2, expected2, equal_nan=True)


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


def test_get_background_raises_on_unknown_method():
    """Test that get_background raises a ValueError on unknown method."""
    img = np.zeros((5, 5))
    with pytest.raises(ValueError):
        get_background(img, polyx=0, polyy=0, method="not-a-method", mask=None)


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


# ---------------------------------
# apply_level: real data tests
# ---------------------------------


def nrmse_range(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the root-mean-square difference normalised by dynamic range."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return 0.0
    rmse = np.sqrt(np.mean((a[m] - b[m]) ** 2))
    denom = float(np.nanmax(b[m]) - np.nanmin(b[m]))
    return float(rmse / denom) if denom != 0 else float(rmse)


def test_apply_level_plane_runs_on_real_resource(load_npz):
    """Test that apply_level with 'plane' method runs on real data."""
    z = load_npz("afm_0_00003_raw.npz")
    img = z.get("data")

    out = apply_level(img, polyx=1, polyy=1, method="plane", mask=None)

    assert out.shape == img.shape
    assert np.isfinite(out).all()

    # trend reduction: row/col mean range should drop
    row_before = np.ptp(np.nanmean(img, axis=1))
    col_before = np.ptp(np.nanmean(img, axis=0))
    row_after = np.ptp(np.nanmean(out, axis=1))
    col_after = np.ptp(np.nanmean(out, axis=0))

    assert row_after <= row_before * 0.8
    assert col_after <= col_before * 0.8


@pytest.mark.parametrize(
    "method,polyx,polyy,ref_file,tol",
    [
        ("plane", 1, 1, "afm_0_00003_nanolocz_plane_1_1.npz", 1e-14),
        ("line", 1, 0, "afm_0_00003_nanolocz_line_1_0.npz", 1e-14),
        ("med_line", 1, 0, "afm_0_00003_nanolocz_medline_1_0.npz", 1e-14),
    ],
)
def test_apply_level_matches_matlab_reference(
    load_npz, method, polyx, polyy, ref_file, tol
):
    """Apply_level matches MATLAB reference outputs on a real AFM resource."""
    # Load raw image and force to 2D if stored as (1,H,W)
    z = load_npz("afm_0_00003_raw.npz")
    img = z["data"]
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    assert img.ndim == 2

    # Load MATLAB reference (should be 2D)
    ref = load_npz(ref_file)["data"]
    if ref.ndim == 3 and ref.shape[0] == 1:
        ref = ref[0]
    assert ref.ndim == 2

    out = apply_level(img, polyx=polyx, polyy=polyy, method=method, mask=None)

    assert out.shape == ref.shape
    assert nrmse_range(out, ref) < tol


def test_get_background_consistency(load_npz):
    """Test that img - apply_level == get_background for 'plane' method."""
    img = load_npz("afm_0_00003_raw.npz")["data"]

    out = apply_level(img, polyx=1, polyy=1, method="plane", mask=None)
    bg = get_background(img, polyx=1, polyy=1, method="plane", mask=None)

    # by definition: img - out == bg
    assert np.allclose(img - out, bg, rtol=0, atol=1e-12)
