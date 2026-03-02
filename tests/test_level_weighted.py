"""Tests for the level_weighted module."""

import numpy as np
import pytest

from pnanolocz.level_weighted import (
    _find_regions,
    _polyfit_centered,
    _polyval_centered,
    apply_level_weighted,
    level_weighted_line,
    level_weighted_med_line,
    level_weighted_med_line_y,
    level_weighted_plane,
    level_weighted_smed_line,
)


def test_polyfit_and_polyval_centered():
    """Test polynomial fitting and evaluation with centering/scaling."""
    x = np.array([0, 1, 2, 3, 4], dtype=float)
    y = x**2
    coeffs, cent_scale = _polyfit_centered(x, y, 2)
    y_fit = _polyval_centered(coeffs, cent_scale, x)
    np.testing.assert_allclose(y_fit, y, atol=1e-12)


def test_find_regions_simple():
    """Test that connected foreground regions are correctly identified."""
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:3, 1:3] = True
    mask[4, 4] = True
    regions = _find_regions(mask, min_area=1)
    assert len(regions) == 2
    assert all(isinstance(r, np.ndarray) for r in regions)


def test_level_weighted_plane_basic():
    """Test plane leveling on a small synthetic image."""
    img = np.arange(16).reshape(4, 4).astype(float)
    regions = [np.arange(16)]
    leveled = level_weighted_plane(img, regions, 1, 1)
    assert leveled.shape == img.shape


def test_level_weighted_line_basic():
    """Test line leveling on a small synthetic image."""
    img = np.arange(16).reshape(4, 4).astype(float)
    regions = [np.arange(16)]
    leveled = level_weighted_line(img, regions, 1, 1)
    assert leveled.shape == img.shape


def test_level_weighted_med_line_basic():
    """Test median line leveling along rows."""
    img = np.arange(16).reshape(4, 4).astype(float)
    regions = [np.arange(16)]
    leveled = level_weighted_med_line(img, regions)
    assert leveled.shape == img.shape


def test_level_weighted_med_line_y_basic():
    """Test median line leveling along columns."""
    img = np.arange(16).reshape(4, 4).astype(float)
    regions = [np.arange(16)]
    leveled = level_weighted_med_line_y(img, regions)
    assert leveled.shape == img.shape


def test_level_weighted_smed_line_basic():
    """Test smoothed median line leveling along rows."""
    img = np.arange(16).reshape(4, 4).astype(float)
    regions = [np.arange(16)]
    leveled = level_weighted_smed_line(img, regions, smoothing_window=2)
    assert leveled.shape == img.shape


def test_apply_level_weighted_dispatch_plane():
    """Test dispatcher applies plane leveling to an image."""
    img = np.arange(16).reshape(4, 4).astype(float)
    leveled = apply_level_weighted(img, 1, 1, method="plane")
    assert leveled.shape == img.shape


def test_apply_level_weighted_dispatch_line():
    """Test dispatcher applies line leveling to an image."""
    img = np.arange(16).reshape(4, 4).astype(float)
    leveled = apply_level_weighted(img, 1, 1, method="line")
    assert leveled.shape == img.shape


def test_apply_level_weighted_dispatch_med_line():
    """Test dispatcher applies med_line leveling to an image."""
    img = np.arange(16).reshape(4, 4).astype(float)
    leveled = apply_level_weighted(img, 0, 0, method="med_line")
    assert leveled.shape == img.shape


def test_apply_level_weighted_dispatch_med_line_y():
    """Test dispatcher applies med_line_y leveling to an image."""
    img = np.arange(16).reshape(4, 4).astype(float)
    leveled = apply_level_weighted(img, 0, 0, method="med_line_y")
    assert leveled.shape == img.shape


def test_apply_level_weighted_dispatch_smed_line():
    """Test dispatcher applies smed_line leveling to an image."""
    img = np.arange(16).reshape(4, 4).astype(float)
    leveled = apply_level_weighted(img, 0, 0, method="smed_line", smoothing_window=2)
    assert leveled.shape == img.shape


def test_apply_level_weighted_with_mask():
    """Test that apply_level_weighted works correctly with a mask."""
    img = np.arange(16).reshape(4, 4).astype(float)
    mask = np.zeros_like(img, dtype=bool)
    mask[1:3, 1:3] = True
    leveled = apply_level_weighted(img, 1, 1, method="plane", mask=mask)
    assert leveled.shape == img.shape


@pytest.mark.parametrize(
    "img,mask_coords,method",
    [
        (np.ones((3, 3), dtype=float), [(0, 0)], "plane"),
        (np.ones((3, 3), dtype=float), [(0, 0), (1, 1)], "line"),
        (np.arange(9, dtype=float).reshape(3, 3), [(0, 1), (2, 2)], "med_line"),
        (np.arange(25, dtype=float).reshape(5, 5), [(1, 1), (3, 3)], "med_line_y"),
        (np.arange(16, dtype=float).reshape(4, 4), [(0, 0), (0, 1)], "smed_line"),
    ],
)
def test_apply_level_weighted_basic_masks(img, mask_coords, method):
    """Test apply_level_weighted on small images with simple masks."""
    mask = np.zeros_like(img, dtype=bool)
    for i, j in mask_coords:
        mask[i, j] = True

    leveled = apply_level_weighted(img, polyx=1, polyy=1, method=method, mask=mask)

    # Assert the output shape matches input
    assert leveled.shape == img.shape

    # Assert that masked pixels are influenced but not NaN
    assert np.all(np.isfinite(leveled[mask]))


@pytest.mark.parametrize(
    "method", ["plane", "line", "med_line", "med_line_y", "smed_line"]
)
def test_apply_level_weighted_full_mask(method):
    """Fully masked image should either return unchanged or handle gracefully."""
    img = np.ones((3, 3), dtype=float)
    mask = np.ones_like(img, dtype=bool)

    leveled = apply_level_weighted(img, polyx=1, polyy=1, method=method, mask=mask)

    # In your implementation, full mask should not crash; output should be finite
    assert np.all(np.isfinite(leveled))
    assert leveled.shape == img.shape


@pytest.mark.parametrize(
    "method", ["plane", "line", "med_line", "med_line_y", "smed_line"]
)
def test_apply_level_weighted_no_mask(method):
    """Test that leveling runs without a mask (whole image considered)."""
    img = np.arange(9, dtype=float).reshape(3, 3)
    leveled = apply_level_weighted(img, polyx=1, polyy=1, method=method)

    assert leveled.shape == img.shape
    assert np.all(np.isfinite(leveled))
