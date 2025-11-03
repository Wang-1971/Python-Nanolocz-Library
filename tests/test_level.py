"""Tests for the level module."""

import numpy as np

from pnanolocz_lib.level import level_med_line

# --- Level functions ---


def test_level_med_line_basic():
    """Test for the level_med_line function."""
    img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)

    mask = np.array([[1, 1, 0], [1, 0, 1], [1, 1, 1]], dtype=int)

    # polyx = 1, polyy ignored
    leveled = level_med_line(img, mask, polyx=1, polyy=0)

    # Compute expected values manually
    # Row medians over masked pixels:
    # row 0: median([1,2]) = 1.5
    # row 1: median([4,6]) = 5.0
    # row 2: median([7,8,9]) = 8.0
    bg = np.median([1, 2, 4, 6, 7, 8, 9])  # median of all masked pixels
    expected = img - np.array([[1.5], [5.0], [8.0]]) + bg

    # Compare
    np.testing.assert_allclose(leveled, expected)


def test_level_med_line_default_mask():
    """Test the level_med_line function with no mask."""
    # Image with some NaNs
    img = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])

    # Call function with mask=None
    leveled = level_med_line(img, mask=None, polyx=1, polyy=0)

    # Compute expected manually
    bg = np.nanmedian(img)  # median of all non-NaN values
    row_medians = np.array(
        [np.nanmedian(img[0, :]), np.nanmedian(img[1, :]), np.nanmedian(img[2, :])]
    )

    expected = img - row_medians[:, None] + bg  # apply per-row median subtraction + bg

    np.testing.assert_allclose(leveled, expected)
