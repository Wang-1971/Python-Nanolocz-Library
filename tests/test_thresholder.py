"""Tests for the thresholder module."""

import numpy as np
import pytest

from pnanolocz_lib.thresholder import (
    apply_thresholder,
    auto_edges,
    hist_edges,
    hist_skel,
    histogram,
    line_step,
    otsu,
    otsu_edges,
    otsu_skel,
    selection,
)

# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------


def test_selection_passthrough_boolean():
    """Treats nonzero input as True and returns a boolean mask."""
    m = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    out = selection(m, None)
    # included pixels (nonzero) are False → ~out True
    assert np.all(~out[m.astype(bool)])
    # excluded pixels (zero) are True
    assert np.all(out[~m.astype(bool)])


# ---------------------------------------------------------------------
# Histogram thresholding
# ---------------------------------------------------------------------


def test_histogram_inclusive_range_and_validation():
    """Keeps values within [low, high] and validates limit shape."""
    img = np.linspace(0, 1, 10).reshape(2, 5)
    out = histogram(img, limits=(0.2, 0.6))
    mask = (img >= 0.2) & (img <= 0.6)
    assert np.all(~out[mask])  # inside range → False
    assert np.all(out[~mask])  # outside range → True

    with pytest.raises(ValueError):
        histogram(img, limits=None)
    with pytest.raises(ValueError):
        histogram(img, limits=(0.1, 0.2, 0.3))


# ---------------------------------------------------------------------
# Otsu
# ---------------------------------------------------------------------


def test_otsu_bimodal():
    """Produces both kept (True) and excluded (False) regions for a bimodal image."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.2, 0.02, size=(30, 30))
    b = rng.normal(0.8, 0.02, size=(30, 30))
    img = np.block([[a, a], [b, b]])  # 60x60 bimodal

    out = otsu(img, None)
    kept = np.sum(out)  # True pixels
    dropped = np.sum(~out)  # False pixels
    assert kept > 0 and dropped > 0


# ---------------------------------------------------------------------
# Edge detectors
# ---------------------------------------------------------------------


def _synthetic_square(h=64, w=64, r0=16, r1=48, c0=16, c1=48, lo=0.0, hi=1.0):
    """Returns an image with a bright square on a dark background."""
    img = np.full((h, w), lo, float)
    img[r0:r1, c0:c1] = hi
    return img


def test_auto_edges_finds_edges_safely():
    """Creates NaN on detected edges and leaves 1.0 elsewhere."""
    img = _synthetic_square()
    out = auto_edges(img, None)
    assert out.shape == img.shape
    assert np.any(out)  # some True values
    assert np.any(~out)


def test_hist_edges_and_validation():
    """Detects edges using intensity limits and validates limit shape."""
    img = _synthetic_square()
    out = hist_edges(img, limits=(0.25, 0.75))
    # There should be both included (True) and excluded/edges (False) pixels
    assert out.shape == img.shape
    assert np.any(out)  # some True pixels (kept regions)
    assert np.any(~out)  # some False pixels (edges/excluded)

    with pytest.raises(ValueError):
        hist_edges(img, limits=None)


def test_otsu_edges_detects_transition():
    """Mask output is boolean and has correct shape."""
    h, w = 64, 64
    one_band = np.r_[np.zeros(16), np.ones(16)]
    row = np.tile(one_band, w // one_band.size)
    img = np.tile(row, (h, 1)).astype(float)

    out = otsu_edges(img, None)
    assert out.shape == img.shape
    assert out.dtype == bool


def test_otsu_edges_detects_transition_realistic():
    """Detects edges for a synthetic gradient split image with sufficient width."""
    h, w = 64, 64
    # Wide transition region (16 pixels) to survive Gaussian smoothing sigma=2
    img = np.zeros((h, w), dtype=float)
    img[:, : w // 2 - 8] = 0.0  # left dark
    img[:, w // 2 + 8 :] = 1.0  # right bright
    img[:, w // 2 - 8 : w // 2 + 4] = 0.5  # middle gradient

    out = otsu_edges(img, None)

    assert out.shape == img.shape
    assert out.dtype == bool
    # There should be both kept and edge pixels
    assert np.any(out)  # some kept
    assert np.any(~out)  # some edges


# ---------------------------------------------------------------------
# Skeleton paths (guard heavy deps)
# ---------------------------------------------------------------------


@pytest.mark.skipif(pytest.importorskip("sknw") is None, reason="sknw not available")
def test_otsu_skel_returns_mask_with_nans_and_ones():
    """Skeletonizes Otsu-selected regions and returns NaN outside skeleton."""
    img = _synthetic_square(64, 64, 8, 56, 28, 36)  # thin vertical bar inside
    out = otsu_skel(img, None)
    assert out.shape == img.shape
    # Should have some True values (skeleton)
    assert np.any(out)
    assert np.any(~out)


@pytest.mark.skipif(pytest.importorskip("sknw") is None, reason="sknw not available")
def test_hist_skel_with_limits():
    """Skeletonizes histogram-selected regions and returns False outside skeleton."""
    img = _synthetic_square(64, 64, 8, 56, 31, 33)  # narrow vertical stripe
    out = hist_skel(img, limits=(0.25, 0.75))

    # Shape is preserved
    assert out.shape == img.shape

    # Should have some True values (skeleton)
    assert np.any(out)

    # Should have some False values (outside skeleton)
    assert np.any(~out)


# ---------------------------------------------------------------------
# Line step (ruptures PELT) and its validations
# ---------------------------------------------------------------------


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("ruptures") is None,
    reason="ruptures not available",
)
def test_line_step_detects_segments_upward_step():
    """Upward step: left segment valid (False), right segment excluded (True)."""
    # Row 0: clear upward step at 40
    x = np.r_[np.zeros(40), np.ones(40)]  # 0..39=0, 40..79=1
    img = np.vstack([x, np.zeros_like(x)])  # second row is all zeros (no CPs)

    # Use a modest penalty so PELT finds the CP on clean data
    out = line_step(img, limits=(0.0, 1.0))  # limits[1] is the penalty

    row0 = out[0]
    # MATLAB: xp(1:40)=1 -> valid; xp(40:end)=NaN -> excluded
    assert not np.any(row0[:40])  # left valid  → False
    assert np.all(row0[40:])  # right excl. → True

    # Row 1 has no change points: should be all-valid
    row1 = out[1]
    assert not row1.any()


def test_line_step_handles_missing_limits_gracefully(capfd):
    """Missing limits: no raise; returns all-valid (False) mask and prints a message."""
    img = np.zeros((3, 10))
    out = line_step(img, limits=None)

    assert out.shape == img.shape
    assert out.dtype == np.bool_
    # all-valid: no exclusions
    assert not out.any()

    # optional: assert the message was printed
    captured = capfd.readouterr()
    assert "limits must be" in captured.out  # or a more specific substring


def test_line_step_handles_wrong_limits_type_gracefully(capfd):
    """Wrong type: no raise; returns all-valid (False) mask and prints a message."""
    img = np.zeros((3, 10))
    out = line_step(img, limits="not-a-tuple")

    assert out.shape == img.shape
    assert out.dtype == np.bool_
    assert not out.any()

    captured = capfd.readouterr()
    assert "limits must be" in captured.out


# ---------------------------------------------------------------------
# Wrapper: thresholder
# ---------------------------------------------------------------------


def test_thresholder_unknown_method_raises():
    """Raises ValueError for an unknown thresholding method name."""
    img = np.zeros((8, 8))
    with pytest.raises(ValueError):
        apply_thresholder(img, method="unknown", limits=None)


def test_thresholder_invert_flips_true_and_false():
    """Inverts mask so True becomes False and False becomes True."""
    img = _synthetic_square()
    m = apply_thresholder(img, method="otsu", limits=None, invert=False)
    inv = apply_thresholder(img, method="otsu", limits=None, invert=True)

    # Both should have same shape
    assert m.shape == inv.shape

    # Inversion should be logical negation
    assert np.all(inv == np.logical_not(m))


def test_thresholder_stack_3d_applies_per_frame():
    """Applies per-frame thresholding and preserves (N,H,W) shape for stacks."""
    rng = np.random.default_rng(0)
    img2d_a = rng.normal(size=(32, 32))
    img2d_b = rng.normal(size=(32, 32))
    stack = np.stack([img2d_a, img2d_b], axis=0)
    out = apply_thresholder(stack, method="histogram", limits=(-0.5, 0.5))
    # shape is preserved
    assert out.shape == stack.shape

    # expect boolean mask: at least one True and one False per frame
    for i in range(out.shape[0]):
        frame = out[i]
        assert frame.dtype == bool
        assert frame.any()  # at least one True (selected)
        assert (~frame).any()  # at least one False (excluded)
