"""End-to-end validation of the Python LAFM workflow on supplied TIFF data."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

from pnanolocz.detector import detector
from pnanolocz.lafm_renderer import _round_significant, lafm_renderer
from pnanolocz.localize import localize


TEST_DATA_ROOT = Path(
    os.environ.get(
        "PNANOLOCZ_TEST_DATA",
        Path(__file__).resolve().parents[1] / "Software_testing_images",
    )
)
LAFM_DIR = TEST_DATA_ROOT / "LAFM testing"
LAFM_IMAGES = sorted(LAFM_DIR.glob("*.tiff"))
requires_test_data = pytest.mark.skipif(
    not TEST_DATA_ROOT.is_dir(),
    reason="Software_testing_images is not part of the repository",
)
LOCALIZATION_METHODS = (
    "bicubic",
    "cvcubic",
    "bilinear",
    "lanczos3",
    "lanczos2",
    "gaussian",
    "sphere",
)


def _representative_rows(locs: np.ndarray, limit: int = 48) -> np.ndarray:
    """Select deterministic rows across the complete localization table."""
    if locs.shape[0] <= limit:
        return locs.copy()
    indices = np.linspace(0, locs.shape[0] - 1, limit, dtype=int)
    return locs[indices].copy()


@pytest.mark.parametrize("image_path", LAFM_IMAGES, ids=lambda path: path.stem)
@requires_test_data
def test_lafm_detector_localize_renderer_pipeline(image_path: Path) -> None:
    """Every supplied image passes detection, localization, and rendering."""
    movie = np.asarray(tifffile.imread(image_path), dtype=np.float64)
    frame_count = 1 if movie.ndim == 2 else movie.shape[0]

    detected = detector(
        movie,
        method="Peak picker",
        ref=5,
        filt_img=1.0,
        filt_ccr=0.0,
        min_thresh=0.1,
        ex_edge=False,
        fastdetect=False,
    )

    assert detected.ndim == 2
    assert detected.shape[1] >= 8
    assert detected.shape[0] > 0
    assert np.all(np.isfinite(detected[:, :5]))
    assert np.all((detected[:, 4] >= 1) & (detected[:, 4] <= frame_count))

    sample = _representative_rows(detected)
    refined_by_method: dict[str, np.ndarray] = {}
    for method in LOCALIZATION_METHODS:
        refined = localize(
            movie,
            sample,
            loc_method=method,
            pixperfeat=1.0,
            frame_axis=0,
            matlab_indexing=True,
        )
        refined_by_method[method] = refined
        assert refined.shape[0] == sample.shape[0]
        assert refined.shape[1] >= 12
        assert np.array_equal(refined[:, 4], sample[:, 4])
        assert np.count_nonzero(np.all(np.isfinite(refined[:, :2]), axis=1)) > 0

    render_locs = refined_by_method["bicubic"]
    render_locs = render_locs[np.all(np.isfinite(render_locs[:, :5]), axis=1)]
    rendered, z_limits = lafm_renderer(
        render_locs,
        img_gus=1.0,
        expand=1.0,
        fullcolormap="afmhot",
        prob=False,
        colorlimits=[0.0, 1.0],
        colorlimit_mode="Max Min",
    )
    probability, probability_z_limits = lafm_renderer(
        render_locs,
        img_gus=1.0,
        expand=1.0,
        fullcolormap="afmhot",
        prob=True,
        colorlimits=[0.0, 1.0],
        colorlimit_mode="Max Min",
    )

    assert rendered.ndim == 3 and rendered.shape[2] == 3
    assert probability.shape == rendered.shape[:2]
    assert np.all(np.isfinite(rendered))
    assert np.all(np.isfinite(probability))
    assert np.min(rendered) >= 0
    assert np.min(probability) >= 0
    assert np.max(rendered) > 0
    assert np.max(probability) > 0
    assert np.array_equal(z_limits, probability_z_limits)


@requires_test_data
def test_lafm_input_directory_contains_expected_images() -> None:
    """Guard against a silently skipped validation caused by missing data."""
    assert len(LAFM_IMAGES) == 13


def test_significant_rounding_matches_matlab_half_away_from_zero() -> None:
    """Automatic colour limits use MATLAB's tie-breaking convention."""
    values = np.array([1.125, -1.125, 0.01225, -0.01225])
    expected = np.array([1.13, -1.13, 0.0123, -0.0123])
    assert np.allclose(_round_significant(values, digits=3), expected)
