"""Tests for the reusable computation behind the LAFM Notebook workbench."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from pnanolocz.lafm_workflow import (
    COLORMAP_NAMES,
    DEFAULT_COLORMAP,
    LAFMWorkflow,
    resolve_lafm_colormap,
)

TEST_DATA_ROOT = Path(
    os.environ.get(
        "PNANOLOCZ_TEST_DATA",
        Path(__file__).resolve().parents[1] / "Software_testing_images",
    )
)
requires_test_data = pytest.mark.skipif(
    not TEST_DATA_ROOT.is_dir(),
    reason="Software_testing_images is not part of the repository",
)


def _synthetic_movie() -> np.ndarray:
    yy, xx = np.indices((32, 32))
    frames = []
    for frame in range(4):
        image = np.zeros((32, 32), dtype=np.float64)
        for x, y, height in ((10 + frame * 0.2, 11, 8), (22, 21 - frame * 0.2, 12)):
            image += height * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.2**2))
        frames.append(image)
    return np.asarray(frames)


def test_user_facing_localization_defaults_use_cvcubic() -> None:
    find_default = inspect.signature(LAFMWorkflow.find_all_peaks).parameters[
        "localization_method"
    ].default
    wrapper_default = inspect.signature(LAFMWorkflow.localize_and_render).parameters[
        "localization_method"
    ].default

    assert find_default == "cvcubic"
    assert wrapper_default == "cvcubic"


def test_defaults_leave_test_movie_unprocessed() -> None:
    movie = _synthetic_movie()
    workflow = LAFMWorkflow.from_array(movie, source_name="synthetic.tiff")

    processed = workflow.preprocess()

    assert np.array_equal(processed, movie)
    assert workflow.settings["use_leveling"] is False
    assert workflow.settings["use_filtering"] is False


def test_roi_selection_crops_every_frame_and_invalidates_downstream() -> None:
    workflow = LAFMWorkflow.from_array(_synthetic_movie())
    workflow.preprocess()
    roi = workflow.set_roi((5, 6, 27, 28))
    workflow.initial_locs = np.ones((2, 8))

    assert roi.shape == (4, 22, 22)
    workflow.set_roi((7, 8, 20, 21))
    assert workflow.roi_movie.shape == (4, 13, 13)
    assert workflow.initial_locs is None

    with pytest.raises(ValueError, match="non-empty"):
        workflow.set_roi((2, 2, 2, 8))


def test_roi_reference_detects_particles_before_average_reference() -> None:
    workflow = LAFMWorkflow.from_array(_synthetic_movie())
    workflow.preprocess()
    roi_reference = workflow.set_reference_roi((6, 7, 15, 16), frame_index=0)

    assert roi_reference.shape == (9, 9)
    assert workflow.reference is None
    detections = workflow.detect_initial(
        correlation_filter_sigma=0.5,
        threshold=0.2,
    )
    assert set(detections[:, 4].astype(int)) == {1, 2, 3, 4}
    assert workflow.reference is None

    preview = workflow.detected_particle_preview(crop_radius=4)
    assert preview.shape == (9, 9)

    average = workflow.calculate_average_reference(crop_radius=4)
    assert average.shape == (9, 9)
    assert workflow.reference is not None


@requires_test_data
def test_real_500_frame_roi_method_detects_one_particle_per_frame() -> None:
    path = TEST_DATA_ROOT / "LAFM testing" / "025 circ_sim 25.tiff"
    workflow = LAFMWorkflow.from_tiff(path)
    workflow.preprocess()
    workflow.set_reference_roi((4, 4, 21, 21), frame_index=0)
    detections = workflow.detect_initial(
        method="ROI",
        image_filter_sigma=1.0,
        correlation_filter_sigma=1.0,
        fast_find=True,
        exclude_edges=True,
        correlation_min=0.5,
        correlation_max=1.0,
    )
    frames, counts = np.unique(detections[:, 4].astype(int), return_counts=True)

    assert len(detections) == 500
    assert np.array_equal(frames, np.arange(1, 501))
    assert np.all(counts == 1)

    localized = workflow.find_all_peaks(
        localization_method="bicubic",
        low_pass_sigma=0.0,
        high_pass_sigma=0.0,
        min_separation=1,
        height_threshold=0.0,
        prominence_threshold=0.0,
    )
    peak_frames, peak_counts = np.unique(
        localized[:, 4].astype(int), return_counts=True
    )
    assert len(localized) > 500
    assert len(peak_frames) == 500
    assert np.median(peak_counts) > 1


def test_requested_colormap_registry_and_luts() -> None:
    assert DEFAULT_COLORMAP == "LAFM color"
    assert COLORMAP_NAMES == (
        "LAFM color",
        "magma",
        "plasma",
        "inferno",
        "viridis",
        "gray",
        "Rainbow",
        "hot",
        "jet",
        "AFM brown",
        "AFM dark gold",
        "AFM gold",
        "fire",
    )
    for name in COLORMAP_NAMES:
        cmap = resolve_lafm_colormap(name)
        assert cmap.ndim == 2
        assert cmap.shape[1] == 3
        assert np.all((cmap >= 0) & (cmap <= 1))


def test_step3_postprocessing_samples_unfiltered_particle_stack() -> None:
    movie = np.arange(2 * 6 * 7, dtype=float).reshape(2, 6, 7)
    workflow = LAFMWorkflow.from_array(movie)
    locs = np.array(
        [
            [2.4, 3.4, -99.0, 0.0, 1.0],
            [6.0, 5.0, -99.0, 0.0, 2.0],
            [7.0, 3.0, -99.0, 0.0, 1.0],
        ]
    )

    processed = workflow.postprocess_lafm_z(locs, movie)

    assert processed.shape == (2, 5)
    assert processed[0, 2] == movie[0, 2, 1]
    assert processed[1, 2] == movie[1, 4, 5]
    assert np.array_equal(locs[:, 2], [-99.0, -99.0, -99.0])


def test_lafm_z_filter_uses_inclusive_limits_without_mutating_all_locs() -> None:
    workflow = LAFMWorkflow.from_array(np.zeros((1, 8, 8)))
    workflow.localized_locs = np.array(
        [
            [2.0, 2.0, 5.0, 0.0, 1.0],
            [3.0, 3.0, 10.0, 0.0, 1.0],
            [4.0, 4.0, 15.0, 0.0, 1.0],
        ]
    )
    original = workflow.localized_locs.copy()

    included = workflow.filter_lafm_localizations(5.0, 10.0)

    assert np.array_equal(included[:, 2], [5.0, 10.0])
    assert np.array_equal(workflow.localized_locs, original)
    assert np.array_equal(workflow.localization_include, [True, True, False])
    assert np.array_equal(workflow.included_localizations[:, 2], [5.0, 10.0])

    reset = workflow.reset_lafm_filter()
    assert np.array_equal(reset, original)
    assert np.all(workflow.localization_include)


def test_independent_render_z_ranges_preserve_original_localizations() -> None:
    workflow = LAFMWorkflow.from_array(np.zeros((1, 8, 8)))
    workflow.localized_locs = np.array(
        [
            [2.0, 2.0, 5.0, 0.0, 1.0],
            [3.0, 3.0, 10.0, 0.0, 1.0],
            [4.0, 4.0, 15.0, 0.0, 1.0],
        ]
    )
    original = workflow.localized_locs.copy()

    workflow.render_lafm(
        colorlimit_mode="Max Min",
        delete_outliers=99.0,
        lafm_z_range=(5.0, 10.0),
        probability_z_range=(10.0, 15.0),
    )

    assert np.array_equal(workflow.rendered_lafm_locs[:, 2], [5.0, 10.0])
    assert np.array_equal(
        workflow.rendered_probability_locs[:, 2], [10.0, 15.0]
    )
    assert np.array_equal(workflow.localized_locs, original)
    assert workflow.settings["lafm_render_z_range"] == [5.0, 10.0]
    assert workflow.settings["probability_render_z_range"] == [10.0, 15.0]
    assert np.array_equal(workflow.z_limits, [5.0, 10.0])


def test_empty_render_z_range_does_not_block_the_other_panel() -> None:
    workflow = LAFMWorkflow.from_array(np.zeros((1, 8, 8)))
    workflow.localized_locs = np.array(
        [[2.0, 2.0, 5.0, 0.0, 1.0], [3.0, 3.0, 10.0, 0.0, 1.0]]
    )

    rgb, probability, z_limits = workflow.render_lafm(
        delete_outliers=99.0,
        lafm_z_range=(6.0, 7.0),
        probability_z_range=(5.0, 10.0),
    )

    assert not np.any(rgb)
    assert np.any(probability)
    assert len(workflow.rendered_lafm_locs) == 0
    assert len(workflow.rendered_probability_locs) == 2
    assert np.array_equal(z_limits, [6.0, 7.0])


def test_detection_reference_alignment_localization_and_render() -> None:
    workflow = LAFMWorkflow.from_array(_synthetic_movie(), source_name="synthetic.tiff")
    workflow.preprocess()
    workflow.set_roi((2, 2, 30, 30))

    initial = workflow.detect_initial(
        method="Peaks",
        peak_size=5,
        image_filter_sigma=0.5,
        threshold=1.0,
    )
    assert initial.shape[0] >= 8
    assert np.all((initial[:, 4] >= 1) & (initial[:, 4] <= 4))

    reference = workflow.calculate_average_reference(crop_radius=4)
    assert reference.shape == (9, 9)
    assert np.max(reference) > 0

    correlated = workflow.detect_with_reference(
        correlation_filter_sigma=0.5,
        threshold=0.2,
    )
    assert correlated.shape[0] > 0

    aligned_reference = workflow.align_translation(
        iterations=1,
        method="Cross corr",
        max_drift=3,
        auto_update_reference=True,
    )
    assert aligned_reference.shape == reference.shape

    recalculated = workflow.recalculate_correlation(threshold=0.2)
    assert recalculated.shape[0] > 0

    localized = workflow.find_all_peaks(
        localization_method="bicubic",
    )
    assert localized.shape[0] > 0
    assert workflow.rendered_rgb is None

    rgb, probability, z_limits = workflow.render_lafm(
        colormap_name="LAFM color",
        img_gus=1.0,
        expand=1.0,
        colorlimit_mode="Max Min",
    )
    assert rgb.ndim == 3 and rgb.shape[2] == 3
    assert probability.shape == rgb.shape[:2]
    assert z_limits.shape == (2,)

    output_root = (
        Path.cwd() / ".test_artifacts" / f"workflow_unit_{uuid4().hex}"
    )
    output = workflow.save_results(output_root)
    expected = {
        "localizations_csv",
        "localizations_all_csv",
        "localizations_rendered_csv",
        "rgb_png",
        "rgb_tiff",
        "probability_png",
        "probability_tiff",
        "roi_movie_tiff",
        "reference_tiff",
        "settings_json",
    }
    assert set(output) == expected
    assert all(path.exists() for path in output.values())
    all_locs = np.atleast_2d(
        np.loadtxt(output["localizations_all_csv"], delimiter=",", skiprows=1)
    )
    rendered_locs = np.atleast_2d(
        np.loadtxt(
            output["localizations_rendered_csv"],
            delimiter=",",
            skiprows=1,
        )
    )
    assert len(all_locs) == len(workflow.localized_locs)
    assert len(rendered_locs) == len(workflow.rendered_locs)
