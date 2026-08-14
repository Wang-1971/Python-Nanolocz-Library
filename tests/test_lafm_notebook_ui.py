"""Checks for the staged LAFM Notebook controller."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from pnanolocz.lafm_notebook_ui import (
    LAFM_RENDER_COLORMAPS,
    LAFMWorkbench,
)

TEST_DATA_ROOT = Path(
    os.environ.get(
        "PNANOLOCZ_TEST_DATA",
        Path(__file__).resolve().parents[1] / "Software_testing_images",
    )
)
pytestmark = pytest.mark.skipif(
    not TEST_DATA_ROOT.is_dir(),
    reason="Software_testing_images is not part of the repository",
)


def test_workbench_defaults_and_stage_gates() -> None:
    workbench = LAFMWorkbench(Path(__file__).resolve().parents[1])

    assert workbench.use_leveling.value is False
    assert workbench.use_filtering.value is False
    assert workbench.translation_iterations.value == 2
    assert workbench.auto_reference.value is True
    assert tuple(workbench.localization_method.options) == (
        "bicubic",
        "cvcubic",
        "bilinear",
        "lanczos3",
        "lanczos2",
        "gaussian",
        "sphere",
    )
    assert workbench.localization_method.value == "cvcubic"
    assert workbench.colormap_dropdown.value == "LAFM color"
    assert tuple(workbench.colormap_dropdown.options) == LAFM_RENDER_COLORMAPS
    assert LAFM_RENDER_COLORMAPS == (
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
    assert workbench.detection_method.value == "ROI"
    assert workbench.fast_find.value is True
    assert workbench.image_filter_sigma.value == 1.0
    assert workbench.correlation_sigma.value == 1.0
    assert workbench.exclude_edges.value is True
    assert workbench.tracking_max_step.value == 10.0
    assert workbench.tracking_max_missing.value == 3
    assert workbench.correlation_max.value == 1.0
    assert workbench.correlation_threshold.value == 0.5
    assert workbench.detect_initial_button.disabled is True
    assert workbench.reference_button.disabled is True
    assert workbench.localize_button.description == "1. Find all peaks"
    assert workbench.render_button.description == "2. Render LAFM"
    assert workbench.peak_low_pass.value == 0.0
    assert workbench.peak_high_pass.value == 0.0
    assert workbench.peak_min_separation.value == 1
    assert workbench.peak_height.value == 0.0
    assert workbench.peak_prominence.value == 0.0
    assert workbench.img_gus.value == 1.0
    assert workbench.expand.value == 5.0
    assert workbench.delete_outliers.value == 4.0
    assert workbench.colorlimit_mode.value == "Exc outliers"
    assert workbench.localization_scope.value == "Included peaks"
    assert workbench.lafm_z_min.value == 0.0
    assert workbench.lafm_z_max.value == 1.0
    assert workbench.apply_lafm_filter.description == "Apply LAFM z filter"
    assert workbench.reset_lafm_filter.description == "Reset LAFM z filter"
    assert workbench.localization_view.value == "All particles (MATLAB)"
    assert tuple(workbench.localization_view.options) == (
        "All particles (MATLAB)",
        "Current particle",
    )
    assert workbench.localize_button.disabled is True
    assert workbench.render_button.disabled is True
    assert workbench.save_button.disabled is True


def test_find_peaks_defaults_to_matlab_all_particle_overlay() -> None:
    import numpy as np

    from pnanolocz.lafm_workflow import LAFMWorkflow

    workbench = LAFMWorkbench(Path(__file__).resolve().parents[1])
    workflow = LAFMWorkflow.from_array(np.zeros((2, 9, 9)))
    workflow.preprocess()
    workflow.particle_movie = np.zeros((2, 9, 9))
    workflow.localized_locs = np.array(
        [
            [2.0, 3.0, 4.0, 0.0, 1.0, 1.0, 1.0, 0.8],
            [7.0, 6.0, 8.0, 0.0, 2.0, 2.0, 2.0, 0.9],
        ]
    )
    workbench.workflow = workflow

    workbench._update_localization_frame()

    offsets = workbench.localization_scatter.get_offsets()
    assert offsets.shape == (2, 2)
    assert np.allclose(offsets, [[-2.5, -1.5], [2.5, 1.5]])
    assert "2 included peaks" in workbench.localization_peak_ax.get_title()

    workbench.localization_scope.value = "All peaks"
    assert "2 all peaks" in workbench.localization_peak_ax.get_title()


def test_small_image_display_is_resized_from_image_shape() -> None:
    workbench = LAFMWorkbench(Path(__file__).resolve().parents[1])
    workbench._resize_input_figure((32, 32), (16, 16))
    width, height = workbench.input_fig.get_size_inches()

    assert width == 5.5
    assert height == 5.5
    assert workbench.input_ax.get_aspect() == 1.0


def test_wide_image_fits_inside_box_without_changing_aspect_ratio() -> None:
    workbench = LAFMWorkbench(Path(__file__).resolve().parents[1])
    workbench._resize_input_figure((25, 100))
    width, height = workbench.input_fig.get_size_inches()

    assert width == 9.0
    assert height == 2.25
    assert width / height == 4.0


def test_input_and_roi_are_always_strict_afmhot() -> None:
    workbench = LAFMWorkbench(Path(__file__).resolve().parents[1])
    workbench.colormap_dropdown.value = "viridis"
    workbench._show_current_frame()

    assert workbench.input_artist.get_cmap().name == "afmhot"
    assert workbench.roi_artist.get_cmap().name == "afmhot"


def test_updated_image_extent_fills_the_axes() -> None:
    workbench = LAFMWorkbench(Path(__file__).resolve().parents[1])
    image = __import__("numpy").zeros((25, 25))
    workbench._set_image(
        workbench.input_artist,
        workbench.input_ax,
        image,
        cmap="afmhot",
    )

    assert tuple(workbench.input_artist.get_extent()) == (
        -0.5,
        24.5,
        24.5,
        -0.5,
    )


def test_render_controls_have_independent_z_ranges_and_scale_bars() -> None:
    workbench = LAFMWorkbench(Path(__file__).resolve().parents[1])

    assert workbench.lafm_render_z_range.continuous_update is False
    assert workbench.probability_render_z_range.continuous_update is False
    assert workbench.lafm_render_z_range is not workbench.probability_render_z_range
    assert workbench.lafm_z_colorbar is not None
    assert workbench.probability_density_colorbar is not None

    import numpy as np
    from pnanolocz.lafm_workflow import LAFMWorkflow

    workflow = LAFMWorkflow.from_array(np.zeros((1, 8, 8)))
    workflow.localized_locs = np.array(
        [[2.0, 2.0, 5.0, 0.0, 1.0], [3.0, 3.0, 15.0, 0.0, 1.0]]
    )
    workbench.workflow = workflow
    workbench._sync_render_z_ranges()

    assert workbench.lafm_render_z_range.value == (5.0, 15.0)
    assert workbench.probability_render_z_range.value == (5.0, 15.0)
