"""Structural checks for the non-dashboard LAFM Notebook."""

from pathlib import Path

import nbformat
import pytest


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "LAFM_test.ipynb"
)
pytestmark = pytest.mark.skipif(
    not NOTEBOOK.is_file(),
    reason="notebooks are not part of this submission",
)


def test_block_notebook_exposes_the_particle_workflow_in_order() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)

    stages = [
        "workflow.set_reference_roi",
        "workflow.detect_initial",
        "workflow.calculate_average_reference",
        "workflow.detect_with_reference",
        "workflow.align_translation",
        "workflow.recalculate_correlation",
        "workflow.find_all_peaks",
        "workflow.render_lafm",
        "workflow.save_results",
    ]
    positions = [source.index(stage) for stage in stages]
    assert positions == sorted(positions)
    assert "LAFMWorkbench" not in source
    assert "widgets.Dropdown" in source
    assert "widgets.FloatRangeSlider" in source
    assert "lafm_display" in source
    assert "lafm_z_colorbar" in source
    assert "probability_density_colorbar" in source
    assert "peak_size=" not in source
    assert "ROI Method" in source
    assert "cmap=\"afmhot\"" in source
    assert "USE_LEVELING = False" in source
    assert "USE_FILTERING = False" in source
