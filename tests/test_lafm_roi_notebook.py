"""Structural checks for the interactive LAFM ROI notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "LAFM interactive ROI.ipynb"
)
pytestmark = pytest.mark.skipif(
    not NOTEBOOK.is_file(),
    reason="notebooks are not part of this submission",
)


def test_lafm_roi_notebook_contains_interactive_roi_workflow() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    code_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code" and cell.source.strip()
    ]

    assert "%matplotlib widget" in source
    assert "launch_lafm_workbench" in source
    assert "workbench =" in source
    assert len(code_cells) == 1
    assert "650px" not in source
    assert "import ipywidgets as widgets" not in source
