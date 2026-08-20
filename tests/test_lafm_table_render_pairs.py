from pathlib import Path
from uuid import uuid4

import numpy as np

from pnanolocz.lafm_table_render_pairs import _table_directories, render_table_pair


def test_table_directories_accept_validation_layout() -> None:
    """Resolve the self-contained Gaussian/sphere report layout."""
    run_path = Path.cwd() / ".test_artifacts" / f"render_layout_{uuid4().hex}"
    (run_path / "matlab").mkdir(parents=True)
    (run_path / "python").mkdir()

    assert _table_directories(run_path) == (
        run_path / "matlab",
        run_path / "python",
    )


def test_pair_render_uses_shared_grid_and_z_limits() -> None:
    """Use one shared grid and height range for both paired renders."""
    matlab = np.array([[2.0, 3.0, 5.0], [8.0, 9.0, 15.0]], dtype=float)
    python = np.array([[1.0, 2.0, 4.0], [10.0, 11.0, 20.0]], dtype=float)

    mat_rgb, py_rgb, metadata = render_table_pair(
        matlab, python, img_gus=1.0, expand=5.0
    )

    assert mat_rgb.shape == py_rgb.shape
    assert mat_rgb.shape[2] == 3
    assert metadata["z_range"] == [4.0, 20.0]
    assert metadata["xy_bounds"] == [1.0, 10.0, 2.0, 11.0]
    assert np.all(np.isfinite(mat_rgb))
    assert np.all(np.isfinite(py_rgb))
    assert 0.0 <= mat_rgb.min() <= mat_rgb.max() <= 1.0
    assert 0.0 <= py_rgb.min() <= py_rgb.max() <= 1.0
