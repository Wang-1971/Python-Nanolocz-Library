from pathlib import Path
from uuid import uuid4

import numpy as np

from pnanolocz.lafm_table_parity import (
    _gated_assignment,
    _matlab_round,
    create_run_directory,
    direct_localize_stack,
    match_localization_tables,
    summarize_matches,
)


def test_matching_is_frame_aware_and_gated() -> None:
    matlab = np.array(
        [
            [1.0, 1.0, 10.0, 1.0],
            [5.0, 5.0, 20.0, 1.0],
            [1.0, 1.0, 30.0, 2.0],
        ]
    )
    python = np.array(
        [
            [1.3, 1.4, 11.0, 1.0],
            [8.0, 8.0, 21.0, 1.0],
            [1.1, 1.0, 29.0, 2.0],
        ]
    )

    matched, matlab_only, python_only = match_localization_tables(
        matlab, python, max_distance=0.75
    )

    assert len(matched) == 2
    assert len(matlab_only) == 1
    assert len(python_only) == 1
    assert set(matched["frame"]) == {1, 2}
    assert np.allclose(sorted(matched["distance_xy"]), [0.1, 0.5])

    metrics = summarize_matches(matched, len(matlab), len(python))
    assert metrics["matched_count"] == 2
    assert metrics["matlab_only_count"] == 1
    assert metrics["python_only_count"] == 1
    assert metrics["x_mae"] == 0.2
    assert metrics["z_rmse"] == 1.0


def test_direct_stack_localization_preserves_frames_and_samples_raw_z() -> None:
    movie = np.zeros((2, 9, 9), dtype=float)
    movie[0, 4, 4] = 10.0
    movie[1, 4, 5] = 20.0

    table = direct_localize_stack(movie)

    assert np.array_equal(table[:, 4], [1.0, 2.0])
    xs = np.rint(table[:, 0]).astype(int) - 1
    ys = np.rint(table[:, 1]).astype(int) - 1
    frames = table[:, 4].astype(int) - 1
    assert np.array_equal(table[:, 2], movie[frames, ys, xs])


def test_create_run_directory_never_overwrites() -> None:
    root = Path.cwd() / ".test_artifacts" / f"parity_{uuid4().hex}"
    first = create_run_directory(root, timestamp="20260727_120000")
    second = create_run_directory(root, timestamp="20260727_120000")

    assert first.name == "20260727_120000"
    assert second.name == "20260727_120000_2"


def test_matlab_round_and_gated_assignment_match_reference_rules() -> None:
    assert np.array_equal(
        _matlab_round(np.array([1.5, 2.5, -1.5])),
        [2.0, 3.0, -2.0],
    )
    rows, cols = _gated_assignment(
        np.array([[0.1, 0.7], [0.7, 0.8]]), 0.75
    )
    assert set(zip(rows, cols)) == {(0, 1), (1, 0)}
