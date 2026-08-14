"""MATLAB-specific regression checks for LAFM rendering."""

import numpy as np

from pnanolocz.lafm_renderer import lafm_renderer


def test_duplicate_localizations_use_binary_pixel_occupancy() -> None:
    one = np.array([[5.0, 6.0, 10.0, 0.0, 1.0]])
    repeated = np.repeat(one, 100, axis=0)

    single, _ = lafm_renderer(
        one, 1.0, 5.0, "LAFM color", False, [0.0, 20.0], "Manual"
    )
    duplicate, _ = lafm_renderer(
        repeated, 1.0, 5.0, "LAFM color", False, [0.0, 20.0], "Manual"
    )

    assert np.allclose(single, duplicate)
