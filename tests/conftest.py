"""Pytest-wide configuration and fixtures (warning filters) shared by all tests."""

import warnings


def pytest_configure(config) -> None:
    # NumPy: nanmedian over all-NaN slices during background/median computations
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="All-NaN slice encountered",
        module=r"pnanolocz_lib\.level",
    )

    # scikit-image 0.26: deprecation for remove_small_holes(area_threshold=...)
    # Broaden the module pattern to catch submodules like skimage.morphology.misc,
    # and drop the message constraint to avoid regex mismatches.
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"skimage\.morphology(\..*)?$",
    )
