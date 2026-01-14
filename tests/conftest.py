"""Test fixtures for pnanolocz_lib."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

RESOURCES = Path(__file__).parent / "resources"


@pytest.fixture
def load_npz():
    """Return a loader function so tests can do: data = load_npz('file.npz')."""

    def _load(name: str) -> dict[str, np.ndarray]:
        p = RESOURCES / name
        with np.load(p, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    return _load
