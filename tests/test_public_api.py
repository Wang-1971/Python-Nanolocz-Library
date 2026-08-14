"""Tests for the deliberately small package-root API."""

from __future__ import annotations

import inspect

import pnanolocz
from pnanolocz.detector import detector
from pnanolocz.lafm_renderer import lafm_movie_renderer, lafm_renderer
from pnanolocz.lafm_workflow import LAFMWorkflow
from pnanolocz.localize import localize, localize_matlab


def test_package_root_exposes_only_the_stable_api() -> None:
    assert pnanolocz.__all__ == [
        "__version__",
        "detector",
        "localize",
        "localize_matlab",
        "LAFMWorkflow",
        "lafm_renderer",
        "lafm_movie_renderer",
        "level",
        "level_auto",
        "level_weighted",
        "thresholder",
    ]
    assert pnanolocz.detector is detector
    assert pnanolocz.localize is localize
    assert pnanolocz.localize_matlab is localize_matlab
    assert pnanolocz.LAFMWorkflow is LAFMWorkflow
    assert pnanolocz.lafm_renderer is lafm_renderer
    assert pnanolocz.lafm_movie_renderer is lafm_movie_renderer


def test_collaborator_modules_are_public() -> None:
    assert pnanolocz.level.apply_level is not None
    assert pnanolocz.level_auto.apply_level_auto is not None
    assert pnanolocz.level_weighted.apply_level_weighted is not None
    assert pnanolocz.thresholder.apply_thresholder is not None


def test_collaborator_parameter_names_are_stable() -> None:
    assert list(inspect.signature(pnanolocz.level.apply_level).parameters)[:5] == [
        "img",
        "polyx",
        "polyy",
        "method",
        "mask",
    ]
    assert list(
        inspect.signature(pnanolocz.level_auto.apply_level_auto).parameters
    )[:2] == ["img_stack", "routine"]
    assert list(
        inspect.signature(pnanolocz.level_weighted.apply_level_weighted).parameters
    )[:5] == ["img", "polyx", "polyy", "method", "mask"]
    assert list(
        inspect.signature(pnanolocz.thresholder.apply_thresholder).parameters
    ) == ["img", "method", "limits", "invert"]
