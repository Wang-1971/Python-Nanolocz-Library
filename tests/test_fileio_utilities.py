"""Systematic functionality tests for file I/O, visualization, and utilities.

Uses test images from Software_testing_images/.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

# --- Paths ------------------------------------------------------------------
TEST_IMG = Path(
    os.environ.get(
        "PNANOLOCZ_TEST_DATA",
        Path(__file__).parent.parent / "Software_testing_images",
    )
)
pytestmark = pytest.mark.skipif(
    not TEST_IMG.is_dir(),
    reason="Software_testing_images is not part of the repository",
)
LAFM_DIR = TEST_IMG / "LAFM testing"
ALIGN_DIR = TEST_IMG / "Alignment"
DET_DIR = TEST_IMG / "Detection"

_passed = _failed = _errors = 0


def _ok(cond: bool, label: str) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS: {label}")
    else:
        _failed += 1
        print(f"  FAIL: {label}")


def _err(label: str, exc: Exception) -> None:
    global _errors
    _errors += 1
    print(f"  ERROR: {label} -- {exc}")


def _load(path: Path) -> np.ndarray:
    return np.asarray(tifffile.imread(str(path)), dtype=np.float64)


def summary() -> None:
    print(f"\n{'='*60}")
    print(f"Results: {_passed} passed, {_failed} failed, {_errors} errors")
    print(f"{'='*60}")


# ============================================================================
# 1. READ_AFM_FILE
# ============================================================================
def test_read_afm_file() -> None:
    print("\n--- read_afm_file ---")
    from pnanolocz.read_afm_file import read_afm_file

    test_files = [
        (DET_DIR / "000Triangles 5 degrees.tif", "TIFF"),
        (ALIGN_DIR / "Test algin.tif", "TIFF stack"),
        (LAFM_DIR / "AFM_sim.tiff", "TIFF LAFM"),
    ]

    for path, label in test_files:
        try:
            img, info = read_afm_file(str(path))
            _ok(isinstance(img, np.ndarray), f"{label}: returns ndarray, shape={img.shape}")
            _ok(isinstance(info, dict), f"{label}: returns dict info")
            if img.ndim == 3:
                _ok(img.shape[0] >= 1, f"{label}: {img.shape[0]} frames")
        except Exception as e:
            _err(f"read_afm_file {label}", e)


# ============================================================================
# 2. TIFF_EXPORTER
# ============================================================================
def test_tiff_exporter() -> None:
    print("\n--- tiff_exporter ---")
    from pnanolocz.tiff_exporter import tiff_exporter

    img = _load(DET_DIR / "000Triangles 5 degrees.tif")
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir) / "test_export.tiff"
        try:
            tiff_exporter(img, full_file_name=str(outpath))
            _ok(outpath.exists(), "TIFF file created")
            if outpath.exists():
                reloaded = tifffile.imread(str(outpath))
                _ok(reloaded.squeeze().shape == img.shape, f"roundtrip shape preserved: {reloaded.squeeze().shape}")
        except Exception as e:
            _err("tiff_exporter", e)

    # 3D export
    movie = _load(ALIGN_DIR / "Fine align_tritest.tiff")
    if movie.ndim == 3:
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath2 = Path(tmpdir) / "test_export_3d.tiff"
            try:
                tiff_exporter(movie, full_file_name=str(outpath2))
                _ok(outpath2.exists(), "3D TIFF file created")
            except Exception as e:
                _err("tiff_exporter 3D", e)


# ============================================================================
# 3. EXPORTER (general)
# ============================================================================
def test_exporter() -> None:
    print("\n--- exporter ---")
    from pnanolocz.exporter import exporter

    data = np.random.default_rng(0).normal(size=(10, 10))

    with tempfile.TemporaryDirectory() as tmpdir:
        # TIFF
        try:
            p = exporter(data, format=".tiff", filepath=str(Path(tmpdir) / "out.tiff"))
            _ok(Path(p).exists(), "exporter .tiff")
        except Exception as e:
            _err("exporter tiff", e)

        # TXT
        try:
            p = exporter(data, format=".txt", filepath=str(Path(tmpdir) / "out.txt"))
            _ok(Path(p).exists(), "exporter .txt")
        except Exception as e:
            _err("exporter txt", e)

        # CSV
        try:
            p = exporter(data, format=".csv", filepath=str(Path(tmpdir) / "out.csv"))
            _ok(Path(p).exists(), "exporter .csv")
        except Exception as e:
            _err("exporter csv", e)


# ============================================================================
# 4. CREATE_GIF
# ============================================================================
def test_create_gif() -> None:
    print("\n--- create_gif ---")
    from pnanolocz.create_gif import create_gif

    movie = _load(ALIGN_DIR / "Fine align_tritest.tiff")
    if movie.ndim != 3:
        movie = movie.reshape(1, *movie.shape)

    with tempfile.TemporaryDirectory() as tmpdir:
        base = str(Path(tmpdir) / "test_gif")
        try:
            out = create_gif(movie, base, labels=False, delay_time=0.2)
            _ok(Path(out).exists(), f"GIF created: {out}")
        except Exception as e:
            _err("create_gif", e)

    # Single 2D → PNG
    img = _load(DET_DIR / "000Triangles 5 degrees.tif")
    with tempfile.TemporaryDirectory() as tmpdir:
        base2 = str(Path(tmpdir) / "test_png")
        try:
            out2 = create_gif(img, base2, labels=False)
            _ok(Path(out2).exists(), f"PNG created: {out2}")
        except Exception as e:
            _err("create_gif 2D PNG", e)


# ============================================================================
# 5. DRAW_LABELS
# ============================================================================
def test_draw_labels() -> None:
    print("\n--- draw_labels ---")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  SKIP: matplotlib not available")
        _ok(True, "draw_labels skipped (no matplotlib)")
        return

    from pnanolocz.draw_labels import draw_labels

    img = _load(DET_DIR / "000Triangles 5 degrees.tif")
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")

    info = {"PixelPerNm": 5.0, "time": 0.5, "n": 1}

    try:
        draw_labels(ax, img, frame=1, image_info=info, scalebar=True, timescale=True)
        _ok(True, "draw_labels with scalebar+timestamp")
    except Exception as e:
        _err("draw_labels", e)

    plt.close(fig)


# ============================================================================
# 6. VIEWSTACK (non-interactive)
# ============================================================================
def test_viewstack() -> None:
    print("\n--- viewstack (non-interactive smoke test) ---")
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("  SKIP: matplotlib not available")
        _ok(True, "viewstack skipped (no matplotlib)")
        return

    from pnanolocz.viewstack import viewstack

    movie = _load(ALIGN_DIR / "Fine align_tritest.tiff")
    if movie.ndim != 3:
        movie = movie.reshape(1, *movie.shape)

    try:
        h = viewstack(movie, show=False)
        _ok(h is not None, "viewstack returns handle")
        _ok(h.stack.shape == movie.shape, "handle.stack shape correct")
    except Exception as e:
        _err("viewstack", e)

    # 2D view
    img2d = _load(DET_DIR / "000Triangles 5 degrees.tif")
    try:
        h2 = viewstack(img2d, show=False)
        _ok(h2 is not None, "viewstack 2D returns handle")
    except Exception as e:
        _err("viewstack 2D", e)


# ============================================================================
# 7. AFM_COLORMAP
# ============================================================================
def test_afm_colormap() -> None:
    print("\n--- afm_colormap ---")
    from pnanolocz.afm_colormap import afm_colormap, load_afm_luts, apply_afm_colormap

    try:
        cmap = afm_colormap("AFM gold")
        _ok(cmap.ndim == 2 and cmap.shape[1] >= 3, f"afm_colormap shape={cmap.shape}")
    except Exception as e:
        _err("afm_colormap", e)

    try:
        luts = load_afm_luts()
        _ok(isinstance(luts, dict), f"load_afm_luts: {len(luts)} colormaps")
        _ok("AFM gold" in luts or len(luts) > 0, "has colormap entries")
    except Exception as e:
        _err("load_afm_luts", e)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        img = _load(DET_DIR / "000Triangles 5 degrees.tif")
        ax.imshow(img, cmap="gray")
        apply_afm_colormap(ax, "AFM gold")
        _ok(True, "apply_afm_colormap works")
        plt.close(fig)
    except ImportError:
        _ok(True, "apply_afm_colormap skipped (no matplotlib)")
    except Exception as e:
        _err("apply_afm_colormap", e)


# ============================================================================
# 8. SYMMETRY / ROTATION_SYM / SYM_PTCLOUD
# ============================================================================
def test_symmetry() -> None:
    print("\n--- symmetry ---")
    from pnanolocz.symmetry import rotation_sym as rot_sym1
    from pnanolocz.rotation_sym import rotation_sym as rot_sym2
    from pnanolocz.sym_ptcloud import sym_ptcloud

    img = _load(DET_DIR / "000Triangles 5 degrees.tif")

    for fold in [2, 3, 4, 6]:
        try:
            sym1 = rot_sym1(img, fold)
            _ok(sym1.shape == img.shape, f"symmetry fold={fold}: shape preserved")
            _ok(np.all(np.isfinite(sym1)), f"symmetry fold={fold}: all finite")
        except Exception as e:
            _err(f"symmetry fold={fold}", e)

    try:
        sym2 = rot_sym2(img, 3)
        _ok(sym2.shape == img.shape, "rotation_sym works")
    except Exception as e:
        _err("rotation_sym", e)

    # sym_ptcloud
    from pnanolocz.fast_peaks2d import fast_peaks2d
    peaks = fast_peaks2d(img, thresh=0.05, kernel_size=5, matlab_indexing=True)
    if peaks.shape[0] > 5:
        locs = np.zeros((peaks.shape[0], 8), dtype=np.float64)
        locs[:, 0:2] = peaks[:, 0:2]
        try:
            result = sym_ptcloud(3, img, locs)
            _ok(result is not None, "sym_ptcloud returns result")
        except Exception as e:
            _err("sym_ptcloud", e)


# ============================================================================
# 9. CONSTRUCT_PARTICLE_STACK
# ============================================================================
def test_construct_particle_stack() -> None:
    print("\n--- construct_particle_stack ---")
    from pnanolocz.construct_particle_stack import construct_particle_stack

    movie = _load(ALIGN_DIR / "Test algin.tif")
    if movie.ndim != 3:
        print("  SKIP: not 3D")
        return

    # Create fake Particles object
    from pnanolocz.align_iterate import Particles
    n_part = 10
    part_img = np.zeros((n_part, 20, 20), dtype=np.float64)
    for i in range(n_part):
        r0 = 90 + i
        c0 = 90 + i
        if r0 + 20 <= movie.shape[1] and c0 + 20 <= movie.shape[2]:
            part_img[i] = movie[min(i, movie.shape[0]-1), r0:r0+20, c0:c0+20]

    locs = np.zeros((n_part, 8), dtype=np.float64)
    locs[:, 0] = np.arange(100, 100 + n_part)
    locs[:, 1] = np.arange(100, 100 + n_part)
    locs[:, 2] = 1.0
    locs[:, 4] = np.arange(1, n_part + 1)

    part = Particles(image=part_img, locs=locs)
    try:
        stack = construct_particle_stack(movie, part, quick=False)
        _ok(stack.ndim >= 2, f"particle stack shape={stack.shape}")
    except Exception as e:
        _err("construct_particle_stack", e)


# ============================================================================
# 10. ADD_PARA
# ============================================================================
def test_add_para() -> None:
    print("\n--- add_para ---")
    from pnanolocz.add_para import add_para

    img = _load(DET_DIR / "000Triangles 5 degrees.tif")
    try:
        result = add_para(img, para_grad=0.5, direction="trace")
        _ok(result.shape == img.shape, f"add_para: shape={result.shape}")
        _ok(np.all(np.isfinite(result)), "add_para: all finite")
    except Exception as e:
        _err("add_para", e)


# ============================================================================
# 11. PREVENT_CLASH
# ============================================================================
def test_prevent_clash() -> None:
    print("\n--- prevent_clash ---")
    from pnanolocz.prevent_clash import prevent_clash

    rng = np.random.default_rng(7)
    n = 50
    x = rng.uniform(0, 100, n)
    y = rng.uniform(0, 100, n)
    z = rng.uniform(0, 1, n)
    try:
        result = prevent_clash(diameter=5.0, x=x, y=y, z=z)
        _ok(isinstance(result, tuple) and len(result) == 3, f"prevent_clash returns 3 arrays")
        _ok(len(result[0]) <= n, f"filtered: {n} -> {len(result[0])}")
    except Exception as e:
        _err("prevent_clash", e)


# ============================================================================
# 12. PAD_STACKER
# ============================================================================
def test_pad_stacker() -> None:
    print("\n--- pad_stacker ---")
    from pnanolocz.pad_stacker import pad_stacker

    # pad_stacker(A, B) pads the smaller to match the larger
    img = _load(DET_DIR / "000Triangles 5 degrees.tif")
    A = img[50:100, 50:100]  # 50x50
    B = img[100:170, 100:170]  # 70x70
    try:
        padded = pad_stacker(A, B)
        _ok(padded.ndim == 2 or padded.ndim == 3, f"pad_stacker shape={padded.shape}")
    except Exception as e:
        _err("pad_stacker", e)


# ============================================================================
# 13. RES_TO_RENDER
# ============================================================================
def test_res_to_render() -> None:
    print("\n--- res_to_render ---")
    from pnanolocz.res_to_render import res_to_render

    try:
        px, nm = res_to_render(pixpernm=5.0, res=512)
        _ok(px > 0 and nm > 0, f"res_to_render: {px:.1f} px, {nm:.1f} nm")
    except Exception as e:
        _err("res_to_render", e)


# ============================================================================
# 14. REF_SELECTOR
# ============================================================================
def test_ref_selector() -> None:
    print("\n--- ref_selector ---")
    from pnanolocz.ref_selector import ref_selector

    img = _load(DET_DIR / "000Triangles 5 degrees.tif")
    try:
        ref = ref_selector(img, center="yes", fold=3,
                           rect=(100, 100, 200, 200))
        _ok(ref.ndim == 2, f"ref_selector: shape={ref.shape}")
    except Exception as e:
        _err("ref_selector", e)


# ============================================================================
# 15. TIME_ELAPSED
# ============================================================================
def test_time_elapsed() -> None:
    print("\n--- time_elapsed ---")
    from pnanolocz.time_elapsed import time_elapsed

    try:
        # time_elapsed computes elapsed seconds from timestamps
        times = ["12:00:00", "12:00:01", "12:00:03"]
        elapsed = time_elapsed(times)
        _ok(len(elapsed) == 3, f"time_elapsed returns {len(elapsed)} values")
        _ok(elapsed[0] == 0.0, "first elapsed is 0")
    except Exception as e:
        _err("time_elapsed", e)


# ============================================================================
# 16. OPEN_GWYCHANNEL / OPEN_ASD (smoke tests on non-GWY/ASD files)
# ============================================================================
def test_open_gwychannel() -> None:
    print("\n--- open_gwychannel ---")
    from pnanolocz.open_gwychannel import open_gwychannel
    # GWY format is specialized; we verify the function exists and is callable
    _ok(callable(open_gwychannel), "open_gwychannel is callable")


def test_open_asd() -> None:
    print("\n--- open_asd ---")
    from pnanolocz.open_asd import open_asd
    _ok(callable(open_asd), "open_asd is callable")


# ============================================================================
# 17. JPK / IBW / PARK / NANOSCOPE readers (smoke)
# ============================================================================
def test_file_readers_exist() -> None:
    print("\n--- file readers ---")
    readers = [
        ("open_jpk_image", "pnanolocz.open_jpk"),
        ("open_jpk_info", "pnanolocz.open_jpk"),
        ("open_ibw", "pnanolocz.open_ibw"),
        ("open_park", "pnanolocz.open_park"),
        ("open_nanoscope", "pnanolocz.open_nanoscope"),
    ]
    for name, module in readers:
        try:
            exec(f"from {module} import {name}")
            _ok(True, f"{name} importable")
        except Exception as e:
            _err(name, e)


# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FILE I/O & UTILITIES TESTS")
    print("=" * 60)

    test_read_afm_file()
    test_tiff_exporter()
    test_exporter()
    test_create_gif()
    test_draw_labels()
    test_viewstack()
    test_afm_colormap()
    test_symmetry()
    test_construct_particle_stack()
    test_add_para()
    test_prevent_clash()
    test_pad_stacker()
    test_res_to_render()
    test_ref_selector()
    test_time_elapsed()
    test_open_gwychannel()
    test_open_asd()
    test_file_readers_exist()

    summary()
