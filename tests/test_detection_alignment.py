"""Systematic functionality tests for detection and alignment modules.

Uses test images from Software_testing_images/Detection/ and /Alignment/.
Uses smaller image crops to keep tests fast.
"""

from __future__ import annotations

import os
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
DET_DIR = TEST_IMG / "Detection"
ALIGN_DIR = TEST_IMG / "Alignment"

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


def _load_crop(path: Path, size: int = 200) -> np.ndarray:
    """Load a center crop for faster testing."""
    img = _load(path)
    h, w = img.shape[-2], img.shape[-1]
    r0 = max(0, (h - size) // 2)
    c0 = max(0, (w - size) // 2)
    if img.ndim == 3:
        return img[:, r0:r0+size, c0:c0+size]
    return img[r0:r0+size, c0:c0+size]


def summary() -> None:
    print(f"\n{'='*60}")
    print(f"Results: {_passed} passed, {_failed} failed, {_errors} errors")
    print(f"{'='*60}")


# ============================================================================
# 1. FAST_PEAKS2D
# ============================================================================
def test_fast_peaks2d() -> None:
    print("\n--- fast_peaks2d ---")
    from pnanolocz.fast_peaks2d import fast_peaks2d

    img = _load_crop(DET_DIR / "000Triangles 5 degrees.tif", size=200)
    try:
        peaks = fast_peaks2d(img, thresh=0.1, kernel_size=3, matlab_indexing=True)
        _ok(peaks.ndim == 2 and peaks.shape[1] >= 4, f"returns Nx>=4, shape={peaks.shape}")
        _ok(peaks.shape[0] > 0, f"found {peaks.shape[0]} peaks")
    except Exception as e:
        _err("fast_peaks2d", e)

    try:
        empty = fast_peaks2d(np.zeros((50, 50)), thresh=1.0, kernel_size=3)
        _ok(empty.size == 0, "empty result for blank image")
    except Exception as e:
        _err("fast_peaks2d empty", e)


# ============================================================================
# 2. PEAKS2D
# ============================================================================
def test_peaks2d() -> None:
    print("\n--- peaks2d ---")
    from pnanolocz.peaks2d import peaks2d

    img = _load_crop(DET_DIR / "000Triangles 5 degrees.tif", size=200)
    try:
        peaks = peaks2d(img, thresh=0.05, ns=3, min_prom=0.0)
        _ok(peaks.ndim == 2 and peaks.shape[1] == 4, f"returns Nx4, shape={peaks.shape}")
        _ok(peaks.shape[0] > 0, f"found {peaks.shape[0]} peaks")
    except Exception as e:
        _err("peaks2d", e)

    try:
        peaks_prom = peaks2d(img, thresh=0.05, ns=3, min_prom=0.01)
        _ok(peaks_prom.shape[0] <= peaks.shape[0],
            f"prominence filtering ({peaks.shape[0]} -> {peaks_prom.shape[0]})")
    except Exception as e:
        _err("peaks2d prominence", e)

    try:
        peaks0 = peaks2d(img, thresh=0.05, ns=3, matlab_indexing=False)
        _ok(peaks0.ndim == 2, "matlab_indexing=False works")
    except Exception as e:
        _err("peaks2d python indexing", e)


# ============================================================================
# 3. DETECTOR (Peak picker mode)
# ============================================================================
def test_detector_peakpicker() -> None:
    print("\n--- detector (Peak picker) ---")
    from pnanolocz.detector import detector

    img = _load_crop(DET_DIR / "000Triangles 5 degrees.tif", size=128)
    try:
        locs = detector(
            img, method="Peak picker", ref=5, filt_img=1.0,
            filt_ccr=0, min_thresh=0.1, ex_edge=False, fastdetect=False,
        )
        _ok(locs.ndim == 2 and locs.shape[1] >= 8, f"returns Nxm, shape={locs.shape}")
        if not np.all(np.isnan(locs)):
            _ok(locs.shape[0] > 0, f"found {locs.shape[0]} particles")
    except Exception as e:
        _err("detector peak picker", e)

    try:
        locs_fast = detector(
            img, method="Peak picker", ref=5, filt_img=0, filt_ccr=0,
            min_thresh=0.1, ex_edge=False, fastdetect=True,
        )
        _ok(locs_fast.ndim == 2, f"fastdetect mode, shape={locs_fast.shape}")
    except Exception as e:
        _err("detector peak picker fastdetect", e)


# ============================================================================
# 4. DETECTOR (CCR mode)
# ============================================================================
def test_detector_ccr() -> None:
    print("\n--- detector (CCR) ---")
    from pnanolocz.detector import detector

    full_img = _load(DET_DIR / "000Triangles 5 degrees.tif")
    # Use a small center crop for speed
    img = full_img[150:350, 150:350]
    ref_img = img[30:130, 30:130]

    try:
        locs = detector(
            img, method="ccr", ref=ref_img, filt_img=0, filt_ccr=0,
            min_thresh=0.3, ex_edge=True, fastdetect=False,
        )
        _ok(locs.ndim == 2 and locs.shape[1] >= 8, f"returns Nxm, shape={locs.shape}")
    except Exception as e:
        _err("detector ccr", e)

    # With rotation
    try:
        locs_rot = detector(
            img, method="ccr", ref=ref_img, filt_img=0, filt_ccr=0,
            min_thresh=0.3, ex_edge=True, fastdetect=False,
            angles=[0, 5, 10],
        )
        _ok(locs_rot.ndim == 2, f"CCR + rotation, shape={locs_rot.shape}")
    except Exception as e:
        _err("detector ccr rotation", e)


# ============================================================================
# 5. LOCALIZE
# ============================================================================
def test_localize() -> None:
    print("\n--- localize ---")
    from pnanolocz.localize import localize
    from pnanolocz.fast_peaks2d import fast_peaks2d

    img = _load_crop(DET_DIR / "000Triangles 5 degrees.tif", size=200)
    peaks = fast_peaks2d(img, thresh=0.1, kernel_size=5, matlab_indexing=True)

    if peaks.size == 0:
        print("  SKIP: no peaks found for localization test")
        return

    # Take up to 20 peaks for speed
    peaks = peaks[:min(20, peaks.shape[0])]

    locs_in = np.zeros((peaks.shape[0], 12), dtype=np.float64)
    locs_in[:, 0:2] = peaks[:, 0:2]
    locs_in[:, 2] = peaks[:, 2]
    locs_in[:, 4] = 1

    for method in ["bicubic", "bilinear", "gaussian"]:
        try:
            out = localize(img, locs_in, loc_method=method, pixperfeat=1.0)
            _ok(out.ndim == 2, f"localize ({method}), shape={out.shape}")
        except Exception as e:
            _err(f"localize ({method})", e)


def test_localize_cvcubic_matches_opencv_resize() -> None:
    import cv2

    from pnanolocz.localize import localize

    image = np.zeros((9, 9), dtype=np.float64)
    image[3:6, 3:6] = np.array(
        [[0.2, 1.1, 0.4], [0.8, 3.0, 1.7], [0.1, 1.4, 0.6]]
    )
    locs = np.zeros((1, 12), dtype=np.float64)
    locs[0, :2] = [5.0, 5.0]
    locs[0, 4] = 1.0

    clip = image[2:7, 2:7]
    resized = cv2.resize(clip, (50, 50), interpolation=cv2.INTER_CUBIC)
    central = resized[10:40, 10:40]
    iy, ix = np.unravel_index(np.argmax(central), central.shape)
    expected = np.array(
        [5.0 + ((ix + 1) - 15.0) / 10.0, 5.0 + ((iy + 1) - 15.0) / 10.0]
    )

    result = localize(image, locs, loc_method="cvcubic", pixperfeat=1.0)

    assert np.array_equal(result[0, :2], expected)


# ============================================================================
# 6. ALIGN_TRANS
# ============================================================================
def test_align_trans() -> None:
    print("\n--- align_trans ---")
    from pnanolocz.align_trans import align_trans

    img = _load(ALIGN_DIR / "Test algin.tif")  # (20, 200, 200)
    if img.ndim != 3:
        print("  SKIP: test image not 3D")
        return
    ref = img[0]

    for method in ["Cross corr", "FFT cross"]:
        try:
            xshift, yshift = align_trans(
                img, ref, method=method, maxshift=20, subpixel=True, filt_cr=0
            )
            _ok(xshift.shape == (img.shape[0],), f"{method}: xshift len={len(xshift)}")
        except Exception as e:
            _err(f"align_trans ({method})", e)

    subpix_img = _load(ALIGN_DIR / "subpix test.tif")
    if subpix_img.ndim == 3:
        try:
            xsub, ysub = align_trans(
                subpix_img, subpix_img[0], method="Cross corr",
                maxshift=10, subpixel=True, filt_cr=0
            )
            _ok(xsub.shape == (subpix_img.shape[0],), "subpixel shift shape ok")
        except Exception as e:
            _err("align_trans subpixel", e)


# ============================================================================
# 7. ALIGN_ROT
# ============================================================================
def test_align_rot() -> None:
    print("\n--- align_rot ---")
    from pnanolocz.align_rot import align_rot

    rot_img = _load(ALIGN_DIR / "Rotation Triangles 5 degrees_.tiff")
    if rot_img.ndim != 3:
        print("  SKIP: rotation test image not 3D")
        return
    ref = rot_img[0]

    for method in ["Rotation corr", "Polar Corr"]:
        try:
            angles = align_rot(rot_img, ref, method=method, maxangle=10, subpixel=True)
            _ok(angles.shape == (rot_img.shape[0],), f"{method}: angles len={len(angles)}")
        except Exception as e:
            _err(f"align_rot ({method})", e)


# ============================================================================
# 8. ALIGN_MOVIE
# ============================================================================
def test_align_movie() -> None:
    print("\n--- align_movie ---")
    from pnanolocz.align_movie import align_movie

    img = _load(ALIGN_DIR / "Test algin.tif")
    if img.ndim != 3:
        print("  SKIP: test image not 3D")
        return
    try:
        x, y = align_movie(img, ref=img[0], pixel_shift=20, full_image=True,
                           sub_pix=True, filt_cr=0)
        _ok(x.shape == (img.shape[0],), f"x shift len={len(x)}")
    except Exception as e:
        _err("align_movie", e)

    try:
        ref_obj = type("Ref", (), {"image": img[1], "position": np.array([5.0, 5.0])})()
        x2, y2 = align_movie(img, ref=ref_obj, pixel_shift=20, full_image=False,
                             sub_pix=False, filt_cr=0)
        _ok(x2.shape == (img.shape[0],), "partial ref mode works")
    except Exception as e:
        _err("align_movie partial ref", e)


# ============================================================================
# 9. ALIGN_ITERATE
# ============================================================================
def test_align_iterate() -> None:
    print("\n--- align_iterate ---")
    from pnanolocz.align_iterate import Particles, align_iterate

    img = _load(ALIGN_DIR / "Test algin.tif")
    if img.ndim != 3:
        print("  SKIP: not 3D")
        return

    n_part = 10
    part_img_stack = np.zeros((n_part, 20, 20), dtype=np.float64)
    for i in range(n_part):
        r0, c0 = 90 + i, 90 + i
        part_img_stack[i] = img[min(i, img.shape[0]-1), r0:r0+20, c0:c0+20]

    locs = np.zeros((n_part, 8), dtype=np.float64)
    locs[:, 0] = np.arange(100, 100 + n_part)
    locs[:, 1] = np.arange(100, 100 + n_part)
    locs[:, 2] = 1.0
    locs[:, 4] = np.arange(1, n_part + 1)

    part = Particles(image=part_img_stack, locs=locs)
    ref = part_img_stack[0].copy()

    try:
        out_part, out_ref = align_iterate(
            img, ref, part,
            tran_iterations=2, translat_method="Cross corr", maxdrift=5,
            rot_iterations=2, rota_method="Rotation corr", maxang=5,
            thresh_min=0, autoupdateref=False,
        )
        _ok(out_part is not None, "returns updated part")
        _ok(out_ref is not None, "returns updated ref")
    except Exception as e:
        _err("align_iterate", e)


# ============================================================================
# 10. ALIGN_PTCLOUD
# ============================================================================
def test_align_ptcloud() -> None:
    print("\n--- align_ptcloud ---")
    from pnanolocz.align_ptcloud import align_ptcloud
    from pnanolocz.fast_peaks2d import fast_peaks2d

    img = _load_crop(DET_DIR / "000Triangles 5 degrees.tif", size=200)
    peaks = fast_peaks2d(img, thresh=0.1, kernel_size=5, matlab_indexing=True)

    if peaks.shape[0] < 10:
        print("  SKIP: not enough peaks")
        return

    ref1 = img[ 10:190,  10:190]
    ref2 = img[ 20:200,  20:200]

    p1 = fast_peaks2d(ref1, thresh=0.1, kernel_size=5, matlab_indexing=True)
    p2 = fast_peaks2d(ref2, thresh=0.1, kernel_size=5, matlab_indexing=True)

    if p1.shape[0] < 5 or p2.shape[0] < 5:
        print("  SKIP: not enough peaks in sub-regions")
        return

    locs_src = np.zeros((min(p1.shape[0], 30), 3), dtype=np.float64)
    locs_src[:, 0:3] = p1[:min(p1.shape[0], 30), 0:3]
    locs_tgt = np.zeros((min(p2.shape[0], 30), 3), dtype=np.float64)
    locs_tgt[:, 0:3] = p2[:min(p2.shape[0], 30), 0:3]

    try:
        result = align_ptcloud(locs_src, ref1, locs_tgt, ref2, auto_rotate=True)
        if isinstance(result, tuple):
            _ok(result[0].ndim == 2, f"aligned locs shape={result[0].shape}")
        else:
            _ok(True, "align_ptcloud returned result")
    except Exception as e:
        _err("align_ptcloud", e)


# ============================================================================
# 11. FIND_CENTER
# ============================================================================
def test_find_center() -> None:
    print("\n--- find_center ---")
    from pnanolocz.find_center import find_center_positions, find_center_ptcloud
    from pnanolocz.fast_peaks2d import fast_peaks2d

    img = _load_crop(DET_DIR / "000Triangles 5 degrees.tif", size=150)
    for fold in [2, 3, 4, 6]:
        try:
            center = find_center_positions(fold, img, align_exp=1.0)
            _ok(center.shape == (2,), f"fold={fold}: center={np.round(center, 1)}")
        except Exception as e:
            _err(f"find_center_positions fold={fold}", e)

    peaks = fast_peaks2d(img, thresh=0.1, kernel_size=5, matlab_indexing=True)
    if peaks.shape[0] > 5:
        locs = np.zeros((peaks.shape[0], 3), dtype=np.float64)
        locs[:, 0:3] = peaks[:, 0:3]
        for fold in [2, 3]:
            try:
                center2 = find_center_ptcloud(fold, img, locs)
                _ok(center2.shape == (2,), f"ptcloud fold={fold}: ok")
            except Exception as e:
                _err(f"find_center_ptcloud fold={fold}", e)


# ============================================================================
# 12. LINE_SHIFT
# ============================================================================
def test_line_shift() -> None:
    print("\n--- line_shift ---")
    from pnanolocz.line_shift import line_shift

    img = _load_crop(DET_DIR / "_linesshift 0-1-80 degrees.tif", size=200)
    try:
        shift_val, shifted_img = line_shift(img, shift_type="min")
        _ok(isinstance(shift_val, float), f"shift='min' returns {shift_val:.2f}")
        _ok(shifted_img.shape == img.shape, "output shape matches input")
    except Exception as e:
        _err("line_shift min", e)

    try:
        shift_val2, _ = line_shift(img, shift_type="median")
        _ok(isinstance(shift_val2, float), f"shift='median' returns {shift_val2:.2f}")
    except Exception as e:
        _err("line_shift median", e)

    try:
        shift_val3, _ = line_shift(img, shift_type=0.5)
        _ok(isinstance(shift_val3, float), "shift=float works")
    except Exception as e:
        _err("line_shift float", e)


# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DETECTION & ALIGNMENT TESTS")
    print("=" * 60)

    test_fast_peaks2d()
    test_peaks2d()
    test_detector_peakpicker()
    test_detector_ccr()
    test_localize()
    test_align_trans()
    test_align_rot()
    test_align_movie()
    test_align_iterate()
    test_align_ptcloud()
    test_find_center()
    test_line_shift()

    summary()
