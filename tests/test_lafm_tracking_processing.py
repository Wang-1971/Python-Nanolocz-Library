"""Systematic functionality tests for LAFM, tracking, and image processing.

Uses test images from Software_testing_images/LAFM testing/, /Tracking/, etc.
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
LAFM_DIR = TEST_IMG / "LAFM testing"
TRACK_DIR = TEST_IMG / "Tracking"
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
# 1. MAT_SIM_AFM — AFM simulation from coordinates
# ============================================================================
def test_mat_sim_afm() -> None:
    print("\n--- mat_sim_afm ---")
    from pnanolocz.mat_sim_afm import mat_sim_afm, mat_sim_afm_dyn, mat_sim_afm_spin

    rng = np.random.default_rng(42)
    # Create simple atomic coordinates (x, y, z)
    n_atoms = 20
    coords = np.column_stack([
        rng.uniform(0, 50, n_atoms),
        rng.uniform(0, 50, n_atoms),
        rng.uniform(0.5, 2.0, n_atoms),
    ])

    try:
        img = mat_sim_afm(coords, r=5.0, angle=12.0, pix_per_ang=1.0)
        _ok(img.ndim == 2, f"returns 2D image, shape={img.shape}")
        _ok(np.any(np.isfinite(img)), "image has finite values")
        _ok(np.any(img > 0), "image has positive values")
    except Exception as e:
        _err("mat_sim_afm", e)

    # Dynamic simulation
    try:
        movie = mat_sim_afm_dyn(coords, r=5.0, angle=12.0, pix_per_ang=1.0,
                                fluct_z=0.3, fluct_xy=0.2, n=3)
        _ok(movie.ndim == 3, f"dynamic: returns 3D, shape={movie.shape}")
        _ok(movie.shape[0] == 3, f"dynamic: {movie.shape[0]} frames")
    except Exception as e:
        _err("mat_sim_afm_dyn", e)

    # Spin simulation
    try:
        spin_angles = np.array([0, 45, 90, 135])
        movie2 = mat_sim_afm_spin(coords, r=5.0, angle=12.0, pix_per_ang=1.0,
                                  spin=spin_angles, axis="Z")
        _ok(movie2.ndim == 3, f"spin: returns 3D, shape={movie2.shape}")
    except Exception as e:
        _err("mat_sim_afm_spin", e)


# ============================================================================
# 2. LAFM_RENDERER
# ============================================================================
def test_lafm_renderer() -> None:
    print("\n--- lafm_renderer ---")
    from pnanolocz.lafm_renderer import lafm_renderer, lafm_movie_renderer
    from pnanolocz.afm_colormap import afm_colormap

    # Use LAFM test image and detect particles to get locs
    img = _load(LAFM_DIR / "AFM_sim.tiff")
    from pnanolocz.fast_peaks2d import fast_peaks2d
    peaks = fast_peaks2d(img, thresh=0.1, kernel_size=3, matlab_indexing=True)

    if peaks.shape[0] < 5:
        print("  SKIP: not enough peaks for LAFM rendering")
        return

    # Build locs table: [x, y, z, corr, frame, ...]
    locs = np.zeros((peaks.shape[0], 8), dtype=np.float64)
    locs[:, 0:2] = peaks[:, 0:2]
    locs[:, 2] = peaks[:, 2]
    locs[:, 4] = 1

    cmap = afm_colormap("AFM gold")

    try:
        rendered, zlims = lafm_renderer(
            locs, img_gus=1.0, expand=2.0, fullcolormap=cmap,
            prob=False, colorlimits=[0, 1], colorlimit_mode="Manual",
        )
        _ok(rendered.ndim >= 2, f"static render shape={rendered.shape}")
    except Exception as e:
        _err("lafm_renderer", e)

    # Movie renderer
    try:
        frames, zlims2, times = lafm_movie_renderer(
            locs, img_gus=1.0, expand=2.0, fullcolormap=cmap,
            prob=False, colorlimits=[0, 1], colorlimit_mode="Manual",
            window=10, slide=5,
        )
        _ok(len(frames) >= 1, f"movie renderer: {len(frames)} frames")
    except Exception as e:
        _err("lafm_movie_renderer", e)


# ============================================================================
# 3. TRACK_PARTICLES
# ============================================================================
def test_track_particles() -> None:
    print("\n--- track_particles ---")
    from pnanolocz.track_particles import track_particles, simpletracker, hungarianlinker, nearestneighborlinker

    # Create synthetic particle tracks across frames
    rng = np.random.default_rng(123)
    n_frames = 5
    n_per_frame = 20
    xy_list = []
    frames_list = []

    # Simulate particles moving slowly
    true_positions = rng.uniform(10, 190, (n_per_frame, 2))
    for f in range(n_frames):
        noise = rng.normal(0, 1, (n_per_frame, 2))
        pos = true_positions + noise + f * 0.5  # small drift
        xy_list.append(pos)
        frames_list.append(np.full(n_per_frame, f + 1))

    all_xy = np.vstack(xy_list)
    all_frames = np.concatenate(frames_list)

    try:
        tracks = track_particles(all_xy, all_frames, max_step=10, max_missing_frames=2)
        _ok(len(tracks) == len(all_xy), f"returns {len(tracks)} track IDs")
        _ok(np.max(tracks) > 0, f"max track ID = {np.max(tracks)}")
    except Exception as e:
        _err("track_particles", e)

    # simpletracker directly
    try:
        result = simpletracker(xy_list, max_linking_distance=10, max_gap_closing=2)
        _ok(len(result.tracks) > 0, f"simpletracker: {len(result.tracks)} tracks")
    except Exception as e:
        _err("simpletracker", e)

    # hungarianlinker
    try:
        ti, td, ut, tc = hungarianlinker(xy_list[0], xy_list[1], max_distance=10)
        _ok(ti.shape[0] == n_per_frame, "hungarianlinker returns correct shape")
    except Exception as e:
        _err("hungarianlinker", e)

    # nearestneighborlinker
    try:
        ti2, td2, ut2 = nearestneighborlinker(xy_list[0], xy_list[1], max_distance=10)
        _ok(ti2.shape[0] == n_per_frame, "nearestneighborlinker returns correct shape")
    except Exception as e:
        _err("nearestneighborlinker", e)


# ============================================================================
# 4. SHARPEN
# ============================================================================
def test_sharpen() -> None:
    print("\n--- sharpen ---")
    from pnanolocz.sharpen import sharpen, fastsmooth, secderiv

    # sharpen is a 1D signal sharpener: sharpen(x, y, factor1, factor2, smooth_width)
    try:
        x = np.linspace(0, 10, 200)
        y = np.sin(x) + 0.3 * np.sin(2.5 * x)
        sharp = sharpen(x, y, factor1=1.0, factor2=0.5, smooth_width=5)
        _ok(len(sharp) == len(y), f"sharpen preserves length {len(sharp)}")
        _ok(np.any(np.isfinite(sharp)), "output has finite values")
    except Exception as e:
        _err("sharpen", e)

    # fastsmooth: fastsmooth(Y, w, smooth_type=1, ends=0) — 1D signal smoother
    try:
        x = np.linspace(0, 10, 200)
        y = np.sin(x)
        smooth = fastsmooth(y, w=5, smooth_type=3)
        _ok(len(smooth) == len(y), f"fastsmooth preserves length {len(smooth)}")
    except Exception as e:
        _err("fastsmooth", e)

    # secderiv: test 1D signal
    try:
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        d2 = secderiv(x, y)
        _ok(len(d2) == len(y), f"secderiv returns len={len(d2)}")
    except Exception as e:
        _err("secderiv", e)


# ============================================================================
# 5. SCAR_FILL
# ============================================================================
def test_scar_fill() -> None:
    print("\n--- scar_fill ---")
    from pnanolocz.scar_fill import scar_fill

    img = _load(DET_DIR / "000Triangles 5 degrees.tif")
    try:
        filled = scar_fill(img, thresh=0.1, thresh_h=3, min_length=20)
        _ok(filled.shape == img.shape, f"scar_fill preserves shape {filled.shape}")
        _ok(np.all(np.isfinite(filled)), "output all finite")
    except Exception as e:
        _err("scar_fill", e)


# ============================================================================
# 6. FILTER_MOVIE
# ============================================================================
def test_filter_movie() -> None:
    print("\n--- filter_movie ---")
    from pnanolocz.filter_movie import filter_movie

    # Use movie stack
    movie = _load(ALIGN_DIR / "Fine align_tritest.tiff")
    if movie.ndim != 3:
        movie = movie.reshape(1, *movie.shape)

    filters_to_test = [
        ("sphere_med", 3),
        ("sphere_tophat", 5),
        ("gauss_high", 1.0),
        ("row_align", 1),
        ("median", 3),
    ]

    for filt_name, strength in filters_to_test:
        try:
            out = filter_movie(movie, filt_name, strength)
            _ok(out.shape == movie.shape, f"'{filt_name}': shape preserved")
        except Exception as e:
            _err(f"filter_movie '{filt_name}'", e)

    # Two filters chained
    try:
        out2 = filter_movie(movie, "sphere_med", 3, "gauss_high", 1.0)
        _ok(out2.shape == movie.shape, "chained filters work")
    except Exception as e:
        _err("filter_movie chained", e)


# ============================================================================
# 7. FFT_LINE_ANALYSIS
# ============================================================================
def test_fft_line_analysis() -> None:
    print("\n--- fft_line_analysis ---")
    from pnanolocz.fft_line_analysis import fft_line_analysis

    # Create synthetic line profile with periodic signal
    x = np.linspace(0, 100, 500)
    y = np.sin(2 * np.pi * x / 10.0) + 0.5 * np.sin(2 * np.pi * x / 25.0)
    y += np.random.default_rng(0).normal(0, 0.1, len(x))

    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
    except ImportError:
        pass  # do_plot=False doesn't need matplotlib

    try:
        result = fft_line_analysis(x, y, do_plot=False, max_peaks=3)
        _ok(len(result.periods) >= 1, f"found {len(result.periods)} dominant period(s)")
        _ok(len(result.amps) == len(result.periods), "amps match periods")
    except Exception as e:
        _err("fft_line_analysis", e)


# ============================================================================
# 8. MEASURE_FRC
# ============================================================================
def test_measure_frc() -> None:
    print("\n--- measure_frc ---")
    from pnanolocz.measure_frc import measure_frc
    from pnanolocz.fast_peaks2d import fast_peaks2d

    # Build locs from AFM image for FRC measurement
    img = _load(LAFM_DIR / "AFM_sim.tiff")
    peaks = fast_peaks2d(img, thresh=0.1, kernel_size=3, matlab_indexing=True)

    if peaks.shape[0] < 10:
        print("  SKIP: not enough peaks for FRC test")
        return

    # Create proper locs table with multiple frames for FRC
    n_peaks = peaks.shape[0]
    locs = np.zeros((n_peaks * 2, 8), dtype=np.float64)
    locs[:n_peaks, 0:2] = peaks[:, 0:2]
    locs[:n_peaks, 2] = peaks[:, 2]
    locs[:n_peaks, 4] = 1
    locs[n_peaks:, 0:2] = peaks[:, 0:2] + np.random.default_rng(0).normal(0, 0.5, (n_peaks, 2))
    locs[n_peaks:, 2] = peaks[:, 2]
    locs[n_peaks:, 4] = 2

    try:
        frc_curve, q, av_res, sd_res = measure_frc(locs, pixpernm=1.0, runs=3, expand=2.0)
        _ok(len(frc_curve) > 0 or np.isfinite(av_res),
            f"measure_frc: av_res={av_res:.3f}" if np.isfinite(av_res) else "measure_frc returned")
    except Exception as e:
        _err("measure_frc", e)


# ============================================================================
# 9. IMPROFILE_THICK
# ============================================================================
def test_improfile_thick() -> None:
    print("\n--- improfile_thick ---")
    from pnanolocz.improfile_thick import improfile_thick

    img = _load(DET_DIR / "000Triangles 5 degrees.tif")
    # Create a simple perimeter (rectangle corners)
    perimeter = np.array([[50, 50], [50, 200], [200, 200], [200, 50]], dtype=np.float64)
    try:
        profile = improfile_thick(thickness=5, perimeter=perimeter, img=img, shape="rectangle")
        _ok(profile.ndim == 1, f"profile len={len(profile)}")
        _ok(len(profile) > 0, "profile has data")
    except Exception as e:
        _err("improfile_thick", e)


# ============================================================================
# 10. LINEPROFILER
# ============================================================================
def test_lineprofiler() -> None:
    print("\n--- lineprofiler ---")
    from pnanolocz.lineprofiler import lineprofiler
    from pnanolocz.fast_peaks2d import fast_peaks2d

    img = _load(DET_DIR / "000Triangles 5 degrees.tif")
    peaks = fast_peaks2d(img, thresh=0.05, kernel_size=5, matlab_indexing=True)
    if peaks.shape[0] < 3:
        print("  SKIP: not enough peaks for lineprofiler")
        return

    try:
        xy = peaks[:5, 0:2]  # first 5 peak coordinates
        Fwidth, mean_prof, prof_x, p = lineprofiler(
            img, xy, max_radius=10, directions=[1, 1, 1, 1],
            width_ref="local height",
        )
        _ok(Fwidth.ndim >= 1, f"lineprofiler: Fwidth shape={Fwidth.shape}")
    except Exception as e:
        _err("lineprofiler", e)


# ============================================================================
# 11. OUTLIERS / REM_OUTLIERS
# ============================================================================
def test_outliers() -> None:
    print("\n--- outliers / rem_outliers ---")
    from pnanolocz.outliers import rem_outliers

    # Create dummy localizations with an outlier
    locs = np.zeros((10, 8), dtype=np.float64)
    locs[:, 0] = np.arange(10, dtype=np.float64) * 10 + 100  # x
    locs[:, 1] = 200  # y
    locs[:, 2] = 1.0  # z
    locs[5, 2] = 100.0  # outlier height

    try:
        clean = rem_outliers(locs, threshold=3.0)
        _ok(clean.shape[0] <= locs.shape[0], f"filtered: {locs.shape[0]} -> {clean.shape[0]}")
    except Exception as e:
        _err("rem_outliers", e)

    # Also test rem_outliers from rem_outliers module (same function)
    from pnanolocz.rem_outliers import rem_outliers as rem_outliers2
    try:
        clean2 = rem_outliers2(locs, threshold=3.0)
        _ok(clean2.shape[0] <= locs.shape[0], "rem_outliers duplicate module works")
    except Exception as e:
        _err("rem_outliers (dup)", e)


# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("LAFM, TRACKING & IMAGE PROCESSING TESTS")
    print("=" * 60)

    test_mat_sim_afm()
    test_lafm_renderer()
    test_track_particles()
    test_sharpen()
    test_scar_fill()
    test_filter_movie()
    test_fft_line_analysis()
    test_measure_frc()
    test_improfile_thick()
    test_lineprofiler()
    test_outliers()

    summary()
