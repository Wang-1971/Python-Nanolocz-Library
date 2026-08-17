# Gaussian Localization Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate Python Gaussian localization against known synthetic centers and fresh MATLAB results for every LAFM testing TIFF.

**Architecture:** A focused pytest module owns deterministic Gaussian accuracy checks. Separate MATLAB and Python batch exporters consume identical TIFFs and write compatible tables; the existing frame-aware matching utilities compare those tables, while a Gaussian-only report command writes metrics and plots beneath its own run directory.

**Tech Stack:** Python 3.11+, NumPy, SciPy, tifffile, pandas, matplotlib, pytest, MATLAB.

---

### Task 1: Synthetic Gaussian localization tests

**Files:**
- Create: `tests/test_localize_gaussian.py`
- Exercise: `src/pnanolocz/localize.py`

- [ ] **Step 1: Write parametrized failing accuracy tests**

Create a deterministic 17x17 image with `z = 2 + 12 * exp(-((x-cx)^2/(2*sx^2) + (y-cy)^2/(2*sy^2)))`, initialize a 12-column MATLAB-indexed row at the rounded center, and parameterize `(cx, cy, pixperfeat)` over `(8.2, 8.7, 1.0)`, `(8.65, 8.25, 0.5)`, and `(9.1, 7.8, 1.0)`. Assert finite x/y, absolute coordinate error below the measured stable tolerance, positive width in column 10, positive amplitude in column 11, and preservation of frame and sentinel column 12.

- [ ] **Step 2: Add boundary rejection tests**

Pass peaks at `[2, 2]` and `[16, 16]`; assert both output x/y pairs are NaN and non-coordinate columns remain unchanged.

- [ ] **Step 3: Prove the tests detect displacement**

Run the focused test once with the expected center deliberately shifted by 0.5 px. Expected: FAIL on coordinate error. Restore the true center.

- [ ] **Step 4: Run the focused tests**

Run: `& 'C:\Users\hzfq0862\AppData\Local\anaconda3\envs\pnanolocz\python.exe' -m pytest tests/test_localize_gaussian.py -v`

Expected: PASS with no warning from the Gaussian optimizer.

- [ ] **Step 5: Commit**

Run: `git add tests/test_localize_gaussian.py && git commit -m "test: validate Gaussian localization accuracy"`

### Task 2: Gaussian-only real-image exporters

**Files:**
- Create: `src/pnanolocz/gaussian_localization_validation.py`
- Create: `matlab/run_gaussian_localization_export.m`
- Test: `tests/test_localize_gaussian.py`

- [ ] **Step 1: Write failing exporter tests**

Add a temporary two-frame TIFF test that calls `export_python_gaussian_tables(input_dir, output_dir)`. Assert both `.tif` and `.tiff` are discovered, manifest method is `gaussian`, tables use `TABLE_COLUMNS`, frame IDs are retained, and an individual bad file becomes a manifest error without stopping the other file.

- [ ] **Step 2: Run the exporter tests and verify RED**

Run the focused pytest command. Expected: import failure because `gaussian_localization_validation` does not exist.

- [ ] **Step 3: Implement the Python exporter**

Implement `direct_gaussian_localize_stack(movie, pixperfeat=1.0)` by reusing the existing `fast_peaks2d(..., thresh=0.0, kernel_size=1, matlab_indexing=True)` table construction and calling `localize(..., 'gaussian', pixperfeat, frame_axis=0, matlab_indexing=True)`. Filter only rows with finite required coordinates, refresh z from the raw rounded pixel, and export every sorted `*.tif`/`*.tiff` with a manifest that records method and parameters.

- [ ] **Step 4: Implement the MATLAB exporter**

Clone the input enumeration, table schema, and manifest contract from `matlab/run_direct_lafm_table_export.m`, but call `localize(stack, locs, 'gaussian', 1)`. Write results only beneath the caller-supplied Gaussian output directory.

- [ ] **Step 5: Run tests and commit**

Run the focused pytest command. Expected: PASS.

Run: `git add tests/test_localize_gaussian.py src/pnanolocz/gaussian_localization_validation.py matlab/run_gaussian_localization_export.m && git commit -m "feat: export Gaussian localization parity tables"`

### Task 3: Gaussian comparison report and plots

**Files:**
- Modify: `src/pnanolocz/gaussian_localization_validation.py`
- Modify: `tests/test_localize_gaussian.py`

- [ ] **Step 1: Write failing report tests**

Build tiny MATLAB/Python CSV fixtures with one matched and one unmatched row. Assert the command uses `match_localization_tables`, writes matched/MATLAB-only/Python-only CSVs, per-file and overall metrics, an overlay PNG, a residual PNG, and `comparison_report.md` beneath `gaussian_localization_validation/<timestamp>`.

- [ ] **Step 2: Verify RED**

Run the focused pytest command. Expected: failure because the report function is absent.

- [ ] **Step 3: Implement the minimum report command**

Use `create_run_directory`, `match_localization_tables`, and `summarize_matches`. Extend matched rows with Gaussian width/amplitude deltas from table columns 10/11 when available. Use matplotlib for a source-image coordinate overlay and x/y residual scatter. Record failures per input and return non-zero when a manifest entry failed or required outputs are absent.

- [ ] **Step 4: Verify GREEN and commit**

Run the focused pytest command. Expected: PASS.

Run: `git add tests/test_localize_gaussian.py src/pnanolocz/gaussian_localization_validation.py && git commit -m "feat: report Gaussian MATLAB parity"`

### Task 4: Run all TIFFs and regressions

**Files:**
- Generate only: `Software_testing_images/test_output/gaussian_localization_validation/<timestamp>/`

- [ ] **Step 1: Run fresh MATLAB export**

Run MATLAB with `Software_testing_images/LAFM testing` as input and a new run's `matlab` subdirectory as output. Expected: manifest lists every direct `.tif` and `.tiff` with status `ok`.

- [ ] **Step 2: Run Python export and comparison**

Use the `pnanolocz` Python executable. Expected: matching input set, per-file tables/plots, summary CSV/JSON, and Markdown report under the Gaussian-only root.

- [ ] **Step 3: Run regressions**

Run: `& 'C:\Users\hzfq0862\AppData\Local\anaconda3\envs\pnanolocz\python.exe' -m pytest tests/test_localize_gaussian.py tests/test_lafm_table_parity.py tests/test_detection_alignment.py -v`

Expected: PASS.

- [ ] **Step 4: Inspect the worst discrepancies**

Sort per-file metrics by x/y RMSE and inspect the top overlays. Record exact failures and largest deviations in the Gaussian report; do not change tolerances merely to make results pass.

