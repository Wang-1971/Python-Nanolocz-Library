# Sphere Localization Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate Python spherical-cap localization against known synthetic centers and fresh MATLAB results for every LAFM testing TIFF.

**Architecture:** A sphere-only pytest module validates both fitting-window branches using analytic spherical caps. Independent MATLAB and Python batch exporters write compatible localization tables; a sphere-only comparison command reuses existing frame-aware row matching while reporting radius differences and its own artifacts.

**Tech Stack:** Python 3.11+, NumPy, SciPy, tifffile, pandas, matplotlib, pytest, MATLAB.

---

### Task 1: Synthetic spherical-cap localization tests

**Files:**
- Create: `tests/test_localize_sphere.py`
- Exercise: `src/pnanolocz/localize.py`

- [ ] **Step 1: Write parametrized failing accuracy tests**

Create a deterministic 17x17 spherical cap centered at non-integer `(cx, cy)`, using `z = zc + sqrt(max(R^2 - (x-cx)^2 - (y-cy)^2, 0))` inside the cap. Parameterize `(cx, cy, pixperfeat)` to exercise `w=2,const=5` and `w=3,const=8`. Initialize MATLAB-indexed 12-column rows at rounded centers; assert finite x/y, coordinate error below a measured stable tolerance, positive fitted radius in column 10, and preserved frame/sentinel columns.

- [ ] **Step 2: Add boundary rejection tests**

Pass points outside the sphere fitting window and assert NaN x/y with untouched unrelated columns.

- [ ] **Step 3: Prove the tests detect displacement**

Temporarily shift the asserted center by 0.5 px. Expected: coordinate assertion FAIL. Restore the true center.

- [ ] **Step 4: Run focused tests**

Run: `& 'C:\Users\hzfq0862\AppData\Local\anaconda3\envs\pnanolocz\python.exe' -m pytest tests/test_localize_sphere.py -v`

Expected: PASS with finite radius for all valid synthetic cases.

- [ ] **Step 5: Commit**

Run: `git add tests/test_localize_sphere.py && git commit -m "test: validate sphere localization accuracy"`

### Task 2: Sphere-only real-image exporters

**Files:**
- Create: `src/pnanolocz/sphere_localization_validation.py`
- Create: `matlab/run_sphere_localization_export.m`
- Modify: `tests/test_localize_sphere.py`

- [ ] **Step 1: Write failing exporter tests**

Add temporary `.tif` and `.tiff` inputs and assert `export_python_sphere_tables` discovers both, records method `sphere`, retains frame IDs, writes `TABLE_COLUMNS`, and records a bad file without preventing valid exports.

- [ ] **Step 2: Verify RED**

Run the focused pytest command. Expected: import failure because the sphere validation module is absent.

- [ ] **Step 3: Implement the Python exporter**

Construct initial rows with the existing direct pipeline, call `localize(..., 'sphere', pixperfeat, frame_axis=0, matlab_indexing=True)`, retain finite required rows, refresh raw z, and emit a sphere-only manifest and tables.

- [ ] **Step 4: Implement the MATLAB exporter**

Follow `matlab/run_direct_lafm_table_export.m` for input enumeration/schema, call `localize(stack, locs, 'sphere', 1)`, and write only to the caller-provided sphere output directory.

- [ ] **Step 5: Verify GREEN and commit**

Run the focused pytest command. Expected: PASS.

Run: `git add tests/test_localize_sphere.py src/pnanolocz/sphere_localization_validation.py matlab/run_sphere_localization_export.m && git commit -m "feat: export sphere localization parity tables"`

### Task 3: Sphere comparison report and plots

**Files:**
- Modify: `src/pnanolocz/sphere_localization_validation.py`
- Modify: `tests/test_localize_sphere.py`

- [ ] **Step 1: Write failing report tests**

Use tiny fixture tables to assert separate matched/unmatched CSVs, per-file/overall metrics, source overlay, residual plot, radius-delta metrics from column 10, JSON metadata, and Markdown report beneath `sphere_localization_validation/<timestamp>`.

- [ ] **Step 2: Verify RED**

Run the focused pytest command. Expected: failure because the report function is absent.

- [ ] **Step 3: Implement the minimum report command**

Reuse `create_run_directory`, `match_localization_tables`, and `summarize_matches`; add radius bias/MAE/RMSE, per-input error capture, plots, and non-zero exit semantics. Do not import the Gaussian validation module.

- [ ] **Step 4: Verify GREEN and commit**

Run the focused pytest command. Expected: PASS.

Run: `git add tests/test_localize_sphere.py src/pnanolocz/sphere_localization_validation.py && git commit -m "feat: report sphere MATLAB parity"`

### Task 4: Run all TIFFs and regressions

**Files:**
- Generate only: `Software_testing_images/test_output/sphere_localization_validation/<timestamp>/`

- [ ] **Step 1: Run fresh MATLAB export**

Run the sphere MATLAB exporter over every direct TIFF in `Software_testing_images/LAFM testing`. Expected: every input appears once with status `ok`.

- [ ] **Step 2: Run Python export and comparison**

Use the `pnanolocz` Python executable. Expected: sphere-only tables, matches, plots, metrics, metadata, and report.

- [ ] **Step 3: Run regressions**

Run: `& 'C:\Users\hzfq0862\AppData\Local\anaconda3\envs\pnanolocz\python.exe' -m pytest tests/test_localize_sphere.py tests/test_lafm_table_parity.py tests/test_detection_alignment.py -v`

Expected: PASS.

- [ ] **Step 4: Inspect the worst discrepancies**

Sort by x/y and radius RMSE, inspect the worst overlays/residuals, and record concrete discrepancies without weakening tolerances to hide them.
