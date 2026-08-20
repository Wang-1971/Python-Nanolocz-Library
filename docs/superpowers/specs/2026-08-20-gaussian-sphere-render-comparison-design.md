# Gaussian and Sphere MATLAB/Python Render Comparison Design

## Goal

Generate the same final LAFM render comparison artifacts used by the existing
bicubic parity run for every Gaussian and sphere localization table pair.

## Inputs

- Gaussian run: `Software_testing_images/test_output/gaussian_localization_validation/20260819_174126`
- Sphere run: `Software_testing_images/test_output/sphere_localization_validation/20260819_174309`
- Each run contains self-contained `matlab/` and `python/` manifests and 13 CSV localization tables.

## Design

Reuse `pnanolocz.lafm_table_render_pairs` so MATLAB and Python tables are
rendered on the same XY grid, with the same Z range, LAFM colormap, expansion,
and Gaussian rendering parameters. Extend its run-directory discovery to
accept both the legacy bicubic directory names (`matlab_tables`,
`python_tables`) and the Gaussian/sphere directory names (`matlab`, `python`).
Do not create method-specific rendering implementations.

Each validation run receives a new non-overwriting `renders` directory with:

- `matlab/`: 13 MATLAB RGB LAFM TIFF renders.
- `python/`: 13 Python RGB LAFM TIFF renders.
- `comparisons/`: 13 side-by-side PNG comparisons with a shared height colorbar.
- `render_manifest.json`: source names, row counts, render paths, shared bounds,
  Z range, image shape, colormap, `img_gus`, and `expand`.

## Failure Handling

Rendering stops with a clear error if a manifest is missing, source sets differ,
an entry failed, a required table is missing or empty, or the input does not
contain exactly 13 paired TIFF results. Existing render directories are never
overwritten; a numeric suffix is used instead.

## Verification

Add a regression test for the alternate `matlab/` and `python/` run layout,
observe it fail before implementation, then make the minimum compatibility
change. Run the focused renderer tests and generate both complete render sets.
Verify each method has 13 MATLAB TIFFs, 13 Python TIFFs, 13 comparison PNGs,
and a 13-entry manifest.
