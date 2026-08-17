# Gaussian and Sphere Localization Validation Design

## Goal

Validate the Python `gaussian` and `sphere` localization methods against MATLAB on every TIFF in `Software_testing_images/LAFM testing`, and independently measure localization accuracy on synthetic peaks with known subpixel centers.

## Scope

- Discover every `.tif` and `.tiff` file directly under `Software_testing_images/LAFM testing`.
- Run Gaussian and spherical-cap localization with equivalent MATLAB and Python inputs and parameters.
- Compare localization tables numerically and render diagnostic images.
- Test Python localization accuracy on synthetic Gaussian and spherical-cap peaks.
- Preserve existing notebooks, source edits, and prior test outputs.

The Python API names are `gaussian` and `sphere`; reports may use the human-readable label "spherical".

## Real-image validation

For each TIFF, load the same pixel data in MATLAB and Python. Use one shared configuration for detection thresholds, feature size, indexing convention, and frame selection. Record the resolved configuration in the output metadata so every comparison is reproducible.

Both implementations will produce localization tables for `gaussian` and `sphere`. Rows will be matched within the same frame by one-to-one nearest-neighbour assignment in x/y, subject to an explicit maximum distance. Unmatched rows remain visible in the report rather than being discarded.

For each file and method, report:

- MATLAB and Python localization counts;
- matched, MATLAB-only, and Python-only counts and match rate;
- x/y signed bias, median absolute error, RMSE, and error percentiles;
- Gaussian width/amplitude or sphere-radius differences where both tables expose the field;
- non-finite or rejected localization counts.

Generate an overlay showing the MATLAB and Python coordinates on the source image and a residual plot for matched particles. Produce a machine-readable per-particle CSV and a per-image summary CSV.

## Synthetic accuracy validation

Create deterministic images whose true centers are non-integer coordinates. The Gaussian dataset will use the same axis-aligned 2-D Gaussian model expected by the Python fitter. The sphere dataset will use a sampled spherical-cap height model compatible with the spherical fit.

Each method will be exercised at multiple subpixel offsets, with both `pixperfeat` branches and with valid points near the localization boundary. Assertions will cover:

- finite localized coordinates for valid interior peaks;
- x/y error relative to the known center;
- populated Gaussian width/amplitude or sphere-radius output;
- documented rejection of points outside the valid fitting window;
- preservation of frame and unrelated localization-table columns.

Accuracy tolerances will be derived from exact synthetic inputs and fixed only after confirming that the test fails under a deliberately perturbed center. Tolerances must be strict enough to detect a meaningful regression and loose enough to avoid platform-specific optimizer noise.

## Separate test code and outputs

Keep the two methods separate. Gaussian artifacts go beneath a new timestamped directory under `Software_testing_images/test_output/gaussian_localization_validation`. Sphere artifacts go beneath a new timestamped directory under `Software_testing_images/test_output/sphere_localization_validation`. Each method directory independently contains:

- resolved configuration and environment metadata;
- MATLAB and Python localization tables;
- per-particle match tables;
- per-image overlays and residual plots;
- method summary CSV/JSON files;
- a concise Markdown report listing failures and largest discrepancies.

Automated synthetic tests live in two separate files in the existing `tests` tree: one Gaussian localization test file and one sphere localization test file. Real-image/MATLAB comparison entry points and method-specific assertions likewise remain separate. Both may reuse existing localization-table parity helpers where their matching contract fits, but no combined Gaussian/sphere test module or combined output directory will be created. The tests avoid new dependencies.

## Failure handling

One unreadable image, optimizer failure, or MATLAB failure must be recorded against that file and method without hiding results for other inputs. Gaussian and sphere validation commands have independent exit statuses. Each exits non-zero when one of its required cases fails, its comparison cannot run, or its agreed tolerances are exceeded.

## Verification

Run each focused synthetic test file first, then its corresponding complete real-image MATLAB comparison in the `pnanolocz` Conda environment. Finally run the existing localization-related regression tests. Each final report distinguishes test failures, missing MATLAB reference results, unmatched localizations, and numerical tolerance failures.
