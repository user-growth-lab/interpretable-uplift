# Public distribution preparation — 31 October 2026

This distribution was prepared from the code supplement actually submitted to
Wiley on 13 October 2026 for manuscript 8651290 (article EXSY70471).
The original submitted ZIP is retained unchanged by the authors.

Submitted archive SHA256:
`2ceb36568253bfa728e892245a4d5becf01ad12d328fa8034d7d5f1d28c37e3f`

## Distribution changes

- Replaced the confidential-review README with public-facing article metadata,
  installation, data acquisition and reproduction instructions.
- Preserved the current implementation, historical implementation and execution
  snapshots in their existing directories so relative paths remain valid.
- Preserved experimental configurations, aggregate results and historical audit
  reports. Removed only the `rows` arrays from 60 recourse result JSONs (3,000
  point-pair records) to avoid redistributing sample-level dataset features.
  Reproduction regenerates these records from separately acquired datasets.
- Added `CITATION.cff`, `.gitignore` and `scripts/smoke_test.py`.
- Preserved the submitted manifest as `audit/submitted_package_manifest.json`.
  The root `PACKAGE_MANIFEST.json` describes this distribution, including all
  changed and added files. Neither manifest includes its own checksum.
- Documented the required empty `results/` directory for a fresh run and the
  explicit creation of `paper/figures/` before plotting.

No original algorithm or experiment Python file was edited. No new scientific
experiment was conducted, and the included historical validation reports are
not claims that this lightweight ZIP alone reproduces their full checks.
The added smoke check verifies only a small, deterministic installation example.

## Publication details still to be supplied

The public repository URL, a fixed repository version/commit, and any selected
software license are not specified in this prepared package. The article DOI
identifies the paper; it is not a software archive DOI. Add the real repository
information after publication of the code, then use that verified address in the
paper's Code Availability update. This package does not include or modify the
accepted manuscript, its supplementary PDF, or the Wiley proof.
