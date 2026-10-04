# CRM R1: one checkout, external results

Use the existing `LLM-PathwayCurator` checkout for code. Use the existing
`CRM_R1` directory for inputs, analysis outputs, ratings, and manuscript files.
Historical smoke-test checkouts and the copy under `CRM_R1/projects/` remain
reference material; do not edit or delete them as part of this workflow.

Code and tests are tracked by Git. Runtime inputs and outputs are not added to
Git. Pull tested updates from `origin/main`; no new local clones or per-run
branches are required. Do not use `git add .`, reset, clean, or force push to
resolve synchronization problems.

If a tested code bundle is supplied before it has been pushed, its `APPLY.py`
uses `install_revision_bundle.py`. The default is a read-only preflight.
`--apply` accepts only absent new files, exact base versions, or already
identical versions. Any differing local file stops the entire preflight before
code changes. Updated originals and the install log are saved in
`CRM_R1/output/revision_v17/code_install_<UTC timestamp>/`.
`--run` executes the supplied milestone using the calling Python. R01 bundles
execute R01; R02 bundles execute R02 with their separately supplied metadata
snapshot. R02 metadata files are checked before installation and are not added
to Git. Place R02 bundles beneath `CRM_R1/input/` before using `--run`.
Optional `--publish` requires
the existing `main` checkout to equal the fetched `origin/main`; it stages and
commits only listed code files, leaves unrelated staged files out of the commit,
and uses an ordinary `HEAD:main` push. Git history conflicts stop the operation.

## R01: saved-result inventory and deterministic adapter

Run from the existing checkout, with its existing Python environment:

```bash
export CRM_R1_DATA_ROOT="/path/to/CRM_R1"
python paper/revision/CRM_R1/scripts/60_revision_r01.py
```

The entry point imports the package from this checkout's `src/`, preventing a
different editable smoke-test installation from supplying the implementation.
It creates a new directory under:

```text
CRM_R1/output/revision_v17/r01_<UTC timestamp>/
```

It verifies recorded hashes of the saved P2B figure sources, discovery statistics,
and Hallmark snapshot. Historical absolute paths are rebased to the explicit
data root using their full relative paths, not just filenames. Every accessed
input is hashed again before successful completion.

It inventories P1, P3 grading progress, and P4 ratings without changing them.
Missing optional inventory files are reported as missing. Missing or changed
required frozen inputs stop the run; a `FAILED.json` is retained.

It checks all saved discovery partitions for numerical validity, direction,
complete term census, and leading-edge membership in the frozen gene sets.
It exports the fixed **ACC/S001 50-term census**, checks it against the saved
discovery job and Sample Card, and runs the opt-in contract API in deterministic
mode. Standard statements derive from canonical evidence, with exact source
locators and hashes. Stability and context scores are not invented or exported.
The frozen statistics supply the exported numerical values. Job copies may
differ by at most one binary64 step from TSV reserialization, provided neither
the NES sign nor the FDR decision changes; such differences are recorded rather
than silently rounded. Larger differences stop the run.

Important files:

| File | Purpose |
| --- | --- |
| `summary.json` | Completion status, row counts, candidate retention, and scope |
| `inventory/summary.json` | Saved-result inventory and P3/P4 progress |
| `adapter/all_discovery_schema_qc.tsv` | Checks for all frozen discovery partitions |
| `adapter/source_mapping.private.tsv` | Exact source rows and standard statements |
| `adapter/job_numeric_alignment.private.tsv` | Job/frozen numeric comparisons, including one-step reserialization |
| `adapter/contract_run/summary.json` | Candidate retention and execution status |
| `INPUT_MANIFEST.private.json` | Hashes of accessed code and data |
| `project_inventory.private.json` | Read-only inventory of similarly named checkouts |

R01 makes **zero LLM calls**, does not rerun enrichment, does not change historical
memberships, and is not confirmatory biological validation. `NOT_RUN` semantic
status in deterministic mode is expected. Retaining 50 candidates establishes
input interoperability, not accuracy or a benefit over simpler reporting.

## Version boundary

The V16.1 opt-in API is integrated from the previously tested package bundle.
Its cached-response/schema infrastructure and deterministic controls are useful,
but its known real-model semantic failures remain unresolved. Integration and
developer tests do not establish that real-model review works.

The V17 revision direction separates candidate preservation, factual reporting,
semantic review, and independent biological evidence. Historical P1-P5 protocols
and negative results remain intact. A new natural-text comparison and external
dataset protocol must be frozen separately after development. Large repeated
expert review and completion of all 620 P3 rows are not R01 prerequisites.

## R02: historical figure sources and external metadata

After applying the R02 bundle, run from the same checkout:

```bash
python paper/revision/CRM_R1/scripts/61_revision_r02.py \
  --metadata-snapshot "$CRM_R1_BUNDLE/metadata_snapshot"
```

`CRM_R1_DATA_ROOT` is reused. Results go to a new
`CRM_R1/output/revision_v17/r02_<UTC timestamp>/` directory. The script uses
pandas and the standard library. It needs no LLM, plotting or PDF dependencies
and makes no network requests. All accessed code/data hashes are rechecked.

`figures/historical_run_inventory.tsv` retains unsuccessful and uncertified
records. `figures/reviewed_panel_routes.tsv` distinguishes reviewed visual layout
from unrecovered PDF assembly lineage. The source map's missing paths and
templates remain visible in `figures/declared_dependencies.tsv`; the script
never fixes a declared path by matching only a basename.

`metadata/sample_design.tsv` and `metadata/discovery_pairs.tsv` record the
public sample census and verified discovery pairing. GSE34313's replicate
suffixes do not establish donor pairing. `COMPLETE_WITH_FINDINGS` means the
audit ran successfully and retained unresolved scientific/source issues. It
does not certify biological validity or an evaluation protocol.

See `R02_FINDINGS.md` for the concrete figure corrections and external design
limitations. Next is bounded semantic development and a separately frozen
comparison. Do not generate new held-out biological outcomes from R01/R02.
