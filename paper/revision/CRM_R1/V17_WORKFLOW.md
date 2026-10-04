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
R03 bundles run **offline preparation only** under `--run`; the local-model
pilot is a separate explicit `62_revision_r03.py --live` invocation. Its
historical baseline snapshot is checked before installation and stays outside
Git. Place R03 bundles under `CRM_R1/input/` as well.
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

## R03: bounded atomic development

`R03_DEVELOPMENT.md` defines the fixed development scope and scoring rules.
Apply/publish the R03 bundle in the same checkout:

```bash
python "$CRM_R1_BUNDLE/APPLY.py" \
  --repo "$CRM_R1_REPO" --data-root "$CRM_R1_DATA_ROOT" \
  --apply --run --publish
```

The offline status is `PREPARED_NOT_LIVE_TESTED`. It checks all 66 historical
export hashes, replays the 16 original responses with the unchanged V16.1
strict parser, and prepares 96 atomic requests for the same known controls.
There are 20 candidates, four deterministic cases, and 16 semantic cases.
Preparing requests is not a successful semantic test.

Start the bounded local pilot with the existing Ollama server:

```bash
python "$CRM_R1_REPO/paper/revision/CRM_R1/scripts/62_revision_r03.py" \
  --data-root "$CRM_R1_DATA_ROOT" --source-bundle "$CRM_R1_BUNDLE" --live
```

The script verifies the original `llama3.1:8b` weight digest and records the
current server version. It never pulls a model, starts a server, or substitutes
a different model. Generation uses seed 42, temperature 0, context 16384,
output limit 512, and HTTP timeout 120 seconds. It starts at most 96 new atomic
requests and stops starting new requests after the default 1800-second budget.
An in-flight request may finish after that budget. Smaller limits are available
as `--max-new-requests` and `--wall-budget-seconds`.

Every exact request has one immutable first outcome, including malformed and
interrupted outcomes. Repeating the command reuses those outcomes and starts
only previously unattempted requests, such as cases skipped by the budget.
Do not delete the cache to obtain a different judgment. Prompt/model changes
produce new request identities and are separate development changes.

Results are new `CRM_R1/output/revision_v17/r03_<UTC>/` directories. The shared
R03 cache is `.../r03_atomic_cache/`, separate from all old caches. Live runs
also create a sibling `r03_<UTC>.zip` containing this run's results and only its
own exact request/response records. The summary records its path as
`results_archive`. No manual cache collection is needed.

`scores.private.json` retains complete, incomplete and unattempted cases with
required and forbidden concerns. `baseline_vs_r03.private.json` keeps the old
and new outcomes together without altering the old results. `NOT_RUN` means
unattempted; `INCOMPLETE` means at least one attempted aspect did not complete
or a partially attempted case remains unfinished. All canonical candidates
are retained in either case.

Passing all 20 known controls is only a development gate. Before submission,
freeze a separate natural-text comparison and external biological protocol.
The existing negative P2B results, human ratings, source issues, and P3 missing
grades remain unchanged. No new expert review is requested by R03.
