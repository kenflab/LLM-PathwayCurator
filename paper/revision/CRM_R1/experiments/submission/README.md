# Revision source assembly

Run `assemble.py --data-root "$CRM_R1_DATA_ROOT"` in the existing checkout.
It reads saved R06/R07/dex archives, checks their export hashes, collects declared
P1/P2B/P5 output tables, and copies the current R1 Word for manuscript integration.
It does not run R, fit a model, grade literature, request ratings or edit Word.

The new folder and compact ZIP are written under
`output/revision_v17/submission_assembly_<UTC>/`. Read `START_HERE_JA.txt`,
`SUMMARY.json`, `RESPONSE_MATRIX.private.tsv` and `PUBLICATION_ACTIONS.private.tsv`.
Missing or unverified components remain explicit; successful collection does
not establish submission readiness or a full-audit interpretive advantage.

The exact existing inputs are:

- R06 archive `r06_20261005T163843024404Z.zip`.
- R07 archive `r07_20261005T190838308028Z.zip`.
- Dex archive referenced by `external_dex_design_v1/COMPLETED_ANALYSIS.json`.
- `LLM-PathwayCurator_CRM_R1.docx` at the data-root top level.
- P1 `priority1_replication.run_meta.json` and its declared outputs.
- P2B `final_v14_1_3/figure2_source_manifest.json` and its declared outputs.
- P5 `ontology_evaluation.run_meta.json` and its declared output hashes.
- The P3 grading-working table, inspected for completeness without recoding.

Original absolute CRM_R1 paths are rebased by their complete relative path.
The program does not search for a newer run or a similarly named alternative.
It copies only bounded reporting artifacts, not raw expression data.

Saved outputs above 8 MiB are checked with a streaming SHA-256 calculation and
listed as `HASH_VERIFIED_NOT_COPIED_COMPACT_SIZE_LIMIT`. They remain in place;
their size does not invalidate a complete, matching output manifest. Smaller
declared outputs and the manifest are copied. `copied_paths` and
`verified_but_not_copied` distinguish available table bytes from hash-only checks.
An actual hash mismatch or missing file still leaves the component unverified.

Word extraction separates deleted text and comments from the final-view body.
Phrase flags identify passages for author review; they are not automatic error
labels or replacements for visual review of the manuscript and figures.

The collected ZIP contains private manuscript and rating records. Keep it in
the existing data directory. Public Git contains only this code and its guides.

## Render verified main-figure drafts

Run the reporting step against one explicitly selected, complete collected ZIP:

```bash
python paper/revision/CRM_R1/experiments/submission/report.py \
  --data-root "$CRM_R1_DATA_ROOT" \
  --source-archive "$CRM_R1_DATA_ROOT/output/revision_v17/<exact-collected-run>.zip"
```

`report.py` validates every exported byte before creating a new
`output/revision_v17/manuscript_report_<UTC>/` folder. It checks the 22-cohort,
20-split P2B census, identical selected K, cohort aggregation and the 12 saved
fixed-seed cohort-bootstrap intervals. It checks the saved P1, dex, hierarchy
and original-text counts without fitting expression models, changing a protocol,
querying models or adding labels. The large claim ledger and raw expression
payloads absent from the compact ZIP are not certified by these checks.

The four numbered figures show the evidence contract/source examples, matched
baselines/original raters, temporal/non-cancer components and hierarchy
diagnostics. `Source_Data/`, `SOURCE_FILE_MAP.tsv` and `SOURCE_VERIFICATION.json`
record the routes, denominators, plotting code and limits. Parent/child matrices
use the named status fields; the saved `status_pattern` remains child-to-parent.
Uncertain labels and unavailable intervals are retained. Incomplete numeric
coverage is not recoded as incorrect prose. Post hoc source examples are not
independent expert labels or automatic full-audit correction demonstrations.

The manuscript, author-response drafts and original labels stay in private data
storage. Source verification and rendering do not establish a general full-audit
advantage or submission readiness. Complete author review, revision-specific
release archiving and remaining provenance/scientific requirements before
submission. This reporting step does not change `src/` or the frozen analyses.
