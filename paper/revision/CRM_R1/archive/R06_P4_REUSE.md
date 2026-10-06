# R06: existing expert evaluations and frozen baseline comparisons

R06 implements the first analysis from the 2026-10-05 revision refocus:
authenticate the original P4 packet/ratings and P2 candidate memberships, then
compare the same, unchanged candidate statements across the saved methods.
It makes **zero model/network requests** and does not require P3 grading.
The paid R05 live-backend recommendation is withdrawn. R05 installation is
not a prerequisite for R06.

## Inputs and linkage

The default benchmark is PANCAN_TP53_v1_HNSC_R1. R06 reads:

- output/priority2/benchmark/metrics/priority2_freeze_manifest.json
- output/priority4/benchmark/packet_v1/priority4_packet_manifest.json
- output/priority4/benchmark/ratings_lock_v1/priority4_ratings_lock_manifest.json
- Their SHA256 companions and declared candidate, membership, sampling,
  packet, template, returned-rating and locked-rating records.
- The original P4 question protocol referenced by the ratings lock.

Historical Mac paths are rebased by their complete relative path. A basename
search, automatic choice among alternative locks, or automatic label recoding
is not used. Hashes, candidate/review/rater census, template assignment, original
return fields, packet order and wording/evidence fields must match.
All accessed inputs are hashed again before completion.

### R06.1 spreadsheet-export compatibility fix

The returned R06 run `r06_20261005T160801990532Z` stopped before comparison
at the second returned rating TSV. Every recorded frozen hash up to that file
matched. The result archive contains no returned TSV bytes, so it establishes
a header-reading block, not the exact duplicate/empty header layout.
A previously shared rating workbook has the eight assigned fields plus two
unnamed supplementary columns, including side notes. That layout reproduces
the R06 rejection; the original P4 importer accepted it with pandas.

R06.1 preserves the original TSV bytes and pandas' positional `Unnamed` columns.
It skips only the leading empty/space-only lines also skipped by pandas.
All named fields retain their original names; duplicate named columns still
block instead of selecting or recoding a response. The eight assigned fields
must reproduce the frozen ratings, and all original hashes still have to match.
No frozen rating, manifest, selection, wording, endpoint or bootstrap setting
is edited. Supplementary notes are retained and do not become rating fields.

`table_read_checks.private.json` records the actual raw/pandas headers, leading
blank lines, unnamed-column positions/nonempty counts and parse status.
It is included in both completed and blocked result ZIPs. R06.1 is a reader
fix within the same R06 analysis, not a new validation experiment.

The analysis verifies the operational linkage needed for P4 reuse, not every
upstream run or every P3 retrieval input. The original blinded literature packet
is hash-checked as a packet output. P3 grading is neither inspected nor required.
Blinding flags are recorded metadata; hashes do not prove what each expert
read outside the saved files.

## Evaluation and scope

Use the existing environment:

    python paper/revision/CRM_R1/scripts/65_revision_r06.py --data-root "$CRM_R1_DATA_ROOT"

The entry point has no live/model flag and imports no model backend.
It writes a new immutable folder and returned ZIP under
CRM_R1/output/revision_v17/r06_UTCtimestamp/.

The four saved methods are all frozen candidates (descriptive), matched
q-value, matched stability and **legacy** full audit. No selection is retuned.
These candidate statements were standardized in P2 and rated in P4.
They are not newly generated unaudited LLM prose. The comparison addresses
same-corpus selection/reporting assessments, not rewriting quality.

Every rater is reported separately. Six fixed category-based endpoints include
confirmed major/any overstatement, statistical support and external evidence.
UNCERTAIN is retained in the selected denominator and is reported explicitly,
with lower/upper partial-information bounds. Absence of literature support is
not equated with biological falsity.

Wilson intervals describe individual-rater category fractions. All method
differences and agreement intervals jointly resample the same candidate IDs,
preserving overlap and rater dependence. The equal-weight rater mean conditions
on these three fixed raters, not a population of independent evaluators.
Claim-level resampling does not model pathway overlap; all intervals are
descriptive. Collapsed bootstrap distributions have unavailable CIs.
No majority labels, p-values or superiority conclusions are generated.
K=0 is retained with undefined fractions and differences.

R06 is post hoc and exploratory because the ratings and earlier outcomes have
already been examined. It preserves the original locked majority protocol
without replacing or rewriting it. It is not independent validation, an
estimate of biological accuracy, or a performance test of the corrected audit.

## Results and blocked inputs

Successful linkage yields:

- summary.json: status, counts, scope, zero model calls and returned ZIP.
- linkage_checks.private.tsv: original text hashes and evidence correspondence.
- ratings_linked.private.tsv: unchanged labels with saved method membership.
- rater_method_endpoints.tsv: all raters, category counts, UNCERTAIN bounds and CIs.
- paired_method_differences.tsv: shared-candidate full-minus-baseline comparisons.
- interrater_agreement.tsv, category counts and pairwise confusion matrices.
- descriptive_q_value_summary.tsv: statistical significance only.
- Fig_R06_P4_major_overstatement_by_rater.pdf and .png.
- READOUT_JA.md, source snapshot and input/output manifests.
- table_read_checks.private.json: original headers and parser compatibility.

Missing/mismatched frozen records yield P4_REUSE_BLOCKED, a precise reason and
no performance tables or figure. Exit zero means the diagnostic completed;
it does **not** mean input reuse or publication readiness is certified.
Unexpected execution failures and inputs changed during execution exit nonzero.
An existing output directory is never overwritten.

The low P4 agreement and unfavorable P2B replication remain findings. New
ratings, all 620 P3 grading rows, another model, or a rerun of 440 partitions are
not automatic responses to a block or an unfavorable result. Inspect the
returned ZIP and the actual evidence gap first.
