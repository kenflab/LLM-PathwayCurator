# R07: one local natural-text generation and reporting comparison

R06 reused the final three-rater P4 ratings and found no demonstrated legacy
audit advantage. The final R2 workbook matches all 50 rows and all eight locked
fields; the older Q3 source-version hold is resolved. R07 does not change those
ratings, selections or negative findings.

The public historical LLM proposal table contains a structured claim and an LLM
context rationale. Its rationale is not a naturally generated enrichment report.
Its historical backend identity is also uncertified. Neither that rationale nor
a deterministic replacement will be relabelled as an unaudited natural report.

R07 therefore freezes **one new short paragraph per each of the 50 Hallmark
terms in the existing HNSC/S001 discovery partition**, in lexicographic term
order. It uses the previously authenticated local `llama3.1:8b` weights and
Ollama version from the returned R04 receipt. No paid backend is supported.
No second semantic judge is called. Previously inspected discovery statistics
remain development data; new generated paragraphs are not independent biological
validation. Do not retune the policy after seeing responses.

## Comparators and endpoints

| Comparator | Reported text and selection |
| --- | --- |
| Unaudited LLM prose | Exact first successfully completed returned paragraph |
| Explicit numeric gate | Same paragraph retained only when the existing V16.1 checker finds explicit NES and q relations and no detected mismatch |
| Matched q-value | Same paragraphs, selecting K by ascending source q and term ID among successfully generated candidates; K is the gate's retained count |
| Source template | Existing `factual_statement` compiled from every canonical source record, without an LLM |

The matched q comparison is conditional on available generated paragraphs, not
the historical P2B selection. Ties use term IDs; K=0 remains visible. All 50
canonical candidates are retained regardless of report availability. Coverage
uses all 50, including errors, incomplete generations and unattempted requests.

Numeric gates detect only their explicit syntax. An exact decimal mismatch may
be benign rounding, and other wording may be uncheckable. A checked paragraph
can still contain a wrong direction, metadata error or unsupported interpretation.
Do not call its retained state PASS, safe, full audit, or biological correctness.
The unchanged checker will not be widened after viewing these outputs.

Coverage and numeric-check status are software observations, not accuracy labels.
Do not score the gate with ground truth produced by the same gate. The optional
`source_fact_check.optional.private.tsv` starts with blank labels and presents
only original prose and independently authenticated source records; gate and
selection decisions are absent. It is a bounded **author check of reported source
facts**, not a new expert evaluation. Annotate each available paragraph once,
before inspecting method comparisons, with CHECKED_FACTS_MATCH,
CHECKED_FACT_ERROR or UNCLEAR and the author identifier. An error requires an
exact error quote; UNCLEAR requires a reason. Check reported NES/q including
legitimate rounded display, NES direction under the specified contrast, cohort,
and comparison against the referenced original source row. Do not grade biological
interpretations or literature support under this label. Missing claims or scope
that cannot be resolved are UNCLEAR. No label is inferred automatically.

Unannotated/UNCLEAR items remain unknown. Conditional error proportions, false
withholding and missed error counts may only be reported with these unknown
counts and denominators. The source template is the simple reporting baseline;
its construction and matching numbers are not independent accuracy or a novel
audit advantage. P4 labels apply to their original standardized statements and
will not be attached to the new paragraphs or templates.

If source-fact annotation is later needed, save a separate copy of the optional
packet outside the immutable result directory. Evaluate it with
`66_revision_r07.py --data-root "$CRM_R1_DATA_ROOT" --evaluate-archive
"/path/to/r07_<UTC>.zip" --annotations "/path/to/source_facts.tsv"`.
The evaluator authenticates the complete archived output, original source rows,
candidate IDs and exact paragraph hashes. Unknown labels remain unknown. Its
lower/upper error proportions are partial-information bounds, not confidence
intervals. No bootstrap p-value, semantic-safety estimate or template accuracy
is produced. Annotation does not call any model.

## Execution and preservation

`66_revision_r07.py --data-root "$CRM_R1_DATA_ROOT"` freezes the design without
network/model calls, using the R01 adapter's source hashes, full partition census,
gene memberships, sample card and job/numeric alignment. The design and all code
inputs are hashed. Repeating preparation verifies and reuses the immutable design.

`66_revision_r07.py --data-root "$CRM_R1_DATA_ROOT" --live` explicitly starts
local generation: at most 50 new requests and 1800 seconds for starting requests,
with a 120-second timeout per in-flight request. Smaller budgets may be passed.
The local backend must match the frozen weights/version; no model pull, server
start or fallback is attempted. An unavailable/mismatched backend stops before
generation.

Every first outcome is retained, including truncation, malformed wire responses,
backend mismatches and interruptions. The paragraph string is preserved exactly.
Repeating generation reuses complete/failed first outcomes and only starts
unattempted candidates. The cache is not deleted to improve responses.

New output directories and ZIPs live in `CRM_R1/output/revision_v17/` outside Git.
They include raw first outcomes, canonical source records, method coverage,
selection/number-check records, optional fact-check packet, code/policy snapshots
and input/output hashes. No previous P1/P2B/P4/P5 output is overwritten.

R07 answers a limited report-consistency comparison. It does not complete external
biological validation or guarantee that the revised method merits acceptance.
After the first actual output, review the saved error/coverage patterns before
deciding whether further data are necessary. R03/R04's known controls are not a
progression gate. Large expert Round 2 and all 620 P3 rows are not prerequisites.
