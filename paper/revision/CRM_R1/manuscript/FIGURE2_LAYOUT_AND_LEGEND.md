# Main comparative figure: source and layout guide

The earlier independent-support/majority-risk layout must not be finalized
from incomplete literature grading or interpreted as a biological truth test.
Build the comparative figure from the verified saved-result tables and retain
unfavorable findings. Final numbering follows the current R1 manuscript.

## Proposed panels

- **A — Design:** distinguish the unchanged standardized statement pool and its
  matched-K historical selections from newly generated natural-language prose.
- **B — Multi-cohort comparison:** show cohort-level full-minus-baseline
  replication results, including adverse contrasts. Aggregate splits within
  cohort; do not use split/pathway counts as independent sample size.
- **C — Rater-specific assessment:** show each existing evaluator separately
  for all candidates and the matched selections, with numerators, selected
  denominators and UNCERTAIN counts. State that the analysis is post hoc.
- **D — Agreement:** report Fleiss kappa and descriptive uncertainty for the
  three questions. Keep negative estimates visible and identify undefined or
  degenerate intervals.

Use the R06 `rater_method_endpoints.tsv`, `paired_method_differences.tsv`,
`interrater_agreement.tsv` and method-overlap table as saved sources. The raw
pool is descriptive; its coverage differs from the selected sets. Shared-claim
resampling and fixed-rater summaries do not model all biological dependencies.
A majority label is not an independently established correct answer.

The independent literature-support panel remains unavailable while grading is
incomplete. Do not fill it with hit counts or interpret blank cells as negative.
R07 explicit-number coverage belongs in a separate reporting diagnostic; its
new prose cannot inherit the existing labels. Include the absence of a general
interpretive advantage in the Results and legend.

## Draft legend structure

Describe the pool and exact method memberships, then the biological endpoint
and cohort-level analysis unit, followed by evaluator-specific categories and
agreement. Use the actual tables to insert numbers and interval status. State
which software version was evaluated. Biological correctness, causal effects
and performance of a revised audit are not established by this figure.

Keep source data and generated figures in the existing data directory. The
submission collector supplies original tables and a private figure plan; it
is not a final renderer or a replacement for the frozen analysis protocol.
