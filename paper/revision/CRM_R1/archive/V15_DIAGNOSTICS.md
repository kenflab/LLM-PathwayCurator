# V15 read-only diagnostics and development boundary

This addition examines the already frozen V14.1.3 analysis. It does not rerun
Ollama, change memberships, recalculate enrichment, or replace the original
negative comparison. All diagnostics are post hoc.

## Commands and dependencies

Python >=3.11, pandas, numpy; pytest for tests. The scripts are standalone and
do not import the historical package or require the uncommitted V14 helper files.

- `92_diagnose_selection_v15.py`: join existing audit logs to the frozen joined
  outcome table; validate keys, replicated fields, four method copies, matched K,
  directions, PASS membership, finite confidence, nonblank reasons, and available
  raw cache files. Check the joined table against the supplied source-manifest hash.
- `93_trace_utility_v15.py`: inspect `ranked.py` with Python AST, compare stored
  utility with the three-component product, search for candidate historical inputs,
  and list metadata/code references. Matching values do not prove historical lineage.
- `94_build_explicit_utility_v15.py`: optional development-only utility builder
  requiring explicit fields and declared provenance. No proxy/hash columns or
  missing-to-one fallback. It is not automatically used by the pipeline or figures.

Each command requires a new output directory. Outputs cannot be placed inside a
frozen lock/final bundle. Input hashes are rechecked at the end. A run manifest
records code/input hashes and output hashes. It does not replace upstream freeze
checkers or certify biological truth. Private derived tables remain external.

## Selection diagnosis

The existing source defines replication. The diagnostic verifies that definition
but uses the recorded labels; no new endpoint or selection threshold is introduced.
For full audit A and comparator B, each split has identical positive K:

`difference = sum_i[(selected_A_i - selected_B_i) * replicated_i] / K`.

Decompose this identity by recorded gate and selection group. Fill absent groups
with zero within every cohort/split, average over splits within each cohort, then
average cohorts equally. This exactly reproduces the primary paired estimate.
Contributions are an accounting identity, not causal effects of removing a gate.
Do not attach an independent-pathway or independent-split P value.

`selection_gate_appearance_summary.tsv` reports pooled appearances and is explicitly
descriptive. `gate_contribution_summary.tsv` uses equal-cohort averaging.
`pathway_selection_by_cohort.tsv` shows concentration by term/cohort, and
`context_repeat_by_cohort_pathway.tsv` shows consistency over repeated splits.

The lexical rules in `diagnostic_policy_v15.json` flag text patterns, can overlap,
and do not classify biological errors. The policy was written after the negative
result was known. Stored reasons of >=160 characters are flagged as potentially
truncated. Reasons that are shorter can also lack context; inspect original cached
responses and prompts before making substantive assertions.

Examples are selected by SHA-256 order from q-value-only context FAILs, two per
cohort and replication outcome where available. This is an explicitly outcome-
stratified diagnostic sample, not a prevalence sample or independent annotation.
Do not choose only striking examples for the response letter. Retain the selection
rule and both outcome strata. Reviewer-facing claims require case-specific checks
against the original evidence packet; replication alone does not establish context truth.

## Utility provenance and sensitivity

The historical code auto-selects a context column. The trace reports its actual
order from the supplied source and numerical matches against each available audit.
It preserves ambiguity, even if a single candidate matches. Resolve historical
lineage using the original ranking command, code version, audit/evidence inputs,
hashes, and figure input references. Do not infer the chosen field from its name.

The sensitivity analysis uses stored components E, S, C: E*S*C, E*S, E*C, S*C.
It reports top-10/top-25 overlap with deterministic claim-ID tie breaking. This
measures rank dependence only, not usefulness or biological accuracy. It does not
justify interpreting a hash proxy as contextual relevance.

The V14.1.3 discovery membership rule used PASS status, not utility rank. The
historical utility issue cannot itself invalidate that held-out comparison.

## Scope of the development-only builder

The template deliberately cannot run until an investigator supplies input
provenance and changes its status to DEVELOPMENT_ONLY. The declaration is not
cryptographically authenticated and does not prove calibration. An LLM-derived
score, including confidence, remains a model output, not independent truth.
No historical package source, frozen ranking table, or figure is overwritten.

See `V15_CORRECTION_AND_VALIDATION_PLAN.md` for the separate redesign plan.
