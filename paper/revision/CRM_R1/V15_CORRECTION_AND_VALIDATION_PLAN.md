# V15 correction and additional-validation plan

Status: DEVELOPMENT DESIGN; not a preregistered or completed biological experiment.
The existing V13 and V14.1.3 results remain reportable in their original form.

## 1. Resolve what the current evidence can establish

Finish the read-only gate/outcome join and retain every cohort and pathway.
For the deterministic example set, inspect original claim direction, statistics,
supporting genes, context card, full response, and prompt. Separate four questions:

1. Does the rationale accurately describe the supplied context and evidence?
2. Does it demand a tissue/pathway association that the stated contract never required?
3. Does it confuse absence of provided evidence with a documented contradiction?
4. Is the stored explanation too truncated to assess?

These questions are a case audit, not a new three-rater endpoint. No replacement
raters or compulsory Round 2 are requested. Use UNKNOWN for insufficient evidence.
Retain the outcome-stratified sampling rule and do not relabel locked outcomes.

For legacy utility, recover inputs and original rendering/ranking commands from
the two dataset directories identified in `missing_provenance.tsv`. If unavailable,
state that exact field lineage is unresolved; do not reconstruct an invented history.

## 2. Engineering corrections that can be tested without biological labels

The independent development utility builder implements an explicit-column contract,
no hash proxy, no missing-to-one fallback, finite ranges, and deterministic ties.
Keep this opt-in until field meaning and scale are documented. Do not silently
replace the package's historical column precedence or switch to raw LLM confidence.

A future audit-runner patch should fail before membership creation for connection
errors, incomplete cache responses, blank reasons, nonfinite confidence, or wrong
model identity. Retrying such jobs is a technical recovery, not selection on K.
Retain valid completed jobs and technical-failure records. A biologically/semantically
valid K=0 is different from technical failure and must not be retried for a favorable K.
The current frozen positive-K design is unchanged; any new design must specify
coverage and conditional precision separately, with K=0 precision undefined.

## 3. Candidate semantic redesign, to be evaluated separately

Do not use unsupported pathway-name plausibility as an automatic rejection rule.
A candidate reporting workflow can retain statistical ordering and expose context
assessment as a documented annotation, reserving blocking decisions for explicitly
testable contract violations. This changes the method's role. Matching q-value
selection by construction would not demonstrate added selection performance.

Before new data are evaluated, fix a typed contract covering cohort/context identity,
comparison orientation, gene-set/evidence linkage, statistical wording, uncertainty,
and conditions for abstention. Separate grounded contradictions from unknown relevance.
Document the full prompt, model digest, temperature/seed, raw outputs, and versions.

## 4. Two distinct validation stages

### A. Contract tests with controlled inputs

Construct a versioned test suite before running the revised model. Start from
valid evidence-linked claims, then change one field at a time: cohort identity,
comparison orientation, q-value transcription, gene linkage, or certainty wording.
Also include scientifically equivalent formatting/context-name transformations.
Derive labels from the typed input contract, not the tested LLM. Separate templates
used for development from held-out templates and report both error detection and
false rejection of unchanged/equivalent claims, abstention, and coverage.

This stage tests faithful reporting and contract checking. It cannot establish that
a pathway interpretation is biologically correct. No aggregate 'safety' claim follows.

### B. Independent biological evaluation, only for a biological claim

An additional dataset has not yet been selected or approved. Before observing new
outcomes, document accession/version, independence from all development cohorts,
reference-label provenance, candidate universe, exclusions, sample/group eligibility,
method baselines, matched-K and zero-K policy, endpoint, uncertainty unit, and a
sample-size/precision rationale. Use experimentally supported labels for the stated
context; lack of a literature record is UNKNOWN, not a negative label. Literature
retrieval alone is not independent reference grading.

The current 22 cohorts and their held-out results have now been inspected and are
development/diagnostic data for any redesign. They cannot be reused as an untouched
confirmatory evaluation of the revised gate. Do not add endpoints, seeds, thresholds,
or subsets until an advantageous result appears. Report all planned comparisons.

## 5. Decision and reporting rules

- Keep the original endpoint and its unfavorable estimates/intervals visible.
- Release the diagnostic code and synthetic tests separately from private inputs.
- If contract-stage benefits are demonstrated, restrict the corresponding claim
  to reporting fidelity/contract checking and report the false-rejection cost.
- If biological-stage benefits are absent, do not claim better biological selection.
- Do not require a new benchmark merely to continue inspecting existing artifacts.
  Its need depends on the revised manuscript claim; engineering fixes alone do not
  supply the missing biological evidence.
- Preserve P1 empirical stability and P5 ontology evidence within their own scopes;
  neither automatically establishes superiority of a revised full audit.

Completion of this document is completion of the design draft, not completion of
new validation or resolution of the historical utility provenance gap.
