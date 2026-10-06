# V15.1 development and validation boundary

## What is finished

- Legacy hash origin reconstructed on both supplied 50-row tables.
- Repository mapping to historical Fig. 2e–f and EDFig4e–f identified.
- Separate strict artifact verification, all-candidate E*S diagnostics, and structured
  reporting checks implemented with regression tests.
- Existing V13/V14.1.3 outputs retained. No new superiority endpoint or human round added.

## What remains open

1. **Rationale assessment:** V15 lexical/outcome joins are complete, but whether a
   context rejection accurately reflects the supplied evidence is not settled by a
   keyword count. Review the already selected case set using the original prompt,
   full response, sample card, evidence genes and claim. Use UNKNOWN if material is
   absent. No replacement raters or mandatory Round 2 is implied.
2. **Final submission impact:** inspect the submitted panel assembly and Source Data
   workbook. The repository declares the affected panels; exact submitted byte lineage
   remains to be checked. Metadata from other legacy variants/tau values is also needed
   before describing those runs as actual LLM evaluations.
3. **New evaluation:** unit tests prove that the new checks enforce their stated
   contract; they do not prove better biology, greater safety, or better replication.

## Next experiment, if the manuscript claims reporting fidelity

Freeze a corpus before scoring: canonical evidence, original structured claims,
single-field perturbations (cohort, comparison/direction, q-value, gene linkage,
certainty assertion), unchanged controls and equivalent formatting controls.
Derive reference labels from the canonical contract rather than the tested model.
Separate development templates from held-out templates and source contexts. Report
each violation category, false rejection of unchanged controls, unassessable cases,
and coverage. Do not call synthetic contract tests biological validation.

Use the same corpus for: no-check baseline, deterministic structured checker, and
any separately specified LLM annotation/checker. Prompt/model digest, response schema,
seed/temperature, retry policy and technical-failure policy must be frozen before
scoring. Preserve all failures. Context-name plausibility alone is not a ground-truth
violation. A passing technical receipt is not an outcome label.

The current script 98 is the deterministic comparator for this development design.
The included synthetic example and regression tests are development fixtures; they
are not a held-out benchmark and should not be reported as model accuracy.

## If the manuscript still claims better biological selection

Additional independent evidence is required. Before new outcomes are inspected,
specify dataset/accession, independence, supported positive/negative labels for the
target context, candidate universe, eligibility, matched-K and zero-K handling,
endpoint, inference unit, and precision rationale. Missing literature is UNKNOWN.
The 22 inspected V14.1.3 cohorts are now development data for any revised gate.
Reusing their outcomes cannot provide untouched confirmation of a redesigned method.

No new dataset has been selected in this package. No claim of biological superiority
is authorized by these engineering corrections. P1 empirical stability and P5
ontology findings retain their original, separate scope.
