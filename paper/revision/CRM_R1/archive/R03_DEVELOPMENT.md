# R03: repair the review interface, test real semantics separately

## Reason for this milestone

The authenticated V16.1 local run returned 16 model responses: two passed the
response/reference contract and 14 remained incomplete. All eight faithful
prose controls were incomplete. Saved raw decisions also confused explicit
denials of a causal inference or literal allograft-rejection event with
assertions of those inferences. Fixing JSON paths alone cannot establish that
the model understands the sentences.

R03 keeps V16.1, its API, and its historical results intact. The opt-in
`contract_v17.atomic` implementation has a distinct method and prompt version.
No calibrated context probability or E*S*C utility is introduced.

## Fixed development changes

1. Review one of the six aspects per request, using the whole unmodified prose.
2. First distinguish an affirmed assertion, denied inference, non-mention,
   ambiguity, or mixed statement. An asserted negative biological fact still
   requires evidence; it is different from saying the evidence does not
   establish an inference. A disclaimer does not cancel another affirmative
   unsupported assertion.
3. Select sentence IDs and aspect-specific fact IDs generated from canonical
   inputs. Dynamic response enums constrain the IDs; local validation still
   rejects invented IDs, blank reasons, inconsistent polarity/verdicts, and
   malformed/truncated responses. Exact sentence spans and selected facts are
   resolved locally and saved. No model-created JSON path or rewritten quote
   is repaired into a successful result.
4. Retain all six aspect slots, all first responses, and all 20 known controls.
   Incompletion cannot count as CLEAR or disappear from the denominator.
5. Use one first attempt per exact request. Reuse valid, invalid and interrupted
   results on resume; never regenerate a completed judgment to obtain a pass.

Sentence segmentation is a reversible punctuation/whitespace rule, not a
linguistic parser. The full prose is also supplied. Fixed IDs prove reference
integrity, not that the model selected the right sentence or fact. The polarity
field is a model judgment, not an independent label. The local pilot must test
whether the substantive failures improve.

## Scoring before new model output

`config/r03_development_plan.json` pins the existing case bytes, exact order,
historical ZIP hash, generation settings, request/time budget, and required/
allowed concern sets. Expectations are outside the model-visible payload.
Critical denial and affirmative examples also have fixed expected assertion
modes. A correct overall verdict cannot hide misreading a denied inference as
an affirmative assertion; polarity mismatches are reported separately.
Eight faithful controls require no concerns. Semantic negative controls require
their specified concern, with no unrelated concerns. For A06, causality is
required and metadata is additionally allowed because the text invents an
experimental knockout in an observational dataset. This allowance is declared
before R03 output; it is not inferred from a later response.

The four deterministic cases remain deterministic and make zero model calls.
The other 16 cases require all six aspects to complete. The development gate
requires all 20 expectations, including the forbidden-concern checks, to pass.
Report faithful controls as eligible, complete but withheld, incomplete, and
unattempted. Do not infer safety from zero erroneous passes when most cases
are incomplete. Required-issue detection alone is not sufficient.

The inputs are known synthetic development controls previously reviewed during
V16.1 development. They are neither an independent test set nor an estimate of
the prevalence of errors in natural model output. Differences from the old run
also include the review interface/output length and possibly server version;
they do not isolate a causal effect of one prompt sentence.

## What follows a successful development gate

Freeze the method and scoring before collecting a separate natural-text corpus.
Specify the complete candidate census, source hashes, sample context, prompt,
model identity, ordering, output length, one-generation policy, reporting
endpoints, and treatment of failures. Preserve the unaudited prose once and
reuse exactly that prose in both audit comparators:

| Arm | Input / role |
| --- | --- |
| A | One natural LLM interpretation, kept before any audit |
| B | Same A text with a prespecified simple numerical/consistency audit |
| C | Same A text with the frozen full audit |
| D | Deterministic canonical statement as a separate generation control |

Keep full candidate denominators, report coverage and faithful text withheld,
and compare paired outcomes with uncertainty. Artificial challenges can support
development but cannot substitute for naturally occurring errors. Agreement
between model judges is not expert or external biological ground truth.

Decide whether a small, tightly specified expert task is needed only after the
new corpus and endpoint are frozen; do not reassign the full prior rating task
or automatically fill the 620 blank P3 grading rows. The existing first-round
expert ratings stay exploratory.

The external expression protocol needs its own freeze. R02's dexamethasone
case has unresolved GSE34313 probe mapping/measured-universe specifications and
single-cell-line culture replication, with unverified between-study donor
disjointness. The current V16 evidence schema is TCGA/TP53-specific and must not
be populated with dexamethasone records by relabeling its fields. A versioned
contrast adapter is needed before applying the reporting method to that case.
R03 reads no new external expression or pathway-replication outcomes.
