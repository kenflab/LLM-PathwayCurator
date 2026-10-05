# R04: returned-response diagnosis and source-only reading probe

The returned R03 pilot does not qualify the semantic auditor for a frozen
comparison. The four deterministic cases pass; only two of the 16 semantic
cases meet the original development rules. R04 diagnoses every unchanged
first response and tests a narrower reading task before building another
evidence-comparison prompt. It does not produce new audit verdicts.

## Authenticated R03 findings

Returned archive: `r03_20261004T231625491225Z.zip`, SHA256
`b80d8e01f07e173c916c63126cd38d6a72ba84f731501549895387d549572a3d`.
All 419 exported file hashes and 15 recorded code/fixture hashes match.
All 96 request envelopes and backend receipts are checked, and the unchanged
R03 parser, case aggregation, scores and summary are reproduced.

| Observation | Result | Meaning |
| --- | --- | --- |
| Completed generations | 96/96 report `done_reason=stop` | No recorded token truncation |
| Completion tokens | 49–101, limit 512 | Increasing the limit does not address these failures |
| Valid atomic responses | 90/96 | Reference/schema validity alone is insufficient |
| Invalid responses | 4 `NOT_MENTIONED` with sentence IDs; 2 `DENIED` without IDs | Keep all six as the original technical failures |
| Complete semantic cases | 10/16 | Six cases remain incomplete |
| Original semantic case matches | 2/16 | Four additional matches in 6/20 are deterministic controls |
| Faithful wording eligible | 2/8 | Five complete cases withheld; one incomplete |
| Extra concern cases | 14/20 | Concerns transfer between aspects |

The model invents an explicit disclaimer in P01's evidence-scope review,
although P01 has no such disclaimer. In N03 it treats an observational
description as a denial of causality despite the explicit causal assertion.
In N06 it attributes the evidence-side enrichment limitation to the claim.
P02's valid failure to meet FDR is treated as a problem in unrelated aspects.
N04's metadata response clears an unsupported oncocytic subtype; A05's numeric
response clears a direction statement opposite to the recorded negative NES.
These last two are failures of individual aspect judgments; both whole cases
are incomplete, so neither is a falsely eligible interpretation in R03.

A01 and A03 pass the original aggregate rules, but even their responses assign
causal/FDR limitations to unrelated aspects. Their original scores stay intact;
the examples show why those scores do not certify grounded reading.
These are developer-reviewed known synthetic controls, not independent expert
annotations or estimates of real-world error prevalence.

## One bounded probe

The R04 locator sees **only the unchanged claim text and its full sentence
spans**, plus the task definitions and response schema. Canonical evidence,
statistical values from evidence, assertion flags, past responses, case IDs and
expected labels do not reach the model. The response selects supplied sentence
IDs under `asserted`, `limited`, or `uncertain` for each of the six aspects.
There are no model reasons, support verdicts or confidence values.

`limited` means that the text says an inference is not established. It does not
mean a negative statistical fact or a claim that an effect does not exist.
All references resolve to full unchanged sentences, preserving negation and
nearby qualifications. Lists can be empty or select the same sentence in several
aspects. A sentence can contain both a claim and a limitation of the same aspect.
Syntactically valid locations do not establish semantic coverage; the fixed
developer annotations assess all 18 lists in every semantic case.

The same 20 controls remain in the output. Four cases preserve their existing
deterministic violations without a model call. The other 16 each receive one
source-only request. All-six-aspect sentence annotations were reviewed **after
seeing R03**, and are fixed in `config/r04_development_plan.json` before the R04
live run. This is disclosed development, not a held-out evaluation. Expected
labels are used only by the scoring code. No case is dropped because of failure.

Use the same `llama3.1:8b` weight digest as R03, verify the current local Ollama
version, and record before/after identity for each request. Keep temperature 0,
seed 42, context 16384, output limit 512, timeout 120 seconds. Start at most 16
new requests, with a 600-second default budget for starting requests. A request
already in flight may finish after the budget. No model download or substitution
occurs. No failed, interrupted or invalid first outcome is retried or repaired.

The new `claim_locator_r04` namespace and cache leave the R03 implementation
digest, request keys and original cache unchanged. A new run directory is
created for every invocation; repeated exact requests reuse their first outcome.

## Decisions after the probe

All 16 cases must parse and match all annotated sentence sets before moving to
the separately designed evidence-comparison stage. An empty output, transferred
negation, missed sentence or extra aspect fails the reading gate. The stage must
then check every relevant assertion, including mismatched prose direction and
unsupported histology, against canonical facts. Reading success alone does not
resolve N04/A05 or demonstrate that any evidence comparison works.

If substantive errors remain, stop after this bounded pilot. Review all outcomes
and reassess backend capacity or the role assigned to LLM auditing; do not repeat
prompt changes until these exposed controls pass. Do not start a held-out corpus
or ask experts to rate unfinished methods.

The scientific revision still needs a separately frozen natural-text comparison
of unaudited, simple-audit and full-audit treatment of the **same generated text**,
coverage and error reporting with all candidates retained, and independently
qualified biological evidence. Standard deterministic statements are a separate
generation control. Existing negative P1/P2B results remain reportable. P4 ratings
remain exploratory, and the 620 empty P3 screening rows do not become negative
evidence or a new mandatory grading workload.

Any small targeted expert follow-up should follow method and corpus freeze. No
expert ratings, external expression outcomes or biological performance are
generated by R04. GSE34313 remains a one-cell-line culture case; replicate suffixes
do not prove donor pairing, and donor disjointness from discovery is unverified.
