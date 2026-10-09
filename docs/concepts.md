# Concepts and scope

## Three different questions

| Question | What this software supplies | What still needs evaluation |
| --- | --- | --- |
| What does the supplied enrichment table record? | Statistical statements with source identity, adjusted value, signed direction where supplied, and comparison | Correct upstream analysis and study metadata |
| Does a draft contradict a stated source fact? | Limited numeric, direction, declared context and evidence identity checks; quoted wording flags | Broader semantic assessment and domain expert review |
| Does a biological interpretation generalize? | Preserved provenance to support follow-up | Independent biological evidence and an appropriate study design |

A pathway enrichment direction does not establish activation of a molecular
event, causation, gene-level expression changes, or clinical efficacy.

## Selection and dispositions

The source workflow selects estimable adjusted values at or below the declared
cutoff. An optional top-K cap orders these values and identifiers. This is a
simple statistical selection rule; no superiority to the same q-value rule is
claimed. The report retains candidates outside the selected set.

`PASS` applies to eligibility of a **source statistical statement**.
`ABSTAIN` applies to source statements outside the declared rule or missing
adjusted values. Neither is a biological truth label.

For submitted free prose, an explicit limited-check contradiction gives
`FAIL`. Other prose remains `ABSTAIN` for human review. Causal, mechanistic and
clinical expressions produce review flags; they do not automatically establish
a biological error. No draft receives automatic semantic approval.

## Limited wording rules

The current rules recognize specified English NES/ES/statistic and q/FDR/adjusted
p-value expressions, signed enrichment wording, significance expressions, and
some mechanistic or clinical expressions. Displayed numeric rounding and bounds
are checked; unqualified p values are not interpreted as adjusted values.

The fgsea adapter preserves whether the input statistic was NES or ES. Explicit
unadjusted, uncorrected, or nominal significance wording within the same clause
is sent for review because an adjusted value alone cannot verify it. A draft
context attribute absent from the Sample Card is also unverified; it is not a
proven context contradiction. These limited rules do not resolve all statistical
bases, sentence scopes, or study-context synonyms.

Coverage is recorded separately from contradictions. A missing explicit number
is not a proven wrong number. Negation handling is heuristic. The rules can miss
paraphrases or flag wording that a researcher can justify with other evidence.
Unflagged text is not certified as faithful or biologically correct.

## Historical modes

The current source workflow can also generate descriptive support-overlap
modules with `--modules` and import typed proposals with `--proposals`.
These options retain all source candidates and do not use the historical
context/stability ranking score. See [Structured curation](structured-curation.md).

The earlier module/proposal pipeline is available only by explicit selection.
Hash-derived context gates and synthetic gene perturbations in that pipeline
are not independent semantic assessment or empirical donor stability. Keep
their historical receipts and unfavorable or null comparisons.

Research-specific protocols, model experiments and revision tests are outside
the installed package. Their version names remain in archival records so past
results can be traced; they are not public runtime API names.
