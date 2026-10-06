# Response assembly guide

Generate the current response matrix privately with
`experiments/submission/assemble.py --data-root "$CRM_R1_DATA_ROOT"`.
It writes `RESPONSE_MATRIX.private.tsv` next to the saved-result snapshots,
current Word and draft text. Reviewer correspondence and point-by-point working
responses remain outside public Git.

The matrix distinguishes completed source checks, partially answered topics,
missing data and the remaining scientific evidence gap. Completion of collection
or rendering does not mean all concerns have been resolved.

Before finalizing each response:

1. Match the exact unchanged statement or selection to its original source.
2. State the comparator, denominator, analysis unit and uncertainty.
3. Separate methodological consistency from biological correctness.
4. Preserve low rater agreement, unknown labels, incomplete checks and adverse
   or null results.
5. Add figure/page references from the current assembled manuscript.

Do not transfer existing expert labels to new LLM prose or rewritten templates.
Do not describe ungraded literature retrieval as independent biological support.
Do not say the current full audit improves interpretive quality without data
that directly measure that advantage.

The source assembly produces draft text and an explicit unresolved-item list.
It neither sends correspondence nor submits the manuscript.
