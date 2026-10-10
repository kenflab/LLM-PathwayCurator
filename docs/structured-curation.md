# Structured curation candidate

This additive source workflow creates typed observations and descriptive
supporting-gene modules. Model- or human-authored interpretations remain separate
proposals. This implementation does not estimate biological or semantic accuracy.

## Compile source observations and modules

```bash
llm-pathway-curator run \
  --evidence-table examples/source_report/evidence_table.tsv \
  --sample-card examples/source_report/sample_card.json \
  --modules --module-min-shared-genes 3 --module-jaccard-min 0.10 \
  --outdir out/curation_packet
```

The normal report retains every source term. `claims.structured.jsonl` contains
compiler-created statistical observations and support summaries. Module views
link to source records in `report.html`; `modules.json` includes every member and
qualifying edge, union genes, genes shared by at least two members, genes common
to all members, source directions and structural review flags.

Modules use exact, case-preserving gene tokens. Normalize identifier namespaces
before combining sources. They are connected components of edges that satisfy
both fixed cutoffs. Connections can be transitive; neither a common biological
mechanism nor pairwise similarity of every member follows. Missing-support terms
remain singletons. Mixed enrichment directions are flagged for review, not called
contradictions. There is no hub deletion, adaptive threshold, proxy context score,
or synthetic stability score. More than 2,000,000 distinct candidate pairs causes
an explicit error, not substitution of another grouping algorithm.

Grouping uses all input candidates and does not change q-value eligibility or
the declared `--k-claims` cap. Content-based module IDs identify membership and
support genes, not independent biological entities. Source hashes also encode
input-file bytes and record position: regenerating or reordering an input requires
fresh proposal references even when module membership is unchanged.

## Import a saved structured proposal

`proposal_packet.json` supplies source facts, Sample Card, modules, instructions
and the strict JSON schema. Give the packet to the chosen local/external model
or authoring system and save its response as JSONL. This tool performs no model
transport and never executes the response. Generator metadata is self-reported.
Preserve model/version/prompt/decoding settings and raw response independently
when evaluating LLM behavior; this importer alone does not establish provenance.

Every proposal requires `claim_id`, `claim_type`, unchanged `text`, `comparison`
and one or more exact `evidence_refs` (`term_uid`, `evidence_sha256`). Optional
`supporting_genes` must occur in the union of the linked support sets. The claim
types are `statistical_observation`, `support_summary` and `hypothesis`. The
declared type is recorded, not semantically verified. Duplicate IDs/references,
unknown fields and malformed schema fail before report creation.

```bash
llm-pathway-curator run \
  --evidence-table examples/source_report/evidence_table.tsv \
  --sample-card examples/source_report/sample_card.json \
  --modules --proposals saved_proposals.jsonl \
  --outdir out/curation_review
```

Use the same source files as the packet. `proposals.submitted.jsonl` preserves
the exact submitted bytes. `proposals.checked.jsonl` preserves the original text
and metadata beside binding results and limited findings. Invalid/stale links,
declared-comparison differences and unsupported declared supporting genes produce
explicit violations without dropping the proposal. Gene membership checks do
not check every gene mentioned in free prose. The HTML displays proposed text
and links separately from compiler-written source observations.

Single-source non-hypothesis prose receives the existing limited English numeric,
signed-direction and inferential-wording checks. Multi-source text is not checked
against each source indiscriminately: assigning individual clauses to evidence
records requires human review. Hypothetical numbers in a hypothesis are also not
treated as assertions about present statistics; only its structured bindings are
checked. Hypothesis support, omitted facts, term-name fidelity in free prose,
causal interpretation and the truth of biological explanations need human review.

Valid bindings never approve free prose. Explicit violations receive `FAIL` in
the scope of binding/limited checks; all other prose receives `ABSTAIN` pending
human review. Source-compiler `PASS` means statistical reporting eligibility only.
Support summaries are `DESCRIPTIVE`, with no accuracy score. No numerical warning
rate or module count should be interpreted as utility, sensitivity or precision.

## Reproducible software demonstration

```bash
python examples/structured_curation/run_demo.py --outdir out/curation_demo
```

This creates a source packet and imports three clearly labeled synthetic examples
(observation, support summary, hypothesis). It makes no LLM calls. The script also
accepts `--evidence-table` and `--sample-card` to demonstrate saved datasets. These
demonstrations are software checks, not new study results or independent validation.

## Revision-study boundaries

This is a new version, not a repair to frozen historical outputs or study locks.
Record its source hashes, packet, proposal inputs and fresh output directory.
The version identifier is `source-linked-curation/1`; the underlying source-text
rules are recorded separately in `run_meta.json`. New metadata hashes include
`curation.py`; consumers assuming only two implementation files need to accept
the additional entry. Source core hashes change, so old frozen-study checks must
not be bypassed by relocking.

To establish added value, compare final reports under declared matched tasks and
information budgets, measuring discrepancies and useful information retained.
Separately ablate modules and structured proposals if attributing benefit to
either. Use independent cases/raters for claims of generalization. Preserve the
previous unfavorable results and report endpoint/version changes explicitly.

The archived-label inventory can be reproduced without model calls:

```bash
python paper/scripts/audit_saved_label_provenance.py --outdir out/saved_label_inventory
```

Use a new output directory. The diagnostic records source hashes, available
raters, unmatched claim IDs, and source-record field differences. It does not
equate a shared claim ID with the same text and context shown to a rater, pool
multiple raters by discarding duplicates, or estimate a new performance score.
Its saved WNT example distinguishes the original model rationale from the
source enrichment statistic and adjusted q-value. Archived outputs are preserved.

## Significance wording

Source checks recognize “significant up-regulation” and “significantly downregulated”,
including joined, spaced and hyphenated up/down variants, against the declared
adjusted-value cutoff. Negation, explicit unadjusted significance and explicitly
biological or clinical significance retain separate handling. These checks neither
establish gene-expression regulation from enrichment nor approve free prose.
Unsupported wording still requires human review; the supported expressions are not
a complete natural-language parser.
