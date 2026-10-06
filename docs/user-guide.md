# User guide

## EvidenceTable

Provide a UTF-8 TSV with the following columns.

| Column | Meaning |
| --- | --- |
| `term_id` | Stable pathway identifier |
| `term_name` | Display name |
| `source` | Enrichment source; participates in `source:term_id` identity |
| `stat` | Source statistic, or blank/`NA` if unavailable |
| `qval` | Adjusted value in [0,1], or blank/`NA` if unavailable |
| `direction` | `up`, `down`, or `na` |
| `evidence_genes` | Semicolon-separated supporting genes |
| `stat_kind` (optional) | Use `NES` for normalized enrichment scores; fgsea sources infer it |

A supplied `term_uid` must equal `source:term_id`. Repeated identities and
inconsistent signed NES values are rejected. Use separate runs for different
contrasts. Raw inputs are hashed; normalized records have evidence identities.
Adapters remain available through `llm-pathway-curator adapt --help`.

## Sample Card

The JSON object requires a nonblank `comparison`. Explicitly describe the
positive group and reference group; do not infer their order from a file name.

```json
{
  "comparison": "treated (positive group) versus vehicle (reference group)",
  "condition": "your study condition",
  "tissue": "your tissue or cell type",
  "perturbation": "your perturbation",
  "study_design": "experimental or observational"
}
```

These fields are user-supplied metadata, not inferred facts. No TCGA code or TP53
comparison is required. Historical Sample Card tuning keys do not control the
new source workflow.

## Optional drafts

Provide a TSV with `text` and `term_uid`, or an unambiguous `term_id`.
Include `source` to resolve duplicate term IDs across sources. Optional
`claim_id` identifies multiple drafts for one pathway. Quoted multiline TSV
cells are supported and their wording is preserved.

Optional `comparison`, `condition`, `tissue`, `perturbation`, and
`evidence_sha256` check explicitly declared attributes. Missing attributes do
not imply that free prose was checked for them. Unknown evidence links stop
the run before output creation.

## Run and inspect

```bash
llm-pathway-curator run \
  --evidence-table evidence.tsv --sample-card sample_card.json \
  --claims drafts.tsv --q-threshold 0.05 --outdir out/run_001
```

`--k-claims` optionally limits eligible source statements. Every pathway and
every supplied draft stays in the exports. Source decisions and prose
dispositions are separate fields. Open `report.html` and inspect the original
wording and quoted findings. `run_meta.json` records inputs, settings, implementation
hashes and output hashes. No model requests or hash-based context scores are used.

Existing nonempty output directories are rejected. `--force`, `--tau`,
`--seed`, model environment settings and custom metadata paths are legacy options;
they do not enable new semantic checks in the source workflow.

## Historical reproduction

`--workflow legacy` selects the original processing path. Model environment
settings can make API requests there. Source mode never silently switches to a
model or proxy. Exact frozen paper reproductions should pin
[`b069e8a`](https://github.com/kenflab/LLM-PathwayCurator/tree/b069e8ad3d916618adcf26af0202748790c3ca20)
and use their original inputs and recorded environments.

The legacy demo can be inspected without a paid model by using its original
offline settings. Legacy results and new source reports are different workflows;
their performance labels should not be pooled.

