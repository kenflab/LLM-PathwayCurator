# Tutorials

## Inspect existing LLM wording without calling a model again

1. Convert your enrichment result to an EvidenceTable.
2. Describe the contrast direction in the Sample Card.
3. Export the exact existing wording to a draft TSV linked by pathway identifier.
4. Run the source workflow and open the HTML report.
5. Check quoted findings against your analysis and external biological evidence.

Preserve original wording; do not rewrite it before estimating how often its
source facts were correct. A limited rule's flag is not an expert truth label.

## Report enrichment results without draft prose

Omit `--claims`. The tool writes statistical statements from supplied values and
records all candidates. Use `--q-threshold` for your declared cutoff, and
`--k-claims` only when a prespecified reporting cap is needed. The tool does not
perform differential expression, enrichment fitting, or mechanistic validation.

## Reassess a changed implementation

Use a new code snapshot, a separately recorded protocol, and new output folders.
Keep historical results and code hashes. Saved-text development comparisons and
independent semantic or biological validation have different roles.

The repository's
[revision workspace](https://github.com/kenflab/LLM-PathwayCurator/tree/main/paper/revision/CRM_R1)
contains manuscript-specific code and the saved-text reassessment runner.
