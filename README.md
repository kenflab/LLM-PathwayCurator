# LLM-PathwayCurator

**Source-linked enrichment reports and checks of pathway draft wording.**

[Documentation](https://llm-pathwaycurator.readthedocs.io/) ·
[Historical preprint](https://doi.org/10.64898/2026.02.18.706381) ·
[MIT license](LICENSE)

Enrichment output and an interpretation are different records. LLM-PathwayCurator
keeps the complete enrichment table, writes statistical statements tied to that
table, and places supplied draft text beside checks of its numbers, signed
enrichment direction, declared study context, and inferential wording. The HTML
report lets a researcher inspect the original evidence and wording together.

The current default makes no model requests. It does not use hash-derived
context scores or simulated gene survival to select statements. You can inspect
drafts written by a person or any language model without calling that model again.
The historical LLM pipeline remains available through an explicit legacy mode.

## Install the current source

The workflow below is available from this repository. Earlier PyPI releases and
the archived preprint release describe the historical pipeline.

```bash
git clone https://github.com/kenflab/LLM-PathwayCurator.git
cd LLM-PathwayCurator
python -m pip install -e .
```

Python 3.11 or later is required. Existing users can update their checkout and
reinstall it; a second checkout is unnecessary.

## Try a source report

```bash
llm-pathway-curator run \
  --evidence-table examples/source_report/evidence_table.tsv \
  --sample-card examples/source_report/sample_card.json \
  --claims examples/source_report/drafts.tsv \
  --outdir out/source_report
```

Open `out/source_report/report.html`. The example is synthetic software test data,
including deliberately problematic wording; it is not a biological benchmark.
Omit `--claims` to create a report from the enrichment table alone.

Supply a new or empty output directory for each run. Previous reports and input
files are preserved.

## Inputs

The EvidenceTable TSV requires `term_id`, `term_name`, `source`, `stat`, `qval`,
`direction` (`up`, `down`, or `na`), and `evidence_genes` (semicolon-separated).
Use `stat_kind=NES` for a signed normalized enrichment score; `fgsea` sources
infer that label. Missing statistics remain missing. The tool checks unique
`source:term_id` identities and rejects ambiguous evidence links.

The Sample Card JSON must give a nonblank `comparison`, including the positive
group and reference group. Optional `condition`, `tissue`, `perturbation`, and
`study_design` describe your system. No cancer type or TP53 comparison is required.

Optional draft TSVs contain `text` and either `term_uid` (`source:term_id`) or an
unambiguous `term_id`. Optional `claim_id`, study context fields, and
`evidence_sha256` allow explicit identity checks. Submitted wording is preserved.
See the [input guide](docs/user-guide.md) and [adapters](src/llm_pathway_curator/adapters/README.md).

## Outputs and decisions

| Artifact | Purpose |
| --- | --- |
| `report.html` | Searchable source evidence, statistical statements, original drafts and quoted findings |
| `report.md`, `report.jsonl` | Readable and machine-readable records for every input pathway |
| `evidence.source.tsv` | Complete normalized census with source and evidence hashes |
| `audit_log.tsv` | Statistical statement eligibility and reasons |
| `claims.checked.tsv` | Original draft text, limited checks and human-review disposition |
| `run_meta.json` | Input/output hashes, settings, implementation hashes and run counts |

The default adjusted-value cutoff is 0.05; set `--q-threshold` explicitly to change
it. `--k-claims` optionally caps eligible source statements, ordered by adjusted
value and identifier. All other candidates stay in the report. This selection
rule does not claim an advantage over selection by the same adjusted values.

`PASS` applies to a **source statistical statement** under the declared cutoff.
An explicit contradiction in supplied prose gives `FAIL`; other prose remains
`ABSTAIN` for human review. Mechanistic or clinical wording is flagged for review.
Absence of a flag never establishes semantic or biological correctness.
The limited wording rules currently cover specified English expressions, not a
general language understanding system. See [scope and limitations](docs/concepts.md).

## Python API

```python
from llm_pathway_curator import ReviewConfig, review_enrichment

result = review_enrichment(ReviewConfig(
    evidence_table="evidence.tsv",
    sample_card="sample_card.json",
    claims_file="drafts.tsv",  # optional
    outdir="out/my_source_report",
))
print(result.artifacts["report_html"])
```

`run_pipeline(RunConfig(...))` uses the same source workflow by default.

## Historical pipeline and paper reproduction

Use `--workflow legacy`, or `RunConfig(workflow="legacy", ...)`, to request the
historical distill/module/proposal pipeline. Its proxy context scores and
synthetic gene perturbations are historical computational mechanisms, not
empirical stability or biological validation. Model-related environment settings
only affect that explicit workflow; legacy model settings can make API requests.

For an exact historical reproduction, pin commit
[`b069e8a`](https://github.com/kenflab/LLM-PathwayCurator/tree/b069e8ad3d916618adcf26af0202748790c3ca20).
Changing file paths changes frozen code hashes, so archived protocols should not
be silently relocked against the reorganized source tree.

The [paper workspace](paper/README.md) and
[revision README](paper/revision/CRM_R1/README.md) describe research-only
comparisons, saved-result reporting and the updated-tool reassessment.
Development prototypes, study-specific registries and revision tests live under
`paper/revision/CRM_R1/`; they are excluded from the installed package.
Earlier versioned notes are retained in its `archive/` for provenance.

## Development and citation

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m build
```

Software tests establish implementation behavior. They do not estimate expert
agreement, interpretive utility, or biological accuracy. The revised implementation
requires a separately specified evaluation before those performance claims.

[CITATION.cff](CITATION.cff) describes the existing preprint and release. This source
update does not change that historical publication or its archived results.
