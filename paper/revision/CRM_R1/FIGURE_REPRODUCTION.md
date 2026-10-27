# Reproduce manuscript figures and summaries

Run from the repository root with Python 3.11 or later:

```bash
python -m pip install -e ".[dev]"
python paper/revision/CRM_R1/reproduce.py --outdir out/manuscript_reproduction
```

Use a new output directory. The command reads the 64 files recorded in
[`data_manifest.json`](data_manifest.json), verifies their SHA-256 checksums,
and performs the following steps:

1. Repeat the limited source checks on all 50 unchanged DEX responses.
2. Recompute the two-evaluator DEX counts and agreement from 48 returned rows.
3. Recompute three-evaluator HNSC category counts, Fleiss' kappa and bootstrap
   intervals from 150 statement-specific ratings.
4. Aggregate the 1,760 split/method rows into 22-cohort summaries and reproduce
   the six reported method/contrast bootstrap intervals.
5. Generate seven TCGA source reports and compare selected pathway identities
   with the supplied q ≤ 0.05 baseline (159 of 350 records).
6. Render Figures 2–3 and S1–S4; export numerical tables and a hash manifest.

`reproduction_manifest.json` records the environment, code and output hashes and
completion status. A failed run may leave partial outputs, explicitly marked
`failed`. No source records, returned categories or locks are modified. This
command starts from saved records; it does not refit the upstream expression
models. `--figures-only` verifies the supplied files and renders the figures but
omits steps 1–5.

## Figure and table mapping

The full input/script/output map is [`paper/FIGURE_MAP.csv`](../../FIGURE_MAP.csv).

| Manuscript item | Source | Output or authoring route |
| --- | --- | --- |
| Figure 1 | Software workflow | Author-supplied artwork |
| Figure 2A–D | Supplementary Data 1 | `figure2/`, rendered by `plot_figure2.py` |
| Figure 3A–B | Supplementary Data 2 | `figure3/`, rendered by `plot_figure3.py` |
| Figures S1–S4 | Software scope and Supplementary Data 1–4 | `supplement/`, rendered by `plot_supplement.py` |
| Table 1; Tables S2–S3 | Adapter, source and proposal contracts | Authored tables in manuscript/supplement; implementation references in figure map |
| Table S1A–B | TCGA sample counts and source census | `supplement/tables/Table_S1*.tsv` |
| Table S4A–C | Split, temporal, cross-study and ontology comparisons | `supplement/tables/Table_S4*.tsv` |
| Table S5A–B | DEX and HNSC returned ratings | `supplement/tables/Table_S5*.tsv`; recomputed summaries in `recomputed_tables/` |
| Table S6 | Original SRC003/SRC019 responses and source-check outcomes | `supplement/tables/Table_S6_worked_examples.tsv` |
| Supplementary Data 1–4 | Supplied source records | `data/`; original export bytes and provenance retained |

Each renderer writes PNG, PDF, SVG and a receipt. Figure 2 also exports selected
HNSC source values and gene overlaps. The supplementary renderer exports numerical
TSVs; it does not rewrite Word layout or the explanatory prose in the tables.

`viz_ranked.py` supplied the original Figure 2e–f ranked views and remains the
implementation of `plot-ranked`. Current manuscript Figures 2–3 use the dedicated
renderers above. `viz_source.py` supplies the general `plot-source` command.
These are three distinct, supported visualization routes.

## Recorded versions and sampling order

| Component | Recorded analysis version |
| --- | --- |
| TCGA ranking/enrichment | `764b18ec0140d9c878bf7866f129e6513ac751d4` |
| DEX source checker | `674edd05dca863bd424106df4baa07efd2ed911a` |
| 22-cohort membership package | `e4698b836986b10550c626763480678b6766897a`; exact runner and package hashes in [archive](archive/split_replication/PROVENANCE.json) |
| Three-evaluator HNSC analysis | `baf343390dab82dc1ef6759b0d0244ad046004db` |
| Public source reports and main-figure renderer base | `d0af5d74f4276f2a05cdcb926d076e1a8cd8593f` |

The DEX runner verifies the recorded `grounding.py` content hash before using it.
HNSC bootstraps retain statement first-occurrence order, fixed raters, seed
20261005 and 10,000 multinomial resamples from the original policy. Sorting claim
identifiers would change the finite-resample intervals. Split bootstraps use
sorted cohort identifiers and the seeds recorded in each supplied summary table.
A new plotting commit does not replace the versions that produced upstream fits.

## Refit upstream analyses separately

| Analysis | Entry point | Additional requirements |
| --- | --- | --- |
| Seven-cohort TCGA | [TCGA rebuild guide](../../scripts/tcga_rebuild/README.md) | Included input snapshot; expression download with recorded checksum; recorded R environment |
| Temporal replication | `scripts/18_validation_72h.R`, `scripts/19_evaluate_replication.py` | Original GSE146225 input, selections and locked validation design |
| DEX cross-study replication | [experiment guide](experiments/external_dex/README.md) | Original GSE52778/GSE34313 inputs and design lock |
| Ontology comparisons | `scripts/52_evaluate_ontology_hierarchy.py`, `scripts/53_evaluate_utility_sensitivity.py` | Original candidates and versioned ontology mappings |
| 22-cohort split fits | [Archived discovery/validation runners](archive/split_replication/README.md) | R and Python runners and the historical package match recorded hashes; original inputs and environment required for refitting |

The TCGA snapshot is included as
[`paper/source_data/CRM_R1_TCGA_inputs.zip`](../../source_data/CRM_R1_TCGA_inputs.zip).
It contains 11 public input files and their original manifest. Do not replace the
recorded GDC snapshot with a current query. The 331 MB expression matrix remains a
checksum-verified download. The per-cohort provenance files record the manuscript fits (R 4.6.1,
limma 3.68.4, fgsea 1.38.0 and msigdbr 26.1.0). The input bundle's separate
[verification log](../../scripts/tcga_rebuild/BUNDLE_VERIFICATION_R_ENVIRONMENT.txt)
uses a different R environment; it is not the manuscript fit environment.
Dependency availability alone is not an exact environment reconstruction.

The [split-replication archive](archive/split_replication/README.md) contains the
verified discovery and validation R scripts, exact membership Python runner and
24 historical package Python files. Its verification command checks these
against recorded source hashes. The archive guide states the matched-K selection
rules and the additional input and environment requirements for upstream fits.
Neither provenance collector is required for manuscript reproduction; both
remain available for inspecting original local records.

## Historical records and citation

The [original figure map](archive/FIGURE_MAP_original.csv), earlier scripts and
adverse/null results remain available. Historical figure numbers are not current
manuscript panel numbers. The root citation describes the software; cite the
exact release or commit used. DOI 10.5281/zenodo.18625777 identifies v0.1.0. 

LLM-PathwayCurator v0.2.0; DOI: 10.5281/zenodo.23702739; RRID:SCR_027964.
