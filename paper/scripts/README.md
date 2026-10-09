<!-- paper/scripts/README.md -->
# Figure reproduction scripts (canonical)

This directory contains the **canonical, script-based** pipelines used to reproduce manuscript figures and publication Source Data.
For the authoritative mapping of **inputs ↔ scripts ↔ outputs**, see [`paper/FIGURE_MAP.csv`](../FIGURE_MAP.csv).
Notebooks are exploratory and are not required for reproduction.

> **TCGA input correction (2026-10-09):** the archived TCGA rankings contain
> numeric row indices exported as gene identifiers, which were then interpreted
> as Entrez IDs. The old grouping code also treated samples without a qualifying
> mutation record as WT without confirming assessment. These inputs and their
> dependent TCGA figures require regeneration; their q-values do not establish
> a biological absence of enrichment. Archived research outputs are retained.

## Corrected TCGA inputs

- `fig2_make_groups.py` now distinguishes `TP53_mut`, `TP53_wt`, and
  `TP53_unknown`. WT requires a reviewed TSV with `sample` and `tp53_assessed`
  (`true`/`false` or `1`/`0`). This file must be based on assay/sample metadata and
  TP53 assessment, not inferred from the presence of other mutation records.
  Without it, samples lacking qualifying TP53 calls remain UNKNOWN.
- Matching uses the 15-character TCGA sample barcode, including sample type;
  different aliquots can match, but normal and tumor sample types are not merged.
- `fig2_deg_rank.R` retains gene identity in an explicit limma annotation column.
  Numeric Xena IDs are resolved through the versioned `--gene-map` table
  (`gene_id`, `gene_symbol`; default: `resources/gene_id_maps/id_map.tsv.gz`).
  Ambiguous mappings stop the run. Unmapped numeric rows are excluded and listed
  individually in the mapping audit. Symbol rows are retained unchanged; this
  does not automatically resolve historical symbol aliases.
- Duplicate resolved symbols are averaged on the supplied log-expression scale
  per sample, before fitting. Rankings contain unique symbols and an explicit
  `gene_id_type=symbol`; scores are never associated through `topTable` row names.
- Both Hallmark and collection-specific fgsea scripts require the new ranking
  contract and use `msigdbr` **gene symbols**. Numeric row-index rankings and
  rankings without a declared namespace are rejected. Exact ties use gene-name
  order without changing t statistics. Membership snapshots, input checksums,
  mapping decisions, sample counts and package versions accompany new outputs.
- UNKNOWN samples are excluded from differential expression. The fit requires
  at least two assessed samples per arm and ten overall; passing this technical
  minimum does not establish adequate power or a suitable biological comparison.
- All corrected TCGA input scripts refuse to replace existing outputs. Use fresh
  directories. They do not relock studies, run language models, or update figures.

Example from the repository root, after preparing a genuine assessment table:

```bash
export TCGA_RERUN="/absolute/path/to/a/new/tcga_input_run"
export TP53_ASSESSMENT="/absolute/path/to/reviewed_tp53_assessment.tsv"

python paper/scripts/fig2_make_groups.py \
  --assessed-samples "$TP53_ASSESSMENT" --outdir "$TCGA_RERUN/groups"

Rscript paper/scripts/fig2_deg_rank.R HNSC \
  --groups "$TCGA_RERUN/groups/HNSC.groups.tsv" \
  --outdir "$TCGA_RERUN/rankings"

Rscript paper/scripts/fig2_fgsea_to_evidence_table.R HNSC \
  --rank "$TCGA_RERUN/rankings/HNSC.deg_ranking.tsv" \
  --outdir "$TCGA_RERUN/evidence_tables"

Rscript paper/scripts/figS2_fgsea_to_evidence_table.R HNSC \
  --rank "$TCGA_RERUN/rankings/HNSC.deg_ranking.tsv" \
  --collection C2 --subcategory CP:REACTOME \
  --outdir "$TCGA_RERUN/evidence_tables"
```

Use `--mc3`/`--phenotype` for external raw files in the group builder and
`--expression`/`--gene-map` for external inputs in the ranking script. The
example retains the existing unadjusted MUT-minus-WT contrast; it does not add
clinical covariates or change the scientific design. Reassess cohort eligibility
(especially a small OV WT arm) before producing manuscript figures.

The Python fgsea adapter converts already computed enrichment results; it cannot
repair upstream sample or gene identities from a leading-edge list. It rejects
ambiguous column aliases and invalid adjusted p-values and preserves an explicit
`gene_id_type` declaration without certifying its accuracy. Numeric Entrez IDs
remain valid adapter inputs; a short consecutive leading-edge list alone is not
evidence of the TCGA row-index failure.

Regression checks: `pytest tests/test_tcga_inputs.py tests/test_fgsea_input_integrity.py`
and `Rscript tests/test_tcga_inputs.R` (requires limma). The R test reproduces the
duplicate-ID failure with real limma and checks the corrected gene/score mapping.

> Sanity check (recommended): run the deterministic, LLM-free demo first: [`examples/demo/`](../../examples/demo/).

## Conventions
- Run from the repository root (e.g., `/work` in the paper container).
- Do not edit paths inside scripts. Figure inputs/outputs are organized under `paper/source_data/<BENCHMARK_ID>/`.
- Run outputs are written to figure-scoped directories (e.g., [`out_fig2/`](../source_data/PANCAN_TP53_v1/out_fig2), [`out_figS2/`](../source_data/PANCAN_TP53_v1/out_figS2), [`out_figS3/`](../source_data/PANCAN_TP53_v1/out_figS3), [`out_figS4/`](../source_data/BEATAML_TP53_v1/fig/FigS4)) and include `run_meta*.json` for provenance.

---

## Fig. 2 (PANCAN_TP53_v1; Pan-cancer TP53 mut vs wt)

Outputs live under [`paper/source_data/PANCAN_TP53_v1/`](../source_data/PANCAN_TP53_v1):
- Inputs: `raw/`, [`derived/`](../source_data/PANCAN_TP53_v1/derived), [`sample_cards/`](../source_data/PANCAN_TP53_v1/sample_cards), [`evidence_tables/`](../source_data/PANCAN_TP53_v1/evidence_tables)
- Run outputs: [`out_fig2/`](../source_data/PANCAN_TP53_v1/out_fig2)
- Figures: [`fig/`](../source_data/PANCAN_TP53_v1/fig)
- Publication Source Data workbook: [`paper/journal_source_data/SourceData_Fig2.xlsx`](../journal_source_data/SourceData_Fig2.xlsx)

### Pipeline (high level)
1) Fetch inputs: [`fig2_fetch_inputs.py`](../scripts/fig2_fetch_inputs.py)
2) Define TP53 mut/wt groups: [`fig2_make_groups.py`](../scripts/fig2_make_groups.py)
3) Generate Sample Cards: [`fig2_make_sample_cards.py`](../scripts/fig2_make_sample_cards.py)
4) Compute DE rankings (per cancer): [`fig2_deg_rank.R`](../scripts/fig2_deg_rank.R)
5) Build EvidenceTables (Hallmark; per cancer): [`fig2_fgsea_to_evidence_table.R`](../scripts/fig2_fgsea_to_evidence_table.R)
6) Run LLM-PathwayCurator + mechanical audits: [`fig2_run_pipeline.py`](../scripts/fig2_run_pipeline.py)
7) Aggregate + plot panels:  
   [`fig2_collect_risk_coverage.py`](../scripts/fig2_collect_risk_coverage.py), [`fig2_plot_multipanel.py`](../scripts/fig2_plot_multipanel.py), [`fig2_plot_lines_status_by_tau.py`](../scripts/fig2_plot_lines_status_by_tau.py),  
   [`fig2_plot_scatter_human_risk.py`](../scripts/fig2_plot_scatter_human_risk.py) (labels optional), [`fig2_plot_abstain_reasons.py`](../scripts/fig2_plot_abstain_reasons.py)

---

## Human labels (optional; decision-grade risk only)
Human labels are used only to compute **human non-accept risk** among audit-PASS claims
(`(SHOULD_ABSTAIN + REJECT) / audit-PASS among labeled`; Supplementary Table 5).

Helpers:
- Template generator: [`fig2_make_labels_template.py`](../scripts/fig2_make_labels_template.py)
- Merge/validation: [`fig2_check_labels_merge.py`](../scripts/fig2_check_labels_merge.py)

If labels are unavailable or restricted, all panels **except human-risk plots** can be reproduced from audit logs and derived metrics.

---

## Optional: LLM-assisted proposal generation (local Ollama)

Main figure pipelines are reproducible in deterministic mode.
Optionally, we ran an **LLM-assisted proposal** setting where the LLM is used **only** for:
(i) context-conditioned representative selection and (ii) schema-bounded JSON claim typing.
**PASS/ABSTAIN/FAIL decisions are always mechanical (audit suite).**
LLM-assisted runs are stored separately under [`out_figS3/`](../source_data/PANCAN_TP53_v1/out_figS3).

### Example (HNSC; τ=0.8; k=50)
```bash
export LLMPATH_BACKEND=ollama
export LLMPATH_OLLAMA_HOST=http://ollama:11434
export LLMPATH_OLLAMA_MODEL=llama3.1:8b
export LLMPATH_CONTEXT_REVIEW_MODE=llm
export LLMPATH_CONTEXT_GATE_MODE=hard
export LLMPATH_CLAIM_MODE=llm

python paper/scripts/fig2_run_pipeline.py \
  --cancers HNSC \
  --variants ours \
  --gate-modes hard \
  --taus 0.8 \
  --k-claims 50 \
  --context-review-mode llm \
  --out-root paper/source_data/PANCAN_TP53_v1/out_figS3 \
  --force
```

### Outputs (within the run directory)
- [`audit_log.tsv`](../source_data/PANCAN_TP53_v1/out_figS3/HNSC/ours/gate_hard/tau_0.80/audit_log.tsv) (mechanical decisions with reason codes)
- [`run_meta.json`](../source_data/PANCAN_TP53_v1/out_figS3/HNSC/ours/gate_hard/tau_0.80/run_meta.json), [`run_meta.runner.json`](../source_data/PANCAN_TP53_v1/out_figS3/HNSC/ours/gate_hard/tau_0.80/run_meta.runner.json) (configuration + runtime provenance)
- `llm_claims.*.json` (prompt/meta/raw artifacts; present when LLM is enabled)
To compare deterministic vs LLM-assisted runs, aggregate metrics into a single table (e.g., rename the LLM-assisted slice to `out_figS3/`) and reuse the standard plotting scripts.

---

## Extended Data Fig. 2 (PANCAN_TP53_v1; collection sensitivity)

This pipeline evaluates audited outcomes across gene set collections (Hallmark / GO BP / Reactome / KEGG) using the same cohort construction and Sample Cards as Fig. 2.

### Key scripts:
- Build EvidenceTables per collection: [`figS2_fgsea_to_evidence_table.R`](../scripts/figS2_fgsea_to_evidence_table.R)
- Run collections pipeline: [`figS2_run_collections_pipeline.py`](../scripts/figS2_run_collections_pipeline.py)
- Collect metrics: [`figS2_collect_collection_metrics.py`](../scripts/figS2_collect_collection_metrics.py)
- Render panels: [`figS2_plot_collection_panels.py`](../scripts/figS2_plot_collection_panels.py)
- Export per-panel Source Data CSVs: [`figS2_export_panel_source_csv.py`](../scripts/figS2_export_panel_source_csv.py)

### Outputs
- Run outputs (audit logs, intermediates): [`paper/source_data/PANCAN_TP53_v1/out_figS2/`](../source_data/PANCAN_TP53_v1/out_figS2)
- Aggregated metrics (wide/long TSVs): [`paper/source_data/PANCAN_TP53_v1/collection_metrics/`](../source_data/PANCAN_TP53_v1/out_figS2/collection_metrics)
- Figure PDF: [`paper/source_data/PANCAN_TP53_v1/fig/FigS2`](../source_data/PANCAN_TP53_v1/fig/FigS2)
- Source Data workbook: [`paper/journal_source_data/SourceData_EDFig2.xlsx`](../journal_source_data/SourceData_EDFig2.xlsx)

### Notes:
- All acceptance decisions are mechanical (audit suite).
- Human labels are not required for this figure.
