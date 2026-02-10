<!-- paper/scripts/README.md -->
# Figure reproduction scripts (canonical)

This directory contains the **canonical, script-based** pipelines used to reproduce manuscript figures and publication Source Data.
For the authoritative mapping of **inputs ↔ scripts ↔ outputs**, see `paper/FIGURE_MAP.csv`.
Notebooks are exploratory and are not required for reproduction.

## Conventions
- Run from the repository root (e.g., `/work` in the paper container).
- Do not edit paths inside scripts. Figure inputs/outputs are organized under `paper/source_data/<BENCHMARK_ID>/`.
- Run outputs are written to figure-scoped directories (e.g., `out_fig2/`, `out_figS2/`, `out_figS3/`, `out_figS4/`) and include `run_meta*.json` for provenance.

---

## Fig. 2 (PANCAN_TP53_v1; Pan-cancer TP53 mut vs wt)

Outputs live under `paper/source_data/PANCAN_TP53_v1/`:
- Inputs: `raw/`, `derived/`, `sample_cards/`, `evidence_tables/`
- Run outputs: `out_fig2/`
- Figures: `fig/`
- Publication Source Data workbook: `paper/journal_source_data/SourceData_Fig2.xlsx`

### Pipeline (high level)
1) Fetch inputs: `fig2_fetch_inputs.py`
2) Define TP53 mut/wt groups: `fig2_make_groups.py`
3) Generate Sample Cards: `fig2_make_sample_cards.py`
4) Compute DE rankings (per cancer): `fig2_deg_rank.R`
5) Build EvidenceTables (Hallmark; per cancer): `fig2_fgsea_to_evidence_table.R`
6) Run LLM-PathwayCurator + mechanical audits: `run_fig2_pipeline.py`
7) Aggregate + plot panels:  
   `fig2_collect_risk_coverage.py`, `fig2_plot_multipanel.py`, `fig2_plot_lines_status_by_tau.py`,  
   `fig2_plot_scatter_human_risk.py` (labels optional), `fig2_plot_abstain_reasons.py`

---

## Human labels (optional; decision-grade risk only)
Human labels are used only to compute **human non-accept risk** among audit-PASS claims
(`(SHOULD_ABSTAIN + REJECT) / audit-PASS among labeled`; Supplementary Table 5).

Helpers:
- Template generator: `fig2_make_labels_template.py`
- Merge/validation: `fig2_check_labels_merge.py`

If labels are unavailable or restricted, all panels **except human-risk plots** can be reproduced from audit logs and derived metrics.

---

## Optional: LLM-assisted proposal generation (local Ollama)

Main figure pipelines are reproducible in deterministic mode.
Optionally, we ran an **LLM-assisted proposal** setting where the LLM is used **only** for:
(i) context-conditioned representative selection and (ii) schema-bounded JSON claim typing.
**PASS/ABSTAIN/FAIL decisions are always mechanical (audit suite).**
LLM-assisted runs are stored separately under `out_figS3/`.

### Example (HNSC; τ=0.8; k=50)
```bash
export LLMPATH_BACKEND=ollama
export LLMPATH_OLLAMA_HOST=http://ollama:11434
export LLMPATH_OLLAMA_MODEL=llama3.1:8b
export LLMPATH_CONTEXT_REVIEW_MODE=llm
export LLMPATH_CONTEXT_GATE_MODE=hard
export LLMPATH_CLAIM_MODE=llm

python paper/scripts/run_fig2_pipeline.py \
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
- `audit_log.tsv` (mechanical decisions with reason codes)
- `run_meta.json`, `run_meta.runner.json` (configuration + runtime provenance)
- `llm_claims.*.json` (prompt/meta/raw artifacts; present when LLM is enabled)
To compare deterministic vs LLM-assisted runs, aggregate metrics into a single table (e.g., rename the LLM-assisted slice to `out_figS3/`) and reuse the standard plotting scripts.

---

## Extended Data Fig. 2 (PANCAN_TP53_v1; collection sensitivity)

This pipeline evaluates audited outcomes across gene set collections (Hallmark / GO BP / Reactome / KEGG) using the same cohort construction and Sample Cards as Fig. 2.

### Key scripts:
- Build EvidenceTables per collection: `figS2_fgsea_to_evidence_table.R`
- Run collections pipeline: `figS2_run_collections_pipeline.py`
- Collect metrics: `figS2_collect_collection_metrics.py`
- Render panels: `figS2_plot_collection_panels.py`
- Export per-panel Source Data CSVs: `figS2_export_panel_source_csv.py`

### Outputs
- Run outputs (audit logs, intermediates): `paper/source_data/PANCAN_TP53_v1/out_figS2/`
- Aggregated metrics (wide/long TSVs): `paper/source_data/PANCAN_TP53_v1/collection_metrics/`
- Figure PDF: `paper/source_data/PANCAN_TP53_v1/fig/EDFig2.pdf`
- Source Data workbook: `paper/journal_source_data/SourceData_EDFig2.xlsx`

### Notes:
- All acceptance decisions are mechanical (audit suite).
- Human labels are not required for this figure.
