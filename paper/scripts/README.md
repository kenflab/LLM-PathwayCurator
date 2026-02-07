<!-- paper/scripts/README.md -->
# Figure reproduction scripts (canonical)

This directory contains the **canonical, script-based** pipelines used to reproduce figures and tables for the manuscript.  
For a complete mapping of **inputs ↔ scripts ↔ outputs**, see `paper/FIGURE_MAP.csv`. Notebooks are exploratory and are not required for reproduction.

## Quick start (Fig. 2; Pan-cancer TP53 mut vs wt)
Run from the repository root (e.g., `/work` in the paper container). Outputs are written under:
`paper/source_data/PANCAN_TP53_v1/` (intermediates in `out/`, figures in `fig/`).

1) Fetch inputs and define groups
- `fig2_fetch_inputs.py`
- `fig2_make_groups.py`

2) Build Sample Cards and ranked gene lists
- `fig2_make_sample_cards.py`
- `fig2_deg_rank.R` (per cancer)

3) Convert enrichment to EvidenceTable (Hallmark; fgsea/GSEA)
- `fig2_fgsea_to_evidence_table.R` (per cancer)

4) Run LLM-PathwayCurator + audits (Proposed / context swap / evidence-dropout stress)
- `fig2_run_pipeline.py`

5) Collect metrics and render panels
- `fig2_collect_risk_coverage.py`
- `fig2_plot_multipanel.py`
- `fig2_plot_lines_status_by_tau.py`
- `fig2_plot_scatter_human_risk.py` (requires labels; see below)
- `fig2_plot_abstain_reasons.py`

## Supplementary pipelines
- **Fig. S2 (collections / thresholding):**
  - `figS2_run_collections_pipeline.py`, `figS2_collect_collection_metrics.py`,
    `figS2_collections_plot_hallmark_threshold.py`, `figS2_plot_collection_panels.py`
- **Fig. S3 (LLM examples):**
  - `figS3_make_llm_examples.py`
- **Fig. S4 (BeatAML TP53 mut vs wt):**
  - `figS4_beataml_fetch_inputs.py`, `figS4_beataml_make_groups.py`,
    `figS4_beataml_make_sample_cards.py`, `figS4_beataml_deg_rank.R`,
    `figS4_beataml_fgsea_to_evidence_table.R`

## Human labels (optional)
Human labels are used only to compute **decision-grade risk** (fraction of SHOULD_ABSTAIN/REJECT among audit-PASS).
For public release, we provide a **template** generator:
- `fig2_make_labels_template.py`
and a merge/validation helper:
- `fig2_check_labels_merge.py`

If labels are not available, all figure panels except human-risk plots can be reproduced from the audit logs and derived metrics.

## Notes
- These scripts are intended to be run in the paper environment (Docker/Compose) with pinned dependencies.
- Do not edit paths inside scripts; configure locations via `FIGURE_MAP.csv` and per-figure configs where applicable.

---

## Optional: LLM-assisted claim proposal (local Ollama)

The main figure pipelines are fully reproducible in deterministic mode.  
Optionally, we also ran an **LLM-assisted proposal** setting, where the LLM is used **only** for (i) context-conditioned representative selection and (ii) JSON claim typing; **PASS/ABSTAIN/FAIL are still decided mechanically by the audit suite**.

### Run (example: HNSC; Ollama)
Set the LLM backend via environment variables (example uses a local Ollama 8B model):

```bash
export LLMPATH_BACKEND=ollama
export LLMPATH_LLM_ASSUME_SMALL_MODEL=1
export LLMPATH_OLLAMA_HOST=http://ollama:11434
export LLMPATH_OLLAMA_MODEL=llama3.1:8b
export LLMPATH_CONTEXT_REVIEW_MODE=llm
export LLMPATH_CONTEXT_GATE_MODE=hard
export LLMPATH_CLAIM_MODE=llm
export LLMPATH_DEBUG=1

# timeouts / safety (adjust if needed)
export LLMPATH_OLLAMA_READ_TIMEOUT=1500
export LLMPATH_OLLAMA_READ_TIMEOUT_MAX=4000
export LLMPATH_OLLAMA_TIMEOUT_ESCALATIONS=3
export LLMPATH_OLLAMA_TIMEOUT_FACTOR=2.0

# candidate budget
export LLMPATH_LLM_TOPN=80
export LLMPATH_LLM_MAX_CAND_LINES=80
export LLMPATH_LLM_STRICT_K=1

python paper/scripts/fig2_run_pipeline.py \
  --cancers HNSC \
  --variants ours \
  --gate-modes hard \
  --taus 0.8 \
  --k-claims 50 \
  --context-review-mode llm \
  --out-root /work/paper/source_data/PANCAN_TP53_v1/out_figS3 \
  --force
```

### Outputs
- LLM runs are kept separate under `out_figS3/`. Each run directory persists:
- `llm_claims.prompt.json` (exact prompt + candidate pack)
- `llm_claims.raw.json` (raw model output)
- `audit_log.tsv` (mechanical decisions with reason codes)

### Plotting / comparison
To compare deterministic vs LLM-assisted runs in the same plot, merge the derived metrics tables (e.g., rename the LLM variant to `ours_llm`) and reuse the standard plotting scripts (e.g., `fig2_plot_scatter_human_risk.py`, `figS_plot_abstain_reasons.py`)

---

## Extended Data Fig. 2: Gene set collection sensitivity (Hallmark / GO BP / Reactome / KEGG)

This pipeline evaluates how audited decisions change across different gene set collections, using the same cohort construction and Sample Cards as Fig. 2 (Pan-cancer TP53 mut vs wt). We run LLM-PathwayCurator on EvidenceTables derived from each collection and summarize PASS/ABSTAIN/FAIL and module-level stability.

### Outputs
- Run outputs (audit logs, intermediates):  
  `paper/source_data/PANCAN_TP53_v1/out_figS2/`
- Aggregated metrics (wide/long TSVs):  
  `paper/source_data/PANCAN_TP53_v1/collection_metrics/`
- Figure panels:  
  `paper/source_data/PANCAN_TP53_v1/fig/FigS2/`

### Steps (run from repo root, e.g. `/work`)
If you already reproduced Fig. 2, you can skip the fetch/groups/sample-card steps.

1) Fetch inputs, define groups, and generate Sample Cards
- `fig2_fetch_inputs.py`
- `fig2_make_groups.py`
- `fig2_make_sample_cards.py`

2) Create ranked gene lists (per cancer)
- `fig2_deg_rank.R` (HNSC, LUAD, LUSC, BRCA, OV, UCEC, SKCM)

3) Build EvidenceTables for each gene set collection (per cancer)
- Hallmark:
  - `figS2_fgsea_to_evidence_table.R <CANCER> --collection H`
- GO Biological Process:
  - `figS2_fgsea_to_evidence_table.R <CANCER> --collection C5 --subcategory GO:BP`
- Reactome and KEGG:
  - `figS2_fgsea_to_evidence_table.R <CANCER> --collection C2 --subcategory CP:REACTOME`
  - `figS2_fgsea_to_evidence_table.R <CANCER> --collection C2 --subcategory CP:KEGG_MEDICUS`

4) Run the collections pipeline (τ fixed; audited hard gate)
- `run_figS2_collections_pipeline.py`  
  (example collections: `Hallmark,GO_BP,Reactome,KEGG_MEDICUS`)

5) Collect summary metrics and render figure panels
- `figS2_collect_collection_metrics.py`
- `figS2_plot_collection_panels.py`
- Optional Hallmark threshold diagnostic:
  - `figS2_collections_plot_hallmark_threshold.py`

Notes:
- All acceptance decisions are mechanical (audit suite); collection choice affects EvidenceTable content and thus stability/modules and abstention patterns.
- Human labels are not required for this figure.
