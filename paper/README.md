<!-- paper/README.md -->
# Paper artifacts (reproducible figures)

This folder contains the scripts and (when redistributable) frozen/derived artifacts used to reproduce the manuscript figures.

**Canonical reproduction is script-based.**
- How to run: [`paper/scripts/README.md`](./scripts/README.md)
- Figure ↔ inputs ↔ scripts ↔ outputs (source of truth): [`paper/FIGURE_MAP.csv`](./FIGURE_MAP.csv)

This repository includes pipelines for **Fig. 2** and supplementary/extended figures (**ED Fig. 2–4**). BioRender panels are provided as PDFs only.

## What is (and is not) redistributed
- **Redistributed (when allowed):** derived tables, plotting exports, and rendered PDFs under [`paper/source_data/*/fig/`](./source_data/PANCAN_TP53_v1/fig) and [`paper/journal_source_data/`](./journal_source_data).
- **Not redistributed:** controlled-access raw inputs (when applicable) and internal human-label files. When labels cannot be shared, we provide **templates/schemas** and deterministic merge/validation helpers in [`paper/scripts/`](./scripts/).

## Layout (by benchmark)
Artifacts are organized by benchmark under [`paper/source_data/`](./source_data).

### PANCAN_TP53_v1 (TCGA PanCan; Fig. 2 / ED Fig. 2–3)
Root: [`paper/source_data/PANCAN_TP53_v1/`](./source_data/PANCAN_TP53_v1)
- `raw/` public upstream inputs (fetched by scripts when missing)
- [`derived/`](./source_data/PANCAN_TP53_v1/derived) intermediate cohort/ranking products
- [`sample_cards/`](./source_data/PANCAN_TP53_v1/sample_cards) normalized Sample Cards used by runs
- [`evidence_tables/`](./source_data/PANCAN_TP53_v1/evidence_tables) normalized EvidenceTables used by runs
- [`out_fig2/`](./source_data/PANCAN_TP53_v1/out_fig2) run outputs for Fig. 2 (audit logs, reports, run metadata)
- [`out_figS2/`](./source_data/PANCAN_TP53_v1/out_figS2) run outputs for ED Fig. 2 (collection analyses)
- [`out_figS3/`](./source_data/PANCAN_TP53_v1/out_figS3) run outputs for ED Fig. 3 (LLM-assisted slice)
- [`fig/`](./source_data/PANCAN_TP53_v1/fig) rendered figure PDFs

### BEATAML_TP53_v1 (BeatAML2; ED Fig. 4)
Root: [`paper/source_data/BEATAML_TP53_v1/`](./source_data/BEATAML_TP53_v1)
- `raw/` BeatAML2 *processed/harmonized* tables used by the pipeline (fetched by scripts when missing)
- [`derived/`](./source_data/BEATAML_TP53_v1/derived), [`sample_cards/`](./source_data/BEATAML_TP53_v1/sample_cards), [`evidence_tables/`](./source_data/BEATAML_TP53_v1/evidence_tables) as above
- [`out_figS4/`](./source_data/BEATAML_TP53_v1/out_figS4) run outputs for ED Fig. 4
- [`fig/`](./source_data/BEATAML_TP53_v1/fig) rendered figure PDFs

## Publication Source Data (Excel/CSV exports)
[`paper/journal_source_data/`](./journal_source_data) contains “Source Data” workbooks (e.g., [`SourceData_Fig2.xlsx`](./journal_source_data/SourceData_EDFig2.xlsx) and any exported CSVs used to populate them. Each workbook’s sheet mapping is documented inside the workbook and is linked from [`paper/FIGURE_MAP.csv`](./FIGURE_MAP.csv).

## Risk–coverage definition (Fig. 2; reused in ED figures)
- **Coverage (PASS rate):** `n_pass / n_total` under a given stability threshold τ.
- **Human risk (optional):** fraction of **human non-acceptable** labels among audit-PASS claims  
  (`(SHOULD_ABSTAIN + REJECT) / audit-PASS among labeled`; Supplementary Table 5).

Human labels are optional and only required for human-risk panels. Templates and deterministic merge/validation helpers are provided in [`paper/scripts/`](./scripts).

## Provenance (recommended)
Each run directory persists machine-readable metadata (e.g., `run_meta.json`, `run_meta.runner.json`) alongside audit logs and reports. Upstream fetch scripts may additionally record input identifiers/checksums for traceability (see [`paper/FIGURE_MAP.csv`](./FIGURE_MAP.csv)).
