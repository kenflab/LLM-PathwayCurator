<!-- paper/README.md -->
# Paper artifacts (reproducible figures)

This folder contains scripts and (where redistributable) frozen/derived outputs used to reproduce manuscript figures.

**Canonical reproduction is script-based**:
- Run instructions: `paper/scripts/README.md`
- Input ↔ script ↔ output map: `paper/FIGURE_MAP.csv`

This repository includes pipelines for **Fig. 2** and supplementary/extended figures (e.g., **Fig. S2–S4**); all execution details are consolidated in `paper/scripts/README.md`.

## Figure inputs/outputs (PANCAN_TP53_v1)
All figure artifacts are organized under:
`paper/source_data/PANCAN_TP53_v1/`

- Inputs (generated): `sample_cards/`, `evidence_tables/`
- Run outputs: `out/` (audit logs, reports, intermediates)
- Derived metrics: `out/risk_coverage*.tsv`
- Figures: `fig/`

## Risk–coverage definition (Fig. 2)
- **Coverage**: fraction of audit-PASS claims (PASS / TOTAL) under a given τ.
- **Risk (human, optional)**: fraction of **human non-acceptable** labels among audit-PASS claims  
  (non-acceptable = SHOULD_ABSTAIN or REJECT; Supplementary Table 5).

Human labels are optional and are only required for human-risk plots; template generation and merge helpers are provided in `paper/scripts/`.
