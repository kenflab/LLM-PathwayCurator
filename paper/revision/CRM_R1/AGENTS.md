# Codex instructions for CRM_R1

Before changing files in this directory, read `ANALYSIS_PLAN.md` and
`config/priority1_protocol.json`, then use `paper/scripts/README.md` and
`paper/FIGURE_MAP.csv` as the canonical publication-pipeline conventions.

## Scope

- Keep revision-specific code and documentation under `paper/revision/CRM_R1/`.
- Reuse the production package and canonical `paper/scripts/` after an EvidenceTable is built.
- Keep raw data outside Git. Read inputs from `CRM_R1_DATA_ROOT`; write derived artifacts there.
- Do not modify `src/` merely to make this benchmark pass. Any core change needs a separate rationale.

## Priority order

1. Direct-perturbation temporal replication (GSE146225).
2. Same-pool unaudited-versus-audited benchmark with matched baselines.
3. Source-masked external database and literature evidence grading.
4. Narrow blinded human evidence review and inter-rater agreement.
5. GO/Reactome hierarchy validation, robustness, and figure integration.

Do not reorder these priorities to improve a result. Follow the dependencies and stop gates in
`ANALYSIS_PLAN.md`.

## Priority 1 guardrails

- The primary context is ENDO.
- The discovery contrast is evaluated at 48 h only.
- The 72 h data are a held-out temporal replication endpoint.
- Do not inspect or optimize against 72 h pathway results before the frozen manifest is signed.
- The primary input is the raw integer-count matrix.
- Use the repository-proven edgeR TMM + voom-limma workflow; do not add DESeq2 solely for this
  benchmark.
- Determine expression filtering from the 48 h discovery samples only, freeze that gene universe,
  and apply it unchanged to the 72 h validation analysis.
- Fit the same prespecified 2 x 2 interaction separately at 48 h and 72 h. Do not use a joint model
  that allows held-out 72 h outcomes to influence discovery.
- Run `fgseaMultilevel` in R, then use the production fgsea adapter to create the EvidenceTable.
- Use deterministic proposal generation as the primary Priority 1 run. Any LLM-assisted proposal
  run is secondary and stored separately; PASS/ABSTAIN/FAIL always remains mechanical.
- Compare the full audit with coverage-matched q-value and stability-only baselines.
- Raw-versus-PASS alone is not the primary comparison.
- PASS is a reporting disposition, not biological truth or causality.

## Reproducibility

- Every script must accept explicit paths and avoid user-specific hard-coded paths.
- Every analytical output must record configuration, input SHA-256 values, and software versions.
- Figure-facing tables must be TSV/CSV files, not values embedded in plotting code.
- Number new scripts by priority: `1x_`, `2x_`, `3x_`, `4x_`, `5x_`; reserve `9x_` for plots and
  final export checks.
- Plotting scripts may read figure-source tables but may not recompute analytical endpoints.
- Run `ruff format`, `ruff check`, and `pytest` before committing.
