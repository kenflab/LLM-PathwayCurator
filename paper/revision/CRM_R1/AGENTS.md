# Codex instructions for CRM_R1

Before changing files in this directory, read `ANALYSIS_PLAN.md` and
the relevant protocol under `config/`, then use `paper/scripts/README.md` and
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
- Do not inspect or optimize against 72 h pathway results before the τ=0.80 frozen manifest exists
  and `17_check_priority1_freeze.py` passes.
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
- For Priority 1, use the 81-run balanced empirical-resampling stability as the primary audit
  component and keep synthetic evidence-set perturbation secondary.
- Disable context review for Priority 1; a hash proxy or LLM judgment must not define the held-out
  biological validation set.
- Compare empirical-stability PASS membership with coverage-matched q-value selection and a
  leading-edge-size-matched q-value sensitivity analysis.
- The full 48 h tau grid is complete and `tau = 0.80` is frozen. Describe that point as
  discovery-calibrated rather than independently prespecified; do not change it after 72 h release.
- Raw-versus-PASS alone is not the primary comparison.
- PASS is a reporting disposition, not biological truth or causality.

## Priorities 2-4 guardrails

- Priority 2 is frozen at 50 HNSC Hallmark claims, `tau = 0.90`, and matched `K = 25`. Do not
  change claim wording, method membership, or randomized review IDs after the freeze.
- Apply every Priority 3 query family to every claim. Do not adapt query depth or synonyms using
  audit disposition, method membership, or preliminary literature results.
- Keep PubMed abstracts, raw ESearch/EFetch artifacts, contact email, and API credentials outside
  public Git and public Source Data. The API key must never be persisted.
- Do not start evidence grading until `31_check_priority3_retrieval.py` passes.
- P4 packets must mask claim UID, audit status, method membership, stability, and context-review
  fields. Give raters only copied rating templates and keep the frozen blanks unchanged.
- Do not unblind P3/P4 or calculate method-level outcomes until all grading and rating files are
  complete and locked.

## Reproducibility

- Every script must accept explicit paths and avoid user-specific hard-coded paths.
- Every analytical output must record configuration, input SHA-256 values, and software versions.
- Figure-facing tables must be TSV/CSV files, not values embedded in plotting code.
- Number new scripts by priority: `1x_`, `2x_`, `3x_`, `4x_`, `5x_`; reserve `9x_` for plots and
  final export checks.
- Plotting scripts may read figure-source tables but may not recompute analytical endpoints.
- Run `ruff format`, `ruff check`, and `pytest` before committing.
