# CRM_R1 analysis plan

## Core revision claim

At matched reporting coverage, does the full LLM-PathwayCurator audit select pathway claims that
are more likely to replicate, have independent evidence, and avoid overstatement than simpler
reporting rules applied to the same candidate pool?

This is a working analysis protocol, not a rebuttal letter. It may become part of the public
reproducibility record after the analyses are frozen and internal editorial notes are excluded.
Pushing it to any branch of the public repository would make it public immediately.

## Priority 1: empirical discovery stability and held-out temporal replication

This is the first go/no-go analysis because it uses an objective discovery-only resampling measure
and a held-out expression time point. It does not require human plausibility judgments.

### Full 48 h discovery analysis

- Dataset: GSE146225, human hiPSCs with WT or TP53 knockout, untreated or MMS treated.
- Benchmark ID: `GSE146225_TP53_v1`.
- Primary biological context: cells undergoing definitive endoderm differentiation (`ENDO`).
- Discovery time: 48 h.
- Held-out temporal replication time: 72 h.
- Interaction contrast:

  `(WT_MMS - WT_UT) - (TP53_KO_MMS - TP53_KO_UT)`

- Positive values mean a stronger MMS response in WT than in TP53-knockout cells.
- Expression input: supplied raw integer-count matrix.
- Gene filtering: edgeR `filterByExpr` on the twelve 48 h ENDO discovery samples.
- Normalization and model: edgeR TMM followed by a voom-limma 2 x 2 interaction model.
- Ranking statistic: moderated t-statistic for the interaction contrast.
- The full 48 h-filtered gene universe is frozen before resampling and later applied unchanged at
  72 h. Normalization and model fitting are performed separately in each analysis.
- Primary gene-set collection: the full-discovery MSigDB Hallmark snapshot.
- Gene identifiers: NCBI Gene/Entrez IDs.
- Primary proposal mode: deterministic and LLM-free.

### Empirical 48 h stability

The V3 synthetic evidence-set perturbation was associated with pathway size and is not used as the
primary Priority 1 stability measure. It remains a secondary implementation stress test only.

The V4 primary discovery stability is calculated by exhaustive balanced deletion:

1. Each of the four genotype-treatment cells contains three discovery samples.
2. One sample is deleted from each cell and the other two are retained.
3. All `3^4 = 81` balanced combinations are analyzed.
4. Every resample uses the full-discovery frozen gene universe.
5. TMM normalization and the same voom-limma interaction are refitted within each resample.
6. `fgseaMultilevel` is rerun against the frozen Hallmark snapshot.
7. Production `replicates_proxy` distillation compares each resample with the full 48 h baseline.

A pathway survives a resample when its enrichment direction agrees with the full analysis and its
leading-edge evidence meets the frozen Jaccard, recall, and precision thresholds. Empirical term
survival is the fraction of the 81 resamples that survive. This is sample-resampling stability, not
independent biological replication.

Context review is disabled for Priority 1 because neither a deterministic hash proxy nor an LLM
plausibility judgment is an objective held-out biological endpoint. The context gate is nonblocking
(`note`), and the stability gate remains mechanical (`hard`).

### Discovery-only operating-point calibration

The empirical calibration grid is `tau = 0.80, 0.90, 0.95, 0.98`. All four runs must be completed
using only 48 h outputs. The primary tau is then selected and recorded before any 72 h pathway
statistic is calculated. The choice is described as discovery-calibrated, not independently
prespecified. The selected tau must provide nondegenerate coverage and a meaningful membership
difference from the coverage-matched q-value baseline. The full risk-coverage curve remains a
prespecified sensitivity analysis.

The discovery-only calibration was completed on 2026-08-08. `tau = 0.80` is frozen as the primary
operating point: 23 of 50 pathways PASS (coverage 0.46), compared with 15, 10, and 5 at `tau =`
0.90, 0.95, and 0.98. The empirical selection overlaps the coverage-matched q-value selection for
16 of 23 claims and the leading-edge-size-matched selection for 17 of 23 claims. Empirical survival
has Spearman correlations of -0.166 with full pathway size and 0.079 with full-discovery
leading-edge count, resolving the material size association seen with the V3 synthetic proxy.

The machine-readable specification is `config/priority1_protocol.json`. It records
`primary_tau = 0.80` and status `FROZEN`. This protocol status alone does not release the held-out
endpoint. `16_freeze_priority1_membership.py` must write the exact empirical, q-value, size-matched,
and tau-grid memberships plus their SHA-256 inventory, and `17_check_priority1_freeze.py` must pass
before any script may calculate a 72 h pathway statistic.

### Methods compared

All methods start from the identical full-discovery 48 h Hallmark candidate pool.

1. Raw pool: descriptive reference only.
2. q-value matched: top K pathways by 48 h q-value, where K equals empirical-audit PASS coverage.
3. Empirical-stability audit: PASS claims based on 81 balanced 48 h resamples.
4. q-value and leading-edge-size matched: lowest-q pathways within four deterministic
   equal-frequency leading-edge-size strata, matching the empirical selection count in every
   stratum; sensitivity analysis.
5. Synthetic evidence perturbation: implementation stress test only; not a primary biological
   stability measure.
6. Random matched: repeated random K-pathway selections; secondary reference only.

The primary comparison is empirical-stability audit versus q-value matched at identical K. The
size-matched comparator tests whether any difference can be explained by supporting-gene-set size.
Raw versus PASS alone is insufficient because it confounds quality with reporting fewer claims.

### Held-out endpoint

After protocol and membership freeze, the primary replication endpoint is same-direction 48 h and
72 h NES together with 72 h Benjamini-Hochberg FDR below 0.05. The 72 h analysis uses the frozen
full-discovery gene universe but recalculates TMM normalization and fits the interaction model using
72 h samples only.

The primary estimand is the empirical-selection replication fraction minus the q-value-matched
replication fraction at the identical frozen `K = 23`. Method-specific fractions receive Wilson
95% confidence intervals. Because 16 pathways are shared, the methods are not independent groups;
an overlap-aware exact label-randomization reference is calculated on the symmetric difference and
is interpreted descriptively. The Stop gate is based on the direction of the frozen point estimate,
not on post hoc endpoint or threshold changes.

Across all 50 pathways, a frozen continuous secondary analysis asks whether 48 h empirical survival
predicts the binary 72 h replication endpoint. It reports AUROC with a label-permutation reference
and bootstrap 95% confidence interval. A fixed exploratory logistic sensitivity includes empirical
survival, negative log10 48 h q-value, and log1p full-discovery leading-edge count; separation or
non-estimability is reported rather than repaired by changing the model.

### Frozen Priority 1 result

The held-out endpoint was released once after the operational freeze and evaluated without changing
tau, membership, endpoint, or comparators. At `tau = 0.80`, 18 of 23 empirical-stability claims
replicated at 72 h, compared with 17 of 23 coverage-matched q-value claims and 16 of 23
q-value-plus-leading-edge-size-matched claims. The frozen empirical-minus-q-value difference was
`+1/23` (`+0.0435`), so the prespecified point-estimate Stop gate passed. The overlap-aware exact
reference was not significant (one-sided `P = 0.50`; two-sided `P = 1.00`) and remains descriptive.

Across all 50 pathways, empirical survival discriminated the held-out binary replication endpoint
with AUROC `0.713` (stratified-bootstrap 95% CI `0.558-0.857`; label-permutation `P = 0.0055`). The
fixed adjusted logistic coefficient for standardized empirical survival was positive but did not
reach conventional significance (`P = 0.065`). These results support empirical stability as an
objective component-level ranking signal; they do not establish a statistically detectable
discrete-set advantage, independent-cohort replication, or validation of the complete semantic
audit workflow.

### Analysis boundary

- Priority 1 tests whether empirical discovery-resampling stability predicts held-out temporal
  replication after a direct TP53 perturbation.
- It validates one objective audit component, not the complete semantic audit workflow.
- It does not establish independent-cohort replication.
- It does not prove mechanism, causality beyond the experimental contrast, or clinical utility.
- Context review, human ratings, and literature grading belong to later priorities and must not
  define or rescue this endpoint.
- A normalized expression matrix is not used in the primary differential-expression analysis.

### Required outputs

- `output/priority1/GSE146225_TP53_v1/preflight/sample_metadata.normalized.tsv`
- `output/priority1/GSE146225_TP53_v1/derived/rankings/discovery_48h.tsv`
- `output/priority1/GSE146225_TP53_v1/derived/fgsea/discovery_48h.tsv`
- `output/priority1/GSE146225_TP53_v1/derived/empirical_resampling_48h/resample_manifest.tsv`
- `output/priority1/GSE146225_TP53_v1/derived/empirical_resampling_48h/fgsea_resamples.tsv`
- `output/priority1/GSE146225_TP53_v1/evidence_tables/discovery_48h_empirical_replicates.tsv`
- `output/priority1/GSE146225_TP53_v1/sample_cards/discovery_48h_empirical.sample_card.json`
- `output/priority1/GSE146225_TP53_v1/metrics/empirical_stability_calibration_preview.tsv`
- `output/priority1/GSE146225_TP53_v1/metrics/selection_membership_frozen_tau0p80.tsv`
- `output/priority1/GSE146225_TP53_v1/metrics/selection_membership_tau_grid_frozen.tsv`
- `output/priority1/GSE146225_TP53_v1/metrics/priority1_freeze_manifest.json`
- `output/priority1/GSE146225_TP53_v1/metrics/priority1_freeze_manifest.sha256`
- `output/priority1/GSE146225_TP53_v1/validation/pathway_statistics_72h.tsv`
- `output/priority1/GSE146225_TP53_v1/metrics/replication_by_method.tsv`
- `output/priority1/GSE146225_TP53_v1/source_data/figure4.tsv`

No plotting script may recompute differential expression, enrichment, audit decisions, or method
membership.

### Frozen Figure 4 panels

- A: full 48 h discovery, 81 balanced resamples, freeze, and held-out 72 h validation.
- B: empirical survival versus held-out replication across all 50 pathways, with the frozen
  threshold, AUROC, bootstrap interval, and permutation reference shown.
- C: held-out replication fraction for empirical, q-value-matched, and
  q-value-plus-leading-edge-size-matched methods with Wilson 95% confidence intervals; the primary
  overlap-aware exact `P = 0.50` is shown without a significance symbol.
- D: coverage versus non-replication risk across the frozen tau grid, including the non-monotonic
  sensitivity pattern and the frozen primary point.

The Figure 4 message is: **Empirical discovery stability stratifies held-out temporal pathway
replication.** The plotting layer reads only the frozen `source_data/figure4.tsv` and evaluation
summary and must not recalculate an analytical endpoint.

### Stop gate P1

If empirical-stability selection does not improve the held-out replication point estimate over the
coverage-matched q-value baseline, do not use literature or human ratings to rescue Priority 1.
Report the negative component-validation result and narrow the manuscript claim before proceeding.

The observed point estimate was positive (`18/23` versus `17/23`), so Priority 1 passed this
prespecified operational gate. The small effect and descriptive exact result must remain visible;
the gate is not interpreted as proof of statistical superiority.

## Priority 2: same-pool audit benchmark

### Question

Does the audit improve reporting quality beyond simply selecting fewer or more statistically
significant claims?

### Design

- Freeze candidate generation before any audit disposition or validation label is inspected.
- Retain every unique structured claim as
  `pathway x context x comparison x direction x claim_strength`.
- Compare the same candidate pool under raw reporting, coverage-matched q-value selection,
  coverage-matched stability selection, and the full audit.
- Treat the raw pool as descriptive. The primary comparison is full audit versus q-value matched;
  raw versus PASS alone is insufficient.
- Sample human-review claims from the pre-audit pool, never by preferentially sampling PASS claims.

### V8 operationalization (frozen)

- The primary P2 benchmark is the canonical HNSC Hallmark pool: 50 directional claims from
  `PANCAN_TP53_v1`.
- The primary `tau = 0.90` is inherited from the original Figure 2 operating point and is fixed
  before P3/P4 outcomes; it is not tuned to literature or rater results.
- Candidate proposals are deterministic for both runs. The mechanical reference disables context
  review and makes its context gate nonblocking. The full-audit run changes only context review to
  the frozen local Ollama model (`llama3.1:8b`) with a hard context gate.
- Freeze fails unless all 50 structured claims agree across runs. Full-audit PASS count defines K
  for q-value and stability matching.
- P3 and P4 use a census of all 50 pre-audit claims in randomized order. Audit status and method
  membership are withheld from the distributed packet.
- LLM context judgments are a method component under evaluation, not reference truth. Their errors
  are retained for P3/P4 rather than manually corrected.
- The completed freeze contains 50 claims and matched `K = 25`; full-audit overlap is 13/25 with
  q-value matching and 15/25 with mechanical-stability matching.

### Deliverables

- `output/priority2/pool/claims.tsv`
- `output/priority2/membership/selection_membership.tsv`
- `output/priority2/metrics/risk_coverage_source.tsv`
- A locked sampling frame for Priorities 3 and 4.

### Figure target

Figure 2 risk-coverage and disposition panels. Evidence- and human-labeled endpoints are added only
after Priorities 3 and 4 are frozen.

## Priority 3: independent evidence grading

### Question

Are audited claims better supported by evidence sources that were not used to generate or audit the
claim?

### Design

- Freeze query templates, synonyms, search date, publication cutoff, and retrieved records.
- Mask any source used for enrichment or candidate generation when selecting validation evidence.
- Apply three identical query families to every frozen claim: HNSC + TP53 + pathway, HNSC +
  pathway, and TP53 + pathway. Freeze the top ten PubMed best matches per family before grading.
- Grade evidence as E0 (no eligible support in the frozen retrieval), E1 (general relevance), E2
  (partial HNSC-pathway or TP53-pathway context), E3 (direction-matched independent HNSC TP53
  relationship), or E4 (matched TP53 perturbation or prospectively independent validation).
  Record contradiction separately; do not equate E0 with false.
- Primary independent support requires E3/E4, direction match, and independent data. Same or
  possibly overlapping TCGA data cannot qualify for that endpoint.
- Score context, direction, and study-design match rather than publication count.
- Stratify or adjust summaries for literature abundance to expose publication bias.

### Primary outputs

- Direct-evidence-supported fraction at matched coverage.
- Directionally supported fraction at matched coverage.
- Unsupported/refuted risk difference with claim-level bootstrap 95% confidence intervals.
- A reproducible evidence ledger containing query, date, source, identifier, and grade rationale.

### Figure target

Figure 2 external-evidence panels and a supplementary evidence ledger.

## Priority 4: blinded human evidence review

### Question

Given displayed statistics and independently retrieved evidence, is the audit disposition and claim
wording defensible?

### Rater definition

Reviewers need experience interpreting enrichment analyses, GO/Reactome/MSigDB, and biomedical
literature. They do not need to be the world's expert for every pathway. They must be independent of
LM-PathwayCurator development; a disease specialist may adjudicate difficult disagreements.

### Blinded questions

1. Do the displayed statistics support the data-level claim?
2. Does the displayed external evidence directly, indirectly, or not support the contextual claim?
3. Does the wording overstate causality, mechanism, or clinical implication?

Audit status and method identity remain masked during rating. Store ratings in rater-level long
format; never collapse duplicate `claim_id` rows before agreement analysis.

### Primary outputs

- Primary human risk: majority `MAJOR_OVERSTATEMENT` among claims reported by each matched method.
- Secondary human risk: majority `MINOR_OVERSTATEMENT` or `MAJOR_OVERSTATEMENT`.
- Paired claim-level bootstrap 95% confidence interval for matched risk differences.
- Question-level response proportions with Wilson 95% confidence intervals.
- Ordinal inter-rater agreement with a bootstrap 95% confidence interval.

### Figure target

Figure 2 human-evidence panel. This is a narrow evidence review, not an ungrounded global vote on
whether a pathway is biologically true.

## Priority 5: ontology validation, robustness, and integration

### Question

Are audit decisions coherent across an independent pathway hierarchy, and are conclusions robust to
reasonable scoring choices?

### Design

- Freeze GO and Reactome releases. Use safe GO ancestor propagation (`is_a` and `part_of` only) and
  exclude `has_part` and `regulates` from simple ancestor propagation.
- Do not feed hierarchy results into the audit; hierarchy is an external evaluation.
- Measure directional contradiction, hierarchy-consistent gene support, ontology-depth effects, and
  disposition patterns among specific children and broad parents.
- Do not require a child PASS to imply parent PASS; broad parents may appropriately ABSTAIN for lack
  of specificity.
- Compare multiplicative utility, equal-weight arithmetic mean, minimum component, and a
  prespecified log-linear weight grid. Report rank correlation, top-k overlap, and material rank
  shifts.
- If utility ranking is unstable, describe it only as an exploratory ordering aid.

### V10.3 operationalization: parallel ontology work while ratings are pending

- P5 ontology evaluation may proceed after the frozen P2 census and blank P4 packets exist. It does
  not read P3 grades, partial P4 returns, or method-level P3/P4 outcomes.
- New HNSC GO BP and Reactome audits use the P2 operating configuration: deterministic proposals,
  local `llama3.1:8b` context review with a hard gate, `tau = 0.90`, `k = 500`, and seed 42.
- A mapping-only feasibility preflight showed that the original top-50 proposal sets contained too
  few hierarchy pairs. The wider `k = 500` census was fixed before calculating contradiction,
  gene-support, P3, or P4 endpoints. It changes only the evaluation-set width; hierarchy data do
  not enter proposal generation or audit decisions.
- The first full-collection runs exposed a pre-freeze implementation mismatch: LLM context review
  considered at most 500 source terms before deterministic proposal, whereas the final 500 claims
  used a different ranking. The exact resulting `entity x direction` census is therefore locked
  without reading status, then rerun as an exact 500-row EvidenceTable so every fixed claim receives
  context review. The incomplete runs remain QC records and are not frozen as final P5 audits.
- GO and Reactome hierarchy data are not supplied to those audits. The exact audit logs, ontology
  files, releases, code, and hashes are frozen before any hierarchy endpoint is calculated.
- GO uses the frozen `go-basic.obo` snapshot but propagates only `is_a` and `part_of`; all regulation
  relations and `has_part` are ignored. Reactome uses frozen Version 97 human parent-child edges.
- Audit terms map to ontology identifiers by unique normalized exact label. Unmapped and ambiguous
  terms are excluded and reported; no manual remapping follows inspection of hierarchy outcomes.
- Direct parent-child pairs are the primary scope. All safe ancestor-descendant pairs, including
  direct edges, are a fixed
  sensitivity analysis. A collection with fewer than ten direct pairs is reported as not estimable
  for the primary scope; the sensitivity scope is not promoted after results are seen.
- A descriptive null compares each observed hierarchy pair with nonancestor pairs matched as
  closely as possible on parent depth, child depth, and log2 evidence-gene counts. Ten thousand
  draws use seed 20260811. Dependence among pathway claims precludes interpreting this as an
  independent-sample test.
- Figure 3 ontology panels may be finalized before ratings return. Utility code and its weight grid
  are frozen now, but the real utility calculation requires complete locked P3 and P4 manifests.

### Deliverables

- `output/priority5/ontology/hierarchy_metrics.tsv`
- `output/priority5/utility/sensitivity_metrics.tsv`
- `output/priority5/final/figure_manifest.tsv`
- Source tables for every main and supplementary panel.

### Figure target

Figure 3 hierarchy validation and supplementary utility-sensitivity panels.

## Critical path to final figures

1. Complete the Priority 1 48 h analysis and freeze its protocol before touching 72 h outcomes.
2. Run the 72 h endpoint once, then apply the P1 stop gate.
3. Freeze the Priority 2 candidate pool and selection membership.
4. Build the Priority 3 evidence ledger before preparing blinded packets.
5. Freeze the blank Priority 4 packets and conduct independent ratings.
6. In parallel with P3 grading and P4 rating, freeze and run the P5 ontology analysis and render the
   ontology-only Figure 3 panels without reading P3/P4 outcomes.
7. After all P3 grades and three P4 ratings are complete and locked, calculate Figure 2 endpoints
   and the P5 utility sensitivity once.
8. Export one immutable source table per panel; plotting scripts only render those tables.

## Planned figure map

| Figure | Main content | Primary source |
| --- | --- | --- |
| Figure 1 | Workflow, novelty boundary, worked before/after example | Existing workflow plus revised claim language |
| Figure 2 | Same-pool benchmark, evidence risk, human risk, agreement, 95% CIs | Priorities 2-4 |
| Figure 3 | GO/Reactome hierarchy validation | Priority 5 |
| Figure 4 | GSE146225 48 h discovery to held-out 72 h replication; portability context | Priority 1 |
| Supplement | Utility sensitivity, full dataset detail, reason codes, evidence ledger | Priorities 2-5 |

## Interpretation boundary

- The methodological advance is the pathway-specific typed contract, evidence identity, multi-gate
  audit, reason-coded selective abstention, and machine-readable reporting bundle.
- Proposal/verification separation itself is not claimed as novel.
- PASS is a reporting disposition, not biological truth.
- `decision-grade` should not be used as an unqualified headline claim.
- Biology is a validation and utility demonstration; the paper's center remains the reusable method.
