# CRM_R1 analysis plan

## Core revision claim

At matched reporting coverage, does the full LLM-PathwayCurator audit select pathway claims that
are more likely to replicate, have independent evidence, and avoid overstatement than simpler
reporting rules applied to the same candidate pool?

This is a working analysis protocol, not a rebuttal letter. It may become part of the public
reproducibility record after the analyses are frozen and internal editorial notes are excluded.
Pushing it to any branch of the public repository would make it public immediately.

## Priority 1: direct-perturbation temporal replication

This is the first go/no-go analysis because it is objective, the data are available, and the held-out
endpoint does not require human plausibility judgments.

### Prespecified design

- Dataset: GSE146225, human hiPSCs with WT or TP53 knockout, untreated or MMS treated.
- Benchmark ID: `GSE146225_TP53_v1`.
- Primary biological context: cells undergoing definitive endoderm differentiation (`ENDO`).
- Discovery time: 48 h.
- Held-out temporal replication time: 72 h.
- Interaction contrast:

  `(WT_MMS - WT_UT) - (TP53_KO_MMS - TP53_KO_UT)`

- Positive values mean a stronger MMS response in WT than in TP53-knockout cells.
- Expression input: supplied raw integer-count matrix.
- Gene filtering and normalization: edgeR `filterByExpr` and TMM on the 48 h discovery subset.
- Expression model: voom-limma 2 x 2 interaction, ranked by the moderated t-statistic.
- The 48 h-filtered gene universe is frozen and applied unchanged at 72 h; normalization and model
  fitting are performed separately at each time point.
- Primary gene-set collection: MSigDB Hallmark.
- Gene identifiers: NCBI Gene/Entrez IDs in the count matrix and Hallmark gene sets.
- Primary proposal mode: deterministic and LLM-free; LLM-assisted proposal generation is secondary.
- Audit operating point: tau = 0.8.
- Primary replication endpoint: the 72 h pathway has the same NES direction as at 48 h and
  Benjamini-Hochberg FDR < 0.05.

The machine-readable specification is `config/priority1_protocol.json`. Its status must be changed
from `DRAFT_NOT_FROZEN` to `FROZEN` before a script may calculate the 72 h replication endpoint.
All PASS/ABSTAIN/FAIL decisions are mechanical in both deterministic and optional LLM-assisted runs.

### Methods compared

All methods start from the identical 48 h Hallmark candidate pool.

1. Raw pool: descriptive reference only.
2. q-value matched: top K pathways by 48 h q-value, where K equals full-audit PASS coverage.
3. Stability-only matched: top K pathways by supporting-gene stability; q-value breaks ties.
4. Full audit: the prespecified LLM-PathwayCurator PASS set at tau = 0.8.
5. Random matched: repeated random K-pathway selections; secondary reference only.

The primary comparison is full audit versus q-value matched. Stability-only is the mechanistic
ablation. Raw versus PASS is not sufficient because it confounds quality with reporting fewer
claims.

### Analysis boundary

- Priority 1 tests held-out temporal replication after a direct TP53 perturbation.
- It does not establish independent-cohort replication.
- It does not prove mechanism, causality beyond the experimental contrast, or clinical utility.
- Context swap and supporting-gene dropout remain internal stress tests, not external validation.
- Human ratings and literature grading belong to later priorities and must not define or rescue this
  endpoint.
- A normalized expression matrix is not used in the primary differential-expression analysis.

### Required outputs

- `output/priority1/GSE146225_TP53_v1/preflight/sample_metadata.normalized.tsv`
- `output/priority1/GSE146225_TP53_v1/derived/rankings/discovery_48h.tsv`
- `output/priority1/GSE146225_TP53_v1/derived/fgsea/discovery_48h.tsv`
- `output/priority1/GSE146225_TP53_v1/evidence_tables/discovery_48h.tsv`
- `output/priority1/GSE146225_TP53_v1/sample_cards/discovery_48h.sample_card.json`
- `output/priority1/GSE146225_TP53_v1/out_audit/discovery_48h/audit_log.tsv`
- `output/priority1/GSE146225_TP53_v1/metrics/selection_membership.tsv`
- `output/priority1/GSE146225_TP53_v1/validation/pathway_statistics_72h.tsv`
- `output/priority1/GSE146225_TP53_v1/metrics/replication_by_method.tsv`
- `output/priority1/GSE146225_TP53_v1/source_data/figure4.tsv`

No plotting script may recompute differential expression, enrichment, audit decisions, or method
membership.

### Planned Figure 4 panels

- A: experimental design and 48 h discovery / 72 h validation split.
- B: 48 h versus 72 h NES with audit disposition highlighted.
- C: held-out replication fraction for matched methods, with 95% confidence intervals.
- D: coverage versus non-replication risk, with the frozen operating point marked.

### Stop gate P1

If full audit does not improve the point estimate over the coverage-matched q-value baseline, do not
add human rating or literature work to rescue Priority 1. Narrow or revise the manuscript claim
before proceeding.

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

- Freeze database releases, query templates, synonyms, search date, and retrieved records.
- Mask any source used for enrichment or candidate generation when selecting validation evidence.
- Grade evidence as E0 (not established), E1 (general association), E2 (direct contextual
  association), E3 (same comparison and direction), or E4 (perturbation or independent-cohort
  support). Record `contradicted` separately; do not equate E0 with false.
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

- Human non-accept risk among reported claims: `(SHOULD_ABSTAIN + REJECT) / reported`.
- Paired or stratified bootstrap 95% confidence interval for the matched risk difference.
- ACCEPT, SHOULD_ABSTAIN, and REJECT proportions with Wilson 95% confidence intervals.
- Weighted Cohen's kappa or Krippendorff's alpha with 95% confidence interval.

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
5. Freeze and conduct Priority 4 ratings; calculate agreement and confidence intervals.
6. Run Priority 5 ontology and utility robustness analyses.
7. Export one immutable source table per panel; plotting scripts only render those tables.

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
