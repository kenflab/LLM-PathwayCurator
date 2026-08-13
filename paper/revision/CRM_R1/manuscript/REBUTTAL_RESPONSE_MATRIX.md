# Rebuttal response matrix (working draft)

This is an internal response matrix, not the final point-by-point letter. Entries marked **pending**
must not be converted to result statements until the corresponding immutable lock passes.

| Editorial/reviewer concern | Revision response and new evidence | Manuscript/figure change | State |
|---|---|---|---|
| Editor: substantial new data are required before rereview | Added an objective perturbational validation (GSE146225), a same-pool unaudited comparison, frozen PubMed evidence retrieval/grading, three-rater method-blinded review, and external ontology analyses. | New Figures 2–4; revised Results, STAR Methods, limitations, and Source Data. | P1/P5 complete; P2 frozen; P3/P4 outcomes pending |
| R1-1: “decision-grade” implies biological correctness beyond the evidence | Narrow the terminology to **audit-gated pathway claims**. State that PASS means the claim satisfied the specified contract, not that mechanism, causality, novelty, or clinical utility was established. | Title, Summary, Introduction, Discussion, and figure wording. | Text can be revised now |
| R1-1/R2-3: limited human evaluation and no agreement statistic | Replaced the prior two-rater analysis with a fixed 50-claim census rated independently by three masked reviewers. Prespecified majority outcomes, Fleiss' kappa, unanimous agreement, mean pairwise agreement, and Wilson intervals. | Figure 2C–D; STAR Methods; Supplementary rating table. | Packets frozen; ratings pending |
| R1-2/R2-1: no standard or unaudited baseline | All methods use the identical frozen 50-claim HNSC pool. Compare the full audit with top-K q-value reporting and top-K stability reporting at K=25; show the raw pool descriptively. Membership was frozen before P3/P4 outcomes. | Figure 2A–C and Source Data. | P2 frozen; outcome comparison pending |
| R2-1: reporting fewer claims confounds quality | Primary comparisons are coverage matched (K=25). The raw 50-claim pool is explicitly descriptive and is not used as the primary inferential comparator. | Figure 2 legend and Methods. | Design frozen |
| R1/R2: internal stability is not external validity | P1 uses 81 exhaustive balanced discovery resamples and a held-out 72 h expression time point after a signed freeze. At tau=0.80, empirical selection replicated 18/23 versus 17/23 for q-value matching and 16/23 for the size-matched sensitivity analysis. | Figure 4 and P1 Results. | Complete and frozen |
| R1/R2: show uncertainty and do not overstate a small binary difference | Report Wilson 95% CIs and the overlap-aware exact reference. The primary binary difference is +0.0435 and the one-sided exact P is 0.50; it is shown rather than hidden. | Figure 4C and legend. | Complete and frozen |
| R2: internal stability may still be informative continuously | Report the prespecified secondary continuous analysis: AUROC 0.713 (bootstrap 95% CI 0.558–0.857; permutation P=0.0055). Phrase this as stratification, not proof of superior binary selection. | Figure 4B and Results. | Complete and frozen |
| R1-3: validate against GO/Reactome hierarchy | Froze GO (2026-07-26) and Reactome (release 97); used only safe ancestor relations; kept hierarchy external to audit decisions; assessed directional contradiction, matched nonedges, leading-edge support, and depth. | Figure 3 and P5 STAR Methods/Results. | Complete and frozen |
| R1-2: compare multiplicative utility with alternatives | Prespecified multiplicative, arithmetic-mean, minimum-component, and log-linear weight-grid sensitivity. Utility remains an exploratory ordering aid if rankings materially shift. | Supplementary figure/table after P3/P4 lock. | Pending P3/P4 lock; does not block ontology Figure 3 |
| R2 novelty: propose/verify separation is not itself novel; cite PMID 41071041 | Cite Khan et al. and explicitly state that the contribution is the pathway-specific typed claim contract, evidence-hash linkage, multi-gate audit, reason-coded abstention, and reproducible source artifacts—not invention of propose/verify separation. | Introduction and Discussion. | Text can be revised now |
| R2-4: production behavior is mainly PASS/ABSTAIN | State directly that, in these production runs, the audit operated chiefly as a stability/context filter with contradiction and contract checks as backstops. Report all dispositions and reason codes. | Results and Discussion. | Text can be revised now |
| R2-6: practical “what changed” example is buried | Promote one frozen HNSC before/after vignette selected without changing membership or outcomes. Show the same raw claim and the audit disposition/reason. | Compact Results vignette or supplementary panel. | Claim can be selected now from frozen audit trail |
| Generalizability beyond TP53/cancer | Add the direct TP53 perturbation dataset while retaining explicit limits: one perturbational system, temporal rather than independent-cohort validation, and no clinical validation. | Discussion limitations. | Complete for current scope |

## Language guardrails for the response letter

- Say “addressed with new data” only for P1 and P5 until P3/P4 locks pass.
- For P1, state both the improved point estimate and exact P=0.50 in the same paragraph.
- Treat P1 AUROC as an important secondary result, not a replacement primary endpoint.
- Do not describe PubMed hit counts as biological support; only locked grades define support.
- Do not call the three raters “disease experts” unless their qualifications justify that wording.
- Do not say GO/Reactome decisions are “hierarchically correct.” Parent and child dispositions need not match.
