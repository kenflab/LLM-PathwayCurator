# Figure 4 legend draft

## Figure 4 | Empirical discovery stability stratifies held-out temporal pathway replication

**a,** Frozen Priority 1 design. The pathway candidate pool was defined from the 12 ENDO samples at
48 h. Empirical stability was measured using all 81 balanced combinations obtained by deleting one
sample from each genotype-treatment cell. The operating point (`tau = 0.80`; 23 of 50 pathways) and
all comparator memberships were frozen before any 72 h pathway statistic was calculated. The
held-out analysis used the 12 ENDO samples at 72 h and the frozen 48 h gene universe and Hallmark
snapshot.

**b,** Empirical survival across the 81 discovery resamples for all 50 candidate pathways, grouped
by held-out replication outcome. Replication required concordant 48 h and 72 h NES direction and a
72 h Benjamini-Hochberg FDR below 0.05. The dashed line marks the frozen `tau = 0.80` operating
point. Empirical survival discriminated held-out replication with an AUROC of 0.713
(stratified-bootstrap 95% CI, 0.558-0.857; one-sided label-permutation `P = 0.0055`; 10,000
bootstrap samples and 10,000 permutations).

**c,** Held-out replication fractions at identical reporting coverage (`K = 23`) for empirical
stability, q-value matching, and q-value plus leading-edge-size matching. Points show observed
fractions and error bars show two-sided Wilson 95% confidence intervals. Eighteen empirical,
17 q-value-matched, and 16 size-matched pathways replicated. Because the empirical and q-value
sets shared 16 pathways, their comparison used an exhaustive overlap-aware label-randomization
reference over the symmetric difference (3,432 assignments; one-sided `P = 0.50`). This reference
is descriptive because pathways are biologically dependent.

**d,** Frozen reporting coverage versus non-replication risk across `tau = 0.80`, `0.90`, `0.95`,
and `0.98`. Error bars are the replication Wilson intervals transformed to the risk scale. The
filled point denotes the discovery-calibrated primary operating point (`tau = 0.80`). The complete
tau sensitivity is shown without selecting a new threshold after release of the 72 h endpoint.

Priority 1 evaluates the empirical-stability component using a held-out time point after direct
TP53 perturbation. It does not constitute independent-cohort validation or validation of the full
semantic audit workflow.
