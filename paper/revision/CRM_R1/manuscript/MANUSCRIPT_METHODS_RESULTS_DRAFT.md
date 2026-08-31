# Revision text modules (working manuscript draft)

The text below is written for direct adaptation into a Cell Reports Methods revision. P1 and P5
numbers are frozen. P2–P4 sections are Methods only; bracketed result fields must remain unresolved
until both blinded locks pass.

## STAR Methods — Priority 1: discovery stability and held-out temporal replication

### Dataset and interaction model

We evaluated pathway-claim replication in GSE146225, a human induced pluripotent stem-cell model
containing wild-type or TP53-knockout cells, with or without methyl methanesulfonate treatment. The
primary context was definitive endoderm differentiation. The discovery analysis used the 12 samples
collected at 48 h, and the held-out temporal analysis used the 12 samples collected at 72 h. The
contrast was (WT_MMS − WT_untreated) − (TP53_KO_MMS − TP53_KO_untreated); positive statistics
therefore indicate a stronger treatment response in wild-type than TP53-knockout cells. We used the
supplied raw integer-count matrix. Genes were filtered with edgeR `filterByExpr` using the complete
48 h discovery design. The resulting 17,394-gene universe was frozen before any held-out outcome was
calculated and was applied unchanged at 72 h. Within each analysis, library normalization and model
fitting were performed independently using TMM normalization followed by voom-limma for the 2 × 2
interaction. Genes were ranked by the moderated t statistic for the interaction contrast.

### Pathway enrichment and empirical discovery stability

Pathway enrichment used `fgseaMultilevel` and the Hallmark collection snapshot frozen from the full
48 h discovery analysis. We quantified discovery stability by exhaustive balanced deletion. Each of
the four genotype-by-treatment cells contained three samples; one sample was deleted from each cell,
yielding all 3^4 = 81 balanced eight-sample analyses. Every resample used the frozen discovery gene
universe, while TMM normalization, voom-limma fitting, ranking, and enrichment were recalculated
within that resample. A pathway survived a resample when its enrichment direction agreed with the
full 48 h analysis and its leading-edge evidence met the frozen Jaccard, recall, and precision
thresholds. Empirical survival was the proportion of the 81 resamples that survived. Context review
was disabled for this analysis, because neither an LLM plausibility score nor a deterministic context
proxy constituted a held-out biological endpoint.

### Operating-point freeze and comparators

We completed a discovery-only grid at tau = 0.80, 0.90, 0.95, and 0.98 before calculating any 72 h
pathway statistic. Tau=0.80 was then frozen, together with all inputs, hashes, and method memberships;
the selected audit set contained 23 of 50 pathways. The primary comparator selected the 23 pathways
with the lowest discovery q values. A sensitivity comparator selected 23 pathways by q value while
matching the audit set within four deterministic leading-edge-size strata. The raw 50-pathway pool
and repeated random selections were descriptive references. No literature or human rating was used
to set the threshold or rescue this endpoint.

### Held-out endpoint and inference

After the freeze bundle passed its hash-consistency gate, the 72 h analysis was executed once using
only the 12 ENDO samples at 72 h. The primary pathway endpoint required the same NES direction at 48 h
and 72 h and a Benjamini–Hochberg FDR below 0.05 at 72 h. Method-specific replication fractions are
reported with Wilson 95% confidence intervals. Because the audit and q-value selections overlapped,
we used an overlap-aware conditional exact reference that held common pathways fixed and permuted
membership only within the symmetric difference. This P value is descriptive because pathway
outcomes are biologically dependent. As a prespecified secondary analysis, we evaluated continuous
empirical survival against held-out replication by AUROC, with a stratified percentile bootstrap
confidence interval and a permutation P value.

## Results — Priority 1

The discovery-only freeze selected 23 of 50 Hallmark pathways at tau=0.80. Eighteen of 23
empirical-stability-selected pathways met the held-out 72 h replication endpoint (78.3%; Wilson 95%
CI, 58.1%–90.3%), compared with 17 of 23 pathways selected by q value (73.9%; 53.5%–87.5%) and 16 of
23 in the q-value and leading-edge-size-matched sensitivity set (69.6%; 49.1%–84.4%). Thus, the frozen
primary replication-fraction difference between empirical selection and q-value matching was +4.35
percentage points. The overlap-aware conditional exact reference did not distinguish the two fixed
sets (one-sided P=0.50; two-sided P=1.00), and we therefore do not interpret the binary comparison as
evidence of a statistically resolved advantage.

Across all 50 pathways, however, empirical survival stratified held-out replication continuously
(AUROC=0.713; stratified bootstrap 95% CI, 0.558–0.857; permutation P=0.0055). The corresponding
Spearman correlation between 48 h and 72 h NES was 0.701 (two-sided P=1.41×10^-8). Together, these
results support empirical discovery stability as an informative pathway-level stratifier in this
temporal perturbation experiment, while the small and nonsignificant frozen binary difference limits
claims that thresholded selection is superior to q-value reporting.

## STAR Methods — Priority 2: same-pool audited and unaudited reporting rules

We constructed a census of the same 50 deterministic HNSC Hallmark claims at tau=0.90. Candidate
wording, direction, statistical evidence, and claim identifiers were identical across methods.
Before external evidence grading or human rating, we froze four memberships: the full 50-claim pool
(descriptive only), the 25 lowest-q claims, the 25 highest-stability claims, and the 25 claims passing
the full audit. The full audit used deterministic proposal generation and LLM-assisted context review;
the accept/abstain/fail decision remained mechanical. The q-value rule represents conventional
unaudited top-ranked reporting. Primary comparisons use identical reporting coverage (K=25) and do
not compare the 50-claim raw pool directly with the 25-claim audit set as an inferential contrast.

For each binary external outcome, we will report method-specific fractions with Wilson 95%
confidence intervals. Full-audit versus q-value and full-audit versus stability comparisons will use
an overlap-aware conditional exact reference that holds common claims fixed and exchanges membership
within the symmetric difference. These tests will be labeled descriptive because pathway claims are
dependent.

## STAR Methods — Priority 3: frozen literature retrieval and independent evidence grading

Before method memberships were disclosed, every frozen claim was queried in PubMed using three
prespecified query families: direct HNSC–TP53–pathway evidence, HNSC–pathway evidence, and
TP53–pathway perturbation evidence. Searches excluded reviews, meta-analyses, editorials, comments,
and letters; results were relevance sorted and capped at ten records per claim-query family. Search
date, publication cutoff, queries, returned identifiers, fetched records, and file hashes were frozen
before grading. The retrieval contained 150 queries and 493 fetched unique PubMed records; absence
from this bounded retrieval was not treated as evidence that a claim was false.

The grader received claim text and retrieved bibliographic evidence but no claim UID, audit status,
stability score, context result, or method membership. Each claim-record pair was coded for
eligibility, evidence grade (E1–E4), direction match, context match, design, dataset independence, and
contradiction. E0 denotes no eligible supporting record in the frozen retrieval. The primary
claim-level external-support endpoint requires at least one E3 or E4 record with matched direction
and independent data; same or possible TCGA overlap cannot satisfy this endpoint. The completed
record table and derived claim-level outcomes are hash locked before any join to Priority 2
membership.

## STAR Methods — Priority 4: three-rater method-blinded review

Three reviewers with experience interpreting pathway enrichment and biological literature
independently rate all 50 claims. Packets show the fixed claim wording, enrichment statistic, q value,
leading-edge genes, and a bounded subset of the frozen literature retrieval. They omit claim UID,
audit disposition, method membership, stability, and context fields. Reviewers do not coordinate and
do not perform additional searches. For each claim they rate statistical support, external-evidence
directness, wording overstatement, and confidence, and provide a concise evidence-linked rationale.

The primary human outcome is major overstatement by majority vote; minor-or-major overstatement by
majority is secondary. We report Fleiss' kappa, unanimous agreement, and mean pairwise agreement for
each categorical question. Method-specific outcome fractions receive Wilson 95% confidence
intervals. Completed ratings are validated against each rater's frozen assignment and are locked
before method membership is disclosed. Any disease-specialist adjudication is restricted to
prespecified ties or direct-evidence disagreements and is reported separately from the independent
ratings.

## STAR Methods — Priority 5: external ontology hierarchy evaluation

We evaluated audit decisions against frozen Gene Ontology Biological Process (release 2026-07-26)
and Reactome (release 97) hierarchies. Candidate censuses were locked before reading audit outcomes,
and complete context review was then obtained for 500 claims per collection. Hierarchy information
was never supplied to the audit and did not alter any disposition. GO ancestor propagation used only
`is_a` and `part_of`; `has_part`, `regulates`, and related regulatory edges were excluded from simple
ancestor propagation. Reactome parent-child relations were evaluated from the frozen release.

The primary scope comprised direct parent-child pairs present in the audited census; all safe
ancestor-descendant pairs formed a sensitivity scope. Directional contradiction was defined as
opposite enrichment directions within a hierarchy pair. A frozen matched-nonedge reference matched
ontology depth and evidence size. We additionally measured the fraction of the child's leading edge
covered by the parent and described ontology depth across PASS, ABSTAIN, and FAIL claims. A child
PASS was not required to imply parent PASS, because broad parents may appropriately be withheld for
lack of specificity. Matched-reference P values are descriptive because hierarchy pairs share terms
and genes and are therefore dependent.

## Results — Priority 5

The frozen census yielded 41 direct and 58 safe-ancestor GO Biological Process pairs, and 213 direct
and 551 safe-ancestor Reactome pairs. Directional contradiction occurred in 11/41 GO direct pairs
(26.8%), 18/58 GO ancestor-descendant pairs (31.0%), 63/213 Reactome direct pairs (29.6%), and
178/551 Reactome ancestor-descendant pairs (32.3%). These observed fractions were lower than the
depth- and evidence-size-matched nonedge references (descriptive one-sided P=0.0032, 0.0034,
0.0001, and 0.0001, respectively). Because the pathway pairs are dependent, these matched-reference
results are interpreted as evidence of directional coherence, not as independent Bernoulli tests or
as a requirement that parent and child dispositions agree.

Leading-edge coverage and ontology-depth distributions further characterized how evidence support
and audit disposition varied across specific children and broader ancestors (Figure 3C–D; Source
Data). These analyses were external evaluations: no hierarchy metric was fed back into the audit.
The results therefore support coherence of the frozen decisions with independent pathway structure,
while not establishing that any individual pathway claim is biologically correct.

## Figure 3 legend (render-only v2)

**Figure 3. External ontology hierarchy evaluation of frozen audit decisions.** (A) Analysis design.
GO Biological Process and Reactome audit outputs were frozen before evaluating hierarchy edges,
ontology depth, and matched nonedges; hierarchy information did not feed back into the audit. The
primary scope is direct parent-child pairs and the sensitivity scope includes all safe
ancestor-descendant pairs. (B) Directional contradiction among observed hierarchy pairs with Wilson
95% confidence intervals (circles) and the mean of the depth- and evidence-size-matched nonedge
reference (open diamonds). Pdesc denotes a descriptive one-sided matched-reference P value; pathway
pairs are biologically dependent. (C) Fraction of each child's leading-edge genes contained in the
corresponding parent, shown for direct and safe-ancestor scopes. Boxes show the interquartile range,
center lines the median, and whiskers the standard boxplot range. (D) Minimum ontology depth by
audit disposition. Points are individual pathways; boxes show the interquartile range and median.
GO release 2026-07-26; Reactome release 97. Sample sizes are printed in each panel.
