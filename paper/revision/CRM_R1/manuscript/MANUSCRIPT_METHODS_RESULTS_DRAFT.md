# Manuscript integration guide

Earlier text in this file mixed completed results with planned literature and
rating analyses. Use the source assembly output for current, explicitly scoped
draft modules. The author's current R1 Word remains authoritative and is not
edited by the assembly program.

```bash
python paper/revision/CRM_R1/experiments/submission/assemble.py \\
  --data-root "$CRM_R1_DATA_ROOT"
```

The private output contains `RESULTS_AND_DISCUSSION_DRAFT_EN.txt`, the current
Word and readable body text, source tables, `RESPONSE_MATRIX.private.tsv`,
and a claim-review list. Insert final figure/page references only after the
actual manuscript and assembled figures have been checked.

## Results modules and scope

| Module | Appropriate result | Boundary |
| --- | --- | --- |
| P1 temporal perturbation | Empirical sample-resampling score and fixed temporal replication comparisons | Same-study time point; binary superiority is not established |
| P2B multi-cohort comparison | Cohort-level full-minus-baseline contrasts, including adverse results | Historical full audit; splits and pathways are not independent sample units |
| R06 existing assessments | Original statements, separate raters, agreement, unknown categories and uncertainty | Low agreement is not evidence of biological truth; no label transfer to new prose |
| R07 natural prose | First-output census, explicit-number check coverage and simple source template | Numeric completeness is not semantic accuracy; preserve the original checker |
| External dex | Statistical donor-stability component and primary cross-study culture-level endpoint | Same selected terms as matched q; one validation cell line |
| P5 ontology | Hierarchy pairs, gene overlap, depth and NES direction | Directional discordance is not automatically a biological contradiction |

## Methods requirements

Separate original designs, technical amendments and post hoc analyses. Report
actual software versions and input/output identities, measured gene universes,
contrasts, selection denominators, incomplete records and the unit of analysis.
Do not treat a source template or internally designed stress test as an
independent accuracy standard. Preserve unfavorable and null comparisons.

Do not include an independent PubMed-support result until the required grading
has been completed, authenticated and locked. Retrieval counts and blank grades
do not supply this endpoint. The assembler only inventories grading completeness.

## Novelty and language

Cite Khan et al. (2025), PMID 41071041, doi:10.1093/bioinformatics/btaf541,
for prior LLM screening with literature-grounded verification. Its Methods
uses Phi-4 and GPT-o3-mini as faithfulness evaluators; describe that work
accurately. Focus the proposed contribution on pathway-specific typed evidence,
source linkage and deterministic disposition rules. Do not claim invention of
proposal–verification separation or validated biological/decision accuracy.

https://pmc.ncbi.nlm.nih.gov/articles/PMC12548045/
