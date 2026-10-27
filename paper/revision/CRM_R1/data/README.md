# Manuscript source records

These files reproduce the source data supplied with the manuscript supplement.
The 57 exported files are unchanged; seven recorded TCGA Sample Cards are added
from the completed analysis package. `../data_manifest.json` lists all 64 hashes.
This README is documentation and is not part of that fixed scientific input set.

| Directory | Contents |
| --- | --- |
| `Supplementary_Data_1_TCGA` | Seven-cohort enrichment, sample census, source-selection comparison and provenance |
| `Supplementary_Data_2_DEX` | 50 original responses, 24-case sources, anonymous returned categories and summaries |
| `Supplementary_Data_3_External_HNSC` | 50 unchanged statements, three anonymous raters' categories and agreement summaries |
| `Supplementary_Data_4_Comparisons` | Saved split, temporal, cross-study and ontology results with provenance |

Private forms, correspondence and identifying evaluator metadata are not included.
The public evaluator IDs preserve the original pairings; they do not denote
independent biological truth labels. Original `README.txt` and JSON export notes
remain with the data to preserve the export history.

From the repository root:

```bash
python paper/revision/CRM_R1/reproduce.py --outdir out/manuscript_reproduction
```

This verifies the records, recomputes summaries, creates source reports and renders
figures in a new directory. Raw expression matrices and the complete upstream
split-study design are separate dependencies; see the
[reproduction guide](../FIGURE_REPRODUCTION.md).
