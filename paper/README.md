# Manuscript analyses and figures

Start from the repository root after installing the project:

```bash
python paper/revision/CRM_R1/reproduce.py --outdir out/manuscript_reproduction
```

The [reproduction guide](revision/CRM_R1/FIGURE_REPRODUCTION.md) describes the
checks and outputs. The [figure map](FIGURE_MAP.csv) links every main figure,
supplementary figure, table and data set to its inputs and code.

| Resource | Contents |
| --- | --- |
| [Source data](revision/CRM_R1/data/README.md) | Enrichment records, original model text, anonymous returned categories and saved comparison results |
| [Reproduction runner](revision/CRM_R1/reproduce.py) | Source checks, rating summaries, cohort bootstrap summaries and manuscript figures |
| [Figure scripts](revision/CRM_R1/figures/) | Figures 2–3, S1–S4 and numerical table exports |
| [TCGA rebuild](scripts/tcga_rebuild/README.md) | Recorded input snapshot and expression/enrichment rerun instructions |
| [Split-replication code](revision/CRM_R1/archive/split_replication/README.md) | Verified discovery, membership and validation code for the 22-cohort comparison |
| [Research scripts](revision/CRM_R1/scripts/README.md) | Retained experiment-specific runners and protocols |

Figure 1 is author-supplied artwork. Tables describing software contracts are
maintained in the manuscript and supplement. Rendering from saved records does
not refit expression models. The guide identifies the upstream data, environments
needed to repeat the upstream study fits.

## Historical analyses

Earlier outputs and scripts retain their original figure numbers. Use the
[original figure map](revision/CRM_R1/archive/FIGURE_MAP_original.csv) and
[historical recipes](scripts/README.md) for those analyses. `viz_ranked.py` supplied
the original Figure 2e–f ranked visualizations; it remains available as the
`plot-ranked` tool feature. Current manuscript figures use the dedicated scripts
listed above.

For exact pre-reorganization paths, use commit
[`b069e8a`](https://github.com/kenflab/LLM-PathwayCurator/tree/b069e8ad3d916618adcf26af0202748790c3ca20).
Original locks, null results and adverse comparisons remain unchanged.
