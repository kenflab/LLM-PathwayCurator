# Manuscript research workspace

The [reproduction guide](FIGURE_REPRODUCTION.md) covers Figures 1–3, Figures S1–S4,
Tables S1–S6 and Supplementary Data 1–4. Run from the repository root:

```bash
python paper/revision/CRM_R1/reproduce.py --outdir out/manuscript_reproduction
```

The public tool is documented in the [repository README](../../../README.md).

| Path | Contents |
| --- | --- |
| `reproduce.py` | Verifies supplied inputs, recomputes summaries and creates source reports and figures |
| `data/`, `data_manifest.json` | Saved source records and their checksums |
| `figures/` | Manuscript and supplementary figure renderers |
| `scripts/`, `config/` | Research runners and versioned analysis protocols |
| `experiments/external_dex/` | Cross-study expression comparison and its original design |
| `experiments/reassessment/` | Checks of unchanged saved model text |
| `experiments/submission/` | Historical source assembly and saved-result reporting |
| `experiments/revision_tools/` | Historical research prototypes |
| `tests/`, `fixtures/` | Research-specific tests and synthetic controls |
| `archive/` | Historical maps, version records and analysis notes |

Research scripts are outside the installed `llm_pathway_curator` package. Returned
categories remain linked to the exact statements evaluated. Frozen comparisons
retain their original methods and outcomes; current source selection is reported
separately. The reproduction runner writes a new output directory and records
input, implementation and output hashes.

Private forms, correspondence and manuscript drafts remain outside public Git.
The public data folder contains the anonymous categories and source records
provided with the supplement. Historical version identifiers remain in analysis
protocols because they identify the method used. For exact historical paths,
use commit [`b069e8a`](https://github.com/kenflab/LLM-PathwayCurator/tree/b069e8ad3d916618adcf26af0202748790c3ca20).
