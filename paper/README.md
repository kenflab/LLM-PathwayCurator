# Paper artifacts (reproducible figures)

This folder contains minimal scripts + frozen outputs used to reproduce paper figures.

## Fig2 (risk–coverage)

### What Fig2 measures (v1)
- **Coverage** = PASS rate (tau-swept)
- **Risk** = human **REJECT** rate within PASS set (**R1_only**)
- Inputs are **report.jsonl** (per cancer/method) + **labels.tsv**.

### Folder layout (v1)
- Inputs (generated):
  - `paper/source_data/PANCAN_TP53_v1/evidence_tables/*.evidence_table.tsv`
  - `paper/source_data/PANCAN_TP53_v1/sample_cards/*.sample_card.json`
- Outputs:
  - `paper/source_data/PANCAN_TP53_v1/out/<CANCER>/<METHOD>/report.jsonl`
  - `paper/source_data/PANCAN_TP53_v1/risk_coverage.tsv`
  - `paper/source_data/PANCAN_TP53_v1/fig/*.pdf`

### Reproduce Fig2 (smoke → full)

From repo root:

```bash
# 0) (optional) start containers (Jupyter + Ollama + MSBio)
docker compose -f docker/docker-compose.yml up -d

# 1) fetch raw inputs (Synapse; requires login/token)
python paper/scripts/fig2_fetch_inputs.py

# 2) build TP53 mut/wt groups from MC3
python paper/scripts/fig2_make_groups.py

# 3) generate sample cards (template per cancer)
python paper/scripts/fig2_make_sample_cards.py

# 4) DEG ranking + fgsea → EvidenceTable (start with one cancer)
Rscript paper/scripts/fig2_deg_rank.R HNSC
Rscript paper/scripts/fig2_fgsea_to_evidence_table.R HNSC

# 5) run pipeline to produce report.jsonl (ours + shuffled_context)
python paper/scripts/run_fig2_pipeline.py --cancers HNSC --variants ours,shuffled_context

# 6) tau-sweep risk–coverage (requires labels.tsv)
python paper/scripts/bench_fig2.py \
  --reports paper/source_data/PANCAN_TP53_v1/out/HNSC \
  --labels  paper/source_data/PANCAN_TP53_v1/labels.tsv \
  --out     paper/source_data/PANCAN_TP53_v1/risk_coverage.tsv \
  --taus    0:1:0.05

# 7) plots
python paper/scripts/fig2_plot.py \
  --in  paper/source_data/PANCAN_TP53_v1/risk_coverage.tsv \
  --out paper/source_data/PANCAN_TP53_v1/fig/Fig2_HNSC.pdf
python paper/scripts/fig2_plot_multipanel.py \
  --in  paper/source_data/PANCAN_TP53_v1/risk_coverage.tsv \
  --out paper/source_data/PANCAN_TP53_v1/fig/Fig2_Supp_multipanel.pdf
```