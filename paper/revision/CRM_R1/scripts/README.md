# Script map

> Build: `CRM_R1_DISCOVERY48_V3_20260808`

Keep scripts small and numbered by analytical priority. Do not create empty placeholder scripts.

| Prefix | Scope | First script or planned entry point |
| --- | --- | --- |
| `00_` | Input-only preflight | `00_preflight.py` |
| `1x_` | P1 perturbation replication | `10_make_sample_card.py`, `11_discovery_48h.R`, `12_fgsea_48h.R` |
| `2x_` | P2 candidate pool and matched benchmark | `20_freeze_claim_pool.py` |
| `3x_` | P3 external evidence ledger | `30_build_evidence_queries.py` |
| `4x_` | P4 blinded review and agreement | `40_make_blinded_packets.py` |
| `5x_` | P5 ontology and utility robustness | `50_ontology_validation.py` |
| `9x_` | Rendering and final export validation | `90_render_figures.R` |

Priority 1 scripts write beneath
`$CRM_R1_DATA_ROOT/output/priority1/GSE146225_TP53_v1/`; later priorities use their own frozen
benchmark IDs. Plotting scripts only read final source tables and never recompute endpoints.

## Priority 1 reuse map

| New step | Reuse from the repository | Responsibility |
| --- | --- | --- |
| `10_make_sample_card.py` | `paper/scripts/fig2_make_sample_cards.py` | Create one frozen, GSE146225-specific Sample Card with deterministic primary proposal mode |
| `11_discovery_48h.R` | `paper/scripts/figS4_beataml_deg_rank.R` | edgeR filtering, TMM, voom-limma, moderated-t ranking; replace the two-group contrast with the frozen 2 x 2 interaction |
| `12_fgsea_48h.R` | `paper/scripts/fig2_fgsea_to_evidence_table.R` | Hallmark retrieval, ID-overlap checks, deterministic tie handling, `fgseaMultilevel`; write raw fgsea results |
| `llm-pathway-curator adapt --format fgsea` | `src/llm_pathway_curator/adapters/fgsea.py` | Convert raw fgsea output to the validated EvidenceTable contract |
| Mechanical audit | production `llm-pathway-curator run` and orchestration patterns from `paper/scripts/fig2_run_pipeline.py` | Run deterministic distill/modules/claims/audit/report and record run metadata |

The existing figure scripts are dataset-specific and must not be copied wholesale. Reuse their
validated analytical pattern while keeping GSE146225 paths, factorial design, and held-out controls
in thin revision-specific wrappers.

## Executable 48 h steps

Run from the repository root after exporting `CRM_R1_DATA_ROOT` and `R_LIBS_USER`:

```bash
python paper/revision/CRM_R1/scripts/10_make_sample_card.py \
  --data-root "$CRM_R1_DATA_ROOT"

Rscript paper/revision/CRM_R1/scripts/11_discovery_48h.R \
  --data-root "$CRM_R1_DATA_ROOT"

Rscript paper/revision/CRM_R1/scripts/12_fgsea_48h.R \
  --data-root "$CRM_R1_DATA_ROOT"
```

| Script | Primary outputs | Held-out protection |
| --- | --- | --- |
| `10_make_sample_card.py` | `sample_cards/discovery_48h.sample_card.json` and run metadata | Encodes that held-out outcomes have not been calculated |
| `11_discovery_48h.R` | ranking, frozen discovery gene universe, sample/design QC, run metadata, session info | Streams only `GeneID` plus the twelve ENDO 48 h columns |
| `12_fgsea_48h.R` | raw fgsea table, Hallmark snapshot, overlap QC, run metadata, session info | Accepts only the 48 h ranking and never reads expression data |

All scripts refuse to overwrite an existing analytical output unless `--force` is supplied. The
raw fgsea table must then be converted with the production adapter; do not replace that adapter with
a revision-only EvidenceTable implementation.
