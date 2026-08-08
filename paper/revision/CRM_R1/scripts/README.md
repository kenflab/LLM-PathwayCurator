# Script map

> Build: `CRM_R1_EMPIRICAL81_V4_20260808`

Keep scripts small and numbered by analytical priority. Do not create empty placeholder scripts.

| Prefix | Scope | First script or planned entry point |
| --- | --- | --- |
| `00_` | Input-only preflight | `00_preflight.py` |
| `1x_` | P1 perturbation replication | `10_make_sample_card.py` through `15_preview_empirical_membership.py` |
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
| `13_resample_discovery_48h.R` | `11_discovery_48h.R` and `12_fgsea_48h.R` | Exhaustively delete one sample per factorial cell, refit 81 discovery-only models, and rerun fgsea against frozen inputs |
| `14_build_empirical_evidence.py` | production `adapters/fgsea.py` | Adapt the full baseline and every resample, then add `replicate_id` to form the production `replicates_proxy` input |
| `15_preview_empirical_membership.py` | production Claim schema and audit outputs | Validate the tau grid, monotone membership, context-off invariants, and matched-method previews without freezing |
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

Rscript paper/revision/CRM_R1/scripts/13_resample_discovery_48h.R \
  --data-root "$CRM_R1_DATA_ROOT"

python paper/revision/CRM_R1/scripts/14_build_empirical_evidence.py \
  --data-root "$CRM_R1_DATA_ROOT"
```

| Script | Primary outputs | Held-out protection |
| --- | --- | --- |
| `10_make_sample_card.py` | `sample_cards/discovery_48h.sample_card.json` and run metadata | Encodes that held-out outcomes have not been calculated |
| `11_discovery_48h.R` | ranking, frozen discovery gene universe, sample/design QC, run metadata, session info | Streams only `GeneID` plus the twelve ENDO 48 h columns |
| `12_fgsea_48h.R` | raw fgsea table, Hallmark snapshot, overlap QC, run metadata, session info | Accepts only the 48 h ranking and never reads expression data |
| `13_resample_discovery_48h.R` | 81-run manifest, long fgsea table, QC, run metadata, session info | Streams only the twelve ENDO 48 h columns and uses the frozen full-discovery universe |
| `14_build_empirical_evidence.py` | replicate-stacked EvidenceTable and run metadata | Reads only full and resampled 48 h fgsea artifacts |
| `15_preview_empirical_membership.py` | calibration and matched-membership preview tables | Reads only 48 h audit and fgsea artifacts; never freezes a choice |

All scripts refuse to overwrite an existing analytical output unless `--force` is supplied. The
Full and resampled raw fgsea tables are converted by calling the production adapter from
`14_build_empirical_evidence.py`; do not replace it with a revision-only conversion contract. The
V4 protocol keeps `primary_tau` null until all four 48 h calibration runs have been reviewed.
