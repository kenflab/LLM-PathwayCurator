# Script map

> Build: `CRM_R1_PRIORITY2_FREEZE_V8_20260809`
> Frozen protocol: `CRM_R1_PRIORITY1_v5` (unchanged)

Keep scripts small and numbered by analytical priority. Do not create empty placeholder scripts.

| Prefix | Scope | First script or planned entry point |
| --- | --- | --- |
| `00_` | Input-only preflight | `00_preflight.py` |
| `1x_` | P1 perturbation replication | `10_make_sample_card.py` through `19_evaluate_replication.py` |
| `2x_` | P2 candidate pool and matched benchmark | `20_freeze_claim_pool.py`, `21_check_priority2_freeze.py` |
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
| `16_freeze_priority1_membership.py` | frozen 48 h audit/fgsea outputs and Git state | Write exact τ=0.80 and comparator memberships, the full tau grid, and a SHA-256 freeze manifest; refuse overwrite or dirty tracked code |
| `17_check_priority1_freeze.py` | frozen manifest and recorded files | Verify protocol, membership, code/data hashes, and the pre-72 h gate before validation is released |
| `18_validation_72h.R` | `11_discovery_48h.R`, `12_fgsea_48h.R`, and the freeze checker | Run the same interaction at ENDO 72 h once using the frozen discovery universe and Hallmark snapshot |
| `19_evaluate_replication.py` | frozen membership, 72 h fgsea, and frozen inference specification | Apply the binary endpoint, matched comparisons, intervals, exact reference, continuous analyses, Stop gate, and Figure 4 source export |
| `90_plot_priority1_figure4.py` | frozen `source_data/figure4.tsv` and evaluation metadata | Verify source hashes and render the four-panel PDF/PNG without recalculating any endpoint |
| `llm-pathway-curator adapt --format fgsea` | `src/llm_pathway_curator/adapters/fgsea.py` | Convert raw fgsea output to the validated EvidenceTable contract |
| Mechanical audit | production `llm-pathway-curator run` and orchestration patterns from `paper/scripts/fig2_run_pipeline.py` | Run deterministic distill/modules/claims/audit/report and record run metadata |

The existing figure scripts are dataset-specific and must not be copied wholesale. Reuse their
validated analytical pattern while keeping GSE146225 paths, factorial design, and held-out controls
in thin revision-specific wrappers.

## Priority 2 executable contract

`20_freeze_claim_pool.py` takes two HNSC runs at the inherited canonical `tau = 0.90`: a
deterministic context-off mechanical reference and a same-pool full audit with LLM context review.
It refuses to freeze unless both logs contain the same 50 `entity x direction` claims. Full-audit
PASS count defines K for q-value and mechanical-stability matching. It writes:

```text
$CRM_R1_DATA_ROOT/output/priority2/PANCAN_TP53_v1_HNSC_R1/
  pool/claims.tsv
  membership/selection_membership.tsv
  metrics/risk_coverage_source.tsv
  review/sampling_frame.locked.tsv
  metrics/priority2_freeze_manifest.json
  metrics/priority2_freeze_manifest.sha256
```

`21_check_priority2_freeze.py` verifies every recorded hash, exact row membership, equal matched K,
and the 50-claim blinded-review census. P3 and P4 must pass this gate before reading the pool. The
risk fields remain explicitly pending until independent P3/P4 outcomes are locked.

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

After all four empirical tau runs and the `tau = 0.80` preview are complete, locally commit V5 and
perform the intentional freeze:

```bash
python paper/revision/CRM_R1/scripts/16_freeze_priority1_membership.py \
  --data-root "$CRM_R1_DATA_ROOT" \
  --freeze-label "KF_20260808"

python paper/revision/CRM_R1/scripts/17_check_priority1_freeze.py \
  --data-root "$CRM_R1_DATA_ROOT"
```

After V6 is tested and committed locally, release and evaluate the held-out endpoint once:

```bash
Rscript paper/revision/CRM_R1/scripts/18_validation_72h.R \
  --data-root "$CRM_R1_DATA_ROOT"

python paper/revision/CRM_R1/scripts/19_evaluate_replication.py \
  --data-root "$CRM_R1_DATA_ROOT"

python paper/revision/CRM_R1/scripts/90_plot_priority1_figure4.py \
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
| `16_freeze_priority1_membership.py` | frozen primary/grid memberships and SHA-256 manifest | Reads only 48 h audit/fgsea artifacts and hashes inputs; rejects known 72 h validation outputs |
| `17_check_priority1_freeze.py` | freeze integrity gate | Recomputes recorded hashes and releases 72 h analysis only when the immutable bundle is consistent |
| `18_validation_72h.R` | 72 h ranking/design/pathway statistics and run metadata | Runs the checker first, streams only twelve ENDO 72 h columns, never refilters the frozen universe, and has no force option |
| `19_evaluate_replication.py` | frozen replication metrics, Stop gate summary, and Figure 4 source table | Rechecks freeze integrity in post-validation mode and refuses all output overwrites |
| `90_plot_priority1_figure4.py` | Figure 4 PDF, 600 dpi PNG, and render metadata | Reads only hash-verified frozen figure source and summary tables; no analytical calculation is permitted |

Analytical scripts refuse to overwrite existing outputs unless explicitly documented. Freeze
outputs are stricter: `16_freeze_priority1_membership.py` never overwrites them. The full and
resampled raw fgsea tables are converted by calling the production adapter from
`14_build_empirical_evidence.py`; do not replace it with a revision-only conversion contract. The
V5 protocol freezes `primary_tau = 0.80` after review of all four 48 h calibration runs.
