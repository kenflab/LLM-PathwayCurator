# Split-replication analysis code

These files document the 22-cohort, 20-split comparison in Figure S4, Table S4A
and Supplementary Data 4. They are separate from the seven-cohort source reports
in Figure 2 and the DEX text checks in Figure 3.

| Stage | File | Recorded identity |
| --- | --- | --- |
| Discovery limma/fgsea | `25c_run_priority2b_discovery_statistics_v14_1_3.R` | SHA-256 matches the discovery run signature |
| Method membership | `28c_run_priority2b_discovery_audit_v14_1_3.py` | SHA-256 matches the audit run signature |
| Membership package | `llm_pathway_curator_e4698b8.zip` | All 24 Python files reproduce the recorded package digest |
| Held-out limma/fgsea | `74d_run_priority2b_validation_v14_1_3.R` | SHA-256 matches the frozen runner and validation signature |

The scripts and package sources retain their original bytes. The package is
from commit [`e4698b8`](https://github.com/kenflab/LLM-PathwayCurator/tree/e4698b836986b10550c626763480678b6766897a/src/llm_pathway_curator).
The membership script was recovered from an author-supplied file rather than
that commit; its content hash matches the execution record.
`PROVENANCE.json` lists file hashes and the package digest algorithm.
`membership_execution.public.json` retains execution settings and source-record
hashes, with workstation paths omitted as documented in that file.

## Selection rules

For each discovery partition, the runner joins all 50 Hallmark terms to their
audit records by term ID and checks agreement of the reported directions.

| Method | Selected terms |
| --- | --- |
| `raw_pool` | All 50 candidate terms |
| `full_audit` | Terms with audit status `PASS`; their count defines K |
| `q_value_matched` | The K terms with the lowest discovery q-values |
| `stability_matched` | The K terms with the highest discovery stability scores |

The matched methods break ties by term ID. The runner requires K > 0 and LLM
context evaluation for every candidate. This matched-K comparison differs from
Figure 2's source-report selection at q ≤ 0.05. The recorded export contains
440 partitions and 88,000 membership rows across the four methods.

## Verify the archived code

From the repository root:

```bash
python paper/revision/CRM_R1/archive/split_replication/verify_code.py
```

This checks the two R runners, Python runner, public execution record and
archived package without importing the historical package or calling a model.

## Reproduce the reported summaries and figures

```bash
python paper/revision/CRM_R1/reproduce.py --outdir out/manuscript_reproduction
```

This recomputes cohort summaries and bootstrap intervals from the supplied split
results and produces the manuscript figures and numerical table exports.

## Requirements for repeating the original fits

The archived runners use the original design, assignments, Hallmark snapshot,
protocols, numerical amendment, expression data and recorded output paths.
The JSON signatures identify the input hashes. The held-out protocol records
R 4.6.1, limma 3.68.4 and fgsea 1.38.0. The membership signature records Python
3.11.9, Ollama 0.33.3 and `llama3.1:8b` model ID `46e0c10c039e`, at temperature 0.

The Python runner imports `v14_1_objective_common.py`. The copy in
`distribution_helpers/` is preserved from its historical distribution; the
execution signature did not hash this separate helper. The package ZIP contains
the 24 hashed Python files, not a complete dependency environment or model cache.
Use the recorded historical package in a separate environment for upstream fits.
The current installed package implements the source-report workflow.

The source hashes identify the execution code. Full refitting also requires the
recorded input archive and environment; the summary reproduction command above
starts from saved study results and needs no new model responses.
