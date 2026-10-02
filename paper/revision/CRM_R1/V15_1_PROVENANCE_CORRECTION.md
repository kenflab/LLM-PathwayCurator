# V15.1: legacy provenance correction and development reporting checks

Status: implementation and engineering tests complete; revised biological validation not performed.

## Confirmed evidence and figure mapping

The supplied BEATAML and HNSC legacy audit logs, normalized cards, run metadata,
and ranked tables were examined together. All 50 context-fit values in each ranked
table are reproduced by the historical SHA256 context payload. E × S × that hash
reproduces all 50 utility values in each table within floating-point tolerance.
Both run records specify proxy review, disabled LLM proposal/review, and no attached
LLM backend. `context_score` and `context_confidence` also reproduce the same hash;
renaming or prioritizing those columns would not remove the proxy.

The repository's `paper/FIGURE_MAP.csv` declares the following dependencies:

| Historical figure | Input/output relationship | Interpretation/action |
|---|---|---|
| Fig. 2e–f, HNSC | `fig/Fig2/claims_ranked.tsv` to packed-circle/bar plots | Utility order, circle sizes and module sums depend on hash C; do not label C as measured biological context fit. |
| Extended Data Fig. 4e–f, BEATAML | `fig/FigS4/claims_ranked.tsv` to packed-circle/bar plots | Same direct dependency. |
| Historical Fig. 2a–d / EDFig4a–d | Aggregated audit statuses or reason codes; not ranked utility | Removing C from ranking does not correct proxy-dependent status. Inventory run metadata by variant/tau before making LLM/biological claims. |
| EDFig3 and other legacy runs | Different directories and run identities | Not certified by the two supplied legacy runs; inspect their own metadata/cache. |
| CRM_R1 objective Fig. 2 V14.1.3 | Separately frozen 440-partition membership/outcome join | Not replaced or recomputed by this package; membership used audit PASS, not ranked utility. |

The figure map is a declared repository dependency, not proof of the exact PDF
bytes assembled into the submitted manuscript. The original ranking command and
final submission assembly are not recovered. Do not conflate old Fig. 2 with the
new objective CRM_R1 Fig. 2.

The old plotter defaults to PASS-only display. Merely setting C=1 would leave
proxy-gated selection in place. The new E*S diagnostic keeps all supplied rows,
with no decision filter. It is a stored-component sensitivity, not a corrected
full-audit method, new biological result, or replacement publication figure.

## New entry points

1. `95_reconstruct_legacy_proxy_v15_1.py`: reconstruct hash C, check utility and
   normalized-card provenance, and extract the declared figure-map dependencies.
2. `96_verify_llm_audit_v15_1.py`: require active recorded LLM mode/backend, a plain
   HTTP URL, expected model name, full candidate coverage, normalized-input hashes,
   and a one-to-one raw-cache join. Reject errors, missing reasons, invalid confidence,
   status disagreements, missing/extraneous/duplicate keys, and proxy fallback.
   Preserve full cache reasons. A valid K=0 receives a distinct technical status;
   it is not permission to retry or bypass V14.1.3's frozen zero-K policy.
3. `97_build_stored_es_diagnostic_v15_1.py`: explicit E and S columns, finite ranges,
   deterministic ties, all input rows, no context factor or implicit missing-to-one.
   A new manifest records inputs, script hashes and output hashes, including optional
   development PDF/PNG. No legacy plotter fallback is used.
4. `98_check_structured_report_v15_1.py`: a separate deterministic reporting contract.
   Check cohort, comparison, direction, term/evidence identity, exact q-value transfer,
   gene linkage and explicit FDR-significance assertion. Context plausibility,
   biological correctness and free-text overstatement are explicitly NOT_ASSESSED.

Script 96 verifies internal consistency of supplied artifacts, not the authenticity
of a remote model or calibration of its confidence. It does not verify a model-weight
digest, recover missing prompts, or turn self-reported confidence into a utility factor.
It is a new opt-in gate; it is not retroactively inserted into old entry points.
Future orchestration must call it successfully before treating its outputs as verified.

Script 98 requires canonical evidence from a trusted independent data source.
Putting a model's own values on both sides would not be validation. The contract is
deliberately exact: no guessed aliases, rounded-q tolerance, or inference from prose.

## Migration and manuscript actions

- Keep all old files, manifests, thresholds, memberships and negative results.
- Treat the two old hash-driven examples as deterministic engineering demonstrations
  only, if retained. Correct any legend describing their C as biological context fit
  or their proxy behavior as an actual LLM assessment.
- Record both the scoring and selection provenance when replacing a demonstration.
  A new actual-LLM run is a new experiment; no automatic rerun is included here.
- Do not use the E*S diagnostic to claim biological superiority or to erase the
  original full-audit result.
- The V14.1.3 primary result remains 43.1% full-audit replication versus 61.3% q-value
  matched and 50.4% stability matched. Full audit underperformed both comparators
  on that endpoint. The legacy utility correction does not explain this away.
- The earlier V15 explicit E*S*C development builder accepted declared provenance;
  it should not be used to certify biological C. This package supplies no new C.

Only new versioned files are installed. `pipeline.py`, `ranked.py`, `viz_ranked.py`,
V14 runners and AGENTS.md are not modified or recreated. Therefore old source locks
retain their original scope. These tools do not silently change existing CLI behavior.
