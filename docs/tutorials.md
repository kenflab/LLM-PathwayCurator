## `docs/tutorials.md`

# Tutorials

Short, practical end-to-end recipes.

---

## Tutorial 1 — Run the demo
```bash
llm-pathway-curator run \
  --sample-card examples/demo/sample_card.json \
  --evidence-table examples/demo/evidence_table.tsv \
  --out out/demo/
````

Check:

* `out/demo/audit_log.tsv`
* `out/demo/report.md`

---

## Tutorial 2 — From fgsea to audited claims

1. Run fgsea in your pipeline.
2. Convert results → EvidenceTable (adapter or your own export).
3. Run:

```bash
llm-pathway-curator run \
  --sample-card sample_card.json \
  --evidence-table HNSC.evidence_table.tsv \
  --out out/hnsC/
```

---

## Tutorial 3 — Context swap (stress test)

Run the same EvidenceTable under a swapped Sample Card (e.g., BRCA → LUAD).
Expected behavior: PASS coverage decreases; ABSTAIN reasons shift toward context issues.

(Exact CLI depends on your stress runner / paper scripts. See `paper/scripts/` for reproducible sweeps.)

---

## Tutorial 4 — Evidence dropout (stress test)

Drop supporting genes with probability p (seeded).
Expected behavior: PASS coverage decreases; ABSTAIN reasons shift toward stress-inconclusive / instability.

---

## Tutorial 5 — Pick a τ operating point

Do a τ sweep and summarize:

* PASS/ABSTAIN/FAIL rates
* reason code distribution
* (optional) human-labeled risk among audit-PASS

Then lock τ for your analysis or deployment run.
