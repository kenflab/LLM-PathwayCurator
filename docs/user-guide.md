## `docs/user-guide.md`

# User guide

This guide shows how to use LLM-PathwayCurator on your own enrichment results.

---

## 1) Create an EvidenceTable
You can generate an EvidenceTable via:
- adapters (recommended), or
- manual TSV export if your pipeline already has term × genes.

**Minimum required columns**
- `term_id`, `term_name`, `source`, `stat`, `qval`, `direction`, `evidence_genes`

**Notes**
- `evidence_genes` should be a delimiter-joined list (tool accepts common delimiters; canonical export uses `;`).
- ORA often has `direction=na`. Rank-based EA may have `up/down`.

---

## 2) Create a Sample Card
A Sample Card is structured study context. Keep it explicit and minimal:
- condition / disease
- tissue
- perturbation
- comparison

Use the schema documented in the package docs (and examples).

---

## 3) Run the pipeline
```bash
llm-pathway-curator run \
  --sample-card sample_card.json \
  --evidence-table evidence_table.tsv \
  --out out/run1/
````

---

## 4) Read outputs

### `audit_log.tsv`

Contains:

* decision: PASS / ABSTAIN / FAIL
* reason codes (stable, finite set)
* pointers to evidence identities

### `report.md` / `report.jsonl`

Decision objects for downstream consumption:

* typed claim fields
* evidence links (term/module identifiers + hashes)
* audit outcome and reason codes
* provenance metadata

---

## 5) Tune conservativeness (τ)

τ controls the stability gate operating point.
Conceptually:

* low τ: higher coverage, potentially higher risk
* high τ: lower coverage, more abstention

Use τ sweeps for analysis; lock a τ for deployment.

---

## 6) Optional: enable proposal-only LLM

When enabled, the LLM can:

* choose context-consistent representatives
* emit schema-bounded typed claims

It must never:

* invent evidence
* output free text as “evidence”
* decide PASS/ABSTAIN/FAIL

All decisions remain mechanical and are logged.

---

## 7) Reproducibility checklist

* pin tool version (tag / release)
* record `run_meta.json`
* archive inputs (EvidenceTable + Sample Card)
* prefer Docker / pinned environment for paper matching

