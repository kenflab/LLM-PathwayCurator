<!-- docs/index.md -->

# LLM-PathwayCurator

<p align="left">
  <img src="assets/LLM-PathwayCurator_logo.png" width="90" alt="LLM-PathwayCurator"
       style="vertical-align: middle; margin-right: 10px;">
  <span style="font-size: 28px; font-weight: 700; vertical-align: middle;">
     Enrichment interpretations → audited, decision-grade pathway claims.
  </span>
</p>

LLM-PathwayCurator is a **quality-assurance (QA) layer** for enrichment analysis (EA) interpretation.  
It **does not** introduce a new enrichment statistic. Instead, it turns EA outputs (ranked term lists) into **evidence-linked, typed claims** and assigns **PASS/ABSTAIN/FAIL** via a **mechanical audit suite**.

**Core promise:** we **abstain** when claims are unstable, under-supported, contradictory, or context-nonspecific.

<p align="center">
  <img src="assets/LLM-PathwayCurator_Fig1_bioRxiv_2026.png" width="88%"
       alt="EvidenceTable → distill → modules → claims → audits → report">
</p>

---

## Why this exists (the practical pain)
Enrichment tools return ranked term lists. In practice, interpretation becomes non-reproducible because:

- representative terms are ambiguous under the **study context**
- gene support is opaque → **cherry-picking risk**
- related terms share / bridge evidence in non-obvious ways
- there is no mechanical stop condition for fragile narratives

LLM-PathwayCurator converts “plausible narratives” into **auditable decision objects**.

---

## What you get
- **EvidenceTable**: term × supporting-genes contract (works for ORA and rank-based EA)
- **Evidence distillation**: supporting-gene perturbations → stability proxies (survival-like scores)
- **Evidence modules**: factorization of the term–gene graph (shared vs distinct evidence)
- **Typed claims (JSON)**: schema-bounded, evidence-linked (no free text required)
- **Mechanical audits**: predefined gates → PASS/ABSTAIN/FAIL + reason codes
- **Decision-grade report**: audit log + provenance + reproducible outputs

---

## Quick start
```bash
pip install llm-pathway-curator

llm-pathway-curator run \
  --sample-card examples/demo/sample_card.json \
  --evidence-table examples/demo/evidence_table.tsv \
  --out out/demo/
```

Key outputs:
* `audit_log.tsv` (PASS/ABSTAIN/FAIL + reason codes)
* `report.md`, `report.jsonl` (decision objects)
* `distilled.tsv`, `modules.tsv`, `term_modules.tsv`, `term_gene_edges.tsv`
* `run_meta.json` (pinned parameters + provenance)

---

## Next

* Start here: **[Getting started](getting-started.md)**
* Learn the contracts: **[Concepts](concepts.md)**
* End-to-end usage: **[User guide](user-guide.md)**
* Adapters (inputs → EvidenceTable): see [package](https://github.com/kenflab/LLM-PathwayCurator/tree/main/src/llm_pathway_curator/adapters)
* API docs: **[API reference](api-reference.md)**

---

## Notes

* **LLM is proposal-only (optional)**: representative selection + typing.
* **Acceptance is never delegated**: PASS/ABSTAIN/FAIL is decided by **mechanical audits**.
* **Counterfactual stress tests are internal** (e.g., context swap, evidence dropout): no external knowledge required.

