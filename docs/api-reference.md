# API reference

This page documents the **public surface** of LLM-PathwayCurator.  
Most users should start with the **CLI** (`llm-pathway-curator run ...`). The Python API exists for
integration and reproducible orchestration.

---

## CLI

Primary entry point:
- `llm-pathway-curator run ...`

The CLI runs the end-to-end pipeline:

**EvidenceTable → distill → modules → claims → audit → report**

---

## Pipeline

End-to-end orchestration (recommended integration point).

::: llm_pathway_curator.pipeline
    options:
      members:
        - RunConfig
        - run_pipeline
      show_source: false
      show_if_no_docstring: false

---

## Contracts

### EvidenceTable (TSV contract)

`EvidenceTable` is the normalized **term × supporting-genes** table used by all downstream stages.
It is the stability boundary: if the EvidenceTable is valid, distill/modules/select/audit/report
should not break.

::: llm_pathway_curator.schema
    options:
      members:
        - EvidenceTable
      show_source: false
      show_if_no_docstring: false

### Sample Card (study context contract)

The Sample Card is a structured record of study intent/context (e.g., condition/tissue/perturbation/comparison),
used by proposal steps and context validity gates.

::: llm_pathway_curator.sample_card
    options:
      show_source: false
      show_if_no_docstring: false

### Claim schema (typed JSON)

Claims are schema-bounded decision objects with resolvable evidence links (term/module identifiers + hashes).
Free-text narratives are not treated as evidence.

::: llm_pathway_curator.claim_schema
    options:
      show_source: false
      show_if_no_docstring: false

---

## Core stages (A → B → C)

### A) Stability distillation (evidence hygiene)

Generates stability proxies from supporting-gene perturbations (e.g., LOO/jackknife-like survival scores).
This stage does **not** decide PASS/ABSTAIN/FAIL.

::: llm_pathway_curator.distill
    options:
      show_source: false
      show_if_no_docstring: false

### B) Evidence modules (term–gene factorization)

Constructs the term–gene bipartite graph and extracts evidence modules that preserve shared vs distinct support.
This stage does **not** decide PASS/ABSTAIN/FAIL.

::: llm_pathway_curator.modules
    options:
      show_source: false
      show_if_no_docstring: false

### C1) Proposal (deterministic baseline / LLM proposal-only)

Proposes **typed, evidence-linked** candidate claims from distilled evidence and modules.
Final acceptance is not decided here.

::: llm_pathway_curator.select
    options:
      show_source: false
      show_if_no_docstring: false

::: llm_pathway_curator.llm_claims
    options:
      show_source: false
      show_if_no_docstring: false

### C2) Mechanical audit (decider)

Assigns PASS/ABSTAIN/FAIL with precedence (FAIL > ABSTAIN > PASS) using predefined audit gates.
Produces standardized reason codes and audit logs.

::: llm_pathway_curator.audit
    options:
      show_source: false
      show_if_no_docstring: false

::: llm_pathway_curator.audit_reasons
    options:
      show_source: false
      show_if_no_docstring: false

### C3) Reporting (decision-grade outputs)

Writes decision objects (`report.jsonl` / `report.md`) and renders audit logs with provenance.

::: llm_pathway_curator.report
    options:
      show_source: false
      show_if_no_docstring: false

---

## Backends (proposal-only LLM)

LLM backends are used only for proposal steps (representative selection + typing) when enabled.
Backends should support deterministic settings where possible and persist prompt/raw/meta artifacts.

::: llm_pathway_curator.backends
    options:
      show_source: false
      show_if_no_docstring: false

---

## Adapters (Input → EvidenceTable)

Adapters normalize upstream enrichment outputs into the EvidenceTable contract.
They are intentionally conservative: preserve evidence identity (term × genes), avoid destructive parsing,
and keep TSV round-trips stable.

::: llm_pathway_curator.adapters.fgsea
    options:
      members:
        - FgseaAdapterConfig
        - read_fgsea_table
        - fgsea_to_evidence_table
        - convert_fgsea_table_to_evidence_tsv
      show_source: false
      show_if_no_docstring: false

::: llm_pathway_curator.adapters.metascape
    options:
      members:
        - MetascapeAdapterConfig
        - read_metascape_table
        - metascape_to_evidence_table
        - convert_metascape_table_to_evidence_tsv
      show_source: false
      show_if_no_docstring: false

---

## Calibration (risk–coverage)

Utilities for selecting an operating point (e.g., τ) along the risk–coverage trade-off.
This stage does not change evidence identity; it tunes conservativeness.

::: llm_pathway_curator.calibrate
    options:
      show_source: false
      show_if_no_docstring: false

---

## Shared utilities (spec-level)

Spec-critical helpers for contract stability (NA handling, gene parsing/joining, stable hashes).
If you need to compare outputs across versions, this is the layer that prevents drift.

::: llm_pathway_curator._shared
    options:
      show_source: false
      show_if_no_docstring: false

---

## Noise modules (gene noise dictionaries)

Curated gene-noise patterns used by masking/evidence hygiene steps.

::: llm_pathway_curator.noise_lists
    options:
      show_source: false
      show_if_no_docstring: false
```
