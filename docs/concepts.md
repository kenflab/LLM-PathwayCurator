## `docs/concepts.md`

# Concepts

LLM-PathwayCurator is an **interpretation QA layer** for enrichment analysis (EA).  
It transforms EA outputs into **audited, decision-grade claims**.

---

## What it is (and is not)
**It is:**
- a framework to convert term lists → **typed, evidence-linked claims**
- a **mechanical audit suite** producing PASS/ABSTAIN/FAIL with reason codes
- a way to tune conservativeness via **risk–coverage** (abstention is a feature)

**It is not:**
- a new enrichment statistic
- a free-text summarizer
- a biological truth oracle (it audits internal consistency and evidence linkage)

---

## Objects
### EvidenceTable (term × gene contract)
One row = one enriched term with explicit supporting genes.
This enables:
- term–term overlap (e.g., Jaccard)
- term–gene bipartite graph construction
- evidence factorization (modules)
- stable evidence linkage (hashable gene sets)

### Sample Card (study context contract)
A structured record of study intent and context (e.g., condition/tissue/perturbation/comparison).
Used for:
- context-conditioned representative selection
- context validity gates
- context stress tests (e.g., context swap)

### Claim (typed JSON; evidence-linked)
A claim is a **decision object**, not prose.
It must contain resolvable references:
- `term_id` / `module_id`
- supporting-gene set identity (hash)
- typed fields (schema-bounded)

---

## Pipeline responsibilities (A → B → C)
### A) Distill (stability distillation; “evidence hygiene”)
- supporting-gene perturbations (seeded dropout / jitter)
- survival-like stability proxies (LOO/jackknife, optional extras)
- **does not decide PASS/ABSTAIN/FAIL**

### B) Modules (evidence factorization)
- build term–gene graph
- extract evidence modules (shared vs distinct support)
- attach module ids / summarize structure
- **does not decide PASS/ABSTAIN/FAIL**

### C) Claims → Audit → Report
**C1 (proposal):** select representatives + type claims (LLM optional)  
**C2 (audit):** mechanical gates assign PASS/ABSTAIN/FAIL + reason codes  
**C3 (report):** decision-grade report + provenance

---

## Decisions
### PASS / ABSTAIN / FAIL
- **FAIL**: auditable violations (evidence drift, contradictions, schema violations)
- **ABSTAIN**: under-supported / unstable / context-nonspecific / stress-inconclusive
- **PASS**: survives the predefined gate suite

### τ (stability threshold) as an operating point
Higher τ → more conservative (more ABSTAIN, less PASS).  
This enables a **risk–coverage trade-off**.

---

## Stress tests (internal counterfactuals)
Stress tests are *specification-driven perturbations* (no external knowledge):
- **context swap**: swap Sample Card context keys
- **evidence dropout**: remove supporting genes with probability p

Expected outcome: coverage should decrease and ABSTAIN reasons should shift in a stress-specific way.

