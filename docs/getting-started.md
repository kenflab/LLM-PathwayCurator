<!-- docs/getting-started.md -->

# Getting started
This page gets you from “EA term list” to “audited decision objects” in minutes.

---

## Install
### Option A: PyPI (recommended)
```bash
pip install llm-pathway-curator
```

### Option B: from source

```bash
git clone https://github.com/<ORG>/LLM-PathwayCurator.git
cd LLM-PathwayCurator
pip install .
```

---

## Minimal run (demo)

```bash
llm-pathway-curator run \
  --sample-card examples/demo/sample_card.json \
  --evidence-table examples/demo/evidence_table.tsv \
  --out out/demo/
```

Inspect:

* `out/demo/audit_log.tsv`
* `out/demo/report.md`

---

## Your first real run

You need two inputs:

1. **EvidenceTable** (TSV): `term_id`, `term_name`, `source`, `stat`, `qval`, `direction`, `evidence_genes`
2. **Sample Card** (JSON): structured study context (condition/tissue/perturbation/comparison)

Then:

```bash
llm-pathway-curator run \
  --sample-card sample_card.json \
  --evidence-table evidence_table.tsv \
  --out out/run1/
```

---

## Determinism & provenance

Every run writes `run_meta.json`. Treat it as the “receipt”:

* inputs (paths + hashes when available)
* parameters (τ, k, seed, gates)
* tool version and backend info

For paper-level reproduction, see `paper/README.md` and `paper/FIGURE_MAP.csv`.

---

## Optional: enable LLM proposal mode

LLM-PathwayCurator can use an LLM for proposal-only steps. The audit remains mechanical.

Example (conceptual):

```bash
export OPENAI_API_KEY="..."
export LLMPATH_BACKEND="openai"

llm-pathway-curator run \
  --sample-card sample_card.json \
  --evidence-table evidence_table.tsv \
  --out out/llm/
```

If no API key is provided, the tool should still support a deterministic baseline proposal path.

---

## Build docs locally

From repo root:

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

Open: [http://127.0.0.1:8000](http://127.0.0.1:8000)


---

## Next

* Learn the contracts: **[Concepts](concepts.md)**
* End-to-end usage: **[User guide](user-guide.md)**
* Adapters (inputs → EvidenceTable): see [package](https://github.com/kenflab/LLM-PathwayCurator/tree/main/src/llm_pathway_curator/adapters)
* API docs: **[API reference](api-reference.md)**

---
