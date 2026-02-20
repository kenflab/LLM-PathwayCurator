# LLM-PathwayCurator

<p>
  <img src="https://raw.githubusercontent.com/kenflab/LLM-PathwayCurator/main/docs/assets/LLM-PathwayCurator_logo.png" width="90" alt="LLM-PathwayCurator">
</p>
<h2>Enrichment interpretations → audit-gated, decision-grade claims.</h2>

[![Docs](https://img.shields.io/badge/docs-latest-blue?style=flat&logo=readthedocs&logoColor=white)](https://llm-pathwaycurator.readthedocs.io/)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-preprint-B31B1B.svg)](https://doi.org/10.64898/2026.02.18.706381)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18625777-6f42c1?style=flat&logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.18625777)
[![RRID](https://img.shields.io/badge/RRID-SCR_027964-2ea44f?style=flat)](https://scicrunch.org/resolver/SCR_027964)
[![License: MIT](https://img.shields.io/badge/License-MIT-f1c40f?style=flat)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11%2B-555555?style=flat&logo=python&logoColor=white)](https://www.python.org/downloads/)

- **Docs:** [https://llm-pathwaycurator.readthedocs.io/](https://llm-pathwaycurator.readthedocs.io/)
- **Paper reproducibility (canonical):** [`paper/`](https://github.com/kenflab/LLM-PathwayCurator/tree/main/paper) (see [`paper/README.md`](https://github.com/kenflab/LLM-PathwayCurator/blob/main/paper/README.md); panel map in [`paper/FIGURE_MAP.csv`](https://github.com/kenflab/LLM-PathwayCurator/blob/main/paper/FIGURE_MAP.csv))
- **Cite:** bioRxiv preprint (DOI: [10.64898/2026.02.18.706381](https://doi.org/10.64898/2026.02.18.706381)).

---

## 🚀 What this is

**LLM-PathwayCurator is an interpretation quality-assurance (QA) layer for enrichment analysis.**  
It **does not** introduce a new enrichment statistic. Instead, it turns EA outputs into **auditable decision objects**:

- **Input:** enrichment term lists from ORA (e.g., Metascape) or rank-based enrichment (e.g., fgsea, an implementation of the GSEA method)
- **Output:** **typed, evidence-linked claims** + **PASS/ABSTAIN/FAIL** decisions + **reason-coded audit logs**
- **Promise:** we **abstain** when claims are **unstable**, **under-supported**, **contradictory**, or **context-nonspecific**

> **Selective prediction for pathway interpretation:** calibrated abstention is a feature, not a failure.

<p align="center">
  <img src="https://raw.githubusercontent.com/kenflab/LLM-PathwayCurator/main/docs/assets/LLM-PathwayCurator_Fig1_bioRxiv_2026.png" width="85%"
       alt="LLM-PathwayCurator workflow: EvidenceTable → modules → claims → audits">
</p>


<p align="center">
  <em style="max-width: 600px; display: inline-block; line-height: 1.6;">
    Fig. 1a. Overview of LLM-PathwayCurator workflow:<br>
    <strong>EvidenceTable</strong> → <strong>modules</strong> → <strong>claims</strong> → <strong>audits</strong>
    (<a href="https://doi.org/10.64898/2026.02.18.706381">bioRxiv preprint</a>)
  </em>
</p>

---

## 🧭 Why this is different (and why it matters)

Enrichment tools return ranked term lists. In practice, interpretation breaks because:

1) **Representative terms are ambiguous** under study context  
2) **Gene support is opaque**, enabling cherry-picking  
3) **Related terms share / bridge evidence** in non-obvious ways  
4) There is **no mechanical stop condition** for fragile narratives

**LLM-PathwayCurator replaces narrative endorsement with audit-gated decisions.**  
We transform ranked terms into **machine-auditable claims** by enforcing:

- **Evidence-linked constraints:** claims must resolve to valid term/module identifiers and supporting-gene evidence
- **Stability audits:** supporting-gene perturbations yield stability proxies (operating point: **τ**)
- **Context validity stress tests:** context swap reveals context dependence without external knowledge
- **Contradiction checks:** internally inconsistent claims fail mechanically
- **Reason-coded outcomes:** every decision is explainable by a finite audit code set

---

## 🔍 What this is not

- Not an enrichment method; it **audits** enrichment outputs.
- Not a free-text summarizer; **claims are schema-bounded** (typed JSON; no narrative prose as “evidence”).
- Not a biological truth oracle; it checks **internal consistency and evidence integrity**, not mechanistic truth.

---

## 🧩 Core pipeline (A → B → C)

**A) Stability distillation (evidence hygiene)**  
Perturb supporting genes (seeded) to compute stability proxies (e.g., LOO/jackknife-like survival scores).  
Output: [`distilled.tsv`](https://github.com/kenflab/LLM-PathwayCurator/blob/main/examples/demo/expected/distilled.tsv)

**B) Evidence factorization (modules)**  
Factorize the term–gene bipartite graph into **evidence modules** that preserve shared vs distinct support.  
Outputs: [`modules.tsv`](https://github.com/kenflab/LLM-PathwayCurator/blob/main/examples/demo/expected/modules.tsv), [`term_modules.tsv`](https://github.com/kenflab/LLM-PathwayCurator/blob/main/examples/demo/expected/term_modules.tsv), [`term_gene_edges.tsv`](https://github.com/kenflab/LLM-PathwayCurator/blob/main/examples/demo/expected/term_gene_edges.tsv)

**C) Claims → audit → report**  
- **C1 (proposal-only):** deterministic baseline or optional LLM proposes **typed claims** with resolvable evidence links  
- **C2 (audit/decider):** mechanical rules assign **PASS/ABSTAIN/FAIL** with precedence (FAIL > ABSTAIN > PASS)  
- **C3 (report):** decision-grade report + audit log ([`audit_log.tsv`](https://github.com/kenflab/LLM-PathwayCurator/blob/main/examples/demo/expected/audit_log.tsv)) + provenance

---

## ⚡ Quick start (library entrypoint)

```bash
llm-pathway-curator run \
  --sample-card examples/demo/sample_card.json \
  --evidence-table examples/demo/evidence_table.tsv \
  --out out/demo/
````

### Key outputs (stable contract)

* [`audit_log.tsv`](https://github.com/kenflab/LLM-PathwayCurator/blob/main/examples/demo/expected/audit_log.tsv) — PASS/ABSTAIN/FAIL + reason codes (mechanical)
* [`report.jsonl`](https://github.com/kenflab/LLM-PathwayCurator/blob/main/examples/demo/expected/report.jsonl), [`report.md`](https://github.com/kenflab/LLM-PathwayCurator/blob/main/examples/demo/expected/report.md) — decision objects (evidence-linked)
* [`claims.proposed.tsv`](https://github.com/kenflab/LLM-PathwayCurator/blob/main/examples/demo/expected/claims.proposed.tsv) — proposed candidates (proposal-only; auditable)
* [`distilled.tsv`](https://github.com/kenflab/LLM-PathwayCurator/blob/main/examples/demo/expected/distilled.tsv) — stability proxies / evidence hygiene outputs
* [`modules.tsv`](https://github.com/kenflab/LLM-PathwayCurator/blob/main/examples/demo/expected/modules.tsv), [`term_modules.tsv`](https://github.com/kenflab/LLM-PathwayCurator/blob/main/examples/demo/expected/term_modules.tsv), [`term_gene_edges.tsv`](https://github.com/kenflab/LLM-PathwayCurator/blob/main/examples/demo/expected/term_gene_edges.tsv) — evidence structure
* [`run_meta.json`](https://github.com/kenflab/LLM-PathwayCurator/blob/main/examples/demo/expected/run_meta.json) (+ optional `manifest.json`) — pinned params + provenance

---

## 📊 Rank & visualize ranked terms (`rank` / `plot-ranked`)

LLM-PathwayCurator includes two small post-processing commands for **ranking** and **publication-ready visualization**
of ranked terms/modules:

- [`llm-pathway-curator rank`](https://github.com/kenflab/LLM-PathwayCurator/blob/main/src/llm_pathway_curator/ranked.py) — produces a **ranked table** (`claims_ranked.tsv`) for downstream plots and summaries.
- [`llm-pathway-curator plot-ranked`](https://github.com/kenflab/LLM-PathwayCurator/blob/main/src/llm_pathway_curator/viz_ranked.py) — renders ranked terms/modules as either:
  - **bars** (Metascape-like horizontal bars), or
  - **packed circles** (module-level circle packing with term circles inside).

### A) Rank (produce `claims_ranked.tsv`)

Use `rank` to generate a deterministic ranked table from a run output directory.

```bash
llm-pathway-curator rank --help
# Typical workflow: point rank to a run directory and write claims_ranked.tsv
# (See --help for the exact flags supported by your installed version.)
````

### B) Plot (bars or packed circles)

`plot-ranked` auto-detects `claims_ranked.tsv` (recommended) or falls back to `audit_log.tsv`
under `--run-dir`.

> Packed circles require an extra dependency:
> `python -m pip install circlify`

#### Bars (Metascape-like)

```bash
llm-pathway-curator plot-ranked \
  --mode bars \
  --run-dir out/demo \
  --out-png out/demo/plots/ranked_bars.png \
  --decision PASS \
  --group-by-module \
  --left-strip \
  --strip-labels \
  --bar-color-mode module
```

#### Packed circles (modules → terms)

```bash
llm-pathway-curator plot-ranked \
  --mode packed \
  --run-dir out/demo \
  --out-png out/demo/plots/ranked_packed.png \
  --decision PASS \
  --term-color-mode module
```

#### Packed circles (direction shading)

```bash
llm-pathway-curator plot-ranked \
  --mode packed \
  --run-dir out/demo \
  --out-png out/demo/plots/ranked_packed.direction.png \
  --decision PASS \
  --term-color-mode direction
```

### Consistent module labels/colors across plots

`plot-ranked` assigns a single module display rank (**M01, M02, ...**) and a stable module color per `module_id`,
so **bars** and **packed circles** can be placed side-by-side without label/color drift.

---

## ⚖️ Inputs (contracts)

### EvidenceTable (minimum required columns)

Each row is one enriched term.

Required columns:

* `term_id`, `term_name`, `source`
* `stat`, `qval`, `direction`
* `evidence_genes` (supporting genes; TSV uses `;` join)

### Sample Card (study context)

Structured context record used for proposal and context gating, e.g.:

* `condition/disease`, `tissue`, `perturbation`, `comparison`

Adapters for common tools live under [`src/llm_pathway_curator/adapters/`](https://github.com/kenflab/LLM-PathwayCurator/tree/main/src/llm_pathway_curator/adapters).

---

## 🔧 Adapters (Input → EvidenceTable)

Adapters are intentionally conservative:

* preserve **evidence identity** (term × genes)
* avoid destructive parsing
* keep TSV **round-trips stable** (contract drift is treated as a bug)

See: [`src/llm_pathway_curator/adapters/README.md`](https://github.com/kenflab/LLM-PathwayCurator/blob/main/src/llm_pathway_curator/adapters/README.md)

---

## 🛡️ Decisions: PASS / ABSTAIN / FAIL

LLM-PathwayCurator assigns decisions by **mechanical audit gates**:

* **FAIL**: auditable violations (evidence-link drift, schema violations, contradictions, forbidden fields, etc.)
* **ABSTAIN**: non-specific, under-supported, or unstable under perturbations / stress tests
* **PASS**: survives all enabled gates at the chosen operating point (**τ**)

**Important:** the LLM (if enabled) never decides acceptance.
It may propose candidates; **the audit suite is the decider**.

---

## 🧪 Built-in stress tests (counterfactuals without external knowledge)

* **Context swap**: shuffle study context (e.g., BRCA → LUAD) to test context dependence
* **Evidence dropout**: randomly remove supporting genes (seeded; min_keep enforced)
* **Contradiction injection** (optional): introduce internally contradictory candidates to test FAIL gates

These are specification-driven perturbations intended to validate that the pipeline
**abstains for the right reasons**, with **stress-specific reason codes**.

---

## ♻️ Reproducibility by default

LLM-PathwayCurator is deterministic by default:

* fixed seeds (CLI + library defaults)
* pinned parsing + hashing utilities
* stable output schemas and reason codes
* run metadata persisted to [`run_meta.json`](https://github.com/kenflab/LLM-PathwayCurator/blob/main/examples/demo/expected/run_meta.json) (and runner-level `manifest.json` when used)

Paper-side runners (e.g., [`paper/scripts/fig2_run_pipeline.py`](https://github.com/kenflab/LLM-PathwayCurator/blob/main/paper/scripts/fig2_run_pipeline.py)) **orchestrate** reproducible sweeps
and do not implement scientific logic; they call the library entrypoint ([`llm_pathway_curator.pipeline.run_pipeline`](https://github.com/kenflab/LLM-PathwayCurator/blob/main/src/llm_pathway_curator/pipeline.py)).

---

## 📦 Installation
[![PyPI](https://img.shields.io/badge/PyPI-llm--pathway--curator-3776AB?style=flat&logo=pypi&logoColor=white)](https://pypi.org/project/llm-pathway-curator/)
[![PyPI version](https://img.shields.io/pypi/v/llm-pathway-curator?style=flat&logo=pypi&logoColor=white)](https://pypi.org/project/llm-pathway-curator/)
[![Docker (GHCR)](https://img.shields.io/badge/ghcr.io-llm--pathway--curator-2496ED?style=flat&logo=github&logoColor=white)](https://github.com/kenflab/LLM-PathwayCurator/pkgs/container/llm-pathway-curator)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)

### Option A: PyPI (recommended)

```bash
pip install llm-pathway-curator
```
  (See PyPI project page: [https://pypi.org/project/llm-pathway-curator/](https://pypi.org/project/llm-pathway-curator/))
  
### Option B: From source (development)

```bash
git clone https://github.com/kenflab/LLM-PathwayCurator.git
cd LLM-PathwayCurator
pip install -e .
```

---

## 🐳 Docker (recommended for reproducibility)
We provide an official Docker environment (Python + R + Jupyter), sufficient to run LLM-PathwayCurator and most paper figure generation.  
Optionally includes **Ollama** for local LLM annotation (no cloud API key required).

- #### Option A: Prebuilt image (recommended)
  Use the published image from GitHub Container Registry (GHCR).
  
  ```bash
  # from the repo root (optional, for notebooks / file access)
  docker pull ghcr.io/kenflab/llm-pathway-curator:official
  ```

  Run Jupyter:
  ```
  docker run --rm -it \
    -p 8888:8888 \
    -v "$PWD":/work \
    -e GEMINI_API_KEY \
    -e OPENAI_API_KEY \
    ghcr.io/kenflab/llm-pathway-curator:official
  ```
  Open Jupyter:
  [http://localhost:8888](http://localhost:8888) <br>  
  (Use the token printed in the container logs.)
  <br>  
  Notes:
  > For manuscript reproducibility, we also provide versioned tags (e.g., :0.1.0). Prefer a version tag when matching a paper release.


- #### Option B: Build locally (development)

  - ##### Option B-1: Build locally with Compose (recommended for dev)
    ```bash
    # from the repo root
    docker compose -f docker/docker-compose.yml build
    docker compose -f docker/docker-compose.yml up
    ```

    **B-1.1) Open Jupyter**
    - [http://localhost:8888](http://localhost:8888) 
      Workspace mount: `/work`

    **B-1.2) If prompted for "Password or token"**
    - Get the tokenized URL from container logs:
      ```bash
      docker compose -f docker/docker-compose.yml logs -f llm-pathway-curator
      ```
    - Then either:
      - open the printed URL (contains `?token=...`) in your browser, or
      - paste the token value into the login prompt.

  - ##### Option B-2: Build locally without Compose (alternative)
    ```bash
    # from the repo root
    docker build -f docker/Dockerfile -t llm-pathway-curator:official .
    ```

    **B-2.1) Run Jupyter**
    ```bash
    docker run --rm -it \
      -p 8888:8888 \
      -v "$PWD":/work \
      -e GEMINI_API_KEY \
      -e OPENAI_API_KEY \
      llm-pathway-curator:official
    ```

    **B-2.2) Open Jupyter**
    - [http://localhost:8888](http://localhost:8888)
      Workspace mount: `/work`  
 

---
## 🖥️ Apptainer / Singularity (HPC)
- #### Option A: Prebuilt image (recommended)
  Use the published image from GitHub Container Registry (GHCR).
  ```bash
  apptainer build llm-pathway-curator.sif docker://ghcr.io/kenflab/llm-pathway-curator:official
  ```

- #### Option B:  a .sif from the Docker image (development)
  ```bash
  docker compose -f docker/docker-compose.yml build
  apptainer build llm-pathway-curator.sif docker-daemon://llm-pathway-curator:official
  ```

Run Jupyter (either image):
  ```bash
  apptainer exec --cleanenv \
    --bind "$PWD":/work \
    llm-pathway-curator.sif \
    bash -lc 'jupyter lab --ip=0.0.0.0 --port=8888 --no-browser 
  ```

---

## 🤖 LLM usage (proposal-only; optional)

If enabled, the LLM is confined to proposal steps and must emit **schema-bounded JSON**
with **resolvable EvidenceTable links**.

Backends (example):

* Ollama: `LLMPATH_OLLAMA_HOST`, `LLMPATH_OLLAMA_MODEL`
* Gemini: `GEMINI_API_KEY`
* OpenAI: `OPENAI_API_KEY`

Typical environment:

```bash
export LLMPATH_BACKEND="ollama"   # ollama|gemini|openai
```

Deterministic settings are used by default (e.g., temperature=0), and runs persist
prompt/raw/meta artifacts alongside [`run_meta.json`](https://github.com/kenflab/LLM-PathwayCurator/blob/main/examples/demo/expected/run_meta.json).

---

## 📄 Manuscript reproduction

[`paper/`](https://github.com/kenflab/LLM-PathwayCurator/tree/main/paper) contains manuscript-facing scripts, Source Data exports, and frozen/derived artifacts (when redistributable).

* [`paper/README.md`](https://github.com/kenflab/LLM-PathwayCurator/blob/main/paper/README.md) — how to reproduce figures
* [`paper/FIGURE_MAP.csv`](https://github.com/kenflab/LLM-PathwayCurator/blob/main/paper/FIGURE_MAP.csv) — canonical mapping: panel ↔ inputs ↔ scripts ↔ outputs

---

## 🧾 Citation
If you use LLM-PathwayCurator, please cite:
- bioRxiv preprint (doi: [10.64898/2026.02.18.706381](https://doi.org/10.64898/2026.02.18.706381))
- Zenodo archive (v0.1.0): [10.5281/zenodo.18625777](https://doi.org/10.5281/zenodo.18625777)
- GitHub release tag: [v0.1.0](https://github.com/kenflab/LLM-PathwayCurator/releases/tag/v0.1.0)
- Software RRID: [RRID:SCR_027964](https://scicrunch.org/resolver/SCR_027964)
---
