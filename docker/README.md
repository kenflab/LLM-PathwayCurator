# Docker setup (LLM-Pathway-Curator)

This folder provides a **fully reproducible runtime** for the paper and benchmarks:
data download → DEG analysis (R) → enrichment → LLM-Pathway-Curator → figures.

## What this gives you
- **Python + R in one container**
  - R: DESeq2 / edgeR / limma / fgsea / msigdbr (Fig2-ready)
  - Python: llm-pathway-curator + bench utilities
- **Jupyter Lab** for interactive runs
- **Optional local LLM** via Ollama
- **Optional ORA** via Metascape (MSBio2)

---

## Quick start
From the repository root:
```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up
```

Then open Jupyter at:
[http://localhost:8888](http://localhost:8888)

---


## Services
- llm-pathway-curator
  Main analysis environment (Python + R + Jupyter)
-　Ollama (optional)
  Local LLM backend (e.g. llama3.1:8b)
- msbio2 (optional)
Offline Metascape (requires a license)

---

## Metascape (MSBio2) license
- Not required for Fig2 core results (fgsea-based)
- Obtain a license only when you actually run MSBio2
- Do not commit license files to git

---

Notes
For quick checks of the curator alone, you may also build the image directly:
```bash
docker build -t llm-pathway-curator:official -f docker/Dockerfile .
```