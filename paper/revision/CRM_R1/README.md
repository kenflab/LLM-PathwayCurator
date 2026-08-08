# CRM_R1 revision workspace

> Build: `CRM_R1_DISCOVERY48_V3_20260808`  
> Primary expression input: `GSE146225_raw_counts_GRCh38.p13_NCBI.tsv.gz`

This directory contains the minimal code and analysis protocol for the Cell Reports Methods major
revision. Read `ANALYSIS_PLAN.md` before starting an analysis. Raw data and derived outputs remain
outside Git under `CRM_R1_DATA_ROOT`.

## Recommended local paths

Keep the Git repository on the local Mac filesystem and keep large data/outputs in the existing
OneDrive project directory:

```text
Repository: /Users/kfurudate/projects/LLM-PathwayCurator
Data root:  /Users/kfurudate/Library/CloudStorage/OneDrive-InsideMDAnderson/LLMPATH/Revision/CRM_R1
```

For a first clone:

```bash
mkdir -p /Users/kfurudate/projects
cd /Users/kfurudate/projects
git clone https://github.com/kenflab/LLM-PathwayCurator.git
cd /Users/kfurudate/projects/LLM-PathwayCurator
git switch -c feat/crm-r1-scaffold
```

If the repository is already present, do not clone it again; use:

```bash
cd /Users/kfurudate/projects/LLM-PathwayCurator
```

## Priority map

| Priority | Purpose | Status | Main destination |
| --- | --- | --- | --- |
| P1 | GSE146225 direct-perturbation 48 h to held-out 72 h replication | 48 h scripts ready; protocol draft | Figure 4 |
| P2 | Same-pool unaudited/audited benchmark with matched baselines | Planned | Figure 2 |
| P3 | Source-masked external database and literature evidence grading | Planned | Figure 2 |
| P4 | Narrow blinded evidence review and inter-rater agreement | Planned | Figure 2 |
| P5 | GO/Reactome hierarchy, utility robustness, and final integration | Planned | Figure 3 + Supplement |

## What becomes public

`git commit` and `git push` are different actions.

| State | Public? |
| --- | --- |
| Uncommitted local files | No |
| Commits on a local branch | No |
| Branch pushed to a private revision repository | No |
| Any branch pushed to the public `kenflab/LLM-PathwayCurator` repository | Yes |
| Changes merged into public `main` | Yes |

`ANALYSIS_PLAN.md` is a scientific working protocol, not a required manuscript file. During active
R1 work, keep it and incomplete results on a local branch or private revision remote. At analysis
freeze, create a clean public branch from `origin/main` and copy only the reproducibility payload:
scripts, frozen configurations, environment, non-sensitive source tables, and concise run
instructions. Reviewer correspondence, private notes, raw data, credentials, and identifying local
paths must not enter the public history.

## Recommended Git workflow

Use frequent local commits for recoverability and review. During active R1 work, push to an empty
private repository named, for example, `LLM-PathwayCurator-R1-private`; keep the existing public
repository as `origin`.

```bash
git switch feat/crm-r1-priority1-scaffold
git branch -m feat/crm-r1-scaffold

# After creating an empty private repository in GitHub:
git remote add revision git@github.com:kenflab/LLM-PathwayCurator-R1-private.git
git push -u revision feat/crm-r1-scaffold
```

Do not push this working branch to public `origin`. Before resubmission, build a curated public branch
from current public `main`, copy only explicitly approved reproducibility paths, verify them, and then
merge through a pull request. Do not merge the entire private working history into the public branch.

```bash
git fetch origin
git switch -c feat/crm-r1-public origin/main
git restore --source feat/crm-r1-scaffold -- \
  paper/revision/CRM_R1/scripts \
  paper/revision/CRM_R1/config \
  paper/revision/CRM_R1/environment.yml \
  paper/revision/CRM_R1/README.md
```

Add `ANALYSIS_PLAN.md` to that allowlist only after its protocol is frozen and the authors decide it
belongs in the public reproducibility record.

## 1. Create the environment

The validated macOS path uses the existing Python 3.11 installation, a project-specific `venv`, and
a project-specific R package library. From the repository root:

```bash
python3.11 -m venv /Users/kfurudate/.venvs/llmpath-crm-r1-py311
source /Users/kfurudate/.venvs/llmpath-crm-r1-py311/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]" "scipy>=1.12" "pyyaml>=6.0"

export R_LIBS_USER="/Users/kfurudate/.R/llmpath-crm-r1-R4.6"
mkdir -p "$R_LIBS_USER"

./examples/demo/run.sh
```

Install the R packages once into that project-specific library:

```bash
Rscript -e 'install.packages(
  c("BiocManager", "data.table", "jsonlite", "dplyr", "tidyr",
    "ggplot2", "patchwork", "msigdbr"),
  repos = "https://cloud.r-project.org"
)'

Rscript -e 'BiocManager::install(
  version = "3.23", ask = FALSE, update = FALSE
)'

Rscript -e 'BiocManager::install(
  c("limma", "edgeR", "fgsea", "org.Hs.eg.db"),
  ask = FALSE, update = FALSE
)'
```

In each new Terminal session, reactivate both isolated environments before running the pipeline:

```bash
cd /Users/kfurudate/projects/LLM-PathwayCurator
source /Users/kfurudate/.venvs/llmpath-crm-r1-py311/bin/activate
export R_LIBS_USER="/Users/kfurudate/.R/llmpath-crm-r1-R4.6"
```

The validated reference environment is Python 3.11.9, R 4.6.1, and Bioconductor 3.23 on macOS
arm64. The analytical R packages were `edgeR` 4.10.1, `limma` 3.68.4, `fgsea` 1.38.0,
`msigdbr` 26.1.0, and `org.Hs.eg.db` 3.23.1. Each analytical script records the versions actually
used in its run metadata. `environment.yml` remains the optional cross-platform conda specification;
conda is not required for the validated Mac workflow.

Priority 1 uses the supplied raw integer counts. To minimize new code, it follows the existing
BeatAML pattern: edgeR filtering/TMM normalization, voom-limma modeling, and moderated t-statistic
ranking. No normalized expression matrix is used in the primary differential-expression pipeline.

The deterministic demo is the first environment smoke test. Priority 1 likewise uses deterministic,
LLM-free proposal generation as its primary run. Optional LLM-assisted proposals are stored and
reported separately; every audit disposition remains mechanical.

## 2. Point to external data

```bash
export CRM_R1_DATA_ROOT="/Users/kfurudate/Library/CloudStorage/OneDrive-InsideMDAnderson/LLMPATH/Revision/CRM_R1"
```

Expected inputs:

```text
$CRM_R1_DATA_ROOT/input/GSE146225_raw_counts_GRCh38.p13_NCBI.tsv.gz
$CRM_R1_DATA_ROOT/input/GSE146225_series_matrix.txt
```

## 3. Run the Priority 1 preflight

```bash
python paper/revision/CRM_R1/scripts/00_preflight.py \
  --data-root "$CRM_R1_DATA_ROOT"
```

This validates the frozen SHA-256, gzip parsing, 39,376-by-60 non-negative integer counts, exact
sample matching, and the balanced 2 genotype x 2 treatment x 3 replicate design at 48 h and 72 h.
It does not calculate any 72 h biological outcome.

Expected outputs:

```text
$CRM_R1_DATA_ROOT/output/priority1/GSE146225_TP53_v1/preflight/input_manifest.json
$CRM_R1_DATA_ROOT/output/priority1/GSE146225_TP53_v1/preflight/preflight_summary.json
$CRM_R1_DATA_ROOT/output/priority1/GSE146225_TP53_v1/preflight/sample_metadata.normalized.tsv
$CRM_R1_DATA_ROOT/output/priority1/GSE146225_TP53_v1/preflight/design_counts.tsv
```

## 4. Run the 48 h discovery pipeline

The following commands calculate only the prespecified ENDO 48 h discovery analysis. The DE script
streams only the twelve 48 h discovery columns from the raw count matrix; it does not load the 72 h
expression columns.

```bash
python paper/revision/CRM_R1/scripts/10_make_sample_card.py \
  --data-root "$CRM_R1_DATA_ROOT"

Rscript paper/revision/CRM_R1/scripts/11_discovery_48h.R \
  --data-root "$CRM_R1_DATA_ROOT"

Rscript paper/revision/CRM_R1/scripts/12_fgsea_48h.R \
  --data-root "$CRM_R1_DATA_ROOT"
```

`12_fgsea_48h.R` writes the raw fgsea result, the complete Hallmark Entrez gene-set snapshot, the
ranking-overlap table, run metadata, and R session information. Convert the raw result through the
production adapter:

```bash
llm-pathway-curator adapt --format fgsea \
  --input "$CRM_R1_DATA_ROOT/output/priority1/GSE146225_TP53_v1/derived/fgsea/discovery_48h.tsv" \
  --output "$CRM_R1_DATA_ROOT/output/priority1/GSE146225_TP53_v1/evidence_tables/discovery_48h.tsv"
```

Then run the deterministic primary proposal and mechanical audit at the prespecified operating
point:

```bash
llm-pathway-curator run \
  --evidence-table "$CRM_R1_DATA_ROOT/output/priority1/GSE146225_TP53_v1/evidence_tables/discovery_48h.tsv" \
  --sample-card "$CRM_R1_DATA_ROOT/output/priority1/GSE146225_TP53_v1/sample_cards/discovery_48h.sample_card.json" \
  --outdir "$CRM_R1_DATA_ROOT/output/priority1/GSE146225_TP53_v1/out_audit/discovery_48h" \
  --tau 0.8 \
  --k-claims 50 \
  --seed 42
```

Do not add `--force` on a first run. Use it only for an intentional rerun after preserving or
removing the earlier output.

## 5. Development checks

```bash
ruff format --check paper/revision/CRM_R1
ruff check paper/revision/CRM_R1
pytest
```

## Canonical Priority 1 pipeline

This follows the same organization as `paper/scripts/README.md` while keeping raw data and active
revision outputs outside Git.

1. Validate inputs and normalize metadata: `00_preflight.py`.
2. Generate the GSE146225 Sample Card.
3. Compute the 48 h edgeR/voom-limma interaction ranking.
4. Run Hallmark `fgseaMultilevel`, then adapt its result to an EvidenceTable.
5. Run LLM-PathwayCurator and mechanical audits.
6. Freeze claim-pool and matched-method membership without inspecting 72 h outcomes.
7. After protocol freeze, run 72 h replication, aggregate metrics, and export Figure 4 source data.

Active outputs use the canonical benchmark layout below:

```text
$CRM_R1_DATA_ROOT/output/priority1/GSE146225_TP53_v1/
  preflight/
  derived/rankings/
  derived/fgsea/
  sample_cards/
  evidence_tables/
  out_audit/
  validation/
  metrics/
  fig/
  source_data/
```

At publication freeze, copy only the required manifests, derived inputs, source tables, and run
metadata to `paper/source_data/GSE146225_TP53_v1/`, then add the final script/output mapping to
`paper/FIGURE_MAP.csv`.

## Next decision gate

Review the 48 h discovery outputs, freeze the claim-pool and matched-method membership without
calculating a 72 h pathway statistic, and sign the protocol manifest. The production adapter only
converts fgsea output to the EvidenceTable contract; it does not perform differential expression or
fgsea. Do not implement or run 72 h validation until `config/priority1_protocol.json` has been
scientifically reviewed and changed to `FROZEN`.
