# CRM_R1 revision workspace

> Build: `CRM_R1_HELDOUT72_V6_20260808`
> Frozen protocol: `CRM_R1_PRIORITY1_v5` (unchanged)
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
| P1 | GSE146225 empirical 48 h stability to held-out 72 h replication | τ=0.80 frozen; V6 held-out code ready | Figure 4 |
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

## 4. Reuse the verified V3 full-discovery outputs

V4 does not replace the verified full 48 h ranking or fgsea result. Reuse these existing files:

```text
$CRM_R1_DATA_ROOT/output/priority1/GSE146225_TP53_v1/derived/rankings/discovery_48h.tsv
$CRM_R1_DATA_ROOT/output/priority1/GSE146225_TP53_v1/derived/rankings/discovery_48h_gene_universe.tsv
$CRM_R1_DATA_ROOT/output/priority1/GSE146225_TP53_v1/derived/fgsea/discovery_48h.tsv
$CRM_R1_DATA_ROOT/output/priority1/GSE146225_TP53_v1/derived/fgsea/hallmark_gene_sets.tsv
```

If they do not exist, run the V3-compatible full-discovery scripts once. Both scripts load only the
twelve ENDO 48 h discovery columns.

```bash
Rscript paper/revision/CRM_R1/scripts/11_discovery_48h.R \
  --data-root "$CRM_R1_DATA_ROOT"

Rscript paper/revision/CRM_R1/scripts/12_fgsea_48h.R \
  --data-root "$CRM_R1_DATA_ROOT"
```

Do not rerun these verified outputs merely because V4 was installed.

## 5. Run the V4 81-resample discovery analysis

Create a new empirical-resampling Sample Card. Its filename is different from the V3 card, so the
earlier synthetic-perturbation artifact remains intact.

```bash
python paper/revision/CRM_R1/scripts/10_make_sample_card.py \
  --data-root "$CRM_R1_DATA_ROOT"
```

Run all balanced combinations obtained by deleting one sample from each of the four 48 h factorial
cells. Each run retains eight samples, recalculates TMM normalization, refits voom-limma, and reruns
fgsea using the frozen full-discovery gene universe and Hallmark snapshot.

```bash
Rscript paper/revision/CRM_R1/scripts/13_resample_discovery_48h.R \
  --data-root "$CRM_R1_DATA_ROOT"
```

The expected output is 81 resamples x 50 pathways = 4,050 fgsea rows. Build a replicate-stacked
EvidenceTable through the production fgsea adapter:

```bash
python paper/revision/CRM_R1/scripts/14_build_empirical_evidence.py \
  --data-root "$CRM_R1_DATA_ROOT"
```

Expected stacked EvidenceTable size: one full baseline plus 81 resamples, each containing 50
pathways, for 4,100 rows.

## 6. Completed discovery-only empirical calibration

The fixed calibration grid was completed using only 48 h results. Context evaluation was disabled
and did not block a claim. The frozen result is:

| tau | PASS | ABSTAIN | coverage |
| ---: | ---: | ---: | ---: |
| 0.80 | 23 | 27 | 0.46 |
| 0.90 | 15 | 35 | 0.30 |
| 0.95 | 10 | 40 | 0.20 |
| 0.98 | 5 | 45 | 0.10 |

`tau = 0.80` is the discovery-calibrated primary operating point. At `K = 23`, its overlap is 16
with the q-value-matched comparator and 17 with the q-value-plus-leading-edge-size-matched
comparator. Empirical survival has Spearman correlations of -0.166 with pathway size and 0.079 with
leading-edge count, so the V3 size dependence is not present in the V4 empirical measure.

The commands below are retained only to reproduce the completed calibration. Do not use them to
select a different operating point after 72 h is released.

```bash
export CRM_R1_BENCH="$CRM_R1_DATA_ROOT/output/priority1/GSE146225_TP53_v1"

for TAU in 0.80 0.90 0.95 0.98; do
  TAG="${TAU/./p}"
  LLMPATH_CONTEXT_REVIEW_MODE=off \
  LLMPATH_CONTEXT_GATE_MODE=note \
  llm-pathway-curator run \
    --evidence-table "$CRM_R1_BENCH/evidence_tables/discovery_48h_empirical_replicates.tsv" \
    --sample-card "$CRM_R1_BENCH/sample_cards/discovery_48h_empirical.sample_card.json" \
    --outdir "$CRM_R1_BENCH/out_audit/discovery_48h_empirical_ctxoff_note_tau_${TAG}_calibration_v1" \
    --tau "$TAU" \
    --k-claims 50 \
    --seed 42
done
```

Validate monotonic membership and write the discovery-only calibration table:

```bash
python paper/revision/CRM_R1/scripts/15_preview_empirical_membership.py \
  --data-root "$CRM_R1_DATA_ROOT"
```

The completed pre-freeze preview was:

```bash
python paper/revision/CRM_R1/scripts/15_preview_empirical_membership.py \
  --data-root "$CRM_R1_DATA_ROOT" \
  --primary-tau 0.80 \
  --force
```

This preview does not update the protocol or read a 72 h expression outcome.

## 7. Create and verify the immutable freeze bundle

Apply and locally commit the V5 code before freezing. The freeze script rejects tracked uncommitted
changes and refuses to overwrite an existing freeze. `--freeze-label` is a short, public-safe label
for the deliberate lock; it is not a cryptographic signature.

```bash
git status --short

python paper/revision/CRM_R1/scripts/16_freeze_priority1_membership.py \
  --data-root "$CRM_R1_DATA_ROOT" \
  --freeze-label "KF_20260808"

python paper/revision/CRM_R1/scripts/17_check_priority1_freeze.py \
  --data-root "$CRM_R1_DATA_ROOT"
```

The checker must print `[GO]` before any 72 h pathway statistic is calculated. It verifies the
frozen protocol, `tau = 0.80`, all three exact `K = 23` memberships, the complete tau grid, and every
recorded input/code SHA-256. It also confirms that no known 72 h validation output existed at
freeze. The four freeze files beneath `metrics/` are immutable; do not rerun with a different label
or edit them by hand.

## 8. Implement and test V6 without reading 72 h outcomes

V6 adds the one-time validation and frozen evaluation scripts without changing the V5 protocol,
freeze manifest, memberships, scripts `00`-`17`, or production package. Apply V6, run the full test
suite, and commit the code locally before releasing the held-out endpoint. Do not push the working
branch to public `origin`.

```bash
ruff format --check paper/revision/CRM_R1 \
  tests/test_crm_r1_priority1_v6_validation.py
ruff check paper/revision/CRM_R1 \
  tests/test_crm_r1_priority1_v6_validation.py
pytest -q
./examples/demo/run.sh
git diff --check
```

## 9. Run the held-out 72 h endpoint once

Both scripts reject tracked uncommitted changes and immutable output collisions. The R script runs
the pre-validation freeze checker before loading any 72 h expression column. It then loads only the
twelve ENDO 72 h samples, applies the frozen 48 h gene universe without refiltering, recalculates
TMM, refits the same voom-limma interaction, and runs `fgseaMultilevel` against the frozen Hallmark
snapshot.

```bash
Rscript paper/revision/CRM_R1/scripts/18_validation_72h.R \
  --data-root "$CRM_R1_DATA_ROOT"
```

Expected analytical outputs:

```text
$CRM_R1_DATA_ROOT/output/priority1/GSE146225_TP53_v1/validation/ranking_72h.tsv
$CRM_R1_DATA_ROOT/output/priority1/GSE146225_TP53_v1/validation/design_72h.tsv
$CRM_R1_DATA_ROOT/output/priority1/GSE146225_TP53_v1/validation/pathway_statistics_72h.tsv
$CRM_R1_DATA_ROOT/output/priority1/GSE146225_TP53_v1/validation/validation_72h.run_meta.json
$CRM_R1_DATA_ROOT/output/priority1/GSE146225_TP53_v1/validation/validation_72h.session_info.txt
```

Evaluate the already-frozen endpoint and memberships once:

```bash
python paper/revision/CRM_R1/scripts/19_evaluate_replication.py \
  --data-root "$CRM_R1_DATA_ROOT"
```

The evaluation exports pathway-, method-, tau-grid-, continuous-, exact-null-, and random-null
tables, the Stop gate summary, run metadata, and `source_data/figure4.tsv`. The primary Stop gate is
based only on whether the frozen empirical-minus-q-value replication-fraction point estimate is
greater than zero. Exact randomization, AUROC, logistic adjustment, size matching, and other tau
values are secondary and cannot reverse the gate.

Neither script has a `--force` path. Do not delete, rename, overwrite, or manually edit a completed
72 h output to rerun the endpoint.

## 10. Development checks

```bash
ruff format --check paper/revision/CRM_R1
ruff check paper/revision/CRM_R1
pytest
```

## Canonical Priority 1 pipeline

This follows the same organization as `paper/scripts/README.md` while keeping raw data and active
revision outputs outside Git.

1. Validate inputs and normalize metadata: `00_preflight.py`.
2. Reuse or compute the full 48 h ranking, frozen gene universe, and Hallmark fgsea snapshot.
3. Generate the V4 empirical-resampling Sample Card.
4. Run all 81 balanced 48 h delete-one-per-cell analyses.
5. Build the replicate-stacked EvidenceTable with the production fgsea adapter.
6. Run the context-off empirical tau calibration and inspect size dependence.
7. Freeze `tau = 0.80`, the exact matched memberships, and the input/code hashes without inspecting
   72 h outcomes.
8. Pass the immutable freeze checker.
9. Commit V6 without reading 72 h outcomes.
10. Run `18_validation_72h.R` once.
11. Run `19_evaluate_replication.py` once, apply Stop gate P1, and export Figure 4 source data.

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

The scientific choice and operational freeze are complete at `tau = 0.80`. After V6 is tested and
committed locally, run scripts `18` and `19` once. If the empirical-selection replication point
estimate is not greater than the q-value-matched estimate, apply Stop gate P1 without using
literature or human ratings to rescue the result.
