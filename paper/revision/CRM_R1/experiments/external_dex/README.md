# Dexamethasone public-data case

This revision experiment uses existing statistical tools to test a non-cancer
application and a limited cross-study replication endpoint. It is separate from
the installed `llm_pathway_curator` API and does not import the TP53-specific
contract, call any language model, or request new expert ratings.

GSE52778 contributes four paired donor-derived cell lines at 18 hours.
GSE34313 contributes the public HASM1 cultures: four controls, three 24-hour dex
samples and three 4-hour dex samples. Replicate suffixes do not authenticate
pairing. Cross-study donor overlap remains unverified. Published results were
seen during methods review, so this is an exploratory worked case, not a wholly
unseen or independent-donor benchmark.

## Run in the existing checkout

Use the existing CRM_R1 data directory outside Git, with `input/` and `output/`.
The default Hallmark source is the already frozen
`output/priority2b/hallmark_lock_v14_1_2/hallmark_gene_sets.tsv`. No gene-set
download or replacement is performed. If necessary, specify `--hallmark` with
an existing human gene-symbol snapshot within the same data directory.

```bash
python "$CRM_R1_REPO/paper/revision/CRM_R1/experiments/external_dex/run.py" \
  --data-root "$CRM_R1_DATA_ROOT" --freeze

python "$CRM_R1_REPO/paper/revision/CRM_R1/experiments/external_dex/run.py" \
  --data-root "$CRM_R1_DATA_ROOT" --fetch --analyze --install-missing
```

The second command downloads the fixed public GEO files and runs R statistics.
`--install-missing` installs only missing free R packages using Bioconductor;
existing packages are not updated. R itself must already be installed with
`Rscript` on PATH. Omit this flag when all dependencies are present. The packages
are `airway`, `SummarizedExperiment`, `edgeR`, `limma`, `fgsea` and `jsonlite`.
Nothing invokes Ollama, Gemini or any paid API.

The protocol and source hashes are frozen before downloads or expression reads.
The R/airway versions and package data payload hashes are sealed before analysis.
A completed analysis is reused and its output/archive hashes are checked on
subsequent runs. Changed inputs or code cause a stop; a technical correction
requires an explicit documented amendment, not silent replacement of the lock.
Interrupted runs preserve their logs and partial outputs. Identical fixed code
and inputs may run again after an infrastructure failure; a completed result
cannot be overwritten.

## R08.1 technical correction for the recorded probe-filter stop

The original implementation added a GPL6480 `CONTROL_TYPE` numeric-zero gate
that is absent from `protocol.json`. GPL annotation contains textual labels
such as `FALSE`; those are distinct from the numeric raw Agilent `ControlType`.
R08.1 follows the original protocol's raw `ControlType==0` rule and retains the
fixed mapping, detection, finite-value and minimum-probe checks. It does not
change the scientific protocol or relax the 1,000-probe sanity threshold.

For an original R08 run stopped at this filter, install the reviewed correction,
then record the technical amendment and resume using the existing downloads:

```bash
python "$CRM_R1_REPO/paper/revision/CRM_R1/experiments/external_dex/run.py" \
  --data-root "$CRM_R1_DATA_ROOT" --amend-r08-probe-filter

python "$CRM_R1_REPO/paper/revision/CRM_R1/experiments/external_dex/run.py" \
  --data-root "$CRM_R1_DATA_ROOT" --analyze
```

The R08.1 bundle's `APPLY.py --apply --run` performs the first command. The
original `DESIGN_LOCK.json` and its design SHA stay unchanged. A separate
`TECHNICAL_AMENDMENT_R08_1.json` records the old/new source hashes, unchanged
protocol, existing inputs/runtime and all files of the preserved failed run.
Registration stops if any matching original job already has fitted statistics
or if the original code, input receipts or failure record do not match.
Subsequent runs verify these records; no locks or old outputs are removed.

Before any new statistical analysis, a base-R regression test exercises mixed
`FALSE/pos` GPL labels, numeric/logical-only annotations, duplicate spots,
controls, missing mappings, ambiguous symbols and detection/finite-value rules.
`PROBE_FILTER_DIAGNOSTIC.json` compares the original code's retention with the
frozen protocol's retention on the actual normalized data. Per-probe decisions
and condition counts are saved before any further filter stop. A single old log
does not identify each filter's contribution; the diagnostic establishes
whether the unintended GPL condition explains the observed stop.

## Fixed statistical design

`protocol.json` is authoritative. Discovery uses integer airway counts,
gene-symbol aggregation, a label-independent count filter, TMM/voom and a
paired donor-plus-treatment limma model. The array data are reprocessed from
raw Cy3 Agilent signals with normexp offset 50 and quantile normalization. This
avoids guessing the scale of deposited normalized values. Controls, ambiguous
annotations and probes without the fixed detection coverage are excluded by
the locked rules, then probes are combined by per-sample median.

All fits use the same measured gene intersection and all 50 Hallmark terms.
fgseaMultilevel uses the fixed seed, parameters and BH family of 50. Out-of-size
and numerical-failure terms remain in the ledger. Donor stability removes both
samples for each of four donors; at least three folds must reproduce the
discovery NES sign and q<=0.05. This is empirical donor sensitivity, not the
historical synthetic gene-dropout score.

The primary replication criterion is the same nonzero NES direction and
24-hour validation q<=0.05. The 4-hour contrast is secondary and shares controls.
Selections use discovery results only: all 50, q<=0.05, q plus donor stability,
and a size-matched q comparator. Zero selection remains zero. Rates are
descriptive; overlapping pathways and one validation cell line do not support
independent-trial binomial intervals or a full-audit superiority claim.

## Outputs and interpretation

- Frozen design: `output/revision_v17/external_dex_design_v1/`.
- Public input snapshots: `input/external_dex_r08_v1/`.
- Results and compact ZIP: `output/revision_v17/external_dex_<UTC>/` and `.zip`.
- `METHOD_COMPARISON.tsv` gives denominators, coverage and replication counts.
- `TERM_LEDGER.tsv` retains all 50 terms, missingness and selections.
- Gene ranks, measured universes, sample records, all four donor folds,
  normalization QC, runtime identity and input/export hashes are preserved.
- `SOURCE_TEMPLATES.json` is a simple, evidence-linked reporting baseline using
  the exact saved numeric strings. It is not natural LLM prose or a new semantic
  evaluation. Generic sample cards do not masquerade as TCGA/TP53 contracts.

The compact ZIP omits the public raw array files and records their hashes.
They remain in the data directory alongside the immutable original download.
Do not put downloaded data, local paths, reviewer correspondence or private
results into public Git. Only the experiment code and protocol belong here.

This case cannot replace existing expert agreement, ontology analysis, or the
adverse multi-cohort full-audit comparison. Report negative or null outcomes
with the same protocol. Do not change studies, thresholds or time points to
obtain a favorable result.

## Sources

- [GSE52778](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE52778)
- [GSE34313](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE34313)
- [GPL6480](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL6480)
- [airway data construction](https://www.bioconductor.org/packages/release/data/experiment/vignettes/airway/inst/doc/airway.html)
- [limma user guide](https://bioconductor.org/packages/release/bioc/vignettes/limma/inst/doc/usersguide.pdf)
- [fgsea vignette](https://bioconductor.org/packages/release/bioc/vignettes/fgsea/inst/doc/fgsea-tutorial.html)
