# TCGA rebuild with an explicit OV variant policy

This is the public script port of the OVFIX1 diagnostic rebuild. It preserves
its cohort selection, mutation filters, RNA QC, expression values, enrichment
settings, and source-report workflow. The code is specific to this TCGA study;
it is not an fgsea adapter rule or a general-purpose genotype classifier.

## Inputs and execution

Python 3.11+ and the installed project dependencies are required. The recorded
snapshot is included in [CRM_R1_TCGA_inputs.zip](../../source_data/CRM_R1_TCGA_inputs.zip),
with an adjacent SHA-256 file. It contains the 11 original public input files and
`INPUT_MANIFEST.json`, including source URLs and the expression checksum.
The 331 MB expression matrix is downloaded and verified by the runner if absent.
The recorded GDC snapshot is retained; no current query replaces it.

```bash
python -m zipfile -e paper/source_data/CRM_R1_TCGA_inputs.zip out/tcga_inputs
python paper/scripts/tcga_rebuild/RUN.py \
  --repo /path/to/LLM-PathwayCurator \
  --inputs /path/to/LLM-PathwayCurator/out/tcga_inputs \
  --data-root /path/to/CRM_R1
```

The manuscript enrichment/ranking provenance records R 4.6.1, limma 3.68.4,
fgsea 1.38.0 and msigdbr 26.1.0. See the exact per-cohort files in
[Supplementary Data 1](../../revision/CRM_R1/data/Supplementary_Data_1_TCGA/enrichment/).
The original input bundle also contains a separate
[verification environment log](BUNDLE_VERIFICATION_R_ENVIRONMENT.txt) using
R 4.3.3, limma 3.58.1 and msigdbr 26.1.1. That log does not describe the
manuscript fits. `CHECK_R.R` checks installed dependencies; it does not install
or recreate either historical environment.

Without `--run`, this checks inputs and writes a new sample ledger and group
files. Add `--run` to execute ranking, enrichment, source reports, verification,
and figures. Outputs go to a new timestamped directory. An existing output
location is rejected. No checkout, commit, push, model call, or study relock is
performed. The saved analysis uses source code at
`764b18ec0140d9c878bf7866f129e6513ac751d4`; it is exported separately so local
uncommitted edits cannot enter a reproduction. The public runner hashes are
also recorded. This is a reproducibility entry point for that analysis, not a
claim that future tool versions produce identical results.

## OV and comparator definitions

* Accept exact `PASS` TP53 calls in all cohorts and exact bare `wga` calls in OV.
  A compound filter such as `wga,oxog` is not rescued. Original tags are retained.
* OV permits D/W/X/G DNA analytes. Exclude native/WGA mixtures, badseq/contest,
  missing assay context, RNA QC exclusions, and ambiguous duplicate RNA columns.
  The other six cohorts retain the declared native-DNA/PASS policy.
* Exclude MC3-call-negative OV samples with a protein-altering TP53 occurrence
  in the same GDC case in the saved snapshot. This is a case-level discordance
  exclusion; it does not relabel them as MC3 mutants or prove WT from absence.
* `TP53_wt` is the internal compatibility label. Report it as **MC3 call-negative**.
  Public assay metadata do not establish per-base TP53 callability.
* `n_wt` therefore counts eligible call-negative comparators. Small arms are
  flagged, not silently dropped. OV and SKCM are descriptive examples.

The OV policy follows the WGA-only treatment described in Bailey et al., Cell
2018, DOI: [10.1016/j.cell.2018.02.060](https://doi.org/10.1016/j.cell.2018.02.060).
The GDC snapshot is a cross-source positive-call guard, not a population mutation
frequency denominator. The optional historical ledger in `superseded_reference/`
is used only to document changes from the previous diagnostic implementation.

## Interpretation and saved-output comparison

The corrected results describe unadjusted mutation-group associations, not causal
TP53 effects. Tumor subtype, purity and other covariates can explain associations.
Different R/limma environments can yield numerical differences; preserve the
original run environment and input/output hashes.

```bash
python paper/scripts/summarize_tcga_source_baseline.py \
  --run-dir /path/to/completed_rebuild \
  --outdir /path/to/new_baseline_comparison
```

This compares current source selections to the same q<=0.05 rule and inventories
modules and source statements. Identical selected pathways are expected: source
curation adds traceability and explicit review boundaries, not an alternative
pathway significance test. This comparison does not assess LLM text, biological
correctness, curator speed, or superiority. Those require separate evaluations
of the same submitted texts and inputs with independent reference judgments.
