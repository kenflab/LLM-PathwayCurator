# Updated-tool reassessment

This runner applies the **normal public source workflow** to the exact existing
R07 natural-language corpus. It verifies the source ZIP, records the current
implementation and settings, preserves all 50 texts and source records, and
writes fresh results under the existing data root.

The corpus was already examined during development. This is a development
diagnostic, not an independent semantic accuracy or biological validation study.
No model requests, new expert labels, upstream refits or old-result edits occur.

## Run in the existing checkout

After updating main and installing its source, use the exact original R07 ZIP
from the completed run. A complete submission-source collection ZIP is also
accepted. Do not select an input merely because it is the newest ZIP.

```bash
python "$CRM_R1_REPO/paper/revision/CRM_R1/experiments/reassessment/run.py" \
  --data-root "$CRM_R1_DATA_ROOT" \
  --source "$CRM_R1_DATA_ROOT/output/revision_v17/r07_20261005T190838308028Z.zip" \
  --prepare --run
```

Preparation records the design before computing the new limited checks. It
does not erase prior exposure to these texts. To separate the steps, use
`--prepare` first, then `--run --design <printed design_dir>`. If implementation
or prepared inputs change, execution stops and a new development design is
required. Every run has a new output directory.

## Readout

Open `source_report/report.html`. Inspect the unchanged drafts beside source
statistics and quoted findings. `paired_checks.private.tsv` places original
R07 numeric-gate status beside the updated numeric coverage and limited checks.
`SUMMARY.json` reports the census and check counts, without accuracy estimates.

Numeric coverage, detected contradictions and inferential review flags are
different endpoints. Missing explicit numbers are not proven errors. Flags are
not expert labels; an unflagged draft is not automatically accepted.

## What needs a separate study

Before claiming improved interpretation, fix the new method and endpoints,
then evaluate unchanged unaudited outputs and the updated workflow on text not
used to develop the rules, with appropriate human source-fact judgments.
Use the smallest justified review packet; original rater labels cannot be
transferred to new text. Another paid model is not required for this development
step. Biological generalization requires separately suitable biological data.

The existing P1, P2B, expert and dex results remain historical results. Refit an
upstream analysis only when its input/model/endpoint changes. A code reorganization
alone is not a reason to overwrite or relock completed analyses.
