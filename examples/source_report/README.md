# Synthetic source-report example

From the repository root, after `python -m pip install -e .`:

```bash
llm-pathway-curator run \
  --evidence-table examples/source_report/evidence_table.tsv \
  --sample-card examples/source_report/sample_card.json \
  --claims examples/source_report/drafts.tsv \
  --outdir out/source_report

llm-pathway-curator plot-source \
  --run-dir out/source_report --outdir out/source_plot
```

Open `out/source_report/report.html` and `out/source_plot/source_plot.png`.
Use a new output directory for each run. Omit `--claims` to report source values
without supplied draft text.

| Input term | Deliberate example | Limited text-check outcome |
| --- | --- | --- |
| PROGRAM_A | Correct source values with causal wording | `REVIEW_FLAGGED` |
| PROGRAM_B | Significance wording contradicts q = 0.6644 and cutoff 0.05 | `VIOLATION_DETECTED` |
| PROGRAM_C | An unsigned statistic and adjusted P value | `NO_LIMITED_FLAG` |

The report retains all three pathways and the original wording. A source
statistical statement's eligibility differs from a draft's review disposition.
A missing flag does not establish biological correctness. These controlled
examples are not biological labels or accuracy data. No model is called.
