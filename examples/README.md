# Examples

Install the package from the repository root with `python -m pip install -e .`.
Start with the source report. Use a new output directory for each run.

| Example | Purpose | Command from the repository root |
| --- | --- | --- |
| [Source report](source_report/README.md) | Inspect enrichment values, original draft text and findings | `llm-pathway-curator run --evidence-table examples/source_report/evidence_table.tsv --sample-card examples/source_report/sample_card.json --claims examples/source_report/drafts.tsv --outdir out/source_report` |
| [Structured curation](structured_curation/README.md) | Export source references and import typed proposals | `python examples/structured_curation/run_demo.py --outdir out/structured_curation` |
| [Historical pipeline](demo/README.md) | Run the retained distill/module/audit workflow | `bash examples/demo/run.sh out/legacy_demo` |

All three use synthetic inputs and make no model requests. They demonstrate
software behavior; their outputs are not biological validation results.
The historical example selects `workflow="legacy"` explicitly, and its
`expected/` folder preserves an earlier output snapshot.

To visualize the source report:

```bash
llm-pathway-curator plot-source --run-dir out/source_report --outdir out/source_plot
```

For manuscript figures and tables, use the
[manuscript reproduction guide](../paper/revision/CRM_R1/FIGURE_REPRODUCTION.md).
