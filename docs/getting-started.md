# Getting started

## Install current source

Python 3.11 or later is required.

```bash
git clone https://github.com/kenflab/LLM-PathwayCurator.git
cd LLM-PathwayCurator
python -m pip install -e .
```

If you already have a checkout, update it and reinstall there. The new source
workflow is not a claim that a previous PyPI release has changed.

## Run the example

```bash
llm-pathway-curator run \
  --evidence-table examples/source_report/evidence_table.tsv \
  --sample-card examples/source_report/sample_card.json \
  --claims examples/source_report/drafts.tsv \
  --outdir out/source_report
```

Open `out/source_report/report.html`. Search the pathways and inspect the source
statement, original draft and quoted findings. All candidates are retained.
These data and deliberate wording problems are synthetic software examples.

For your data, replace the evidence table and Sample Card. Omit `--claims` if
you have no draft text. Use a new output directory on each run.

The default workflow makes no model calls, regardless of model environment
variables. Historical execution is explicitly selected with `--workflow legacy`.
See the [User guide](user-guide.md) before using legacy model settings.
