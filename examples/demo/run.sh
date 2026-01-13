#!/usr/bin/env bash
set -euo pipefail

llm-pathway-curator run \
  --evidence-table examples/demo/de_table.tsv \
  --sample-card examples/demo/sample_card.json \
  --outdir out/demo
