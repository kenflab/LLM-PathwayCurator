# Structured curation example

From the repository root, after `python -m pip install -e .`:

```bash
python examples/structured_curation/run_demo.py --outdir out/structured_curation
```

Open `out/structured_curation/reviewed/report.html`.
Use a new output directory. The script reads the synthetic source-report inputs,
exports a proposal packet, and imports three typed proposals: a statistical
observation, a supporting-gene summary and a hypothesis.

| Output | Purpose |
| --- | --- |
| `packet/` | Source records, supporting-gene modules and proposal packet |
| `synthetic_proposals.jsonl` | Explicitly labeled synthetic proposals |
| `reviewed/` | Source-linked report and proposal-check results |

References use exact term identifiers and evidence hashes. Supporting-gene
modules describe overlap; they do not establish a shared mechanism.
The script makes no model requests. It is a software example, separate from
manuscript analyses. See the [structured curation guide](../../docs/structured-curation.md)
for the input schema and decision scope.
