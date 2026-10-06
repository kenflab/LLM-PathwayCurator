# Historical pipeline demo

This demo runs the historical distill/module/proposal/audit path with explicit
`workflow="legacy"`, deterministic proposals and proxy context review.
Its proxy scores and synthetic perturbations are not independent biological or
semantic validation. For the current public workflow use
[examples/source_report](../source_report/README.md).

From the repository root:

```bash
./examples/demo/run.sh
```

You may pass a new output directory as its first argument. Existing nonempty
outputs are not overwritten. The script uses the included EvidenceTable, or
adapts the included fgsea result table if that input is absent. It does not run R
or a language model.

The [expected/](expected/) files are an unchanged historical snapshot.
Their decisions and metrics do not describe the updated default source workflow.
