# Research script map

Use the [revision README](../README.md) as the current entry point.
The public installed tool is documented at the repository root.

| Scope | Runners | Status |
| --- | --- | --- |
| P1 perturbation comparison | `10_` through `19_`, plotting `90_` | Historical locked analysis |
| P2 matched candidate comparison | `20_` through `28_` | Historical locked analysis |
| P3 source retrieval | `30_`, `31_` | Retrieval and ungraded evidence ledger |
| P4 original rater records | `40_`, `41_`, `65_revision_r06.py` | Original statements and labels only |
| P5 ontology diagnostics | `49_` through `54_`, plotting `91_` | Diagnostic scope, not biological truth |
| Saved-result preflight | `60_revision_r01.py`, `61_revision_r02.py` | No new scientific performance estimate |
| Historical model development | `62_revision_r03.py`, `63_revision_r04.py` | Exposed development controls |
| First natural text | `66_revision_r07.py` | Completed frozen generation; preserve first outputs |
| Updated public-tool checks | [reassessment runner](../experiments/reassessment/README.md) | Current saved-text development reassessment |

The original detailed script map is retained unchanged in
[archive/SCRIPT_MAP.md](../archive/SCRIPT_MAP.md). Its prior commands and locks
refer to historical versions; exact reproductions use their pinned code.
Reorganized research prototypes live in `experiments/revision_tools/`.
Their imports are available to research runners and tests, not public installs.

Bundle installers are retained for provenance. Do not reinstall an archived
bundle over an updated main checkout. Use an ordinary reviewed Git update in
the same checkout and write new outputs under the existing external data root.
