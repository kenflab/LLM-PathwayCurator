# CRM R1 research workspace

This is the single entry point for revision analyses. The public tool is described
in the [repository README](../../../README.md). Keep using the existing checkout
and the existing external data root; no additional checkout is needed.

## Layout

| Directory | Contents |
| --- | --- |
| `experiments/reassessment/` | Updated public tool applied to unchanged saved texts; development diagnostics |
| `experiments/external_dex/` | Non-cancer worked case, original design, technical amendment and reporting |
| `experiments/submission/` | Private manuscript/source collection and saved-result figure rendering |
| `experiments/revision_tools/` | Historical contract, atomic-review and source-locator prototypes |
| `scripts/`, `config/` | Research runners and versioned protocols needed to trace completed work |
| `tests/`, `fixtures/` | Research-specific tests and synthetic development controls |
| `manuscript/` | Source-linked writing instructions and response/figure scope |
| `archive/` | Unchanged historical workflow notes and analysis plans |

None of the research prototypes or study-specific registries is part of the
installed `llm_pathway_curator` package. Tests still run through repository pytest.

## Current tasks

The updated public implementation writes source-linked statements and checks
supplied prose with limited rules. Use
[experiments/reassessment/README.md](experiments/reassessment/README.md) to
prepare a new development record and inspect the already generated R07 text.
This does not add independent expert judgments or estimate biological accuracy.

Use [experiments/submission/README.md](experiments/submission/README.md) for frozen
source collection and manuscript drafting, and
[experiments/external_dex/README.md](experiments/external_dex/README.md) for the
completed limited cross-study case. Earlier R06 and R07 notes remain in
[archive/](archive/README.md).

## Reanalysis policy

| Existing analysis | Action after the software update |
| --- | --- |
| R07 saved natural text | Recheck all original texts with new source checks; record new implementation and outputs |
| Legacy full-audit comparative endpoints | Preserve old result; a new algorithm requires new selection outputs and a fixed evaluation design |
| Original rater statements and returned labels | Retain their exact text links; labels cannot be transferred to newly written prose |
| P1 and external dex upstream expression/enrichment fits | Retain if inputs, model and endpoint have not changed; refit only if those components change |
| P3 ungraded records and ontology diagnostics | Preserve unknown grades and diagnostic scope; no conversion into biological truth labels |

A new report or passing software test does not resolve the manuscript's need for
comparative semantic/biological evidence. Frozen adverse or null results remain
reportable. Do not tune a new selector against their validation outcomes and
then describe the result as an untouched holdout.

## Storage and reproduction

Inputs, ratings, correspondence, manuscript Word files and analysis outputs stay
under `CRM_R1_DATA_ROOT`, outside Git. New runners write a fresh directory under
`output/revision_v17/` and never overwrite prior results.

For exact historical code paths and hashes use commit
[`b069e8a`](https://github.com/kenflab/LLM-PathwayCurator/tree/b069e8ad3d916618adcf26af0202748790c3ca20).
The layout migration does not rewrite original locks. Archived installers and
workflow notes are historical records, not instructions to replace the current
public package with old bundles.

Version strings in frozen protocol JSON and historical documents link results
to their method. They remain there deliberately. Current user-facing module
names and the README entry point do not require those development version names.
