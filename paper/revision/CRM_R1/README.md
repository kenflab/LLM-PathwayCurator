# CRM_R1 revision workspace

Use the existing checkout and the existing data root. Source code and concise
instructions belong here; manuscripts, ratings, correspondence, raw data and
analysis outputs remain under `CRM_R1_DATA_ROOT` outside Git.

## Current entry points

| Task | Entry point | Role |
| --- | --- | --- |
| Collect the current manuscript and frozen source tables | [experiments/submission/README.md](experiments/submission/README.md) | Next manuscript integration step; no fitting or model requests |
| Render the completed non-cancer example | [experiments/external_dex/README.md](experiments/external_dex/README.md) | Saved-result reporting; preserve the original R08 design and amendment |
| Reuse existing rater assessments | [R06_P4_REUSE.md](R06_P4_REUSE.md) | Same original standardized statements; rater-specific comparisons |
| Inspect first natural-language outputs | [R07_NATURAL_TEXT_COMPARISON.md](R07_NATURAL_TEXT_COMPARISON.md) | Frozen reporting comparison; numeric coverage is not semantic accuracy |
| Manuscript text and figure scope | [manuscript/MANUSCRIPT_METHODS_RESULTS_DRAFT.md](manuscript/MANUSCRIPT_METHODS_RESULTS_DRAFT.md) | Guide to private draft generation and source limitations |
| Main comparative figure | [manuscript/FIGURE2_LAYOUT_AND_LEGEND.md](manuscript/FIGURE2_LAYOUT_AND_LEGEND.md) | Retain rater disagreement and matched-coverage comparisons |

## Scientific boundaries

Empirical sample resampling, historical synthetic gene perturbation, context
assessment, source-fact checks and biological correctness are distinct tasks.
PASS means the implemented checks passed; it is not a biological truth label.
The current full audit has not demonstrated a general interpretive advantage
against simpler reporting rules. Null or adverse results remain in the record.

Existing rater labels apply only to their original statements. Literature
retrieval is not completed evidence grading. Blank grades remain unknown.
Ontology sign differences describe directional discordance, not demonstrated
biological errors. A cross-study culture-level example does not establish
independent-donor validation.

## Run and Git workflow

Keep using `main` in the existing checkout. Review and publish only the explicit
code paths selected by the revision installer. It preserves unrelated local
work and uses a normal commit/push. Do not create another checkout or replay
historical freeze instructions to run a completed analysis.

The submission collector creates a new private folder and ZIP in the existing
output directory. It never overwrites inputs, designs, ratings or manuscripts.
New model comparisons, broad rerating, full P3 grading and expression refitting
are not prerequisites for this collection step.

## Historical material

The V-numbered workflows, earlier development notes and `ANALYSIS_PLAN.md`
record prior designs and attempts. They are retained as history and do not
supersede completed-result receipts or the current entry points above.
Revision prototypes under `src` are not evidence of a validated public release;
module cleanup remains separate from frozen scientific results.
