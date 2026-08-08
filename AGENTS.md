# Repository instructions for Codex

- Preserve the separation between `src/` (package), `tests/` (developer checks),
  `examples/demo/` (reviewer-facing smoke test), and `paper/` (manuscript reproduction).
- Do not commit raw or controlled data.
- Run `ruff format`, `ruff check`, and `pytest` before committing Python changes.
- Keep figure calculations in scripts and export figure-facing source tables.

## Cell Reports Methods major revision

For work on `CRM_R1` or revision Priorities 1-5, first read:

1. `paper/revision/CRM_R1/ANALYSIS_PLAN.md`
2. `paper/revision/CRM_R1/config/priority1_protocol.json`
3. `paper/revision/CRM_R1/AGENTS.md`
4. `paper/scripts/README.md`
5. `paper/FIGURE_MAP.csv`

Do not calculate the Priority 1 held-out 72 h endpoint while the protocol status is
`DRAFT_NOT_FROZEN`. Status `FROZEN` is still insufficient until
`paper/revision/CRM_R1/scripts/17_check_priority1_freeze.py` passes against the immutable external
freeze manifest.
