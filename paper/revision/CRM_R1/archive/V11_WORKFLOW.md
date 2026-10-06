# V11 blinded-evaluation workflow

## Current state

- P1: frozen and complete; no rescue or retuning permitted.
- P2: 50-claim pool and K=25 memberships frozen.
- P3: retrieval frozen; grading outcomes not yet locked.
- P4: three method-blinded packets frozen; returned ratings not yet locked.
- P5: ontology inputs and Figure 3 source tables frozen; ontology panels complete.

## Allowed now

- Revise the manuscript using the supplied P1/P5 text.
- Use the supplied P2–P4 Methods text without result claims.
- Render Figure 3 v2 from the already frozen source tables.
- Inspect the visibly watermarked Figure 2 layout preview.

## Required order after files return

1. Lock and check completed P3 grades (`32`, `33`).
2. Lock and check all three P4 ratings (`42`, `43`).
3. Join locked outcomes to frozen P2 membership exactly once and freeze Figure 2 source (`60`, `61`).
4. Render Figure 2 from source only (`90`).
5. Finalize the Figure 2 Results paragraph and rebuttal response values.

The P3 and P4 locks do not read method membership. The first permitted unblinding occurs in script
`60_build_priority2_figure2_source.py`, and only after all three freeze checkers pass.
