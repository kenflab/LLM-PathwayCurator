# Figure 2 layout and working legend

## Layout

- **A — Design:** one frozen 50-claim HNSC pool; full audit, q-value, and stability rules; matched
  K=25; blinded P3/P4 evaluation; one-time unblinding.
- **B — Independent support:** E3/E4, direction-matched, independent PubMed evidence. Show raw pool
  in gray as descriptive, then q-value, stability, and full audit. Bars show fractions, Wilson 95% CIs,
  and numerator/denominator.
- **C — Major-overstatement risk:** majority of three blinded raters, with the same method order,
  colors, CIs, and denominators.
- **D — Agreement:** Fleiss' kappa for statistical support, external evidence, and overstatement;
  annotate unanimous agreement. Full category counts belong in Source Data/Supplementary Table.

The locked source builder also writes overlap-aware exact comparisons for full audit versus q-value
and stability. The full-audit versus q-value one-sided descriptive P value is printed in B and C. A
layout-only preview carries a visible watermark and contains no study results.

## Color and typography contract

- Full audit: blue `#0072B2`.
- q-value matched: vermillion `#D55E00`.
- Stability matched: bluish green `#009E73`.
- Raw pool: gray `#6B7280`.
- Base font: 12 pt; panel labels: 16 pt; raster export: 600 dpi; PDF text remains editable.

Colors are redundant with labels and fixed panel order. The raw pool must always be labeled
“descriptive”; it must not be presented as a coverage-matched primary contrast.

## Working legend

**Figure 2. Same-pool external evaluation of audited and simpler pathway-reporting rules.**
(A) Fifty deterministic HNSC Hallmark claims were frozen before external evaluation. The full audit,
q-value reporting, and stability reporting each selected 25 claims; the complete pool is shown only
as a descriptive reference. PubMed evidence grading and three independent ratings were completed
without audit status or method membership and locked before outcomes were joined to the reporting
rules. (B) Fraction of claims with independent direction-matched external support, defined as at
least one eligible E3/E4 record from independent data. (C) Fraction rated as major wording
overstatement by majority vote of three blinded raters. In B–C, bars show fractions, error bars show
Wilson 95% confidence intervals, and labels show numerator/denominator. Pdesc is an overlap-aware
conditional exact reference that fixes common claims and exchanges membership only within the
symmetric difference; it is descriptive because pathway outcomes are dependent. (D) Fleiss' kappa
for the three rating questions, with the proportion of claims receiving unanimous ratings annotated.

Do not finalize this legend's result clauses until scripts 32–43 have locked P3/P4 and script 60 has
created the Figure 2 source manifest.
