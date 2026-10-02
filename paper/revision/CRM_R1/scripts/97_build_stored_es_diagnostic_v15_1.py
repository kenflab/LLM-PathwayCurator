#!/usr/bin/env python3
"""All-candidate E*S diagnostic; no context factor, PASS filter, or silent defaults."""

from __future__ import annotations

import argparse
from pathlib import Path

from v15_1_provenance_common import (
    code_records,
    finish,
    fresh_output,
    read_table,
    record,
    require,
    stored_es,
    write_table,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--evidence-column", required=True)
    parser.add_argument("--stability-column", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    inputs = [record(args.input)] + code_records(__file__)
    frame = read_table(args.input)
    require(
        len(frame) == args.expected_rows > 0,
        "Candidate count differs from declared full pool",
    )
    result = stored_es(frame, args.evidence_column, args.stability_column)
    with fresh_output(args.output) as out:
        write_table(out / "stored_es_all_candidates.private.tsv", result)
        if args.plot:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            shown = result.head(15).iloc[::-1]
            labels = shown.term_uid.str.split(":").str[-1].str.replace("HALLMARK_", "", regex=False)
            fig, axis = plt.subplots(figsize=(10, 7), layout="constrained")
            axis.barh(range(len(shown)), shown.stored_es_diagnostic, color="#426A8C")
            axis.set_yticks(range(len(shown)), labels.str.replace("_", " ", regex=False))
            axis.set_xlabel("Stored E × S (context omitted)")
            axis.set_title(
                f"DEVELOPMENT DIAGNOSTIC — top 15 of {len(result)} input candidates\n"
                "No decision filter; no biological validation"
            )
            fig.savefig(out / "stored_es_development.pdf")
            fig.savefig(out / "stored_es_development.png", dpi=150)
            plt.close(fig)
        finish(
            out,
            inputs,
            {
                "rows": len(result),
                "formula": "explicit E*S; not E*S*C with C imputed",
                "evidence_column": args.evidence_column,
                "stability_column": args.stability_column,
                "context_used_for_score_or_filter": False,
                "decision_filter": None,
                "scope": "all supplied rows; upstream selection is not reconstructed",
                "replaces_historical_figure": False,
                "biological_validation": False,
                "component_semantics_independently_verified": False,
                "warning": "Stored-component sensitivity only; not a corrected full-audit result.",
            },
        )
    print("[PASS] New stored-component development output:", args.output)


if __name__ == "__main__":
    main()
