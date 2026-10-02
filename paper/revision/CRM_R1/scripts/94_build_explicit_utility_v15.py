#!/usr/bin/env python3
"""Opt-in development utility with explicit columns and provenance, no automatic fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from v15_diagnostic_common import (
    finish,
    new_output,
    numeric,
    read_tsv,
    record,
    require,
    write_tsv,
)


def explicit_utility(frame, contract):
    require(contract["status"] == "DEVELOPMENT_ONLY", "Only development output is allowed")
    context = contract["context"]
    require(
        context["kind"] in {"llm_derived_score", "externally_measured_score"},
        "Hash proxy or undocumented context provenance is forbidden",
    )
    require(bool(str(context.get("provenance", "")).strip()), "Context provenance is required")
    columns = [contract["evidence_column"], contract["stability_column"], context["column"]]
    require(
        not any("proxy" in c.lower() or "hash" in c.lower() for c in columns),
        "Proxy/hash columns cannot supply development utility",
    )
    require(set(["claim_id"] + columns).issubset(frame), "Explicit input column missing")
    require(
        frame.claim_id.str.strip().ne("").all() and not frame.claim_id.duplicated().any(),
        "Invalid claim IDs",
    )
    e = numeric(frame[columns[0]])
    require(e.ge(0).all(), "Evidence strength must be nonnegative")
    s, c = numeric(frame[columns[1]], bounded=True), numeric(frame[columns[2]], bounded=True)
    out = pd.DataFrame(
        {
            "claim_id": frame.claim_id,
            "utility_score_v15_development": e * s * c,
            "evidence_source_column": columns[0],
            "stability_source_column": columns[1],
            "context_source_column": columns[2],
            "context_source_kind": context["kind"],
        }
    )
    out = out.sort_values(["utility_score_v15_development", "claim_id"], ascending=[False, True])
    out["rank_v15_development"] = range(1, len(out) + 1)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    contract = json.loads(args.contract.read_text())
    result = explicit_utility(read_tsv(args.input), contract)
    with new_output(args.output) as output:
        write_tsv(output / "utility_v15_development.private.tsv", result)
        finish(
            output,
            [
                record(p)
                for p in [
                    args.input,
                    args.contract,
                    Path(__file__),
                    Path(__file__).with_name("v15_diagnostic_common.py"),
                ]
            ],
            {
                "contract": contract,
                "biological_validation": False,
                "historical_figures_replaced": False,
                "limitation": "Declared provenance is required, not independently authenticated.",
            },
        )
    print("[PASS] Development-only utility written:", args.output)


if __name__ == "__main__":
    main()
