#!/usr/bin/env python3
"""Reconstruct legacy SHA256 context and connect it to declared old figure inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from v15_1_provenance_common import (
    code_records,
    finish,
    fresh_output,
    numeric,
    proxy_v2,
    read_json,
    read_table,
    record,
    require,
    sha,
    unique,
    write_table,
)


def reconstruct(audit, ranked, meta):
    for table in (audit, ranked):
        unique(table, "claim_id")
        unique(table, "term_uid")
    require(set(audit.term_uid) == set(ranked.term_uid), "Candidate universe mismatch")
    require(
        {"context_ctx_id", "context_keys"} <= set(audit),
        "Historical hash inputs missing",
    )
    require(
        {"context_fit", "utility_score", "evidence_strength", "stability"} <= set(ranked),
        "Historical utility components missing",
    )
    source = audit.set_index("term_uid").loc[ranked.term_uid].reset_index()
    require(source.claim_id.tolist() == ranked.claim_id.tolist(), "Claim/term mapping drift")
    hashes = pd.Series(
        [
            proxy_v2(r.context_ctx_id, r.context_keys.split(","), r.term_uid)
            for r in source.itertuples()
        ],
        index=ranked.index,
    )
    c = numeric(ranked.context_fit, lower=0, upper=1)
    product = numeric(ranked.evidence_strength, lower=0) * numeric(
        ranked.stability, lower=0, upper=1
    )
    product = product * hashes
    observed_u = numeric(ranked.utility_score, lower=0)
    rows = pd.DataFrame(
        {
            "claim_id": ranked.claim_id,
            "term_uid": ranked.term_uid,
            "reconstructed_hash_C": hashes,
            "historical_context_fit": c,
            "context_absolute_difference": (c - hashes).abs(),
            "context_matches_hash": np.isclose(c, hashes, rtol=1e-12, atol=1e-15),
            "historical_utility": observed_u,
            "reconstructed_E_S_hash": product,
            "utility_absolute_difference": (observed_u - product).abs(),
            "utility_matches_E_S_hash": np.isclose(observed_u, product, rtol=1e-12, atol=1e-15),
        }
    )
    aliases = []
    for name in (
        "context_score_proxy_u01_norm",
        "context_score_proxy_u01",
        "context_score_proxy_u01_from_proposed",
        "context_score",
        "context_confidence",
    ):
        if name in source:
            values = pd.to_numeric(source[name], errors="coerce")
            aliases.append(
                {
                    "column": name,
                    "rows": len(rows),
                    "hash_matches": int(np.isclose(values, hashes, rtol=1e-12, atol=1e-15).sum()),
                }
            )
    inputs = meta.get("inputs", {})
    llm = inputs.get("llm", {})
    flags = {
        "claim_enabled": llm.get("claim", {}).get("enabled"),
        "review_enabled": llm.get("review", {}).get("enabled"),
        "backend_attached": llm.get("select_entrypoint", {}).get("backend_attached"),
    }
    mode = inputs.get("claims", {}).get("context_review_mode")
    summary = {
        "rows": len(rows),
        "hash_context_matches": int(rows.context_matches_hash.sum()),
        "hash_utility_matches": int(rows.utility_matches_E_S_hash.sum()),
        "maximum_context_absolute_difference": float(rows.context_absolute_difference.max()),
        "maximum_utility_absolute_difference": float(rows.utility_absolute_difference.max()),
        "context_review_mode": mode,
        "recorded_llm_flags": flags,
        "recorded_run_is_proxy_without_llm": mode == "proxy"
        and all(v is False for v in flags.values()),
        "hash_derivation_confirmed": bool(
            rows.context_matches_hash.all() and rows.utility_matches_E_S_hash.all()
        ),
        "exact_historical_command_recovered": False,
        "submitted_pdf_assembly_verified": False,
        "interpretation": "Hash-derived values are not measured biological context fit.",
    }
    return rows, pd.DataFrame(aliases), summary


def figure_rows(path, dataset):
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    return table.loc[
        table.pipeline_id.eq(dataset)
        & table.input_paths.str.contains("claims_ranked.tsv", regex=False)
    ].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("audit", "ranked", "run-meta", "sample-card", "figure-map", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    inputs = [
        record(p)
        for p in (
            args.audit,
            args.ranked,
            args.run_meta,
            args.sample_card,
            args.figure_map,
        )
    ] + code_records(__file__)
    meta = read_json(args.run_meta)
    require(
        sha(args.sample_card) == meta["inputs"]["sample_card_normalized_sha256"],
        "Normalized card does not match recorded run",
    )
    rows, aliases, summary = reconstruct(read_table(args.audit), read_table(args.ranked), meta)
    mapping = figure_rows(args.figure_map, args.dataset)
    summary["dataset"] = args.dataset
    summary["declared_figure_panels"] = mapping[["figure_id", "panel"]].to_dict("records")
    summary["mapping_evidence"] = "repository FIGURE_MAP; not final submitted PDF byte lineage"
    with fresh_output(args.output) as out:
        write_table(out / "hash_reconstruction.private.tsv", rows)
        write_table(out / "hash_aliases.tsv", aliases)
        write_table(out / "declared_figure_dependencies.tsv", mapping)
        finish(out, inputs, summary)
    print("[PASS] Legacy derivation checked:", args.output)
    print("Hash C matches:", summary["hash_context_matches"], "/", summary["rows"])


if __name__ == "__main__":
    main()
