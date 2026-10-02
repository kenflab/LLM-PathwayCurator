#!/usr/bin/env python3
"""Trace historical utility inputs without assuming a matching value proves provenance."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from v15_diagnostic_common import (
    CONTEXT_COLUMNS,
    finish,
    new_output,
    numeric,
    read_tsv,
    record,
    require,
    write_json,
    write_tsv,
)


def context_precedence(source):
    tree = ast.parse(Path(source).read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(x, ast.Name) and x.id == "context_fit_col" for x in node.targets):
            continue
        for call in ast.walk(node.value):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                if call.func.id == "_first_existing_col" and len(call.args) == 2:
                    return ast.literal_eval(call.args[1])
    raise ValueError("Cannot determine context column precedence from supplied ranked.py")


def formula_check(frame):
    required = [
        "claim_id",
        "utility_score",
        "evidence_strength_filled",
        "stability",
        "context_fit",
        "score_source",
    ]
    require(set(required).issubset(frame), "Missing ranked table fields")
    require(not frame.claim_id.duplicated().any(), "Duplicate ranked claim_id")
    values = {c: numeric(frame[c]) for c in required[1:5]}
    predicted = values["evidence_strength_filled"] * values["stability"] * values["context_fit"]
    delta = np.abs(predicted - values["utility_score"])
    return {
        "rows": len(frame),
        "three_component_product_matches": bool(
            np.allclose(predicted, values["utility_score"], rtol=1e-12, atol=1e-14)
        ),
        "maximum_absolute_product_difference": float(delta.max()),
        "score_sources": ";".join(sorted(set(frame.score_source))),
    }


def sensitivity(frame, ranked_path):
    e = numeric(frame.evidence_strength_filled)
    s, c = numeric(frame.stability), numeric(frame.context_fit)
    scores = {
        "E*S*C_stored_components": e * s * c,
        "E*S_omit_context": e * s,
        "E*C_omit_stability": e * c,
        "S*C_omit_evidence": s * c,
    }
    rows = []
    ref = pd.DataFrame({"claim_id": frame.claim_id, "score": e * s * c}).sort_values(
        ["score", "claim_id"], ascending=[False, True]
    )
    for name, score in scores.items():
        ordering = pd.DataFrame({"claim_id": frame.claim_id, "score": score}).sort_values(
            ["score", "claim_id"], ascending=[False, True]
        )
        for k in sorted({min(10, len(frame)), min(25, len(frame))}):
            overlap = len(set(ref.head(k).claim_id) & set(ordering.head(k).claim_id))
            rows.append(
                {
                    "ranked_path": str(ranked_path),
                    "variant": name,
                    "k": k,
                    "overlap_with_product_top_k": overlap,
                    "overlap_fraction": overlap / k,
                    "biological_performance_evaluated": False,
                }
            )
    return rows


def compare_candidate(ranked, audit, precedence):
    if "claim_id" not in audit or audit.claim_id.duplicated().any():
        return []
    if not set(ranked.claim_id) <= set(audit.claim_id):
        return []
    aligned = audit.set_index("claim_id").loc[ranked.claim_id].reset_index()
    # Require exact term identities in addition to claim IDs when present.
    for key in ["term_uid", "term_id"]:
        if key in ranked and key in aligned:
            if not ranked[key].reset_index(drop=True).eq(aligned[key]).all():
                return []
    chosen = next((c for c in precedence if c in aligned), "DEFAULT_ONE")
    target = numeric(ranked.context_fit).to_numpy()
    rows = []
    for col in dict.fromkeys(precedence + ["context_confidence"]):
        if col not in aligned:
            continue
        raw = pd.to_numeric(aligned[col], errors="coerce")
        # Reproduce historical missing->1, lower clipping; also expose invalid raw values.
        legacy = raw.fillna(1.0).clip(lower=0).to_numpy()
        rows.append(
            {
                "candidate_column": col,
                "column_exists": True,
                "legacy_precedence_choice": chosen,
                "chosen_by_supplied_code": col == chosen,
                "raw_missing_or_nonfinite": int((~np.isfinite(raw)).sum()),
                "raw_outside_0_1": int((raw.notna() & ~raw.between(0, 1)).sum()),
                "context_fit_matches": bool(np.allclose(legacy, target, rtol=1e-12, atol=1e-14)),
                "max_absolute_difference": float(np.max(np.abs(legacy - target))),
                "historical_input_identity_proven": False,
            }
        )
    return rows


def search_inputs(roots):
    ranked, audits, metadata = set(), set(), set()
    for root in roots:
        require(root.is_dir(), f"Search root missing: {root}")
        ranked.update(p.resolve() for p in root.rglob("claims_ranked.tsv") if p.is_file())
        audits.update(p.resolve() for p in root.rglob("audit_log.tsv") if p.is_file())
        metadata.update(p.resolve() for p in root.rglob("*run_meta.json") if p.is_file())
    return sorted(ranked), sorted(audits), sorted(metadata)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--search-root", type=Path, action="append", required=True)
    p.add_argument("--ranked-code", type=Path, required=True)
    p.add_argument("--code-root", type=Path, action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    require(not args.output.exists(), "Utility output already exists")
    precedence = context_precedence(args.ranked_code)
    ranked_paths, audit_paths, meta_paths = search_inputs(args.search_root)
    require(ranked_paths, "No claims_ranked.tsv found in the specified roots")
    inputs = [
        record(args.ranked_code),
        record(Path(__file__)),
        record(Path(__file__).with_name("v15_diagnostic_common.py")),
    ]
    audits = []
    audit_cols = set(CONTEXT_COLUMNS + ["claim_id", "term_id", "term_uid", "context_confidence"])
    for path in audit_paths:
        audit = pd.read_csv(
            path, sep="\t", dtype=str, keep_default_na=False, usecols=lambda c: c in audit_cols
        )
        audits.append((path, audit))
        inputs.append(record(path))
    summaries, candidates, ranks, unresolved, references = [], [], [], [], []
    for path in ranked_paths:
        ranked = read_tsv(path)
        inputs.append(record(path))
        info = {
            "ranked_path": str(path),
            **formula_check(ranked),
            "provenance_status": "UNRESOLVED_REQUIRES_HISTORICAL_INPUT_AND_RUN_RECORD",
        }
        count = 0
        for audit_path, audit in audits:
            matches = compare_candidate(ranked, audit, precedence)
            for row in matches:
                candidates.append({"ranked_path": str(path), "audit_path": str(audit_path), **row})
                count += int(row["context_fit_matches"] and row["chosen_by_supplied_code"])
        info["matching_candidate_inputs_under_supplied_code"] = count
        if count:
            info["provenance_status"] = "NUMERIC_MATCH_ONLY_HISTORICAL_LINEAGE_UNPROVEN"
        summaries.append(info)
        ranks.extend(sensitivity(ranked, path))
        unresolved.append(
            {
                "ranked_path": str(path),
                "required_artifacts": (
                    "original audit_log.tsv; evidence.normalized.tsv; ranking command; "
                    "ranking source/version and run metadata with input hashes; figure inputs"
                ),
                "reason": "Equal numbers alone do not prove which input was used.",
            }
        )
    for path in meta_paths:
        obj = json.loads(path.read_text())
        inputs.append(record(path))
        text = json.dumps(obj)
        if "ranked" in text or "utility" in text:
            references.append(
                {
                    "path": str(path),
                    "kind": "metadata_reference",
                    "line": 0,
                    "text": "Contains ranked/utility references; inspect",
                }
            )
    for root in args.code_root:
        require(root.is_dir(), f"Code root missing: {root}")
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".R", ".sh"}:
                continue
            text = path.read_text(encoding="utf-8")
            hits = []
            for n, line in enumerate(text.splitlines(), 1):
                if any(
                    token in line
                    for token in [
                        "claims_ranked",
                        "context_score_proxy_u01",
                        "context_fit_col",
                        "build_claims_ranked",
                    ]
                ):
                    hits.append(
                        {
                            "path": str(path.resolve()),
                            "kind": "code_reference",
                            "line": n,
                            "text": line.strip(),
                        }
                    )
            if hits:
                inputs.append(record(path))
                references.extend(hits)
    details = {
        "ranked_tables": len(ranked_paths),
        "audit_candidates": len(audit_paths),
        "context_precedence_in_supplied_code": precedence,
        "historical_provenance_resolved": False,
        "sensitivity_role": "stored-component rank sensitivity, not biological validation",
        "frozen_membership_uses_utility": False,
    }
    with new_output(args.output) as output:
        write_tsv(output / "utility_formula_and_provenance.tsv", pd.DataFrame(summaries))
        write_tsv(
            output / "candidate_input_matches.private.tsv",
            pd.DataFrame(
                candidates,
                columns=[
                    "ranked_path",
                    "audit_path",
                    "candidate_column",
                    "column_exists",
                    "legacy_precedence_choice",
                    "chosen_by_supplied_code",
                    "raw_missing_or_nonfinite",
                    "raw_outside_0_1",
                    "context_fit_matches",
                    "max_absolute_difference",
                    "historical_input_identity_proven",
                ],
            ),
        )
        write_tsv(output / "three_component_rank_sensitivity.tsv", pd.DataFrame(ranks))
        write_tsv(output / "missing_provenance.tsv", pd.DataFrame(unresolved))
        write_tsv(
            output / "figure_and_code_references.private.tsv",
            pd.DataFrame(references, columns=["path", "kind", "line", "text"]),
        )
        write_json(output / "utility_summary.json", details)
        finish(output, inputs, details)
    print("[PASS] Utility trace completed:", args.output)
    print("[OPEN] Historical input lineage still requires the records in missing_provenance.tsv")


if __name__ == "__main__":
    main()
