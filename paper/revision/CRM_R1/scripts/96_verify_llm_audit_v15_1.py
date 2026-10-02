#!/usr/bin/env python3
"""Issue a technical receipt only after checking every audit row against its raw cache."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
from v15_1_provenance_common import (
    code_records,
    finish,
    fresh_output,
    numeric,
    read_json,
    read_table,
    record,
    require,
    sha,
    unique,
    write_table,
)


def valid_reason(value):
    return (
        isinstance(value, str)
        and bool(value.strip())
        and value.strip().lower() not in {"none", "null", "nan", "n/a"}
    )


def validate(audit, evidence, meta, caches, expected_model):
    for frame in (audit, evidence):
        unique(frame, "term_uid")
    unique(audit, "claim_id")
    require(len(audit) > 0, "Empty audit")
    require(
        set(audit.term_uid) == set(evidence.term_uid),
        "Audit does not cover full evidence pool",
    )
    required = {
        "context_review_mode",
        "context_method",
        "context_evaluated",
        "context_status",
        "context_reason",
        "context_confidence",
        "context_signature",
        "status",
    }
    require(required <= set(audit), "Missing audit provenance fields")
    require(meta.get("status") == "ok", "Pipeline did not complete successfully")
    claims = meta.get("inputs", {}).get("claims", {})
    llm = meta.get("inputs", {}).get("llm", {})
    require(claims.get("context_review_mode") == "llm", "Proxy/off/fallback run is not LLM")
    require(
        claims.get("context_review_mode_effective_for_select") == "llm",
        "Selection did not use LLM review",
    )
    require(
        llm.get("review", {}).get("enabled") is True
        and llm.get("review", {}).get("backend_enabled") is True
        and llm.get("select_entrypoint", {}).get("backend_attached") is True,
        "LLM backend was not recorded as active",
    )
    identity = llm.get("backend_identity", {})
    require(identity.get("model_name") == expected_model, "Recorded model name mismatch")
    require(identity.get("class") == "OllamaBackend", "Expected an actual Ollama backend")
    host = str(identity.get("host", ""))
    parts = urlsplit(host)
    require(
        parts.scheme in {"http", "https"}
        and bool(parts.hostname)
        and not any(token in host for token in ("[", "]", "(", ")", "\n", " ")),
        "Malformed backend HTTP URL",
    )
    for col in ("context_method", "context_review_mode"):
        require(audit[col].str.lower().eq("llm").all(), "Proxy or unevaluated context row")
    require(
        audit.context_evaluated.str.lower().isin(["true", "1"]).all(),
        "Unevaluated context row",
    )
    require(audit.context_reason.map(valid_reason).all(), "Missing LLM context reason")
    confidence = numeric(audit.context_confidence, lower=0, upper=1)
    require(
        audit.context_status.isin(["PASS", "WARN", "FAIL"]).all(),
        "Invalid context status",
    )
    require(audit.status.isin(["PASS", "ABSTAIN", "FAIL"]).all(), "Invalid final status")
    require(
        audit.context_signature.str.fullmatch("[0-9a-f]{12}").all(),
        "Invalid context key",
    )
    require(audit.context_signature.nunique() == 1, "Multiple contexts in one run")
    expected_keys = (audit.context_signature + "::" + audit.term_uid).tolist()
    require(
        isinstance(caches, dict) and set(caches) == set(expected_keys),
        "Raw cache key set differs from complete audit pool",
    )
    output = []
    for position, row in enumerate(audit.itertuples(index=False)):
        key = expected_keys[position]
        cached = caches[key]
        require(isinstance(cached, dict), f"Invalid cache record: {key}")
        require(not cached.get("error"), f"Technical backend error: {key}")
        require(
            cached.get("status") in {"PASS", "WARN", "FAIL"},
            f"Invalid raw status: {key}",
        )
        c = cached.get("confidence")
        require(
            isinstance(c, (int, float))
            and not isinstance(c, bool)
            and math.isfinite(c)
            and 0 <= c <= 1,
            f"Invalid raw confidence: {key}",
        )
        reason = cached.get("reason")
        require(valid_reason(reason), f"Missing raw reason: {key}")
        reason = reason.strip()
        require(
            row.context_status == cached["status"],
            f"Cache/audit status mismatch: {key}",
        )
        require(
            abs(confidence.iloc[position] - c) <= 1e-12,
            f"Cache/confidence mismatch: {key}",
        )
        require(
            row.context_reason.strip() in {reason, reason[:160]},
            f"Cache/audit reason mismatch: {key}",
        )
        output.append(
            {
                "claim_id": row.claim_id,
                "term_uid": row.term_uid,
                "cache_key": key,
                "context_status": cached["status"],
                "context_reason_full": reason,
                "context_confidence_self_reported": c,
                "final_status": row.status,
            }
        )
    k = int(audit.status.eq("PASS").sum())
    return pd.DataFrame(output), {
        "technical_status": "VALID_ZERO_K" if k == 0 else "VALID_NONZERO_K",
        "rows": len(audit),
        "final_pass_count": k,
        "expected_model_name": expected_model,
        "recorded_backend": identity,
        "model_digest_verified": False,
        "biological_validity_established": False,
        "confidence_is_calibrated": False,
        "zero_k_retry_authorized": False,
        "confidence_used_as_utility": False,
        "limitation": "Consistency of supplied artifacts, not authentication of server weights.",
    }


def check_bundle(run_dir, expected_model):
    files = [
        run_dir / name
        for name in (
            "audit_log.tsv",
            "evidence.normalized.tsv",
            "run_meta.json",
            "sample_card.normalized.json",
        )
    ]
    cache_files = sorted(run_dir.glob("context_review_cache*.json"))
    require(
        bool(cache_files),
        "Raw context caches missing; no technical receipt can be issued",
    )
    inputs = [record(p) for p in files + cache_files]
    audit, evidence, meta = (
        read_table(files[0]),
        read_table(files[1]),
        read_json(files[2]),
    )
    require(
        sha(files[1]) == meta["inputs"]["evidence_normalized_sha256"],
        "Evidence hash mismatch",
    )
    require(
        sha(files[3]) == meta["inputs"]["sample_card_normalized_sha256"],
        "Card hash mismatch",
    )
    if "term_uid" not in evidence:
        require({"source", "term_id"} <= set(evidence), "Evidence term identity missing")
        evidence["term_uid"] = evidence.source + ":" + evidence.term_id
    caches = {}
    for path in cache_files:
        chunk = read_json(path)
        require(isinstance(chunk, dict), "Cache is not an object")
        require(not (set(caches) & set(chunk)), "Duplicate cache key across files")
        caches.update(chunk)
    annotations, summary = validate(audit, evidence, meta, caches, expected_model)
    return annotations, summary, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    annotations, summary, inputs = check_bundle(args.run_dir, args.expected_model)
    with fresh_output(args.output) as out:
        write_table(out / "verified_context_annotations.private.tsv", annotations)
        finish(out, inputs + code_records(__file__), summary)
    print("[PASS] Technical artifact check:", summary["technical_status"])
    print("[INFO] This is not biological validation or permission to change a frozen lock")


if __name__ == "__main__":
    main()
