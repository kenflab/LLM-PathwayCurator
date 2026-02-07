#!/usr/bin/env python3
# paper/scripts/figS3_make_llm_examples.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
def _u(s) -> str:
    return str(s).strip().upper()


def _is_na(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and pd.isna(x):
        return True
    s = str(x).strip().lower()
    return s in {"", "na", "nan", "none", "null"}


def _as_float_or_none(x) -> float | None:
    try:
        if _is_na(x):
            return None
        return float(str(x).strip())
    except Exception:
        return None


def _as_bool_or_none(x) -> bool | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    if s in {"", "na", "nan", "none", "null"}:
        return None
    return None


def _load_claim_json_from_row(row: pd.Series) -> dict:
    for c in ["claim_json", "claim", "json", "payload_json"]:
        if c in row.index and (not _is_na(row.get(c))):
            try:
                obj = json.loads(row[c])
            except Exception as e:
                raise SystemExit(f"Failed to parse JSON in column={c}: {e}") from e
            if not isinstance(obj, dict):
                raise SystemExit(f"Claim JSON in column={c} is not an object.")
            return obj
    raise SystemExit(
        "No claim JSON column found (expected claim_json / claim / json / payload_json)."
    )


def _require_evidence_ref(claim: dict) -> dict:
    """
    Accept both old/new EvidenceRef shapes.

    We require:
      - term_ids: list[str]
      - module_id: str
      - gene_set_hash: str
      - gene_ids OR gene_symbols OR gene_symbols_str: list[str] (at least one present)
    """
    evref = claim.get("evidence_ref")
    if not isinstance(evref, dict) or not evref:
        raise SystemExit(
            "Claim JSON missing evidence_ref "
            "(expected dict with term_ids/module_id/gene_ids(or gene_symbols)/gene_set_hash)."
        )

    term_ids = evref.get("term_ids")
    module_id = evref.get("module_id")
    gene_set_hash = evref.get("gene_set_hash")

    # Be flexible: your current pipeline writes gene_ids (IDs), not symbols.
    gene_ids = evref.get("gene_ids")
    gene_symbols = evref.get("gene_symbols")  # optional future
    gene_symbols_str = evref.get("gene_symbols_str")  # legacy

    if not isinstance(term_ids, list) or not any(str(x).strip() for x in term_ids):
        raise SystemExit("evidence_ref.term_ids missing/empty in claim JSON.")
    if _is_na(module_id):
        raise SystemExit("evidence_ref.module_id missing/empty in claim JSON.")
    if _is_na(gene_set_hash):
        raise SystemExit("evidence_ref.gene_set_hash missing/empty in claim JSON.")

    # Require at least one gene list present (IDs or symbols)
    has_gene_ids = isinstance(gene_ids, list) and any(str(x).strip() for x in gene_ids)
    has_gene_symbols = isinstance(gene_symbols, list) and any(str(x).strip() for x in gene_symbols)
    has_gene_symbols_str = isinstance(gene_symbols_str, list) and any(
        str(x).strip() for x in gene_symbols_str
    )
    if not (has_gene_ids or has_gene_symbols or has_gene_symbols_str):
        raise SystemExit(
            "evidence_ref missing genes list in claim JSON "
            "(expected gene_ids OR gene_symbols OR gene_symbols_str)."
        )

    return evref


def _pick_examples_from_audit(audit: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Return (fallback_example, llm_example).

    fallback_example:
      - context_method != llm (prefer proxy/none) AND status == PASS/ABSTAIN

    llm_example:
      - context_method == llm AND context_status in PASS/FAIL
    """
    if audit.empty:
        raise SystemExit("audit_log.tsv is empty.")

    st = (
        audit["status"].astype(str).str.strip().str.upper()
        if "status" in audit.columns
        else pd.Series([""] * len(audit), index=audit.index)
    )
    cm = (
        audit["context_method"].astype(str).str.strip().str.lower()
        if "context_method" in audit.columns
        else pd.Series([""] * len(audit), index=audit.index)
    )
    cst = (
        audit["context_status"].astype(str).str.strip().str.upper()
        if "context_status" in audit.columns
        else pd.Series([""] * len(audit), index=audit.index)
    )

    # Fallback example
    cand_fb = audit.copy()
    cand_fb["_st"] = st
    cand_fb["_cm"] = cm
    cand_fb = cand_fb[cand_fb["_st"].isin(["PASS", "ABSTAIN"])].copy()
    cand_fb = cand_fb[cand_fb["_cm"] != "llm"].copy()
    if cand_fb.empty:
        cand_fb = audit.copy()
        cand_fb["_st"] = st
        cand_fb = cand_fb[cand_fb["_st"].isin(["PASS", "ABSTAIN"])].copy()
    fb_row = cand_fb.iloc[0]

    # LLM example
    cand_llm = audit.copy()
    cand_llm["_cm"] = cm
    cand_llm["_cst"] = cst
    cand_llm = cand_llm[
        (cand_llm["_cm"] == "llm") & (cand_llm["_cst"].isin(["PASS", "FAIL"]))
    ].copy()
    if cand_llm.empty:
        cand_llm = audit.copy()
        cand_llm["_cm"] = cm
        cand_llm = cand_llm[cand_llm["_cm"] == "llm"].copy()
    llm_row = cand_llm.iloc[0] if not cand_llm.empty else audit.iloc[0]

    return fb_row, llm_row


def _synthesize_context_review_from_audit_row(row: pd.Series, claim: dict) -> dict:
    """
    FigS1: show that LLM/proxy produces typed review fields;
    audit consumes them mechanically; evidence_ref anchors everything.
    """
    evref = _require_evidence_ref(claim)

    status = _u(row.get("context_status_from_proposed", ""))
    if status not in {"PASS", "WARN", "FAIL"}:
        status = "WARN"

    # Prefer gene_ids (current pipeline), fall back to symbols if present
    gene_ids = evref.get("gene_ids")
    gene_symbols = evref.get("gene_symbols")
    gene_symbols_str = evref.get("gene_symbols_str")

    evidence_ref_out = {
        "term_ids": evref.get("term_ids", []),
        "module_id": evref.get("module_id", ""),
        "gene_set_hash": evref.get("gene_set_hash", ""),
    }
    if isinstance(gene_ids, list) and any(str(x).strip() for x in gene_ids):
        evidence_ref_out["gene_ids"] = gene_ids
    elif isinstance(gene_symbols, list) and any(str(x).strip() for x in gene_symbols):
        evidence_ref_out["gene_symbols"] = gene_symbols
    elif isinstance(gene_symbols_str, list) and any(str(x).strip() for x in gene_symbols_str):
        evidence_ref_out["gene_symbols"] = gene_symbols_str

    out = {
        "context_status": status,
        "context_evaluated": _as_bool_or_none(row.get("context_evaluated", None)),
        "context_method": str(row.get("context_method", "") or "").strip(),
        "context_review_mode": str(row.get("context_method_from_proposed", "") or "").strip(),
        "context_gate_mode": str(row.get("context_gate_mode", "") or "").strip(),
        "context_confidence": _as_float_or_none(row.get("context_confidence", None)),
        "context_reason": str(row.get("context_reason_from_proposed", "") or "").strip(),
        "context_notes": str(row.get("context_notes", "") or "").strip(),
        "ctx_id_effective": str(
            row.get("context_ctx_id_effective", "") or row.get("context_ctx_id", "") or ""
        ).strip(),
        "context_signature": str(row.get("context_signature", "") or "").strip(),
        "context_proxy_u01": _as_float_or_none(row.get("context_score_proxy_u01", None)),
        "evidence_ref": evidence_ref_out,
    }

    # Clean Nones/empties (paper-friendly)
    for k in list(out.keys()):
        v = out.get(k)
        if v is None:
            del out[k]
        elif isinstance(v, str) and v.strip() == "":
            del out[k]

    # Ensure evidence_ref keys exist / types sane
    ev = out.get("evidence_ref", {})
    if not isinstance(ev.get("term_ids", None), list):
        ev["term_ids"] = []
    ev["module_id"] = str(ev.get("module_id", "") or "")
    ev["gene_set_hash"] = str(ev.get("gene_set_hash", "") or "")
    out["evidence_ref"] = ev

    return out


def _audit_decision_record(row: pd.Series) -> dict:
    rec = {
        "status": _u(row.get("status", "")),
        "abstain_reason": str(row.get("abstain_reason", "") or "").strip(),
        "fail_reason": str(row.get("fail_reason", "") or "").strip(),
        "audit_notes": str(row.get("audit_notes", "") or "").strip(),
        "link_ok": _as_bool_or_none(row.get("link_ok", None)),
        "stability_ok": _as_bool_or_none(row.get("stability_ok", None)),
        "stress_evaluated": _as_bool_or_none(row.get("stress_evaluated", None)),
        "stress_ok": _as_bool_or_none(row.get("stress_ok", None)),
        "contradiction_ok": _as_bool_or_none(row.get("contradiction_ok", None)),
        "tau_used": _as_float_or_none(row.get("tau_used", None)),
        "term_survival_agg": _as_float_or_none(row.get("term_survival_agg", None)),
        "stability_scope": str(row.get("stability_scope", "") or "").strip(),
        "evidence_key": str(row.get("evidence_key", "") or "").strip(),
        "gene_set_hash_effective": str(row.get("gene_set_hash_effective", "") or "").strip(),
        "gene_set_hash_match_mode": str(row.get("gene_set_hash_match_mode", "") or "").strip(),
        "term_ids_set_hash": str(row.get("term_ids_set_hash", "") or "").strip(),
        "direction_norm": str(row.get("direction_norm", "") or "").strip(),
    }

    for k in list(rec.keys()):
        v = rec.get(k)
        if v is None:
            del rec[k]
        elif isinstance(v, str) and v.strip() == "":
            del rec[k]

    if rec.get("status") not in {"PASS", "ABSTAIN", "FAIL"}:
        rec["status"] = str(row.get("status", "") or "").strip()
    return rec


def _evidence_excerpt(evidence_table_path: Path, claim: dict, *, n_rows: int) -> pd.DataFrame:
    ev = pd.read_csv(evidence_table_path, sep="\t")
    evref = _require_evidence_ref(claim)

    term_ids = set(str(x).strip() for x in (evref.get("term_ids") or []) if str(x).strip())
    module_id = str(evref.get("module_id") or "").strip()

    keep = pd.Series([False] * len(ev), index=ev.index)

    # Evidence tables may have term_uid or term_id; try both.
    if term_ids:
        if "term_uid" in ev.columns:
            keep |= ev["term_uid"].astype(str).isin(term_ids)
        if "term_id" in ev.columns:
            keep |= ev["term_id"].astype(str).isin(term_ids)

    if module_id and "module_id" in ev.columns:
        keep |= ev["module_id"].astype(str).eq(module_id)

    sub = ev[keep].copy() if keep.any() else ev.head(int(n_rows)).copy()
    sub = sub.head(int(n_rows))

    cols = [
        c
        for c in [
            "term_uid",
            "term_id",
            "term_name",
            "source",
            "qval",
            "direction",
            "module_id",
            "evidence_genes",
        ]
        if c in sub.columns
    ]
    if cols:
        sub = sub[cols]
    return sub


def _write_example_bundle(
    out_dir: Path, *, tag: str, audit_row: pd.Series, evidence_table: Path, n_evidence_rows: int
) -> None:
    claim = _load_claim_json_from_row(audit_row)

    (out_dir / f"claim.proposed.{tag}.json").write_text(
        json.dumps(claim, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    ctx = _synthesize_context_review_from_audit_row(audit_row, claim)
    (out_dir / f"context_review.{tag}.json").write_text(
        json.dumps(ctx, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    dec = _audit_decision_record(audit_row)
    (out_dir / f"audit_decision.{tag}.json").write_text(
        json.dumps(dec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    sub = _evidence_excerpt(evidence_table, claim, n_rows=n_evidence_rows)
    sub.to_csv(out_dir / f"evidence_ref.{tag}.tsv", sep="\t", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help=".../HNSC/ours/gate_hard/tau_0.80")
    ap.add_argument(
        "--evidence-table",
        required=True,
        help="paper/source_data/.../evidence_tables/<COND>.evidence_table.tsv",
    )
    ap.add_argument(
        "--out-examples-dir",
        required=True,
        help="Output root; examples will be written into <out-examples-dir>/examples/",
    )
    ap.add_argument("--n-evidence-rows", type=int, default=8)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_root = Path(args.out_examples_dir)
    out_dir = out_root / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_tsv = run_dir / "audit_log.tsv"
    if not audit_tsv.exists():
        raise SystemExit(
            f"Missing: {audit_tsv}\n"
            "FigS1 examples should be produced from audit_log.tsv to demonstrate "
            "the mechanical cage."
        )

    audit = pd.read_csv(audit_tsv, sep="\t")
    if audit.empty:
        raise SystemExit(f"Empty TSV: {audit_tsv}")

    fb_row, llm_row = _pick_examples_from_audit(audit)

    evidence_table = Path(args.evidence_table)
    if not evidence_table.exists():
        raise SystemExit(f"Missing evidence table: {evidence_table}")

    _write_example_bundle(
        out_dir,
        tag="fallback_proxy",
        audit_row=fb_row,
        evidence_table=evidence_table,
        n_evidence_rows=int(args.n_evidence_rows),
    )
    _write_example_bundle(
        out_dir,
        tag="llm_review",
        audit_row=llm_row,
        evidence_table=evidence_table,
        n_evidence_rows=int(args.n_evidence_rows),
    )

    print("[figS1_make_examples] wrote bundles:")
    for tag in ["fallback_proxy", "llm_review"]:
        print(f"  - {out_dir}/claim.proposed.{tag}.json")
        print(f"  - {out_dir}/context_review.{tag}.json")
        print(f"  - {out_dir}/audit_decision.{tag}.json")
        print(f"  - {out_dir}/evidence_ref.{tag}.tsv")


if __name__ == "__main__":
    main()
