# LLM-PathwayCurator/src/llm_pathway_curator/report.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .audit_reasons import (
    ABSTAIN_INCONCLUSIVE_STRESS,
    ALL_REASONS,
    FAIL_SCHEMA_VIOLATION,
)
from .claim_schema import Claim, Decision
from .sample_card import SampleCard

# optional metrics export (report.md only)
try:
    from .calibrate import risk_coverage_curve, risk_coverage_from_status
except Exception:  # pragma: no cover
    risk_coverage_curve = None
    risk_coverage_from_status = None

from .utils import build_id_to_symbol_from_distilled, load_id_map_tsv, map_ids_to_symbols

# -----------------------------
# Small helpers (stable)
# -----------------------------
_NA_TOKENS = {"", "na", "nan", "none", "NA"}
_ALLOWED_STATUSES = {"PASS", "ABSTAIN", "FAIL"}


def _safe_table_md(df: pd.DataFrame, n: int = 20) -> str:
    head = df.head(n).copy()
    try:
        return head.to_markdown(index=False)
    except Exception:
        return head.to_csv(sep="\t", index=False)


def _prefer_str_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    For each requested column c, if c_str exists use it (but keep the display name as c).
    This stabilizes markdown output for list-like columns.
    """
    out = pd.DataFrame()
    for c in cols:
        c_str = f"{c}_str"
        if c_str in df.columns:
            out[c] = df[c_str]
        elif c in df.columns:
            out[c] = df[c]
    return out if not out.empty else df.copy()


def _is_na_scalar(x: Any) -> bool:
    """pd.isna is unsafe for list-like; only treat scalars here."""
    if x is None:
        return True
    if isinstance(x, (list, tuple, set, dict)):
        return False
    try:
        v = pd.isna(x)
        return bool(v) if isinstance(v, bool) else False
    except Exception:
        return False


def _stringify_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    TSV-safe conversion for list-like columns.
    - If a <col>_str already exists, prefer it.
    - Otherwise stringify list/tuple/set as separator-joined.
    - Scalar NA -> "" to stabilize artifacts.

    IMPORTANT:
      Use ';' as the list separator to reduce Excel auto-formatting risk.
    """
    LIST_SEP = ";"

    out = df.copy()
    for c in list(out.columns):
        if c.endswith("_str"):
            continue
        c_str = f"{c}_str"
        if c_str in out.columns:
            continue

        s = out[c]
        if s.dtype == "object":
            sample = s.dropna().head(20).tolist()
            if any(isinstance(x, (list, tuple, set)) for x in sample):
                out[c_str] = s.map(
                    lambda x: LIST_SEP.join(map(str, x))
                    if isinstance(x, (list, tuple, set))
                    else ("" if _is_na_scalar(x) else str(x))
                )
            else:
                out[c] = s.map(lambda x: "" if _is_na_scalar(x) else x)
        else:
            out[c] = s.map(lambda x: "" if _is_na_scalar(x) else x)
    return out


def _excel_safe_ids(x: Any, *, list_sep: str = ";") -> str:
    """
    Make an ID field safe for Excel:
      - Accept list-like or scalar.
      - Treat ',', ';', '|', whitespace as separators.
      - Normalize to list_sep.
      - Prefix with a single quote to force Text in Excel.
    """
    if _is_na_scalar(x):
        return ""

    # If already list-like, join directly.
    if isinstance(x, (list, tuple, set)):
        parts = [str(t).strip() for t in x if str(t).strip()]
        s = list_sep.join(parts)
        return s if not s else ("'" + s if not s.startswith("'") else s)

    s0 = str(x).strip()
    if not s0 or s0.lower() in {t.lower() for t in _NA_TOKENS}:
        return ""

    # Normalize all common separators to ';'
    s = (
        s0.replace("|", ";")
        .replace(",", ";")
        .replace("\t", ";")
        .replace("\n", ";")
        .replace(" ", ";")
    )
    parts = [p.strip() for p in s.split(";") if p.strip()]
    s_norm = list_sep.join(parts)

    if not s_norm:
        return ""
    return s_norm if s_norm.startswith("'") else ("'" + s_norm)


def _require_columns(df: pd.DataFrame, cols: list[str], who: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{who} missing required columns: {missing}")


def _get_first_present(row: pd.Series, keys: list[str]) -> Any:
    for k in keys:
        if k in row.index:
            v = row.get(k)
            if v is None or _is_na_scalar(v):
                continue
            return v
    return None


def _normalize_status(raw: Any) -> str:
    s = str(raw or "").upper().strip()
    if s not in _ALLOWED_STATUSES:
        raise ValueError(f"invalid status='{s}' (allowed: {sorted(_ALLOWED_STATUSES)})")
    return s


def _decision_from_audit_row(row: pd.Series) -> Decision:
    """
    Decision is derived mechanically from audit_log row (no free-text generation).
    - status: PASS/ABSTAIN/FAIL
    - reason: known reason code (fallbacks are fixed)
    - details: small structured extras (audit_notes, raw reason if unknown, etc.)
    """
    st = _normalize_status(row.get("status", ""))

    reason = "ok"
    details: dict[str, Any] = {}

    if st == "FAIL":
        r = str(row.get("fail_reason", "") or "").strip()
        reason = r if (r in ALL_REASONS) else FAIL_SCHEMA_VIOLATION
        if r and (r not in ALL_REASONS):
            details["raw_fail_reason"] = r

    elif st == "ABSTAIN":
        r = str(row.get("abstain_reason", "") or "").strip()
        reason = r if (r in ALL_REASONS) else ABSTAIN_INCONCLUSIVE_STRESS
        if r and (r not in ALL_REASONS):
            details["raw_abstain_reason"] = r

    note = str(row.get("audit_notes", "") or "").strip()
    if note:
        details["audit_notes"] = note

    for k in ["tau_used", "term_survival_agg", "stability_scope", "module_id_effective"]:
        if k in row.index and (not _is_na_scalar(row.get(k))):
            details[k] = row.get(k)

    return Decision(status=st, reason=str(reason), details=details)


def _get_tau_default(card: SampleCard, default: float = 0.8) -> float:
    if hasattr(card, "audit_tau") and callable(card.audit_tau):
        try:
            return float(card.audit_tau())  # type: ignore[attr-defined]
        except Exception:
            pass

    v = getattr(card, "audit_tau", None)
    if v is not None:
        try:
            return float(v)
        except Exception:
            pass

    try:
        extra = getattr(card, "extra", {}) or {}
        if isinstance(extra, dict) and "audit_tau" in extra:
            return float(extra["audit_tau"])
    except Exception:
        pass

    return float(default)


def _get_schema_version(card: SampleCard, default: str = "v1") -> str:
    """
    Report JSONL schema contract version (paper artifact).
    Source of truth: card.extra['schema_version'] if present, else default.
    """
    try:
        extra = getattr(card, "extra", {}) or {}
        if isinstance(extra, dict):
            v = extra.get("schema_version", None)
            if v is not None and str(v).strip():
                return str(v).strip()
    except Exception:
        pass
    return str(default)


def _reason_summary(audit_log: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "status" not in audit_log.columns:
        return (pd.DataFrame(), pd.DataFrame())

    abs_df = audit_log[audit_log["status"] == "ABSTAIN"]
    fail_df = audit_log[audit_log["status"] == "FAIL"]

    abs_sum = (
        abs_df["abstain_reason"]
        .fillna("")
        .replace({"": "(missing)"})
        .value_counts()
        .rename_axis("abstain_reason")
        .reset_index(name="n")
        if "abstain_reason" in abs_df.columns
        else pd.DataFrame()
    )
    fail_sum = (
        fail_df["fail_reason"]
        .fillna("")
        .replace({"": "(missing)"})
        .value_counts()
        .rename_axis("fail_reason")
        .reset_index(name="n")
        if "fail_reason" in fail_df.columns
        else pd.DataFrame()
    )
    return abs_sum, fail_sum


def _pick_score_col(audit_log: pd.DataFrame) -> str | None:
    for c in ["term_survival_agg", "context_score", "stat"]:
        if c in audit_log.columns:
            return c
    return None


def _to_float_or_none(x: Any) -> float | None:
    if _is_na_scalar(x):
        return None
    try:
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


# -----------------------------
# JSONL export (paper artifact; stable keys)
# -----------------------------
def write_report_jsonl(
    audit_log: pd.DataFrame,
    card: SampleCard,
    outdir: str,
    *,
    run_id: str,
    method: str | None = None,
    tau: float | None = None,
    cancer: str | None = None,
    comparison: str | None = None,
) -> Path:
    """
    Writes out/report.jsonl.

    Philosophy:
      - claim_json is the single source of truth for the claim structure.
      - decision is derived mechanically from audit_log row.
      - export does NOT reconstruct claim IDs, evidence refs, term IDs, etc.

    v1 compatibility:
      - Adds flat alias fields (schema_version, claim_id, decision, survival, etc.)
        WITHOUT removing the nested 'claim'/'decision_obj'/'metrics' objects.

    Required columns in audit_log:
      - status
      - claim_json
      - term_survival_agg (column must exist; row values may be NA -> null in JSON)
    """
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not run_id:
        run_id = datetime.now(timezone.utc).isoformat()

    _require_columns(audit_log, ["status", "claim_json", "term_survival_agg"], "audit_log")

    # Make row-wise access stable even if audit_log index is non-consecutive.
    audit0 = audit_log.reset_index(drop=True)
    df = _stringify_list_columns(audit0)

    surv_s = pd.to_numeric(audit0["term_survival_agg"], errors="coerce")
    if surv_s.isna().all():
        raise ValueError(
            "audit_log term_survival_agg is all-NA (upstream survival missing entirely; "
            "check masking/term_survival export)"
        )

    ctx_s = (
        pd.to_numeric(audit_log.get("context_score"), errors="coerce")
        if "context_score" in audit_log.columns
        else None
    )
    stat_s = (
        pd.to_numeric(audit0.get("stat"), errors="coerce") if "stat" in audit0.columns else None
    )

    method_val = "llm-pathway-curator" if method is None else str(method).strip()
    tau_val = _get_tau_default(card, default=0.8) if tau is None else float(tau)

    default_cancer = str(cancer or getattr(card, "cancer", None) or "").strip()
    default_comparison = str(comparison or getattr(card, "comparison", None) or "").strip()

    schema_version = _get_schema_version(card, default="v1")

    ctx_disease = str(getattr(card, "disease", "") or "").strip()
    ctx_tissue = str(getattr(card, "tissue", "") or "").strip()
    ctx_pert = str(getattr(card, "perturbation", "") or "").strip()
    ctx_comp = str(getattr(card, "comparison", "") or "").strip()

    species = ""
    try:
        extra = getattr(card, "extra", {}) or {}
        if isinstance(extra, dict):
            species = str(extra.get("species", "") or "").strip()
    except Exception:
        species = ""

    jsonl_path = out_path / "report.jsonl"
    created_at = datetime.now(timezone.utc).isoformat()

    with jsonl_path.open("w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            _ = _normalize_status(row.get("status", ""))

            row_raw = audit0.loc[i]
            decision_obj = _decision_from_audit_row(row_raw)

            cj = str(row.get("claim_json", "") or "").strip()
            if not cj:
                raise ValueError("audit_log has empty claim_json for a row (required)")
            claim = Claim.model_validate_json(cj)
            surv_val = _to_float_or_none(surv_s.loc[i])

            ctx_val = None
            if ctx_s is not None:
                ctx_val = _to_float_or_none(ctx_s.loc[i])

            stat_val = None
            if stat_s is not None:
                stat_val = _to_float_or_none(stat_s.loc[i])

            disease_key = str(
                _get_first_present(row, ["disease"])
                or ctx_disease
                or _get_first_present(row, ["cancer", "cancer_type"])
                or default_cancer
            ).strip()

            comp_val = str(default_comparison or row.get("comparison", "") or "").strip()

            rec: dict[str, Any] = {
                "created_at": created_at,
                "run_id": str(run_id),
                "method": method_val,
                "disease": disease_key,
                "cancer": disease_key,  # v1 alias
                "comparison": comp_val,
                "tau": float(tau_val),
                "species": species,
                "claim": claim.model_dump(),
                # Contract: nested decision object exists (stable for paper artifacts).
                # Flat aliases still exist below for v1 compatibility.
                "decision_obj": decision_obj.model_dump(),
                "metrics": {
                    "term_survival_agg": surv_val,
                    "context_score": ctx_val,
                    "stat": stat_val,
                    "tau_used": _to_float_or_none(row_raw.get("tau_used")),
                    "stability_scope": str(row_raw.get("stability_scope") or "").strip(),
                    "module_id_effective": str(row_raw.get("module_id_effective") or "").strip(),
                    "gene_set_hash_effective": str(
                        row_raw.get("gene_set_hash_effective") or ""
                    ).strip(),
                },
                "schema_version": schema_version,
                "claim_id": str(getattr(claim, "claim_id", "") or "").strip(),
                "decision_status": str(getattr(decision_obj, "status", "") or "").strip(),
                "decision": str(getattr(decision_obj, "status", "") or "").strip(),
                "survival": surv_val,
                "claim.entity_type": "term",
                "claim.entity_id": str(getattr(claim, "entity", "") or "").strip(),
                "claim.context.disease": ctx_disease,
                "claim.context.tissue": ctx_tissue,
                "claim.context.perturbation": ctx_pert,
                "claim.context.comparison": ctx_comp or comp_val,
                "evidence_refs.gene_set_hash": str(
                    getattr(getattr(claim, "evidence_ref", None), "gene_set_hash", "") or ""
                )
                .strip()
                .lower(),
            }

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return jsonl_path


# -----------------------------
# Markdown report (human-facing)
# -----------------------------
def write_report(
    audit_log: pd.DataFrame, distilled: pd.DataFrame, card: SampleCard, outdir: str
) -> None:
    """
    Human-facing report.md + TSV artifacts.

    Note:
      - This function does NOT write report.jsonl.
      - JSONL export is explicit via write_report_jsonl(...).
      - gene symbol mapping here is DISPLAY-ONLY and does not affect auditing.
    """
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Display-only symbol mapping (best-effort, reproducible, no network)
    # -------------------------
    id2sym_distilled = build_id_to_symbol_from_distilled(distilled)

    id2sym_user: dict[str, str] = {}
    try:
        ex = getattr(card, "extra", {}) or {}
        map_path = ex.get("gene_id_map_tsv", None) if isinstance(ex, dict) else None
        if map_path:
            id2sym_user = load_id_map_tsv(map_path)
    except Exception:
        id2sym_user = {}

    id2sym = {**id2sym_distilled, **id2sym_user}

    audit_out = audit_log.copy()

    # -------------------------
    # Make ID columns Excel-safe BEFORE exporting
    # -------------------------
    # This prevents Excel from interpreting "4033,7062,..." as a huge number
    # (or scientific notation).
    if "gene_ids" in audit_out.columns:
        audit_out["gene_ids"] = audit_out["gene_ids"].map(_excel_safe_ids)
        audit_out["gene_ids_str"] = audit_out["gene_ids"]

    if "gene_ids_str" in audit_out.columns:
        audit_out["gene_ids_str"] = audit_out["gene_ids_str"].map(_excel_safe_ids)
        audit_out["gene_ids"] = audit_out.get("gene_ids", audit_out["gene_ids_str"]).map(
            _excel_safe_ids
        )

    # Optional: term_ids can also look like structured tokens; keep safe anyway
    if "term_ids" in audit_out.columns:
        audit_out["term_ids"] = audit_out["term_ids"].map(_excel_safe_ids)
        audit_out["term_ids_str"] = audit_out["term_ids"]

    if "term_ids_str" in audit_out.columns:
        audit_out["term_ids_str"] = audit_out["term_ids_str"].map(_excel_safe_ids)
        audit_out["term_ids"] = audit_out.get("term_ids", audit_out["term_ids_str"]).map(
            _excel_safe_ids
        )

    # -------------------------
    # Add display-only gene symbols
    # -------------------------
    if ("gene_ids" in audit_out.columns) or ("gene_ids_str" in audit_out.columns):
        src = "gene_ids_str" if "gene_ids_str" in audit_out.columns else "gene_ids"
        s = audit_out[src].map(lambda x: "" if _is_na_scalar(x) else str(x))
        syms = s.map(lambda x: map_ids_to_symbols(x, id2sym))
        audit_out["gene_symbols"] = syms
        audit_out["gene_symbols_str"] = syms.map(
            lambda xs: ";".join(map(str, xs)) if isinstance(xs, list) else ""
        )

    # stringify after we create *_str columns
    audit_out = _stringify_list_columns(audit_out)
    dist_out = _stringify_list_columns(distilled)

    audit_tsv = out_path / "audit_log.tsv"
    distilled_tsv = out_path / "distilled.tsv"
    audit_out.to_csv(audit_tsv, sep="\t", index=False)
    dist_out.to_csv(distilled_tsv, sep="\t", index=False)

    status_col = "status" if "status" in audit_log.columns else None
    if status_col is None:
        n_pass = n_abs = n_fail = 0
    else:
        n_pass = int((audit_log[status_col] == "PASS").sum())
        n_abs = int((audit_log[status_col] == "ABSTAIN").sum())
        n_fail = int((audit_log[status_col] == "FAIL").sum())

    abs_sum, fail_sum = _reason_summary(audit_log)

    rc_summary: dict[str, float] | None = None
    rc_curve_path: Path | None = None
    score_col = _pick_score_col(audit_log)

    if status_col and (risk_coverage_from_status is not None):
        try:
            rc_summary = risk_coverage_from_status(audit_log[status_col])
        except Exception:
            rc_summary = None

    if status_col and score_col and (risk_coverage_curve is not None):
        try:
            curve = risk_coverage_curve(audit_log, score_col=score_col, status_col=status_col)
            rc_curve_path = out_path / "risk_coverage.tsv"
            curve.to_csv(rc_curve_path, sep="\t", index=False)
        except Exception:
            rc_curve_path = None

    def _top(df: pd.DataFrame) -> pd.DataFrame:
        if "term_survival_agg" in df.columns:
            return df.sort_values(["term_survival_agg"], ascending=[False], kind="mergesort")
        if "stat" in df.columns:
            return df.sort_values(["stat"], ascending=[False], kind="mergesort")
        if "claim_id" in df.columns:
            return df.sort_values(["claim_id"], ascending=[True], kind="mergesort")
        return df

    lines: list[str] = []
    lines.append("# LLM-PathwayCurator report")
    lines.append("")
    lines.append("## Sample Card")
    lines.append(f"- disease: {getattr(card, 'disease', '')}")
    lines.append(f"- tissue: {getattr(card, 'tissue', '')}")
    lines.append(f"- perturbation: {getattr(card, 'perturbation', '')}")
    lines.append(f"- comparison: {getattr(card, 'comparison', '')}")
    notes = getattr(card, "notes", None)
    if notes:
        lines.append(f"- notes: {notes}")
    lines.append("")
    lines.append(f"## Decisions (PASS/ABSTAIN/FAIL): {n_pass}/{n_abs}/{n_fail}")
    lines.append(
        "PASS requires evidence-link integrity and stability; "
        "otherwise the system ABSTAINS or FAILS."
    )
    lines.append("")

    if rc_summary is not None:
        lines.append("## Risk–Coverage summary")
        lines.append("")
        lines.append(
            f"- coverage_pass_total: {rc_summary.get('coverage_pass_total', float('nan')):.3f}"
        )
        risk_decided = rc_summary.get("risk_fail_given_decided", float("nan"))
        lines.append(f"- risk_fail_given_decided: {risk_decided:.3f}")
        risk_total = rc_summary.get("risk_fail_total", float("nan"))
        lines.append(f"- risk_fail_total: {risk_total:.3f}")
        if score_col:
            lines.append(f"- score_col_for_curve: {score_col}")
        lines.append("")

    if not abs_sum.empty or not fail_sum.empty:
        lines.append("## Reason summary")
        lines.append("")
        if not abs_sum.empty:
            lines.append("### ABSTAIN reasons")
            lines.append(_safe_table_md(abs_sum, n=50))
            lines.append("")
        if not fail_sum.empty:
            lines.append("### FAIL reasons")
            lines.append(_safe_table_md(fail_sum, n=50))
            lines.append("")

    # PASS
    lines.append("## PASS (top)")
    lines.append("")
    pass_df = audit_out[audit_out[status_col] == "PASS"] if status_col else audit_out.iloc[0:0]
    pass_df = _top(pass_df)
    lines.append(
        _safe_table_md(
            _prefer_str_cols(
                pass_df,
                [
                    "claim_id",
                    "entity",
                    "direction",
                    "module_id_effective",
                    "term_ids",
                    "gene_ids",
                    "gene_symbols",
                    "gene_set_hash_effective",
                    "evidence_key",
                    "term_ids_set_hash",
                    "term_survival_agg",
                    "context_score",
                    "stat",
                    "audit_notes",
                    "stress_evaluated",
                    "stress_ok",
                ],
            ),
            n=5,
        )
    )
    lines.append("")

    # ABSTAIN
    lines.append("## ABSTAIN (top)")
    lines.append("")
    abs_df = audit_out[audit_out[status_col] == "ABSTAIN"] if status_col else audit_out.iloc[0:0]
    abs_df = _top(abs_df)
    lines.append(
        _safe_table_md(
            _prefer_str_cols(
                abs_df,
                [
                    "claim_id",
                    "entity",
                    "direction",
                    "module_id_effective",
                    "abstain_reason",
                    "gene_ids",
                    "gene_symbols",
                    "gene_set_hash_effective",
                    "stress_evaluated",
                    "stress_ok",
                    "term_survival_agg",
                    "context_score",
                    "audit_notes",
                ],
            ),
            n=5,
        )
    )
    lines.append("")

    # FAIL
    lines.append("## FAIL (top)")
    lines.append("")
    fail_df = audit_out[audit_out[status_col] == "FAIL"] if status_col else audit_out.iloc[0:0]
    fail_df = _top(fail_df)
    lines.append(
        _safe_table_md(
            _prefer_str_cols(
                fail_df,
                [
                    "claim_id",
                    "entity",
                    "direction",
                    "module_id_effective",
                    "fail_reason",
                    "gene_ids",
                    "gene_symbols",
                    "gene_set_hash_effective",
                    "audit_notes",
                ],
            ),
            n=5,
        )
    )
    lines.append("")

    lines.append("## Audit log (top)")
    lines.append("")
    lines.append(_safe_table_md(_stringify_list_columns(audit_log), n=20))
    lines.append("")

    lines.append("## Reproducible artifacts")
    lines.append("")
    lines.append(f"- audit_log.tsv: {audit_tsv.name}")
    lines.append(f"- distilled.tsv: {distilled_tsv.name}")
    if rc_curve_path is not None:
        lines.append(f"- risk_coverage.tsv: {rc_curve_path.name}")
    lines.append("")

    report_path = out_path / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
