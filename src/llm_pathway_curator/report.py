# LLM-PathwayCurator/src/llm_pathway_curator/report.py
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .audit_reasons import (
    ABSTAIN_REASONS,
    FAIL_CONTRADICTION,
    FAIL_EVIDENCE_DRIFT,
    FAIL_REASONS,
    FAIL_SCHEMA_VIOLATION,
)
from .sample_card import SampleCard

# optional metrics export (report.md only)
try:
    from .calibrate import risk_coverage_curve, risk_coverage_from_status
except Exception:  # pragma: no cover
    risk_coverage_curve = None
    risk_coverage_from_status = None


# -----------------------------
# Small helpers (stable)
# -----------------------------
_NA_TOKENS = {"", "na", "nan", "none", "NA"}


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
    if x is None:
        return True
    if isinstance(x, (list, tuple, set, dict)):
        return False
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def _stringify_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    TSV-safe conversion for list-like columns.
    - If a <col>_str already exists, prefer it.
    - Otherwise stringify list/tuple/set as comma-joined.
    - Scalar NA -> "" to stabilize artifacts.
    """
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
                    lambda x: ",".join(map(str, x))
                    if isinstance(x, (list, tuple, set))
                    else ("" if _is_na_scalar(x) else str(x))
                )
            else:
                out[c] = s.map(lambda x: "" if _is_na_scalar(x) else x)
        else:
            out[c] = s.map(lambda x: "" if _is_na_scalar(x) else x)
    return out


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


def _as_list(x: Any) -> list[str]:
    """
    Robust list normalization:
    - list/tuple/set -> list[str]
    - comma-joined string -> split, strip, drop empties
    - scalar -> [str(x)]
    """
    if x is None or _is_na_scalar(x):
        return []
    if isinstance(x, list):
        items = [str(v).strip() for v in x]
    elif isinstance(x, (tuple, set)):
        items = [str(v).strip() for v in list(x)]
    elif isinstance(x, str):
        s = x.strip()
        if not s or s in _NA_TOKENS or s.lower() in _NA_TOKENS:
            return []
        items = [t.strip() for t in s.replace(";", ",").split(",")]
    else:
        items = [str(x).strip()]

    out: list[str] = []
    seen: set[str] = set()
    for t in items:
        if not t:
            continue
        if t in _NA_TOKENS or t.lower() in _NA_TOKENS:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _sorted_list(x: Any) -> list[str]:
    return sorted(_as_list(x))


def _canonical_json(obj: Any) -> str:
    """
    Deterministic JSON for hashing.
    """
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _claim_id_v1(
    *,
    entity_id: str,
    direction: str,
    context: dict[str, str],
    context_keys: list[str],
    gene_set_hash: str,
    module_ids: list[str],
    term_ids: list[str],
) -> str:
    payload = {
        "v": 1,
        "entity_id": (entity_id or "").strip(),
        "direction": (direction or "").strip(),
        "context": {
            k: (context.get(k, "") or "").strip()
            for k in ["disease", "tissue", "perturbation", "comparison"]
        },
        "context_keys": sorted([k.strip() for k in context_keys if k and k.strip()]),
        "gene_set_hash": (gene_set_hash or "").strip(),
        "module_ids": sorted([m.strip() for m in module_ids if m and m.strip()]),
        "term_ids": sorted([t.strip() for t in term_ids if t and t.strip()]),
    }
    return _sha1(_canonical_json(payload))


def _require_columns(df: pd.DataFrame, cols: list[str], who: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{who} missing required columns: {missing}")


def _get_first_present(row: pd.Series, keys: list[str]) -> Any:
    for k in keys:
        if k in row.index:
            v = row.get(k)
            # treat NA-ish as missing
            if v is None or _is_na_scalar(v):
                continue
            return v
    return None


def _reason_codes_and_details_from_row(row: pd.Series) -> tuple[list[str], dict[str, Any]]:
    """
    v1 contract:
    - FAIL -> [fail_reason] (token from FAIL_REASONS)
    - ABSTAIN -> [abstain_reason] (token from ABSTAIN_REASONS)
    - PASS -> ["OK"]
    For unknown tokens, preserve the raw value in details.
    """
    details: dict[str, Any] = {}
    status = str(row.get("status", "")).upper().strip()

    if status == "FAIL":
        raw = "" if row.get("fail_reason") is None else str(row.get("fail_reason")).strip()
        if raw in FAIL_REASONS:
            return [raw], details
        details["raw_fail_reason"] = raw
        # keep token stable; do not silently invent new reasons
        return ["unknown_fail"], details

    if status == "ABSTAIN":
        raw = "" if row.get("abstain_reason") is None else str(row.get("abstain_reason")).strip()
        if raw in ABSTAIN_REASONS:
            return [raw], details
        details["raw_abstain_reason"] = raw
        return ["unknown_abstain"], details

    if status == "PASS":
        return ["OK"], details

    details["raw_status"] = status
    return ["UNKNOWN"], details


def _audit_flags_from_row(row: pd.Series) -> dict[str, bool]:
    """
    v1 contract:
    audit_flags reflect ONLY auditable FAIL reasons.
    """
    fr = str(row.get("fail_reason", "")).strip()
    return {
        "evidence_drift": fr == FAIL_EVIDENCE_DRIFT,
        "contradiction": fr == FAIL_CONTRADICTION,
        "schema_violation": fr == FAIL_SCHEMA_VIOLATION,
    }


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


# -----------------------------
# v1 JSONL export (Fig2-ready)
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
    Fig2-ready JSONL (stable contract).

    Required columns in audit_log (v1):
      - status
      - term_survival_agg
      - gene_set_hash
      - entity_id or term_id (fallback)
    Optional:
      - context_score, stat
      - module_id/module_ids, term_ids/term_id(s), gene_ids/gene_id(s)
      - cancer (row-level)
      - comparison (row-level)
      - context_keys_used (row-level)
    """
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not run_id:
        run_id = datetime.now(timezone.utc).isoformat()

    _require_columns(audit_log, ["claim_id", "status", "gene_set_hash"], "audit_log")

    # stringify list columns for robust I/O (do not use for numerics)
    df = _stringify_list_columns(audit_log)

    surv_s = pd.to_numeric(audit_log.get("term_survival_agg"), errors="coerce")
    ctx_s = (
        pd.to_numeric(audit_log.get("context_score"), errors="coerce")
        if "context_score" in audit_log.columns
        else None
    )
    stat_s = (
        pd.to_numeric(audit_log.get("stat"), errors="coerce")
        if "stat" in audit_log.columns
        else None
    )

    # method/tau: do not require user input (Less is more)
    method_val = "llm-pathway-curator" if method is None else str(method).strip()
    tau_val = _get_tau_default(card, default=0.8) if tau is None else float(tau)

    # grouping keys (prefer explicit args)
    default_cancer = str(cancer or getattr(card, "cancer", None) or "").strip()
    default_comparison = str(comparison or getattr(card, "comparison", None) or "").strip()

    jsonl_path = out_path / "report.jsonl"
    created_at = datetime.now(timezone.utc).isoformat()

    with jsonl_path.open("w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            status = str(row.get("status", "")).upper().strip()
            decision = status  # PASS/ABSTAIN/FAIL

            entity_id = str(
                _get_first_present(row, ["entity_id", "entity", "term_id"]) or ""
            ).strip()

            direction = str(row.get("direction", "") or "").strip()

            # context (do NOT infer context_keys; keep missing visible)
            ctx = {
                "comparison": str(default_comparison or row.get("comparison", "") or "").strip(),
                "disease": str(getattr(card, "disease", "") or "").strip(),
                "tissue": str(getattr(card, "tissue", "") or "").strip(),
                "perturbation": str(getattr(card, "perturbation", "") or "").strip(),
            }

            context_keys = _sorted_list(row.get("context_keys_used"))
            # no inference: if missing, keep empty (audit layer should decide)

            # evidence key (MUST be defined before evidence_refs uses it)
            gene_set_hash = str(row.get("gene_set_hash", "") or "").strip()

            # evidence refs (column-name tolerant)
            module_ids = _sorted_list(_get_first_present(row, ["module_ids", "module_id"]) or [])
            term_ids = _sorted_list(
                _get_first_present(row, ["term_ids", "term_id", "term_id(s)"]) or []
            )
            gene_ids = _sorted_list(
                _get_first_present(row, ["gene_ids", "gene_id", "gene_id(s)"]) or []
            )

            evidence = {
                "module_ids": module_ids,
                "term_ids": term_ids,
                "gene_ids": gene_ids,
                "gene_set_hash": gene_set_hash,
            }

            claim_id = str(row.get("claim_id", "")).strip()
            if not claim_id:
                raise ValueError("audit_log missing claim_id value")

            # numeric metrics (use precomputed series; NA -> 0.0 for safety)
            v = surv_s.loc[i]
            survival_val = 0.0 if pd.isna(v) else float(v)

            context_val = None
            if ctx_s is not None:
                v = ctx_s.loc[i]
                context_val = None if pd.isna(v) else float(v)

            stat_val = None
            if stat_s is not None:
                v = stat_s.loc[i]
                stat_val = None if pd.isna(v) else float(v)

            # reason_codes + details
            reason_codes, reason_details = _reason_codes_and_details_from_row(row)

            # grouping cancer key (prefer row-level cancer, then arg)
            cancer_key = str(
                _get_first_present(row, ["cancer", "cancer_type"]) or default_cancer
            ).strip()

            rec = {
                # provenance
                "schema_version": "v1",
                "created_at": created_at,
                "run_id": str(run_id),
                # fig2 group keys
                "method": method_val,
                "cancer": cancer_key,
                "comparison": str(ctx["comparison"]),
                "tau": float(tau_val),
                # contract keys
                "claim_id": claim_id,
                "decision": decision,
                "survival": survival_val,
                "evidence_refs": evidence,
                "reason_codes": reason_codes,
                "context_keys": context_keys,
                # structured claim (for Fig1/demo; still typed-ish)
                "claim": {
                    "entity_type": str(row.get("entity_type", "pathway")),
                    "entity_id": entity_id,
                    "entity_name": str(
                        _get_first_present(row, ["entity_name", "entity", "term_name"]) or ""
                    ),
                    "direction": direction,
                    "context": ctx,
                },
                # extra stable fields (optional but very useful)
                "metrics": {
                    "survival": survival_val,
                    "context_score": context_val,
                    "stat": stat_val,
                },
                "audit_flags": _audit_flags_from_row(row),
                # disentangle disease vs cancer_type
                "disease": str(getattr(card, "disease", "") or "").strip(),
                "tissue": str(getattr(card, "tissue", "") or "").strip(),
                "perturbation": str(getattr(card, "perturbation", "") or "").strip(),
                # evidence key
                "gene_set_hash": gene_set_hash,
                # details for debugging schema drift / unknown reasons
                "details": reason_details,
            }

            # consistency guard (avoid silent drift)
            if rec["metrics"]["survival"] != rec["survival"]:
                raise AssertionError("inconsistent survival fields in report.jsonl record")

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
    """
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    audit_out = _stringify_list_columns(audit_log)
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
        cols = df.columns
        if "term_survival_agg" in cols:
            return df.sort_values(
                ["term_survival_agg", "claim_id"], ascending=[False, True], kind="mergesort"
            )
        if "stat" in cols:
            return df.sort_values(["stat", "claim_id"], ascending=[False, True], kind="mergesort")
        return df.sort_values(["claim_id"], ascending=[True], kind="mergesort")

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
        "PASS requires evidence-link integrity and stability; otherwise the system "
        "ABSTAINS or FAILS."
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
    pass_df = audit_log[audit_log[status_col] == "PASS"] if status_col else audit_log.iloc[0:0]
    pass_df = _top(pass_df)
    lines.append(
        _safe_table_md(
            _prefer_str_cols(
                pass_df,
                [
                    "claim_id",
                    "entity",
                    "direction",
                    "module_id",
                    "term_ids",
                    "gene_ids",
                    "gene_set_hash",
                    "term_survival_agg",
                    "context_score",
                    "stat",
                    "audit_notes",
                ],
            ),
            n=5,
        )
    )
    lines.append("")

    # ABSTAIN
    lines.append("## ABSTAIN (top)")
    lines.append("")
    abs_df = audit_log[audit_log[status_col] == "ABSTAIN"] if status_col else audit_log.iloc[0:0]
    abs_df = _top(abs_df)
    lines.append(
        _safe_table_md(
            _prefer_str_cols(
                abs_df,
                [
                    "claim_id",
                    "entity",
                    "direction",
                    "module_id",
                    "abstain_reason",
                    "gene_set_hash",
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
    fail_df = audit_log[audit_log[status_col] == "FAIL"] if status_col else audit_log.iloc[0:0]
    fail_df = _top(fail_df)
    lines.append(
        _safe_table_md(
            _prefer_str_cols(
                fail_df,
                [
                    "claim_id",
                    "entity",
                    "direction",
                    "module_id",
                    "fail_reason",
                    "gene_set_hash",
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
