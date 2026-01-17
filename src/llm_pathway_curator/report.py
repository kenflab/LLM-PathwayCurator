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


def _hash12(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


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


def _to_float_or_none(x: Any) -> float | None:
    if x is None or _is_na_scalar(x):
        return None
    try:
        # handle empty strings
        s = str(x).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _reason_codes_from_row(row: pd.Series) -> list[str]:
    """
    v1 contract:
    - FAIL -> [fail_reason] (must be in FAIL_REASONS)
    - ABSTAIN -> [abstain_reason] (must be in ABSTAIN_REASONS)
    - PASS -> ["OK"]
    """
    status = str(row.get("status", "")).upper().strip()
    if status == "FAIL":
        fr = str(row.get("fail_reason", "")).strip()
        return [fr] if fr in FAIL_REASONS else ["unknown_fail"]
    if status == "ABSTAIN":
        ar = str(row.get("abstain_reason", "")).strip()
        return [ar] if ar in ABSTAIN_REASONS else ["unknown_abstain"]
    if status == "PASS":
        return ["OK"]
    return ["UNKNOWN"]


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


def _claim_uid_from_row(row: pd.Series, card: SampleCard) -> str:
    """
    Meaning-stable id (no evidence hash).
    Keep this conservative and stable:
    - disease + comparison + entity_id + direction
    """
    disease = str(getattr(card, "disease", "") or row.get("cancer", "") or "").strip()
    comparison = str(getattr(card, "comparison", "") or row.get("comparison", "") or "").strip()
    entity_id = str(
        row.get("entity_id", "") or row.get("entity", "") or row.get("term_id", "") or ""
    ).strip()
    direction = str(row.get("direction", "") or "").strip()
    return f"{disease}|{comparison}|{entity_id}|{direction}"


# -----------------------------
# v1 JSONL export (Fig2-ready)
# -----------------------------
def write_report_jsonl(
    audit_log: pd.DataFrame,
    card: SampleCard,
    outdir: str,
    *,
    run_id: str,
) -> Path:
    """
    Write Fig2-ready records:
    - metrics.survival is term_survival_agg (required column; values may be NA)
    - audit_flags reflect FAIL reasons (stable)
    - reason_codes uses stable tokens from audit_reasons.py
    """
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    if run_id is None:
        run_id = datetime.now(timezone.utc).isoformat()

    if "term_survival_agg" not in audit_log.columns:
        raise ValueError("audit_log must contain term_survival_agg for v1 (metrics.survival)")

    # Make list-like fields TSV/JSON safe, but keep numeric series from the original frame.
    df = _stringify_list_columns(audit_log)

    # normalize numeric fields (robust to "" from TSV-safe artifacts)
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

    jsonl_path = out_path / "report.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            claim_uid = _claim_uid_from_row(row, card)

            evidence = {
                "module_ids": _as_list(row.get("module_id")),
                "term_ids": _as_list(row.get("term_ids")),
                "gene_ids": _as_list(row.get("gene_ids")),
            }
            evidence_hash = _hash12(json.dumps(evidence, sort_keys=True, ensure_ascii=False))
            claim_id = f"{claim_uid}|{evidence_hash}"

            # numeric metrics (None if missing)
            survival_val: float | None = None
            if surv_s is not None:
                v = surv_s.loc[i]
                survival_val = None if pd.isna(v) else float(v)

            context_val: float | None = None
            if ctx_s is not None:
                v = ctx_s.loc[i]
                context_val = None if pd.isna(v) else float(v)

            stat_val: float | None = None
            if stat_s is not None:
                v = stat_s.loc[i]
                stat_val = None if pd.isna(v) else float(v)

            rec = {
                "run_id": str(run_id),
                "claim_uid": claim_uid,
                "claim_id": claim_id,
                "claim": {
                    # v1: keep coarse; refine later via claim_schema.py
                    "entity_type": str(row.get("entity_type", "pathway")),
                    "entity_id": str(
                        row.get("entity_id", "")
                        or row.get("entity", "")
                        or row.get("term_id", "")
                        or ""
                    ),
                    "entity_name": str(
                        row.get("entity_name", "")
                        or row.get("entity", "")
                        or row.get("term_name", "")
                        or ""
                    ),
                    "direction": str(row.get("direction", "") or ""),
                    "context": {
                        "comparison": str(getattr(card, "comparison", "") or ""),
                        "disease": str(getattr(card, "disease", "") or ""),
                        "tissue": str(getattr(card, "tissue", "") or ""),
                        "perturbation": str(getattr(card, "perturbation", "") or ""),
                    },
                },
                "evidence_refs": evidence,
                "metrics": {
                    "survival": survival_val,
                    "context_score": context_val,
                    "stat": stat_val,
                },
                "audit_flags": _audit_flags_from_row(row),
                "reason_codes": _reason_codes_from_row(row),
                "context_keys_used": _as_list(row.get("context_keys_used")),
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
