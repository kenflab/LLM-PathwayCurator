# LLM-PathwayCurator/src/llm_pathway_curator/report.py
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
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

from . import _shared
from .utils import build_id_to_symbol_from_distilled, load_id_map_tsv, map_ids_to_symbols

# -----------------------------
# Small helpers (stable)
# -----------------------------
_NA_TOKENS = set(_shared.NA_TOKENS)
_NA_TOKENS_L = {t.lower() for t in _NA_TOKENS}

_ALLOWED_STATUSES = {"PASS", "ABSTAIN", "FAIL"}

# Keep stress_tag delimiter consistent across layers.
# masking.apply_evidence_stress uses comma-joined tags;
# modules.attach_module_drift_stress_tag tolerates '+' but canonicalizes to comma.
STRESS_TAG_DELIM = ","


def _is_na_scalar(x: Any) -> bool:
    """Single source of truth: _shared.is_na_scalar."""
    return _shared.is_na_scalar(x)


def _json_sanitize(x: Any) -> Any:
    """
    Make an object JSON-serializable (robust against numpy/pandas scalars, pd.NA, Paths, etc.).
    This is the LAST LINE OF DEFENSE for report.jsonl stability.

    Rules:
      - pd.NA / NaN -> None
      - numpy/pandas scalar -> native Python scalar (via .item() when available)
      - datetime -> ISO string
      - Path -> str
      - set/tuple -> list
      - dict/list -> recurse
      - pydantic models -> model_dump()
      - fallback -> str(x)
    """
    # fast-path for simple primitives
    if x is None or isinstance(x, (str, int, float, bool)):
        if isinstance(x, float) and (pd.isna(x)):
            return None
        return x

    if _is_na_scalar(x):
        return None

    if isinstance(x, datetime):
        try:
            return x.astimezone(timezone.utc).isoformat()
        except Exception:
            return x.isoformat()

    if isinstance(x, Path):
        return str(x)

    if hasattr(x, "model_dump") and callable(x.model_dump):
        try:
            return _json_sanitize(x.model_dump())
        except Exception:
            pass

    if hasattr(x, "item") and callable(x.item):
        try:
            v = x.item()
            return _json_sanitize(v)
        except Exception:
            pass

    if isinstance(x, dict):
        out: dict[str, Any] = {}
        for k, v in x.items():
            out[str(k)] = _json_sanitize(v)
        return out

    if isinstance(x, (list, tuple, set)):
        return [_json_sanitize(v) for v in list(x)]

    if isinstance(x, pd.Series):
        try:
            return [_json_sanitize(v) for v in x.tolist()]
        except Exception:
            return str(x)

    return str(x)


def _strip_excel_text_prefix(s: str) -> str:
    """
    Excel-safe text fields sometimes start with a single quote.
    For JSON payload columns, strip it to allow json.loads.
    """
    ss = str(s or "").strip()
    if ss.startswith("'"):
        return ss[1:].strip()
    return ss


def _split_ids_to_list(x: Any) -> list[str]:
    """
    Parse ID fields that may be:
      - list-like
      - Excel-safe scalar string (may start with a single quote)
      - separator-delimited strings (',', ';', '|', whitespace)
    """
    if _is_na_scalar(x):
        return []
    if isinstance(x, (list, tuple, set)):
        return [str(t).strip() for t in x if str(t).strip()]

    s = str(x).strip()
    if not s:
        return []
    if s.startswith("'"):
        s = s[1:].strip()

    s = re.sub(r"[,\|\t\n ]+", ";", s)
    return [p.strip() for p in s.split(";") if p.strip()]


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


def _stringify_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    TSV-safe conversion for list-like columns.
    - If a <col>_str already exists, prefer it.
    - Otherwise stringify list/tuple/set as separator-joined.
    - Scalar NA handling MUST NOT break numeric dtypes.

    IMPORTANT:
      Use ';' as the list separator to reduce Excel auto-formatting risk.

    Contract (paper/repro):
      - numeric columns stay numeric (missing -> NaN)
      - object scalar columns may use "" for missing (TSV readability)
      - list-like columns get a companion <col>_str, do not overwrite original by default
    """
    from pandas.api.types import is_bool_dtype, is_numeric_dtype

    LIST_SEP = ";"
    out = df.copy()

    # IMPORTANT:
    # If out has duplicate column names, out[c] returns a DataFrame (not Series),
    # which breaks on .dtype. Iterate by *position* to always get a Series.
    n_cols = int(out.shape[1])
    for j in range(n_cols):
        c = out.columns[j]
        c_name = str(c)

        # skip already-string columns
        if c_name.endswith("_str"):
            continue

        # honor pre-existing <col>_str
        c_str = f"{c_name}_str"
        if c_str in out.columns:
            continue

        s = out.iloc[:, j]

        # Decide dtype category robustly
        try:
            is_obj = s.dtype == "object"
        except Exception:
            is_obj = True

        if is_obj:
            # object dtype: either list-like or scalar-like
            sample = s.dropna().head(20).tolist()
            if any(isinstance(x, (list, tuple, set)) for x in sample):
                # Create a stable companion string column; keep original list-like column intact.
                out[c_str] = s.map(
                    lambda x: LIST_SEP.join(map(str, x))
                    if isinstance(x, (list, tuple, set))
                    else ("" if _is_na_scalar(x) else str(x))
                )
            else:
                # scalar object column: keep dtype=object, make missing printable as ""
                out.iloc[:, j] = s.map(lambda x: "" if _is_na_scalar(x) else x)

        else:
            # Non-object columns: keep dtype stable; missing -> NaN for numeric,
            # pd.NA for bool-like.
            try:
                if is_numeric_dtype(s.dtype):
                    mapped = s.map(lambda x: np.nan if _is_na_scalar(x) else x)
                    # Ensure we never assign object series into numeric column
                    out.iloc[:, j] = pd.to_numeric(mapped, errors="coerce")
                elif is_bool_dtype(s.dtype):
                    # boolean extension can hold pd.NA; keep it boolean-ish
                    out.iloc[:, j] = s.map(lambda x: pd.NA if _is_na_scalar(x) else bool(x))
                else:
                    # other extension dtype: best-effort preserve
                    out.iloc[:, j] = s.map(lambda x: pd.NA if _is_na_scalar(x) else x)
            except Exception:
                # last resort: keep original series unchanged
                out.iloc[:, j] = s

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

    if isinstance(x, (list, tuple, set)):
        parts = [str(t).strip() for t in x if str(t).strip()]
        s = list_sep.join(parts)
        if not s:
            return ""
        return s if s.startswith("'") else ("'" + s)

    s0 = str(x).strip()
    if not s0 or s0.lower() in _NA_TOKENS_L:
        return ""

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


def _normalize_stress_tag(raw: Any) -> str:
    """
    Canonicalize stress_tag across historical artifacts.

    Policy:
      - Accept comma-delimited (current) or '+'-delimited (legacy).
      - Normalize to comma-delimited unique tags preserving order.
    """
    if _is_na_scalar(raw):
        return ""
    s = str(raw or "").strip()
    if not s or s.lower() in _NA_TOKENS_L:
        return ""
    s = s.replace("+", STRESS_TAG_DELIM)
    parts = [p.strip() for p in s.split(STRESS_TAG_DELIM) if p.strip()]
    # de-dup preserving order
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return STRESS_TAG_DELIM.join(out)


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

    for k in ["stress_status", "stress_reason", "stress_notes"]:
        if k in row.index and (not _is_na_scalar(row.get(k))):
            details[k] = row.get(k)

    # best-effort: include stress_tag / contradiction flip if present
    if "stress_tag" in row.index and (not _is_na_scalar(row.get("stress_tag"))):
        details["stress_tag"] = _normalize_stress_tag(row.get("stress_tag"))
    if "contradiction_flip" in row.index and (not _is_na_scalar(row.get("contradiction_flip"))):
        try:
            details["contradiction_flip"] = bool(row.get("contradiction_flip"))
        except Exception:
            details["contradiction_flip"] = str(row.get("contradiction_flip"))

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


def _get_dev_meta_enabled() -> bool:
    """
    Developer-only metadata gate.
    Default: OFF (avoid exposing internal versions/knobs in user outputs).
    """
    s = (os.environ.get("LLMPATH_REPORT_INCLUDE_DEV_META", "") or "").strip().lower()
    return s in {"1", "true", "yes", "on"}


def _get_schema_version(card: SampleCard, default: str = "v1") -> str:
    """
    Internal schema contract version.
    IMPORTANT: Do NOT expose this by default in user outputs (see _get_dev_meta_enabled).
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
    for c in ["term_survival_agg", "term_survival", "context_score", "stat"]:
        if c in audit_log.columns:
            return c
    return None


def _pick_claim_json_col(audit_log: pd.DataFrame) -> str | None:
    """
    Prefer canonical 'claim_json'. If missing, accept common fallbacks produced by
    TSV/markdown stringification or older pipeline stages.
    """
    for c in ["claim_json", "claim_json_str", "claim_json_raw"]:
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


def _safe_json_loads(s: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _claim_stub_from_raw(raw: dict[str, Any]) -> dict[str, Any]:
    ev = raw.get("evidence_ref", {}) if isinstance(raw.get("evidence_ref", {}), dict) else {}
    return {
        "entity": str(raw.get("entity", "") or ""),
        "direction": str(raw.get("direction", "") or ""),
        "context_keys": raw.get("context_keys", []),
        "evidence_ref": {
            "module_id": str(ev.get("module_id", "") or ""),
            "gene_ids": ev.get("gene_ids", []),
            "term_ids": ev.get("term_ids", []),
            "gene_set_hash": str(ev.get("gene_set_hash", "") or ""),
        },
    }


def _extract_context_review_fields(claim_raw: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "context_evaluated",
        "context_method",
        "context_status",
        "context_reason",
        "context_notes",
        "context_confidence",
        "context_review_key",
        "context_review_version",
    ]
    out: dict[str, Any] = {}
    for k in keys:
        if k in claim_raw:
            out[k] = claim_raw.get(k)
    return out


def _get_card_context(card: SampleCard) -> dict[str, str]:
    """
    Neutral context block: prefer 'condition' over 'disease'.
    """
    ctx_condition = str(getattr(card, "condition", "") or "").strip()
    ctx_tissue = str(getattr(card, "tissue", "") or "").strip()
    ctx_pert = str(getattr(card, "perturbation", "") or "").strip()
    ctx_comp = str(getattr(card, "comparison", "") or "").strip()

    ctx_disease = str(getattr(card, "disease", "") or "").strip()
    if (not ctx_condition) and ctx_disease:
        ctx_condition = ctx_disease

    return {
        "condition": ctx_condition,
        "tissue": ctx_tissue,
        "perturbation": ctx_pert,
        "comparison": ctx_comp,
        "disease_alias": ctx_disease,
    }


def _synthesize_claim_json_from_audit_log(
    audit_log: pd.DataFrame, card: SampleCard
) -> pd.DataFrame:
    """
    If claim_json payload column is missing, synthesize a minimal claim JSON from audit_log columns.
    This is report-only robustness: auditing is already done upstream.
    """
    df = audit_log.copy()

    def pick_col(cands: list[str]) -> str | None:
        for c in cands:
            if c in df.columns:
                return c
        return None

    entity_col = pick_col(["entity", "term_name", "term", "claim_entity"])
    direction_col = pick_col(["direction", "dir", "claim_direction"])
    module_col = pick_col(["module_id_effective", "module_id", "module"])
    ghash_col = pick_col(["gene_set_hash_effective", "gene_set_hash", "evidence_gene_set_hash"])
    gene_ids_col = pick_col(["gene_ids", "gene_ids_str"])
    term_ids_col = pick_col(["term_ids", "term_ids_str"])
    claim_id_col = pick_col(["claim_id", "id"])

    context_keys = ["condition", "tissue", "perturbation", "comparison"]
    ctx = _get_card_context(card)

    ctx_review_cols = [
        "context_evaluated",
        "context_method",
        "context_status",
        "context_reason",
        "context_notes",
        "context_confidence",
        "context_review_key",
        "context_review_version",
    ]

    payloads: list[str] = []
    for _, row in df.iterrows():
        ev = {
            "module_id": str(row.get(module_col, "") or "").strip() if module_col else "",
            "gene_ids": _split_ids_to_list(row.get(gene_ids_col)) if gene_ids_col else [],
            "term_ids": _split_ids_to_list(row.get(term_ids_col)) if term_ids_col else [],
            "gene_set_hash": str(row.get(ghash_col, "") or "").strip().lower() if ghash_col else "",
        }

        claim_raw: dict[str, Any] = {
            "claim_id": str(row.get(claim_id_col, "") or "").strip() if claim_id_col else "",
            "entity": str(row.get(entity_col, "") or "").strip() if entity_col else "",
            "direction": str(row.get(direction_col, "") or "").strip() if direction_col else "",
            "context_keys": context_keys,
            "evidence_ref": ev,
            "context": {
                "condition": ctx["condition"],
                "tissue": ctx["tissue"],
                "perturbation": ctx["perturbation"],
                "comparison": ctx["comparison"],
            },
        }

        had_any_ctx = False
        for c in ctx_review_cols:
            if c in df.columns and not _is_na_scalar(row.get(c)):
                claim_raw[c] = row.get(c)
                had_any_ctx = True

        if not had_any_ctx:
            claim_raw["context_evaluated"] = False
            claim_raw["context_status"] = "UNEVALUATED"
            claim_raw["context_reason"] = "NOT_EVALUATED"

        payloads.append(json.dumps(claim_raw, ensure_ascii=False))

    df["claim_json_raw"] = payloads
    return df


def _derive_stress_display_cols(audit_out: pd.DataFrame) -> pd.DataFrame:
    out = audit_out.copy()

    # Normalize stress_status
    if "stress_status" in out.columns:
        s = out["stress_status"].map(lambda x: "" if _is_na_scalar(x) else str(x).strip().upper())
        out["stress_status"] = s
        out["stress_evaluated"] = s.map(lambda v: bool(v))
        out["stress_ok"] = s.map(
            lambda v: True if v == "PASS" else (False if v in {"FAIL", "ABSTAIN"} else False)
        )
    else:
        if "stress_evaluated" not in out.columns:
            out["stress_evaluated"] = False
        if "stress_ok" not in out.columns:
            out["stress_ok"] = False

    # Normalize stress_tag for display/debug (do NOT overwrite raw field)
    if "stress_tag" in out.columns:
        out["stress_tag_norm"] = out["stress_tag"].map(_normalize_stress_tag)
    elif "stress_tag_norm" not in out.columns:
        out["stress_tag_norm"] = ""

    return out


def _load_gene_id_map_from_env() -> dict[str, str]:
    """
    Keep consistent with pipeline.py:
      - optional external mapping (TSV) via env LLMPATH_GENE_ID_MAP_TSV
    """
    try:
        p = (os.environ.get("LLMPATH_GENE_ID_MAP_TSV", "") or "").strip()
        if p:
            m = load_id_map_tsv(p)
            return m if isinstance(m, dict) else {}
    except Exception:
        pass
    return {}


# -----------------------------
# JSONL export (tool artifact; stable keys)
# -----------------------------
def write_report_jsonl(
    audit_log: pd.DataFrame,
    card: SampleCard,
    outdir: str,
    *,
    run_id: str,
    method: str | None = None,
    tau: float | None = None,
    condition: str | None = None,
    comparison: str | None = None,
    # Backward-compatible aliases (do not encourage new use)
    cancer: str | None = None,
    disease: str | None = None,
) -> Path:
    """
    Writes out/report.jsonl.

    Robustness:
      - Accept 'claim_json' OR fallbacks as source-of-truth payload.
      - Report should not crash if Claim validation fails (emit stub).
      - Missing metrics columns should NOT crash the export (emit nulls).

    Required columns in audit_log (strict):
      - status
      - <claim_json_col>  (claim_json or claim_json_str or claim_json_raw)
    Optional columns (best-effort):
      - term_survival_agg, context_score, stat, tau_used, stability_scope,
        module_id_effective, gene_set_hash_effective
      - stress_status/stress_reason/stress_notes
      - stress_tag / contradiction_flip (if present)
    """
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not run_id:
        run_id = datetime.now(timezone.utc).isoformat()

    claim_json_col = _pick_claim_json_col(audit_log)
    if claim_json_col is None:
        audit_log = _synthesize_claim_json_from_audit_log(audit_log, card)
        claim_json_col = _pick_claim_json_col(audit_log)

    if claim_json_col is None:
        raise ValueError(
            "audit_log missing claim_json payload column even after synthesis. "
            "Upstream dropped all claim fields needed to reconstruct payload."
        )

    _require_columns(audit_log, ["status", claim_json_col], "audit_log")

    audit0 = audit_log.reset_index(drop=True)
    df = _stringify_list_columns(audit0)

    # Optional metric series (all-null if missing)
    if "term_survival_agg" in audit0.columns:
        surv_s = pd.to_numeric(audit0["term_survival_agg"], errors="coerce")
    elif "term_survival" in audit0.columns:
        surv_s = pd.to_numeric(audit0["term_survival"], errors="coerce")
    elif "module_survival" in audit0.columns:
        surv_s = pd.to_numeric(audit0["module_survival"], errors="coerce")
    else:
        surv_s = pd.Series([pd.NA] * len(audit0))

    ctx_s = (
        pd.to_numeric(audit0.get("context_score"), errors="coerce")
        if "context_score" in audit0.columns
        else None
    )
    stat_s = (
        pd.to_numeric(audit0.get("stat"), errors="coerce") if "stat" in audit0.columns else None
    )

    method_val = "llm-pathway-curator" if method is None else str(method).strip()
    tau_val = _get_tau_default(card, default=0.8) if tau is None else float(tau)

    ctx = _get_card_context(card)

    default_condition = str(
        (condition or "")
        or (disease or "")
        or (cancer or "")
        or ctx["condition"]
        or ctx["disease_alias"]
    ).strip()
    default_comparison = str(comparison or ctx["comparison"] or "").strip()

    species = ""
    try:
        extra = getattr(card, "extra", {}) or {}
        if isinstance(extra, dict):
            species = str(extra.get("species", "") or "").strip()
    except Exception:
        species = ""

    jsonl_path = out_path / "report.jsonl"
    created_at = datetime.now(timezone.utc).isoformat()
    dev_meta_enabled = _get_dev_meta_enabled()
    schema_version_internal = _get_schema_version(card, default="v1")

    with jsonl_path.open("w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            _ = _normalize_status(row.get("status", ""))

            row_raw = audit0.loc[i]
            decision_obj = _decision_from_audit_row(row_raw)

            cj0 = str(row.get(claim_json_col, "") or "").strip()
            cj = _strip_excel_text_prefix(cj0)
            if not cj:
                raise ValueError(f"audit_log has empty {claim_json_col} for a row (required)")

            claim_raw = _safe_json_loads(cj) or {}

            try:
                claim_typed = Claim.model_validate(claim_raw)
                claim_dump = claim_typed.model_dump()
                claim_id_val = str(getattr(claim_typed, "claim_id", "") or "").strip()
                entity_val = str(getattr(claim_typed, "entity", "") or "").strip()
                ev = getattr(claim_typed, "evidence_ref", None)
                gene_set_hash_val = str(getattr(ev, "gene_set_hash", "") or "").strip().lower()
            except Exception:
                claim_dump = _claim_stub_from_raw(claim_raw)
                claim_id_val = str(claim_raw.get("claim_id", "") or "").strip()
                entity_val = str(claim_raw.get("entity", "") or "").strip()
                ev0 = (
                    claim_dump.get("evidence_ref", {})
                    if isinstance(claim_dump.get("evidence_ref"), dict)
                    else {}
                )
                gene_set_hash_val = str(ev0.get("gene_set_hash", "") or "").strip().lower()

            surv_val = _to_float_or_none(surv_s.loc[i])
            ctx_val = _to_float_or_none(ctx_s.loc[i]) if ctx_s is not None else None
            stat_val = _to_float_or_none(stat_s.loc[i]) if stat_s is not None else None

            row_condition = str(_get_first_present(row, ["condition"]) or "").strip()
            row_disease = str(_get_first_present(row, ["disease"]) or "").strip()
            row_cancer = str(_get_first_present(row, ["cancer", "cancer_type"]) or "").strip()
            condition_key = str(
                row_condition or row_disease or row_cancer or default_condition
            ).strip()
            comp_val = str(default_comparison or row.get("comparison", "") or "").strip()

            ctx_review = _extract_context_review_fields(claim_raw)

            # best-effort stress fields
            stress_tag = (
                _normalize_stress_tag(row_raw.get("stress_tag"))
                if ("stress_tag" in audit0.columns)
                else ""
            )
            contradiction_flip = None
            if "contradiction_flip" in audit0.columns and (
                not _is_na_scalar(row_raw.get("contradiction_flip"))
            ):
                try:
                    contradiction_flip = bool(row_raw.get("contradiction_flip"))
                except Exception:
                    contradiction_flip = str(row_raw.get("contradiction_flip"))

            rec: dict[str, Any] = {
                "created_at": created_at,
                "run_id": str(run_id),
                "method": method_val,
                "condition": condition_key,
                "disease": condition_key,
                "cancer": condition_key,
                "comparison": comp_val,
                "tau": float(tau_val),
                "species": species,
                "claim": claim_dump,
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
                    "stress_status": str(row_raw.get("stress_status") or "").strip(),
                    "stress_reason": str(row_raw.get("stress_reason") or "").strip(),
                    "stress_tag": stress_tag,
                    "contradiction_flip": contradiction_flip,
                    "context_status": str(ctx_review.get("context_status", "") or "").strip(),
                    "context_confidence": _to_float_or_none(ctx_review.get("context_confidence")),
                },
                "claim_id": claim_id_val,
                "decision_status": str(getattr(decision_obj, "status", "") or "").strip(),
                "decision": str(getattr(decision_obj, "status", "") or "").strip(),
                "survival": surv_val,
                "claim.entity_type": "term",
                "claim.entity_id": entity_val,
                "claim.context.condition": ctx["condition"],
                "claim.context.tissue": ctx["tissue"],
                "claim.context.perturbation": ctx["perturbation"],
                "claim.context.comparison": ctx["comparison"] or comp_val,
                "claim.context.disease": ctx["disease_alias"],
                "evidence_refs.gene_set_hash": gene_set_hash_val,
                "context_evaluated": ctx_review.get("context_evaluated", None),
                "context_method": ctx_review.get("context_method", None),
                "context_status": ctx_review.get("context_status", None),
                "context_reason": ctx_review.get("context_reason", None),
                "context_notes": ctx_review.get("context_notes", None),
                "context_confidence": ctx_review.get("context_confidence", None),
                "context_review_key": ctx_review.get("context_review_key", None),
                "context_review_version": ctx_review.get("context_review_version", None),
            }

            if dev_meta_enabled:
                rec["dev_meta"] = {"schema_version": schema_version_internal}

            rec_safe = _json_sanitize(rec)
            f.write(json.dumps(rec_safe, ensure_ascii=False) + "\n")

    return jsonl_path


def _ensure_canonical_audit_cols(audit_log: pd.DataFrame, card: SampleCard) -> pd.DataFrame:
    out = audit_log.copy()

    # ---- score / stability (canonical) ----
    if "term_survival_agg" not in out.columns:
        if "term_survival" in out.columns:
            out["term_survival_agg"] = out["term_survival"]
            out["stability_scope"] = out.get("stability_scope", "term")
        elif "module_survival" in out.columns:
            out["term_survival_agg"] = out["module_survival"]
            out["stability_scope"] = out.get("stability_scope", "module")
        else:
            out["term_survival_agg"] = pd.NA
            out["stability_scope"] = out.get("stability_scope", "")

    if "tau_used" not in out.columns:
        try:
            out["tau_used"] = float(_get_tau_default(card, default=0.8))
        except Exception:
            out["tau_used"] = pd.NA

    # ---- effective IDs (canonical) ----
    if "module_id_effective" not in out.columns:
        if "module_id" in out.columns:
            out["module_id_effective"] = out["module_id"]
        else:
            out["module_id_effective"] = ""

    if "gene_set_hash_effective" not in out.columns:
        if "gene_set_hash" in out.columns:
            out["gene_set_hash_effective"] = out["gene_set_hash"]
        else:
            out["gene_set_hash_effective"] = ""

    # ---- stress columns (exist even if empty) ----
    for c in ["stress_status", "stress_reason", "stress_notes"]:
        if c not in out.columns:
            out[c] = ""

    # stress tag / contradiction flip (best-effort, stable presence for report/debug)
    if "stress_tag" not in out.columns:
        out["stress_tag"] = ""
    if "contradiction_flip" not in out.columns:
        out["contradiction_flip"] = pd.NA

    # ---- context review columns (exist even if UNEVALUATED) ----
    for c in ["context_status", "context_reason", "context_confidence"]:
        if c not in out.columns:
            out[c] = ""

    # ---- basic decision columns (exist even if empty) ----
    for c in ["abstain_reason", "fail_reason", "audit_notes"]:
        if c not in out.columns:
            out[c] = ""

    return out


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

    audit_out = _ensure_canonical_audit_cols(audit_log.copy(), card)
    dist_out = distilled.copy()

    # -------------------------
    # Always write TSV artifacts (even if report tables are skipped)
    # -------------------------
    audit_out = _derive_stress_display_cols(audit_out)

    # Excel-safe ID columns
    if "gene_ids" in audit_out.columns:
        audit_out["gene_ids_str"] = audit_out["gene_ids"].map(_excel_safe_ids)
    elif "gene_ids_str" in audit_out.columns:
        audit_out["gene_ids_str"] = audit_out["gene_ids_str"].map(_excel_safe_ids)

    if "term_ids" in audit_out.columns:
        audit_out["term_ids_str"] = audit_out["term_ids"].map(_excel_safe_ids)
    elif "term_ids_str" in audit_out.columns:
        audit_out["term_ids_str"] = audit_out["term_ids_str"].map(_excel_safe_ids)

    if "gene_ids" in audit_out.columns and "gene_ids_str" in audit_out.columns:
        audit_out["gene_ids"] = audit_out["gene_ids_str"]
    if "term_ids" in audit_out.columns and "term_ids_str" in audit_out.columns:
        audit_out["term_ids"] = audit_out["term_ids_str"]

    # -------------------------
    # Display-only symbol mapping (best-effort, reproducible, no network)
    # Priority (consistent with pipeline intent):
    #   1) distilled-derived mapping
    #   2) env LLMPATH_GENE_ID_MAP_TSV
    #   3) card.extra.gene_id_map_tsv (optional override)
    # -------------------------
    id2sym = {}
    try:
        id2sym = build_id_to_symbol_from_distilled(distilled)
    except Exception:
        id2sym = {}

    id2sym = {**id2sym, **_load_gene_id_map_from_env()}

    try:
        ex = getattr(card, "extra", {}) or {}
        map_path = ex.get("gene_id_map_tsv", None) if isinstance(ex, dict) else None
        if map_path:
            id2sym_user = load_id_map_tsv(map_path)
            if isinstance(id2sym_user, dict) and id2sym_user:
                id2sym = {**id2sym, **id2sym_user}
    except Exception:
        pass

    if ("gene_ids_str" in audit_out.columns) or ("gene_ids" in audit_out.columns):
        src = "gene_ids_str" if "gene_ids_str" in audit_out.columns else "gene_ids"
        s = audit_out[src].map(lambda x: "" if _is_na_scalar(x) else str(x))
        syms = s.map(lambda x: map_ids_to_symbols(x, id2sym) if id2sym else [])
        audit_out["gene_symbols"] = syms
        audit_out["gene_symbols_str"] = syms.map(
            lambda xs: ";".join(map(str, xs)) if isinstance(xs, list) else ""
        )

    audit_out = _stringify_list_columns(audit_out)
    dist_out = _stringify_list_columns(dist_out)

    audit_tsv = out_path / "audit_log.tsv"
    distilled_tsv = out_path / "distilled.tsv"
    audit_out.to_csv(audit_tsv, sep="\t", index=False)
    dist_out.to_csv(distilled_tsv, sep="\t", index=False)

    # -------------------------
    # If audit_log has no status, write a minimal report and exit safely.
    # -------------------------
    status_col = "status" if "status" in audit_out.columns else None
    ctx = _get_card_context(card)

    lines: list[str] = []
    lines.append("# LLM-PathwayCurator report")
    lines.append("")
    lines.append("## Sample Card")
    lines.append(f"- condition: {ctx['condition']}")
    lines.append(f"- tissue: {ctx['tissue']}")
    lines.append(f"- perturbation: {ctx['perturbation']}")
    lines.append(f"- comparison: {ctx['comparison']}")
    if ctx["disease_alias"] and (ctx["disease_alias"] != ctx["condition"]):
        lines.append(f"- disease (alias): {ctx['disease_alias']}")
    notes = getattr(card, "notes", None)
    if notes:
        lines.append(f"- notes: {notes}")
    lines.append("")

    if status_col is None:
        lines.append("## Decisions")
        lines.append("")
        lines.append(
            "**WARNING:** audit_log is missing required column `status`. "
            "Skipping PASS/ABSTAIN/FAIL summaries."
        )
        lines.append("")
        lines.append("## Audit log (top)")
        lines.append("")
        lines.append(_safe_table_md(_stringify_list_columns(audit_out), n=20))
        lines.append("")
        lines.append("## Reproducible artifacts")
        lines.append("")
        lines.append(f"- audit_log.tsv: {audit_tsv.name}")
        lines.append(f"- distilled.tsv: {distilled_tsv.name}")
        report_path = out_path / "report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return

    # Normal case
    n_pass = int((audit_out[status_col] == "PASS").sum())
    n_abs = int((audit_out[status_col] == "ABSTAIN").sum())
    n_fail = int((audit_out[status_col] == "FAIL").sum())

    abs_sum, fail_sum = _reason_summary(audit_out)

    rc_summary: dict[str, float] | None = None
    rc_curve_path: Path | None = None
    score_col = _pick_score_col(audit_out)

    if risk_coverage_from_status is not None:
        try:
            rc_summary = risk_coverage_from_status(audit_out[status_col])
        except Exception:
            rc_summary = None

    if score_col and (risk_coverage_curve is not None):
        try:
            curve = risk_coverage_curve(audit_out, score_col=score_col, status_col=status_col)
            rc_curve_path = out_path / "risk_coverage.tsv"
            curve.to_csv(rc_curve_path, sep="\t", index=False)
        except Exception:
            rc_curve_path = out_path / "risk_coverage.tsv"
            pd.DataFrame(
                columns=["threshold", "coverage", "risk_fail_given_decided", "n_decided"]
            ).to_csv(rc_curve_path, sep="\t", index=False)

    def _top(df: pd.DataFrame) -> pd.DataFrame:
        if "term_survival_agg" in df.columns:
            return df.sort_values(["term_survival_agg"], ascending=[False], kind="mergesort")
        if "stat" in df.columns:
            return df.sort_values(["stat"], ascending=[False], kind="mergesort")
        if "claim_id" in df.columns:
            return df.sort_values(["claim_id"], ascending=[True], kind="mergesort")
        return df

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
    pass_df = audit_out[audit_out[status_col] == "PASS"]
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
                    "context_status",
                    "context_reason",
                    "context_confidence",
                    "stress_status",
                    "stress_reason",
                    "stress_tag_norm",
                    "contradiction_flip",
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
    abs_df2 = audit_out[audit_out[status_col] == "ABSTAIN"]
    abs_df2 = _top(abs_df2)
    lines.append(
        _safe_table_md(
            _prefer_str_cols(
                abs_df2,
                [
                    "claim_id",
                    "entity",
                    "direction",
                    "module_id_effective",
                    "abstain_reason",
                    "gene_ids",
                    "gene_symbols",
                    "gene_set_hash_effective",
                    "stress_status",
                    "stress_reason",
                    "stress_tag_norm",
                    "contradiction_flip",
                    "stress_evaluated",
                    "stress_ok",
                    "term_survival_agg",
                    "context_score",
                    "context_status",
                    "context_reason",
                    "context_confidence",
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
    fail_df2 = audit_out[audit_out[status_col] == "FAIL"]
    fail_df2 = _top(fail_df2)
    lines.append(
        _safe_table_md(
            _prefer_str_cols(
                fail_df2,
                [
                    "claim_id",
                    "entity",
                    "direction",
                    "module_id_effective",
                    "fail_reason",
                    "gene_ids",
                    "gene_symbols",
                    "gene_set_hash_effective",
                    "stress_status",
                    "stress_reason",
                    "stress_tag_norm",
                    "contradiction_flip",
                    "context_status",
                    "context_reason",
                    "context_confidence",
                    "audit_notes",
                ],
            ),
            n=5,
        )
    )
    lines.append("")

    lines.append("## Audit log (top)")
    lines.append("")
    # IMPORTANT: show canonicalized audit_out (matches TSV artifacts)
    lines.append(_safe_table_md(_stringify_list_columns(audit_out), n=20))
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
