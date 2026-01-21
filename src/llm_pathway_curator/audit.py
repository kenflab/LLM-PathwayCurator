# LLM-PathwayCurator/src/llm_pathway_curator/audit.py
from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

from .audit_reasons import (
    ABSTAIN_CONTEXT_NONSPECIFIC,
    ABSTAIN_HUB_BRIDGE,
    ABSTAIN_INCONCLUSIVE_STRESS,
    ABSTAIN_MISSING_SURVIVAL,
    ABSTAIN_REASONS,
    ABSTAIN_UNDER_SUPPORTED,
    ABSTAIN_UNSTABLE,
    FAIL_CONTRADICTION,
    FAIL_EVIDENCE_DRIFT,
    FAIL_REASONS,
    FAIL_SCHEMA_VIOLATION,
)
from .claim_schema import Claim
from .sample_card import SampleCard

_NA_TOKENS = {"", "na", "nan", "none", "NA"}


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


def _parse_ids(x: Any) -> list[str]:
    """
    Parse comma/semicolon separated ids into list[str].
    Safe against list-like inputs (won't call pd.isna on list).
    """
    if _is_na_scalar(x):
        return []
    if isinstance(x, (list, tuple, set)):
        items = [str(t).strip() for t in x if str(t).strip()]
    else:
        s = str(x).strip()
        if not s or s in _NA_TOKENS or s.lower() in {t.lower() for t in _NA_TOKENS}:
            return []
        s = s.replace(";", ",")
        items = [t.strip() for t in s.split(",") if t.strip()]

    seen: set[str] = set()
    uniq: list[str] = []
    for t in items:
        if t in _NA_TOKENS or t.lower() in {x.lower() for x in _NA_TOKENS}:
            continue
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _norm_gene_id(g: str) -> str:
    """Minimal normalization to reduce spurious drift (HGNC-like)."""
    return str(g).strip().upper()


def _hash_gene_set(genes: list[str]) -> str:
    """
    Audit-grade gene_set_hash should be SET-stable:
      - same gene set -> same hash, regardless of order
      - normalize IDs to reduce casing drift
    """
    uniq = sorted({_norm_gene_id(g) for g in genes if str(g).strip()})
    payload = ",".join(uniq)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _hash_term_ids(term_ids: list[str]) -> str:
    """Set-stable hash for referenced term IDs (fallback contradiction key)."""
    uniq = sorted({str(t).strip() for t in term_ids if str(t).strip()})
    payload = ",".join(uniq)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _looks_like_hex_hash(x: Any) -> bool:
    """12-hex short hash (sha256[:12])."""
    if _is_na_scalar(x):
        return False
    s = str(x).strip().lower()
    if len(s) != 12:
        return False
    return all(ch in "0123456789abcdef" for ch in s)


# -------- SampleCard knob access (single boundary, with robust fallback) --------
def _get_extra(card: SampleCard) -> dict[str, Any]:
    try:
        ex = getattr(card, "extra", {}) or {}
        return ex if isinstance(ex, dict) else {}
    except Exception:
        return {}


def _as_bool(x: Any, default: bool) -> bool:
    if x is None:
        return bool(default)
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    return bool(default)


def _get_tau_default(card: SampleCard) -> float:
    # Source of truth: SampleCard.audit_tau()
    try:
        return float(card.audit_tau())
    except Exception:
        return 0.8


def _get_min_overlap_default(card: SampleCard) -> int:
    try:
        return int(card.audit_min_gene_overlap())
    except Exception:
        return 1


def _get_min_union_genes(card: SampleCard, default: int = 3) -> int:
    if hasattr(card, "min_union_genes") and callable(card.min_union_genes):
        try:
            return int(card.min_union_genes(default=default))  # type: ignore[attr-defined]
        except Exception:
            pass
    ex = _get_extra(card)
    try:
        v = ex.get("min_union_genes", None)
        return int(default) if v is None else int(v)
    except Exception:
        return int(default)


def _get_hub_term_degree(card: SampleCard, default: int = 200) -> int:
    if hasattr(card, "hub_term_degree") and callable(card.hub_term_degree):
        try:
            return int(card.hub_term_degree(default=default))  # type: ignore[attr-defined]
        except Exception:
            pass
    ex = _get_extra(card)
    try:
        v = ex.get("hub_term_degree", None)
        return int(default) if v is None else int(v)
    except Exception:
        return int(default)


def _get_hub_frac_thr(card: SampleCard, default: float = 0.5) -> float:
    if hasattr(card, "hub_frac_thr") and callable(card.hub_frac_thr):
        try:
            return float(card.hub_frac_thr(default=default))  # type: ignore[attr-defined]
        except Exception:
            pass
    ex = _get_extra(card)
    try:
        v = ex.get("hub_frac_thr", None)
        return float(default) if v is None else float(v)
    except Exception:
        return float(default)


def _get_pass_notes(card: SampleCard, default: bool = True) -> bool:
    if hasattr(card, "pass_notes") and callable(card.pass_notes):
        try:
            return bool(card.pass_notes(default=default))  # type: ignore[attr-defined]
        except Exception:
            pass
    ex = _get_extra(card)
    v = ex.get("pass_notes", None)
    if v is None:
        return bool(default)
    return _as_bool(v, default=bool(default))


def _get_context_gate_mode(card: SampleCard, default: str = "hard") -> str:
    """
    Tool contract (user-facing):
      - "off": ignore context gate
      - "note": annotate context nonspecificity, do not change status
      - "hard": context nonspecific => ABSTAIN_CONTEXT_NONSPECIFIC
    Backward compat:
      - "abstain" => "hard"
    """
    if hasattr(card, "context_gate_mode") and callable(card.context_gate_mode):
        try:
            s = str(card.context_gate_mode(default=default)).strip().lower()  # type: ignore[attr-defined]
        except Exception:
            s = ""
    else:
        ex = _get_extra(card)
        s = str(ex.get("context_gate_mode", default)).strip().lower()

    if s in {"off", "disable", "disabled", "none"}:
        return "off"
    if s in {"note", "warn", "warning"}:
        return "note"
    if s in {"abstain"}:
        return "hard"
    return "hard"


def _get_stress_gate_mode(card: SampleCard, default: str = "off") -> str:
    """
    Tool contract (user-facing):
      - "off" (default): ignore missing stress entirely
      - "note": missing stress => PASS with note; stress FAIL => PASS with note
      - "hard": missing stress => ABSTAIN_INCONCLUSIVE_STRESS;
        stress FAIL => FAIL_EVIDENCE_DRIFT (auditable violation)
    Backward compat:
      - "abstain" => "hard"
    """
    if hasattr(card, "stress_gate_mode") and callable(card.stress_gate_mode):
        try:
            s = str(card.stress_gate_mode(default=default)).strip().lower()  # type: ignore[attr-defined]
        except Exception:
            s = ""
    else:
        ex = _get_extra(card)
        s = str(ex.get("stress_gate_mode", default)).strip().lower()

    if s in {"off", "none", "disable", "disabled"}:
        return "off"
    if s in {"note", "warn", "warning"}:
        return "note"
    if s in {"hard", "abstain"}:
        return "hard"

    d = str(default).strip().lower()
    if d in {"off", "note", "hard"}:
        return d
    return "off"


def _get_stability_gate_mode(card: SampleCard, default: str = "hard") -> str:
    """
    Tool contract (user-facing):
      - "off": do not gate on stability (research only)
      - "note": do not change status; annotate instability
      - "hard": agg < tau => ABSTAIN_UNSTABLE
    Backward compat:
      - "abstain" => "hard"
    """
    if hasattr(card, "stability_gate_mode") and callable(card.stability_gate_mode):
        try:
            s = str(card.stability_gate_mode(default=default)).strip().lower()  # type: ignore[attr-defined]
        except Exception:
            s = ""
    else:
        ex = _get_extra(card)
        s = str(ex.get("stability_gate_mode", default)).strip().lower()

    if s in {"off", "disable", "disabled", "none"}:
        return "off"
    if s in {"note", "warn", "warning"}:
        return "note"
    if s in {"abstain"}:
        return "hard"
    return "hard"


def _get_strict_evidence_check(card: SampleCard, default: bool = False) -> bool:
    """
    If True: missing evidence_genes in distilled => FAIL_SCHEMA_VIOLATION (strict contract).
    If False (default): missing evidence_genes => ABSTAIN_INCONCLUSIVE_STRESS (decision-grade).
    """
    ex = _get_extra(card)
    return _as_bool(ex.get("strict_evidence_check", None), default=default)


def _norm_status(x: Any) -> str:
    if _is_na_scalar(x):
        return ""
    return str(x).strip().upper()


def _append_note(old: str, msg: str) -> str:
    old = "" if _is_na_scalar(old) else str(old)
    return msg if not old else f"{old} | {msg}"


def _extract_evidence_from_claim_json(cj: str) -> tuple[list[str], list[str], Any, Any]:
    """
    Source of truth: claim_json (Claim schema).
    Returns: (term_ids, gene_ids, gene_set_hash, module_id)
    """
    try:
        obj = json.loads(cj)
    except Exception:
        return ([], [], None, None)

    ev = None
    if isinstance(obj, dict):
        ev = obj.get("evidence_ref") or obj.get("evidence_refs")
        if ev is None and isinstance(obj.get("claim"), dict):
            ev = obj["claim"].get("evidence_ref") or obj["claim"].get("evidence_refs")

    if not isinstance(ev, dict):
        return ([], [], None, None)

    term_ids = _parse_ids(ev.get("term_ids") or ev.get("term_id") or "")
    gene_ids = _parse_ids(ev.get("gene_ids") or ev.get("gene_id") or "")
    gsh = ev.get("gene_set_hash")
    mid = ev.get("module_id")
    if isinstance(mid, (list, tuple, set)):
        mid = next(iter(mid), None)
    return (term_ids, gene_ids, gsh, mid)


def _extract_direction_from_claim_json(cj: str) -> str:
    """Robust direction extraction (fallback). Returns lower-case direction in {up,down,na}."""
    try:
        obj = json.loads(cj)
    except Exception:
        return "na"

    d = None
    if isinstance(obj, dict):
        d = obj.get("direction")
        if d is None and isinstance(obj.get("claim"), dict):
            d = obj["claim"].get("direction")

    s = "" if d is None else str(d).strip().lower()
    if s in {"up", "down"}:
        return s
    return "na"


def _inject_stress_from_distilled(out: pd.DataFrame, distilled: pd.DataFrame) -> pd.DataFrame:
    """
    Backfill stress/contradiction columns on claims using distilled annotations.

    Why:
      - Tool contract says audit consumes stress if provided.
      - Fig2 v4 style pipelines may annotate distilled (stress_tag/contradiction_flip)
        but not mutate claims DF.
      - This function turns distilled annotations into "external" stress inputs
        ONLY when claims did not already provide them.

    Rules (minimal + auditable):
      - If claims already include any stress_* columns, do nothing (respect caller).
      - Else if distilled has stress_tag:
          * non-empty tag => stress_status=FAIL, stress_reason="STRESS_TAG"
          * empty tag => leave stress columns absent (not evaluated)
      - If claims already include contradiction_* columns, do nothing.
      - Else if distilled has contradiction_flip:
          * True => contradiction_status=FAIL, contradiction_reason="CONTRADICTION_FLIP"
          * False/NA => leave absent

    Notes:
      - This does NOT change evidence validation logic; it only enables existing gates.
      - Detailed per-term logs belong in masking.py term_events (written by pipeline).
    """
    out2 = out.copy()

    claim_has_stress = any(
        c in out2.columns for c in ["stress_status", "stress_ok", "stress_reason", "stress_notes"]
    )
    claim_has_contra = any(
        c in out2.columns
        for c in [
            "contradiction_status",
            "contradiction_ok",
            "contradiction_reason",
            "contradiction_notes",
        ]
    )

    # Build term_uid in distilled if needed
    dist = distilled.copy()
    if "term_uid" not in dist.columns:
        if {"source", "term_id"}.issubset(set(dist.columns)):
            dist["term_uid"] = (
                dist["source"].astype(str).str.strip()
                + ":"
                + dist["term_id"].astype(str).str.strip()
            )
        elif "term_id" in dist.columns:
            dist["term_uid"] = dist["term_id"].astype(str).str.strip()
        else:
            return out2

    dist["term_uid"] = dist["term_uid"].astype(str).str.strip()

    # We map from term_uid -> tag/flip (first non-empty occurrence wins)
    stress_map: dict[str, str] = {}
    if (not claim_has_stress) and ("stress_tag" in dist.columns):
        for tu, tag in zip(dist["term_uid"].tolist(), dist["stress_tag"].tolist(), strict=False):
            t = "" if _is_na_scalar(tag) else str(tag).strip()
            if not t:
                continue
            if tu and tu not in stress_map:
                stress_map[tu] = t

    contra_map: dict[str, bool] = {}
    if (not claim_has_contra) and ("contradiction_flip" in dist.columns):
        for tu, v in zip(
            dist["term_uid"].tolist(), dist["contradiction_flip"].tolist(), strict=False
        ):
            if not tu:
                continue
            if tu in contra_map:
                continue
            if _is_na_scalar(v):
                continue
            try:
                contra_map[tu] = bool(v)
            except Exception:
                continue

    # If nothing to inject, return as-is
    if not stress_map and not contra_map:
        return out2

    # Ensure schema columns exist only if we are injecting
    if stress_map and (not claim_has_stress):
        out2["stress_status"] = ""
        out2["stress_reason"] = ""
        out2["stress_notes"] = ""
    if contra_map and (not claim_has_contra):
        out2["contradiction_status"] = ""
        out2["contradiction_reason"] = ""
        out2["contradiction_notes"] = ""

    # Inject per-claim using claim_json evidence_ref.term_ids (term_uid space)
    for i, row in out2.iterrows():
        cj = row.get("claim_json", None)
        if cj is None or _is_na_scalar(cj) or (not str(cj).strip()):
            continue

        term_ids, _gene_ids, _gsh, _mid = _extract_evidence_from_claim_json(str(cj))
        if not term_ids:
            continue

        # stress: if ANY referenced term has a stress tag, mark this claim stressed
        if stress_map and (not claim_has_stress):
            tags = [stress_map.get(t, "") for t in term_ids]
            tags = [t for t in tags if t]
            if tags:
                out2.at[i, "stress_status"] = "FAIL"
                out2.at[i, "stress_reason"] = "STRESS_TAG"
                out2.at[i, "stress_notes"] = (
                    f"stress_tag={tags[0]}" if len(tags) == 1 else f"stress_tag={tags}"
                )

        # contradiction: if ANY referenced term has contradiction_flip True, mark FAIL
        if contra_map and (not claim_has_contra):
            flips = [contra_map.get(t, False) for t in term_ids]
            if any(bool(x) for x in flips):
                out2.at[i, "contradiction_status"] = "FAIL"
                out2.at[i, "contradiction_reason"] = "CONTRADICTION_FLIP"
                out2.at[i, "contradiction_notes"] = "distilled.contradiction_flip=True"

    return out2


def _apply_external_contradiction(out: pd.DataFrame, i: int, row: pd.Series) -> None:
    """
    External contradiction check (optional).
    If contradiction_status is present but missing/unknown, we treat it as inconclusive.
    """
    if "contradiction_status" not in out.columns:
        return

    st = _norm_status(row.get("contradiction_status"))
    if not st:
        out.at[i, "status"] = "ABSTAIN"
        out.at[i, "contradiction_ok"] = False
        out.at[i, "abstain_reason"] = ABSTAIN_INCONCLUSIVE_STRESS
        out.at[i, "audit_notes"] = _append_note(
            out.at[i, "audit_notes"], "contradiction_status missing"
        )
        return

    if st == "PASS":
        out.at[i, "contradiction_ok"] = True
        return

    if st == "ABSTAIN":
        out.at[i, "status"] = "ABSTAIN"
        out.at[i, "contradiction_ok"] = False
        out.at[i, "abstain_reason"] = ABSTAIN_INCONCLUSIVE_STRESS
        out.at[i, "audit_notes"] = _append_note(out.at[i, "audit_notes"], "contradiction=ABSTAIN")
        return

    if st == "FAIL":
        out.at[i, "status"] = "FAIL"
        out.at[i, "contradiction_ok"] = False
        out.at[i, "fail_reason"] = FAIL_CONTRADICTION
        rn = str(row.get("contradiction_reason", "")).strip()
        nt = str(row.get("contradiction_notes", "")).strip()
        note = "contradiction"
        if rn:
            note += f": {rn}"
        if nt:
            note += f" ({nt})"
        out.at[i, "audit_notes"] = _append_note(out.at[i, "audit_notes"], note)
        return

    out.at[i, "status"] = "FAIL"
    out.at[i, "contradiction_ok"] = False
    out.at[i, "fail_reason"] = FAIL_SCHEMA_VIOLATION
    out.at[i, "audit_notes"] = _append_note(
        out.at[i, "audit_notes"], f"contradiction_status unknown={st}"
    )


def _apply_external_stress(
    out: pd.DataFrame,
    i: int,
    row: pd.Series,
    *,
    external_cols: set[str],
    gate_mode: str,
) -> None:
    """
    Apply stress results ONLY if they were provided as inputs in the original claims DF.

    Semantics:
      - stress_evaluated indicates whether stress was supplied as an input.
      - stress_ok is:
          True/False when evaluated,
          NA when not evaluated or missing (unless we hard-gate and convert to ABSTAIN/FAIL).
    """
    has_any = any(
        c in external_cols for c in ["stress_status", "stress_ok", "stress_reason", "stress_notes"]
    )
    if not has_any:
        out.at[i, "stress_ok"] = pd.NA
        return

    # If only stress_ok provided
    if ("stress_ok" in external_cols) and ("stress_status" not in external_cols):
        v = row.get("stress_ok")
        if _is_na_scalar(v):
            out.at[i, "stress_ok"] = pd.NA
            if gate_mode == "hard":
                out.at[i, "status"] = "ABSTAIN"
                out.at[i, "abstain_reason"] = ABSTAIN_INCONCLUSIVE_STRESS
                out.at[i, "audit_notes"] = _append_note(
                    out.at[i, "audit_notes"], "stress_ok missing (hard)"
                )
            else:
                out.at[i, "audit_notes"] = _append_note(
                    out.at[i, "audit_notes"], "stress_ok missing"
                )
            return

        ok = bool(v)
        out.at[i, "stress_ok"] = ok

        if ok:
            out.at[i, "audit_notes"] = _append_note(out.at[i, "audit_notes"], "stress_ok=True")
            return

        # stress_ok=False
        if gate_mode == "hard":
            out.at[i, "status"] = "FAIL"
            out.at[i, "fail_reason"] = FAIL_EVIDENCE_DRIFT
            out.at[i, "audit_notes"] = _append_note(
                out.at[i, "audit_notes"], "stress_ok=False (hard)"
            )
        else:
            out.at[i, "audit_notes"] = _append_note(out.at[i, "audit_notes"], "stress_ok=False")
        return

    st = _norm_status(row.get("stress_status"))
    rs_raw = str(row.get("stress_reason", "")).strip()
    nt = str(row.get("stress_notes", "")).strip()

    if not st:
        out.at[i, "stress_ok"] = pd.NA
        if gate_mode == "hard":
            out.at[i, "status"] = "ABSTAIN"
            out.at[i, "abstain_reason"] = ABSTAIN_INCONCLUSIVE_STRESS
            out.at[i, "audit_notes"] = _append_note(
                out.at[i, "audit_notes"], "stress_status missing (hard)"
            )
        else:
            out.at[i, "audit_notes"] = _append_note(
                out.at[i, "audit_notes"], "stress_status missing"
            )
        return

    if st == "PASS":
        out.at[i, "stress_ok"] = True
        out.at[i, "audit_notes"] = _append_note(out.at[i, "audit_notes"], "stress=PASS")
        return

    if st == "ABSTAIN":
        out.at[i, "stress_ok"] = False
        rs = rs_raw if rs_raw in ABSTAIN_REASONS else ABSTAIN_INCONCLUSIVE_STRESS
        if gate_mode == "hard":
            out.at[i, "status"] = "ABSTAIN"
            out.at[i, "abstain_reason"] = rs
        note = "stress=ABSTAIN"
        if rs_raw:
            note += f": {rs_raw}"
        if nt:
            note += f" ({nt})"
        note += f" [{gate_mode}]"
        out.at[i, "audit_notes"] = _append_note(out.at[i, "audit_notes"], note)
        return

    if st == "FAIL":
        out.at[i, "stress_ok"] = False
        note = "stress=FAIL"
        if rs_raw:
            note += f": {rs_raw}"
        if nt:
            note += f" ({nt})"
        note += f" [{gate_mode}]"

        if gate_mode == "hard":
            out.at[i, "status"] = "FAIL"
            out.at[i, "fail_reason"] = FAIL_EVIDENCE_DRIFT
            out.at[i, "audit_notes"] = _append_note(out.at[i, "audit_notes"], note)
            return

        out.at[i, "audit_notes"] = _append_note(out.at[i, "audit_notes"], note)
        return

    out.at[i, "status"] = "FAIL"
    out.at[i, "stress_ok"] = False
    out.at[i, "fail_reason"] = FAIL_SCHEMA_VIOLATION
    out.at[i, "audit_notes"] = _append_note(out.at[i, "audit_notes"], f"stress_status unknown={st}")


def _enforce_reason_vocab(out: pd.DataFrame, i: int) -> None:
    """
    Contract enforcement:
    - If status=FAIL, fail_reason must be one of FAIL_REASONS
    - If status=ABSTAIN, abstain_reason must be one of ABSTAIN_REASONS
    Any violation => FAIL_SCHEMA_VIOLATION
    """
    st = str(out.at[i, "status"]).strip().upper()

    if st == "FAIL":
        fr = (
            "" if _is_na_scalar(out.at[i, "fail_reason"]) else str(out.at[i, "fail_reason"]).strip()
        )
        if fr not in FAIL_REASONS:
            out.at[i, "fail_reason"] = FAIL_SCHEMA_VIOLATION
            out.at[i, "audit_notes"] = _append_note(
                out.at[i, "audit_notes"], f"invalid fail_reason={fr}"
            )

    if st == "ABSTAIN":
        ar = (
            ""
            if _is_na_scalar(out.at[i, "abstain_reason"])
            else str(out.at[i, "abstain_reason"]).strip()
        )
        if ar not in ABSTAIN_REASONS:
            out.at[i, "status"] = "FAIL"
            out.at[i, "fail_reason"] = FAIL_SCHEMA_VIOLATION
            out.at[i, "abstain_reason"] = ""
            out.at[i, "audit_notes"] = _append_note(
                out.at[i, "audit_notes"], f"invalid abstain_reason={ar}"
            )

    if st in {"FAIL", "ABSTAIN"}:
        note = (
            "" if _is_na_scalar(out.at[i, "audit_notes"]) else str(out.at[i, "audit_notes"]).strip()
        )
        if not note:
            out.at[i, "audit_notes"] = f"{st}: reason recorded"


def _apply_intra_run_contradiction(out: pd.DataFrame) -> None:
    """
    Intra-run contradiction check:
      - Among PASS claims only
      - If same evidence key has both direction=up and direction=down => FAIL_CONTRADICTION
    """
    if "evidence_key" not in out.columns or "direction_norm" not in out.columns:
        return

    df = out.copy()
    df["status_u"] = df["status"].astype(str).str.strip().str.upper()
    df = df[df["status_u"] == "PASS"].copy()
    if df.empty:
        return

    grouped = df.groupby("evidence_key", dropna=False)
    for key, g in grouped:
        if _is_na_scalar(key) or str(key).strip() == "":
            continue
        dirs = set(g["direction_norm"].astype(str).tolist())
        if ("up" in dirs) and ("down" in dirs):
            idxs = g.index.tolist()
            note = f"intra-run contradiction: evidence_key={key} has both up and down"
            for i in idxs:
                out.at[i, "status"] = "FAIL"
                out.at[i, "fail_reason"] = FAIL_CONTRADICTION
                out.at[i, "contradiction_ok"] = False
                out.at[i, "audit_notes"] = _append_note(out.at[i, "audit_notes"], note)
                _enforce_reason_vocab(out, i)


def audit_claims(
    claims: pd.DataFrame,
    distilled: pd.DataFrame,
    card: SampleCard,
    *,
    tau: float | None = None,
) -> pd.DataFrame:
    """
    Mechanical audit (tool-facing).

    Status priority: FAIL > ABSTAIN > PASS

    Source of truth for evidence references:
      1) claim_json (Claim schema)
      2) DataFrame columns as fallback (backward compatibility)

    NOTE:
      - This function does NOT generate stress tests; it only consumes them if provided.
      - Missing stress is governed by SampleCard.stress_gate_mode() (default: off).
      - If claims lack stress/contradiction columns but distilled provides
        stress_tag / contradiction_flip,
        we backfill them deterministically (v4 support).
    """
    out = claims.copy()

    # ---- v4 compat: inject stress/contradiction columns from distilled when absent ----
    out = _inject_stress_from_distilled(out, distilled)

    external_cols: set[str] = set(out.columns)

    # Stable audit fields
    out["status"] = "PASS"
    out["link_ok"] = True
    out["stability_ok"] = True
    out["contradiction_ok"] = pd.NA
    out["stress_ok"] = pd.NA
    out["stress_evaluated"] = False
    out["abstain_reason"] = ""
    out["fail_reason"] = ""
    out["audit_notes"] = ""

    # reporting aids
    if "tau_used" not in out.columns:
        out["tau_used"] = pd.NA
    if "term_survival_agg" not in out.columns:
        out["term_survival_agg"] = pd.NA
    if "stability_scope" not in out.columns:
        out["stability_scope"] = ""
    if "module_id_effective" not in out.columns:
        out["module_id_effective"] = ""

    # For intra-run contradiction
    if "gene_set_hash_effective" not in out.columns:
        out["gene_set_hash_effective"] = ""
    if "term_ids_set_hash" not in out.columns:
        out["term_ids_set_hash"] = ""
    if "evidence_key" not in out.columns:
        out["evidence_key"] = ""
    if "direction_norm" not in out.columns:
        out["direction_norm"] = "na"

    # Prepare distilled lookup
    dist = distilled.copy()
    if "term_uid" not in dist.columns:
        if {"source", "term_id"}.issubset(set(dist.columns)):
            dist["term_uid"] = (
                dist["source"].astype(str).str.strip()
                + ":"
                + dist["term_id"].astype(str).str.strip()
            )
        elif "term_id" in dist.columns:
            dist["term_uid"] = dist["term_id"].astype(str).str.strip()
        else:
            raise ValueError("audit_claims: distilled must have term_uid OR (source, term_id)")

    key_col = "term_uid"

    term_key = dist[key_col].astype(str).str.strip()
    term_key = term_key[~term_key.isin(_NA_TOKENS)]
    term_key = term_key[~term_key.str.lower().isin({t.lower() for t in _NA_TOKENS})]
    known_terms = set(term_key.tolist())

    has_survival = "term_survival" in distilled.columns
    if has_survival:
        tmp = distilled.copy()
        tmp["term_key"] = tmp[key_col].astype(str).str.strip()
        tmp["term_survival"] = pd.to_numeric(tmp["term_survival"], errors="coerce")
        term_surv = tmp.groupby("term_key")["term_survival"].min()
    else:
        term_surv = None

    has_module = "module_id" in distilled.columns
    module_surv = None
    if has_survival and has_module:
        tmpm = distilled.copy()
        tmpm["module_id"] = tmpm["module_id"].astype(str).str.strip()
        tmpm["term_survival"] = pd.to_numeric(tmpm["term_survival"], errors="coerce")
        tmpm = tmpm[~tmpm["module_id"].isin(_NA_TOKENS)]
        tmpm = tmpm[~tmpm["module_id"].str.lower().isin({t.lower() for t in _NA_TOKENS})]
        if not tmpm.empty:
            module_surv = tmpm.groupby("module_id")["term_survival"].min()

    # Resolve tau default (contract)
    tau_default = _get_tau_default(card)
    if tau is not None:
        try:
            tau_default = float(tau)
        except Exception:
            pass

    min_overlap_default = _get_min_overlap_default(card)

    # Precompute evidence genes per term_key
    term_to_gene_set: dict[str, set[str]] = {}
    if "evidence_genes" in dist.columns:
        for tk, xs in zip(
            dist[key_col].astype(str).str.strip(),
            dist["evidence_genes"],
            strict=True,
        ):
            if tk in _NA_TOKENS or tk.lower() in {t.lower() for t in _NA_TOKENS}:
                continue
            if isinstance(xs, (list, tuple, set)):
                genes = [_norm_gene_id(g) for g in xs if str(g).strip()]
            else:
                genes = [_norm_gene_id(g) for g in _parse_ids(xs)]
            gs = set([g for g in genes if g])
            if not gs:
                continue
            term_to_gene_set.setdefault(tk, set()).update(gs)

    elif "evidence_genes_str" in dist.columns:
        for tk, s in zip(
            dist[key_col].astype(str).str.strip(),
            dist["evidence_genes_str"].astype(str),
            strict=True,
        ):
            if tk in _NA_TOKENS or tk.lower() in {t.lower() for t in _NA_TOKENS}:
                continue
            genes = [_norm_gene_id(g) for g in _parse_ids(s)]
            gs = set([g for g in genes if g])
            if not gs:
                continue
            term_to_gene_set.setdefault(tk, set()).update(gs)

    # Hub gene heuristic
    gene_to_term_degree: dict[str, int] = {}
    for _tk, gs in term_to_gene_set.items():
        for g in gs:
            gene_to_term_degree[g] = gene_to_term_degree.get(g, 0) + 1

    hub_thr = _get_hub_term_degree(card, default=200)
    hub_genes = {g for g, deg in gene_to_term_degree.items() if deg > int(hub_thr)}

    pass_note_enabled = _get_pass_notes(card, default=True)
    context_gate_mode = _get_context_gate_mode(card, default="hard")
    stability_gate_mode = _get_stability_gate_mode(card, default="hard")

    stress_gate_mode = _get_stress_gate_mode(card, default="off")
    strict_evidence = _get_strict_evidence_check(card, default=False)

    min_union = _get_min_union_genes(card, default=3)
    hub_frac_thr = _get_hub_frac_thr(card, default=0.5)

    stress_cols = {"stress_status", "stress_ok", "stress_reason", "stress_notes"}
    has_stress_any = any(c in external_cols for c in stress_cols)

    for i, row in out.iterrows():
        # 0) schema validation (claim_json -> Claim)
        cj = row.get("claim_json", None)
        if cj is None or _is_na_scalar(cj) or (not str(cj).strip()):
            out.at[i, "status"] = "FAIL"
            out.at[i, "link_ok"] = False
            out.at[i, "fail_reason"] = FAIL_SCHEMA_VIOLATION
            out.at[i, "audit_notes"] = "missing claim_json"
            _enforce_reason_vocab(out, i)
            continue

        cj_str = str(cj)
        try:
            Claim.model_validate_json(cj_str)
        except Exception as e:
            out.at[i, "status"] = "FAIL"
            out.at[i, "link_ok"] = False
            out.at[i, "fail_reason"] = FAIL_SCHEMA_VIOLATION
            out.at[i, "audit_notes"] = f"claim_json schema violation: {type(e).__name__}"
            _enforce_reason_vocab(out, i)
            continue

        # direction (for contradiction check)
        d = str(row.get("direction", "")).strip().lower()
        if d not in {"up", "down", "na"}:
            d = _extract_direction_from_claim_json(cj_str)
        out.at[i, "direction_norm"] = d if d in {"up", "down"} else "na"

        # Per-claim tau override
        tau_row = tau_default
        if "tau" in out.columns and not _is_na_scalar(row.get("tau")):
            try:
                tau_row = float(row.get("tau"))
            except Exception:
                tau_row = tau_default
        out.at[i, "tau_used"] = float(tau_row)

        # Per-claim min_overlap override
        min_overlap = min_overlap_default
        if "min_overlap" in out.columns and not _is_na_scalar(row.get("min_overlap")):
            try:
                min_overlap = int(row.get("min_overlap"))
            except Exception:
                min_overlap = min_overlap_default

        # 1) evidence refs
        term_ids, gene_ids, gsh, module_id = _extract_evidence_from_claim_json(cj_str)

        module_id = "" if module_id is None else str(module_id).strip()
        out.at[i, "module_id_effective"] = module_id

        gene_ids = [_norm_gene_id(g) for g in gene_ids]

        if not term_ids:
            out.at[i, "status"] = "FAIL"
            out.at[i, "link_ok"] = False
            out.at[i, "fail_reason"] = FAIL_EVIDENCE_DRIFT
            out.at[i, "audit_notes"] = "empty term_ids"
            _enforce_reason_vocab(out, i)
            continue

        missing = [t for t in term_ids if t not in known_terms]
        if missing:
            out.at[i, "status"] = "FAIL"
            out.at[i, "link_ok"] = False
            out.at[i, "fail_reason"] = FAIL_EVIDENCE_DRIFT
            out.at[i, "audit_notes"] = f"unknown term_ids={missing}"
            _enforce_reason_vocab(out, i)
            continue

        # union evidence genes across referenced terms
        ev_union: list[str] = []
        ev_seen: set[str] = set()
        for t in term_ids:
            for g in sorted(term_to_gene_set.get(t, set())):
                if g not in ev_seen:
                    ev_seen.add(g)
                    ev_union.append(g)

        computed_gsh = _hash_gene_set(ev_union) if ev_union else ""
        out.at[i, "term_ids_set_hash"] = _hash_term_ids(term_ids)

        gsh_norm = "" if _is_na_scalar(gsh) else str(gsh).strip().lower()
        if not _looks_like_hex_hash(gsh_norm):
            out.at[i, "status"] = "FAIL"
            out.at[i, "link_ok"] = False
            out.at[i, "fail_reason"] = FAIL_SCHEMA_VIOLATION
            out.at[i, "audit_notes"] = "gene_set_hash missing/invalid"
            _enforce_reason_vocab(out, i)
            continue

        if not ev_union:
            # cannot verify hash without distilled evidence genes
            if strict_evidence:
                out.at[i, "status"] = "FAIL"
                out.at[i, "link_ok"] = False
                out.at[i, "fail_reason"] = FAIL_SCHEMA_VIOLATION
                out.at[i, "audit_notes"] = (
                    "evidence_genes unavailable for referenced term_ids (strict)"
                )
            else:
                out.at[i, "status"] = "ABSTAIN"
                out.at[i, "link_ok"] = False
                out.at[i, "abstain_reason"] = ABSTAIN_INCONCLUSIVE_STRESS
                out.at[i, "audit_notes"] = "evidence_genes unavailable for referenced term_ids"
            _enforce_reason_vocab(out, i)
            continue

        if computed_gsh.lower() != gsh_norm:
            out.at[i, "status"] = "FAIL"
            out.at[i, "link_ok"] = False
            out.at[i, "fail_reason"] = FAIL_EVIDENCE_DRIFT
            out.at[i, "audit_notes"] = (
                f"gene_set_hash mismatch: {computed_gsh.lower()} != {gsh_norm}"
            )
            _enforce_reason_vocab(out, i)
            continue

        out.at[i, "gene_set_hash_effective"] = gsh_norm

        if gene_ids:
            n_hit = sum(1 for g in gene_ids if g in ev_seen)
            if n_hit < int(min_overlap):
                out.at[i, "status"] = "FAIL"
                out.at[i, "link_ok"] = False
                out.at[i, "fail_reason"] = FAIL_EVIDENCE_DRIFT
                out.at[i, "audit_notes"] = (
                    f"gene_ids drift: hits={n_hit} < min_overlap={min_overlap}"
                )
                _enforce_reason_vocab(out, i)
                continue

        evk = (
            str(out.at[i, "gene_set_hash_effective"]).strip()
            or str(out.at[i, "term_ids_set_hash"]).strip()
        )
        out.at[i, "evidence_key"] = evk

        # 2) stability gate
        if (not has_survival) or (term_surv is None):
            out.at[i, "status"] = "ABSTAIN"
            out.at[i, "stability_ok"] = False
            out.at[i, "abstain_reason"] = ABSTAIN_MISSING_SURVIVAL
            out.at[i, "audit_notes"] = _append_note(
                out.at[i, "audit_notes"], "term_survival column missing"
            )
            _enforce_reason_vocab(out, i)
            continue

        agg = float("nan")
        scope = "term"

        if module_surv is not None and module_id and module_id in module_surv.index:
            try:
                agg = float(module_surv.get(module_id, float("nan")))
                scope = "module"
            except Exception:
                agg = float("nan")
                scope = "term"

        if pd.isna(agg):
            vals: list[float] = []
            for t in term_ids:
                v = term_surv.get(t, float("nan"))
                try:
                    vals.append(float(v))
                except Exception:
                    vals.append(float("nan"))
            agg = float(pd.Series(vals).min(skipna=True)) if vals else float("nan")
            scope = "term"

        out.at[i, "term_survival_agg"] = agg
        out.at[i, "stability_scope"] = scope

        if pd.isna(agg):
            out.at[i, "status"] = "ABSTAIN"
            out.at[i, "stability_ok"] = False
            out.at[i, "abstain_reason"] = ABSTAIN_MISSING_SURVIVAL
            out.at[i, "audit_notes"] = _append_note(
                out.at[i, "audit_notes"], "term_survival missing for referenced terms/module"
            )
            _enforce_reason_vocab(out, i)
            continue

        if agg < float(tau_row):
            msg = f"survival[{scope}]={agg:.3f} < tau={float(tau_row):.2f}"

            if stability_gate_mode == "off":
                out.at[i, "stability_ok"] = True
                out.at[i, "audit_notes"] = _append_note(
                    out.at[i, "audit_notes"], f"unstable(off): {msg}"
                )
                _enforce_reason_vocab(out, i)
            elif stability_gate_mode == "note":
                out.at[i, "stability_ok"] = True
                out.at[i, "audit_notes"] = _append_note(
                    out.at[i, "audit_notes"], f"unstable(note): {msg}"
                )
                _enforce_reason_vocab(out, i)
            else:
                out.at[i, "status"] = "ABSTAIN"
                out.at[i, "stability_ok"] = False
                out.at[i, "abstain_reason"] = ABSTAIN_UNSTABLE
                out.at[i, "audit_notes"] = _append_note(out.at[i, "audit_notes"], msg)
                _enforce_reason_vocab(out, i)
                continue

        # 3) optional external contradiction
        _apply_external_contradiction(out, i, row)
        if str(out.at[i, "status"]).strip().upper() in {"FAIL", "ABSTAIN"}:
            _enforce_reason_vocab(out, i)
            continue

        # 4) deterministic abstain rules
        if len(ev_union) < int(min_union):
            out.at[i, "status"] = "ABSTAIN"
            out.at[i, "abstain_reason"] = ABSTAIN_UNDER_SUPPORTED
            msg = (
                f"under_supported: |union_evidence_genes|={len(ev_union)} "
                f"< min_union_genes={int(min_union)}"
            )
            out.at[i, "audit_notes"] = _append_note(out.at[i, "audit_notes"], msg)
            _enforce_reason_vocab(out, i)
            continue

        if ev_union and hub_genes:
            hub_hits = [g for g in ev_union if g in hub_genes]
            frac = (len(hub_hits) / len(ev_union)) if ev_union else 0.0
            if frac >= float(hub_frac_thr):
                out.at[i, "status"] = "ABSTAIN"
                out.at[i, "abstain_reason"] = ABSTAIN_HUB_BRIDGE
                out.at[i, "audit_notes"] = _append_note(
                    out.at[i, "audit_notes"],
                    f"hub_bridge: hub_frac={frac:.2f} (n_hub={len(hub_hits)}/{len(ev_union)})",
                )
                _enforce_reason_vocab(out, i)
                continue

        if "context_score" in out.columns:
            cs_raw = row.get("context_score")
            try:
                cs = None if _is_na_scalar(cs_raw) else float(cs_raw)
            except Exception:
                cs = None

            if cs is not None and cs == 0.0:
                if context_gate_mode == "off":
                    out.at[i, "audit_notes"] = _append_note(
                        out.at[i, "audit_notes"], "context_score=0 (context gate off)"
                    )
                elif context_gate_mode == "note":
                    out.at[i, "audit_notes"] = _append_note(
                        out.at[i, "audit_notes"], "context_nonspecific: context_score=0"
                    )
                else:
                    out.at[i, "status"] = "ABSTAIN"
                    out.at[i, "abstain_reason"] = ABSTAIN_CONTEXT_NONSPECIFIC
                    out.at[i, "audit_notes"] = _append_note(
                        out.at[i, "audit_notes"], "context_nonspecific: context_score=0"
                    )
                    _enforce_reason_vocab(out, i)
                    continue

        # 5) stress gate (single place)
        if not has_stress_any:
            out.at[i, "stress_evaluated"] = False

            if stress_gate_mode == "off":
                out.at[i, "stress_ok"] = pd.NA

            elif stress_gate_mode == "note":
                out.at[i, "stress_ok"] = pd.NA
                out.at[i, "audit_notes"] = _append_note(
                    out.at[i, "audit_notes"], "stress_skipped(note_mode)"
                )

            else:
                out.at[i, "status"] = "ABSTAIN"
                out.at[i, "stress_ok"] = False
                out.at[i, "abstain_reason"] = ABSTAIN_INCONCLUSIVE_STRESS
                out.at[i, "audit_notes"] = _append_note(
                    out.at[i, "audit_notes"], "stress_missing(hard_mode)"
                )
                _enforce_reason_vocab(out, i)
                continue
        else:
            out.at[i, "stress_evaluated"] = True
            _apply_external_stress(
                out, i, row, external_cols=external_cols, gate_mode=stress_gate_mode
            )
            if str(out.at[i, "status"]).strip().upper() in {"FAIL", "ABSTAIN"}:
                _enforce_reason_vocab(out, i)
                continue

        if pass_note_enabled and (not str(out.at[i, "audit_notes"]).strip()):
            out.at[i, "audit_notes"] = "ok"

        _enforce_reason_vocab(out, i)

    _apply_intra_run_contradiction(out)
    return out
