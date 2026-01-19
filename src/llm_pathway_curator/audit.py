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
        return bool(pd.isna(x))
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
    """v1 contract uses 12-hex short hash."""
    if _is_na_scalar(x):
        return False
    s = str(x).strip().lower()
    if len(s) != 12:
        return False
    return all(ch in "0123456789abcdef" for ch in s)


def _get_tau_default(card: SampleCard) -> float:
    # Source of truth: SampleCard.audit_tau()
    try:
        return float(card.audit_tau())
    except Exception:
        # v1 default (conservative)
        return 0.8


def _get_min_overlap_default(card: SampleCard) -> int:
    try:
        return int(card.audit_min_gene_overlap())
    except Exception:
        return 1


def _norm_status(x: Any) -> str:
    if _is_na_scalar(x):
        return ""
    return str(x).strip().upper()


def _append_note(old: str, msg: str) -> str:
    old = "" if _is_na_scalar(old) else str(old)
    return msg if not old else f"{old} | {msg}"


def _get_first_present(row: pd.Series, keys: list[str]) -> Any:
    for k in keys:
        if k in row.index:
            v = row.get(k)
            if v is None or _is_na_scalar(v):
                continue
            if isinstance(v, str) and (
                v.strip() == "" or v.strip().lower() in {t.lower() for t in _NA_TOKENS}
            ):
                continue
            return v
    return None


def _extract_evidence_from_claim_json(cj: str) -> tuple[list[str], list[str], Any]:
    """
    Source of truth: claim_json (Claim schema).
    Returns: (term_ids, gene_ids, gene_set_hash)
    """
    try:
        obj = json.loads(cj)
    except Exception:
        return ([], [], None)

    ev = None
    if isinstance(obj, dict):
        ev = (
            obj.get("evidence_refs")
            or obj.get("evidence_ref")
            or (obj.get("claim", {}) if isinstance(obj.get("claim"), dict) else {}).get(
                "evidence_refs"
            )
            or (obj.get("claim", {}) if isinstance(obj.get("claim"), dict) else {}).get(
                "evidence_ref"
            )
        )

    if not isinstance(ev, dict):
        return ([], [], None)

    term_ids = _parse_ids(ev.get("term_ids") or ev.get("term_id") or "")
    gene_ids = _parse_ids(ev.get("gene_ids") or ev.get("gene_id") or "")
    gsh = ev.get("gene_set_hash")
    return (term_ids, gene_ids, gsh)


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


def _apply_external_contradiction(out: pd.DataFrame, i: int, row: pd.Series) -> None:
    if "contradiction_status" not in out.columns:
        return

    st = _norm_status(row.get("contradiction_status"))
    if not st:
        out.at[i, "contradiction_ok"] = False
        out.at[i, "audit_notes"] = _append_note(
            out.at[i, "audit_notes"], "contradiction_status missing"
        )
        return

    if st == "PASS":
        out.at[i, "contradiction_ok"] = True
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
    out: pd.DataFrame, i: int, row: pd.Series, *, external_cols: set[str]
) -> None:
    """
    Apply stress results ONLY if they were provided as inputs in the original claims DF.
    Important: do not treat internally initialized columns (e.g., out['stress_ok']) as evidence
    that stress was evaluated.
    """
    has_any = any(
        c in external_cols for c in ["stress_status", "stress_ok", "stress_reason", "stress_notes"]
    )
    if not has_any:
        # v1 policy: do NOT auto-ABSTAIN just because stress is absent.
        out.at[i, "stress_ok"] = pd.NA
        return

    # Accept either a boolean stress_ok input or a categorical stress_status input.
    if ("stress_ok" in external_cols) and ("stress_status" not in external_cols):
        v = row.get("stress_ok")
        if _is_na_scalar(v):
            out.at[i, "status"] = "ABSTAIN"
            out.at[i, "stress_ok"] = False
            out.at[i, "abstain_reason"] = ABSTAIN_INCONCLUSIVE_STRESS
            out.at[i, "audit_notes"] = _append_note(out.at[i, "audit_notes"], "stress_ok missing")
            return
        ok = bool(v)
        out.at[i, "stress_ok"] = ok
        if not ok:
            out.at[i, "status"] = "ABSTAIN"
            out.at[i, "abstain_reason"] = ABSTAIN_INCONCLUSIVE_STRESS
            out.at[i, "audit_notes"] = _append_note(out.at[i, "audit_notes"], "stress_ok=False")
        return

    st = _norm_status(row.get("stress_status"))
    rs_raw = str(row.get("stress_reason", "")).strip()
    nt = str(row.get("stress_notes", "")).strip()

    if not st:
        out.at[i, "status"] = "ABSTAIN"
        out.at[i, "stress_ok"] = False
        out.at[i, "abstain_reason"] = ABSTAIN_INCONCLUSIVE_STRESS
        out.at[i, "audit_notes"] = _append_note(out.at[i, "audit_notes"], "stress_status missing")
        return

    if st == "PASS":
        out.at[i, "stress_ok"] = True
        out.at[i, "audit_notes"] = _append_note(out.at[i, "audit_notes"], "stress=PASS")
        return

    if st == "ABSTAIN":
        out.at[i, "status"] = "ABSTAIN"
        out.at[i, "stress_ok"] = False
        rs = rs_raw if rs_raw in ABSTAIN_REASONS else ABSTAIN_INCONCLUSIVE_STRESS
        out.at[i, "abstain_reason"] = rs
        note = "stress=ABSTAIN"
        if rs_raw:
            note += f": {rs_raw}"
        if nt:
            note += f" ({nt})"
        out.at[i, "audit_notes"] = _append_note(out.at[i, "audit_notes"], note)
        return

    if st == "FAIL":
        # v1: conservative mapping: stress FAIL -> ABSTAIN unless you add FAIL_* code
        out.at[i, "status"] = "ABSTAIN"
        out.at[i, "stress_ok"] = False
        rs = rs_raw if rs_raw in ABSTAIN_REASONS else ABSTAIN_INCONCLUSIVE_STRESS
        out.at[i, "abstain_reason"] = rs
        note = "stress=FAIL"
        if rs_raw:
            note += f": {rs_raw}"
        if nt:
            note += f" ({nt})"
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
    Any violation => FAIL_SCHEMA_VIOLATION (do not silently coerce).
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

    # Always require a minimal explanation when not PASS
    if st in {"FAIL", "ABSTAIN"}:
        note = (
            "" if _is_na_scalar(out.at[i, "audit_notes"]) else str(out.at[i, "audit_notes"]).strip()
        )
        if not note:
            out.at[i, "audit_notes"] = f"{st}: reason recorded"


def _apply_intra_run_contradiction(out: pd.DataFrame) -> None:
    """
    Intra-run contradiction check (v1):
      - Among PASS claims only
      - If same evidence key has both direction=up and direction=down =>
        FAIL_CONTRADICTION for all involved
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
    Mechanical audit (v1, tool-facing; stress/contradiction are optional inputs).

    Status priority: FAIL > ABSTAIN > PASS

    Source of truth for evidence references:
      1) claim_json (Claim schema)
      2) DataFrame columns as fallback (for backward compatibility)
    """

    # Track which columns were truly provided by the caller (v1 policy).
    external_cols: set[str] = set(claims.columns)

    out = claims.copy()

    # Stable audit fields
    out["status"] = "PASS"
    out["link_ok"] = True
    out["stability_ok"] = True

    # IMPORTANT: "not evaluated" should be NA (not True)
    out["contradiction_ok"] = pd.NA
    out["stress_ok"] = pd.NA

    out["abstain_reason"] = ""
    out["fail_reason"] = ""
    out["audit_notes"] = ""

    # Keep the actual tau used by this row (helps Fig2 debugging)
    if "tau_used" not in out.columns:
        out["tau_used"] = pd.NA

    # Optional numeric feature for reporting/calibration
    if "term_survival_agg" not in out.columns:
        out["term_survival_agg"] = pd.NA

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
    key_col = "term_uid" if "term_uid" in distilled.columns else "term_id"

    term_key = distilled[key_col].astype(str).str.strip()
    term_key = term_key[~term_key.isin(_NA_TOKENS)]
    term_key = term_key[~term_key.str.lower().isin({t.lower() for t in _NA_TOKENS})]
    known_terms = set(term_key.tolist())

    has_survival = "term_survival" in distilled.columns
    if has_survival:
        tmp = distilled.copy()
        tmp["term_key"] = tmp[key_col].astype(str).str.strip()
        tmp["term_survival"] = pd.to_numeric(tmp["term_survival"], errors="coerce")
        surv = tmp.groupby("term_key")["term_survival"].min()
    else:
        surv = None

    # Resolve tau default (tool contract)
    tau_default = _get_tau_default(card)
    if tau is not None:
        try:
            tau_default = float(tau)
        except Exception:
            pass

    min_overlap_default = _get_min_overlap_default(card)

    # Precompute evidence genes per term_key
    term_to_gene_set: dict[str, set[str]] = {}
    if "evidence_genes" in distilled.columns:
        for tk, xs in zip(
            distilled[key_col].astype(str).str.strip(),
            distilled["evidence_genes"],
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

    elif "evidence_genes_str" in distilled.columns:
        for tk, s in zip(
            distilled[key_col].astype(str).str.strip(),
            distilled["evidence_genes_str"].astype(str),
            strict=True,
        ):
            if tk in _NA_TOKENS or tk.lower() in {t.lower() for t in _NA_TOKENS}:
                continue
            genes = [_norm_gene_id(g) for g in _parse_ids(s)]
            gs = set([g for g in genes if g])
            if not gs:
                continue
            term_to_gene_set.setdefault(tk, set()).update(gs)

    # ---- Hub gene heuristic (v1): gene -> #terms it appears in (within this run) ----
    gene_to_term_degree: dict[str, int] = {}
    for _tk, gs in term_to_gene_set.items():
        for g in gs:
            gene_to_term_degree[g] = gene_to_term_degree.get(g, 0) + 1

    hub_thr = 200
    try:
        extra = getattr(card, "extra", {}) or {}
        if isinstance(extra, dict) and extra.get("hub_term_degree") is not None:
            hub_thr = int(extra["hub_term_degree"])
    except Exception:
        pass

    hub_genes = {g for g, deg in gene_to_term_degree.items() if deg > int(hub_thr)}

    # Control PASS note behavior (default True to avoid "empty explanations" warning)
    pass_note_enabled = True
    try:
        extra = getattr(card, "extra", {}) or {}
        if isinstance(extra, dict) and extra.get("pass_notes") is not None:
            pass_note_enabled = bool(extra["pass_notes"])
    except Exception:
        pass

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

        # 1) evidence refs (prefer claim_json)
        term_ids_cj, gene_ids_cj, gsh_cj = _extract_evidence_from_claim_json(cj_str)

        term_ids = term_ids_cj or _parse_ids(
            _get_first_present(
                row, ["term_ids", "evidence_refs.term_ids", "evidence_refs.term_id", "term_id"]
            )
            or ""
        )
        gene_ids = gene_ids_cj or _parse_ids(
            _get_first_present(
                row, ["gene_ids", "evidence_refs.gene_ids", "evidence_refs.gene_id", "gene_id"]
            )
            or ""
        )
        gsh = (
            gsh_cj
            if (gsh_cj is not None and not _is_na_scalar(gsh_cj) and str(gsh_cj).strip())
            else _get_first_present(row, ["gene_set_hash", "evidence_refs.gene_set_hash"])
        )

        gene_ids = [_norm_gene_id(g) for g in gene_ids]

        # 1) evidence-link integrity: term_ids must exist
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

        # compute set-stable gene_set_hash from EvidenceTable if possible
        computed_gsh = ""
        if ev_union:
            computed_gsh = _hash_gene_set(ev_union)

        out.at[i, "term_ids_set_hash"] = _hash_term_ids(term_ids)

        # If user provided gene_set_hash, EvidenceTable MUST allow us to compute/verify it
        if _looks_like_hex_hash(gsh):
            if not ev_union:
                out.at[i, "status"] = "FAIL"
                out.at[i, "link_ok"] = False
                out.at[i, "fail_reason"] = FAIL_SCHEMA_VIOLATION
                out.at[i, "audit_notes"] = "gene_set_hash provided but evidence_genes unavailable"
                _enforce_reason_vocab(out, i)
                continue

            gsh_norm = str(gsh).strip().lower()
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
        else:
            out.at[i, "gene_set_hash_effective"] = computed_gsh

        # Fallback: overlap check if gene_ids available
        if gene_ids:
            if not ev_seen:
                out.at[i, "status"] = "FAIL"
                out.at[i, "link_ok"] = False
                out.at[i, "fail_reason"] = FAIL_SCHEMA_VIOLATION
                out.at[i, "audit_notes"] = "gene_ids provided but evidence_genes unavailable"
                _enforce_reason_vocab(out, i)
                continue
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

        # set evidence_key for intra-run contradiction
        evk = str(out.at[i, "gene_set_hash_effective"]).strip()
        if not evk:
            evk = str(out.at[i, "term_ids_set_hash"]).strip()
        out.at[i, "evidence_key"] = evk

        # 2) stability gate
        if (not has_survival) or (surv is None):
            out.at[i, "status"] = "ABSTAIN"
            out.at[i, "stability_ok"] = False
            out.at[i, "abstain_reason"] = ABSTAIN_MISSING_SURVIVAL
            out.at[i, "audit_notes"] = _append_note(
                out.at[i, "audit_notes"], "term_survival column missing"
            )
            _enforce_reason_vocab(out, i)
            continue

        vals: list[float] = []
        for t in term_ids:
            v = surv.get(t, float("nan"))
            try:
                vals.append(float(v))
            except Exception:
                vals.append(float("nan"))

        agg = float(pd.Series(vals).min(skipna=True)) if vals else float("nan")
        out.at[i, "term_survival_agg"] = agg

        if pd.isna(agg):
            out.at[i, "status"] = "ABSTAIN"
            out.at[i, "stability_ok"] = False
            out.at[i, "abstain_reason"] = ABSTAIN_MISSING_SURVIVAL
            out.at[i, "audit_notes"] = _append_note(
                out.at[i, "audit_notes"], "term_survival missing for referenced terms"
            )
            _enforce_reason_vocab(out, i)
            continue

        if agg < float(tau_row):
            out.at[i, "status"] = "ABSTAIN"
            out.at[i, "stability_ok"] = False
            out.at[i, "abstain_reason"] = ABSTAIN_UNSTABLE
            out.at[i, "audit_notes"] = _append_note(
                out.at[i, "audit_notes"], f"term_survival_agg={agg:.3f} < tau={float(tau_row):.2f}"
            )
            _enforce_reason_vocab(out, i)
            continue

        # 3) optional advanced audits
        _apply_external_contradiction(out, i, row)
        if out.at[i, "status"] == "FAIL":
            _enforce_reason_vocab(out, i)
            continue

        # 4) deterministic abstain rules (v1): under-supported / hub-bridge / context-nonspecific

        # 4a) under_supported: union evidence too small
        min_union = 3
        try:
            extra = getattr(card, "extra", {}) or {}
            if isinstance(extra, dict) and extra.get("min_union_genes") is not None:
                min_union = int(extra["min_union_genes"])
        except Exception:
            pass

        if len(ev_union) < int(min_union):
            out.at[i, "status"] = "ABSTAIN"
            out.at[i, "abstain_reason"] = ABSTAIN_UNDER_SUPPORTED
            out.at[i, "audit_notes"] = _append_note(
                out.at[i, "audit_notes"],
                (
                    f"under_supported: |union_evidence_genes|="
                    f"{len(ev_union)} < min_union_genes={int(min_union)}"
                ),
            )
            _enforce_reason_vocab(out, i)
            continue

        # 4b) hub_bridge: evidence dominated by hub genes (high term-degree)
        hub_frac_thr = 0.5
        try:
            extra = getattr(card, "extra", {}) or {}
            if isinstance(extra, dict) and extra.get("hub_frac_thr") is not None:
                hub_frac_thr = float(extra["hub_frac_thr"])
        except Exception:
            pass

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

        # 4c) context_nonspecific: only if an explicit context_score exists and is 0
        if "context_score" in out.columns:
            cs_raw = row.get("context_score")
            try:
                cs = None if _is_na_scalar(cs_raw) else float(cs_raw)
            except Exception:
                cs = None

            # NEW: allow controlling context gate behavior from SampleCard.extra
            context_gate_mode = "abstain"  # default: backward compatible
            try:
                extra = getattr(card, "extra", {}) or {}
                if isinstance(extra, dict) and extra.get("context_gate_mode") is not None:
                    context_gate_mode = str(extra["context_gate_mode"]).strip().lower()
            except Exception:
                pass

            if cs is not None and cs == 0.0:
                if context_gate_mode in {"off", "disable", "disabled", "none"}:
                    # do nothing
                    out.at[i, "audit_notes"] = _append_note(
                        out.at[i, "audit_notes"], "context_score=0 (context gate disabled)"
                    )
                elif context_gate_mode in {"note", "warn", "warning"}:
                    # NOTE-ONLY: keep tau sweep meaningful for Fig2
                    out.at[i, "audit_notes"] = _append_note(
                        out.at[i, "audit_notes"], "context_nonspecific: context_score=0"
                    )
                else:
                    # default behavior: ABSTAIN
                    out.at[i, "status"] = "ABSTAIN"
                    out.at[i, "abstain_reason"] = ABSTAIN_CONTEXT_NONSPECIFIC
                    out.at[i, "audit_notes"] = _append_note(
                        out.at[i, "audit_notes"], "context_nonspecific: context_score=0"
                    )
                    _enforce_reason_vocab(out, i)
                    continue

        _apply_external_stress(out, i, row, external_cols=external_cols)
        if str(out.at[i, "status"]).strip().upper() in {"FAIL", "ABSTAIN"}:
            _enforce_reason_vocab(out, i)
            continue

        # PASS: keep a minimal explanation if enabled
        if pass_note_enabled:
            # Keep it very short; still gives “why PASS”
            if not str(out.at[i, "audit_notes"]).strip():
                out.at[i, "audit_notes"] = "ok"

        _enforce_reason_vocab(out, i)

        # If PASS and still empty, add a minimal audit breadcrumb (paper-friendly)
        if str(out.at[i, "status"]).strip().upper() == "PASS":
            note = (
                ""
                if _is_na_scalar(out.at[i, "audit_notes"])
                else str(out.at[i, "audit_notes"]).strip()
            )
            if not note:
                out.at[i, "audit_notes"] = "PASS: link+stability ok"

    _apply_intra_run_contradiction(out)
    return out
