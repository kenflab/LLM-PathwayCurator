# LLM-PathwayCurator/src/llm_pathway_curator/select.py
from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from .claim_schema import Claim, EvidenceRef
from .sample_card import SampleCard

_ALLOWED_DIRECTIONS = {"up", "down", "na"}
_NA_TOKENS = {"na", "nan", "none", "", "NA"}


def _is_na_scalar(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, (list, tuple, set, dict)):
        return False
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def _make_id(s: str, *, n: int = 12) -> str:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return h[:n]


def _dedup_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        x = str(x).strip()
        if not x:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _as_gene_list(x: Any) -> list[str]:
    if _is_na_scalar(x):
        return []
    if isinstance(x, (list, tuple, set)):
        genes = [str(g).strip() for g in x if str(g).strip()]
    else:
        s = str(x).strip().replace(";", ",").replace("|", ",")
        if not s or s.lower() in {t.lower() for t in _NA_TOKENS}:
            return []
        genes = [g.strip() for g in s.split(",") if g.strip()]
    return _dedup_preserve_order(genes)


def _hash_gene_set_audit(genes: list[str]) -> str:
    """
    Audit-grade fingerprint should be SET-stable:
      - same gene set -> same hash, regardless of order
    """
    payload = ",".join(sorted({str(g).strip() for g in genes if str(g).strip()}))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _norm_direction(x: Any) -> str:
    if _is_na_scalar(x):
        return "na"
    s = str(x).strip().lower()
    if s in _ALLOWED_DIRECTIONS:
        return s
    if s in {"upregulated", "increase", "increased", "activated", "pos", "positive", "+", "1"}:
        return "up"
    if s in {"downregulated", "decrease", "decreased", "suppressed", "neg", "negative", "-", "-1"}:
        return "down"
    if s in {t.lower() for t in _NA_TOKENS}:
        return "na"
    return "na"


def _context_tokens(card: SampleCard) -> list[str]:
    toks: list[str] = []
    for v in [
        getattr(card, "disease", None),
        getattr(card, "tissue", None),
        getattr(card, "perturbation", None),
        getattr(card, "comparison", None),
    ]:
        if v is None:
            continue
        s = str(v).strip().lower()
        if not s or s in {t.lower() for t in _NA_TOKENS}:
            continue
        if len(s) < 3:
            continue
        toks.append(s)
    return toks


def _context_score(term_name: str, toks: list[str]) -> int:
    name = str(term_name).lower()
    return sum(1 for t in toks if t and t in name)


def _context_keys(card: SampleCard) -> list[str]:
    keys: list[str] = []
    for k in ["disease", "tissue", "perturbation", "comparison"]:
        v = getattr(card, k, None)
        if v is None:
            continue
        s = str(v).strip().lower()
        if not s or s in {t.lower() for t in _NA_TOKENS}:
            continue
        keys.append(k)
    return keys


def _get_card_param(card: SampleCard, key: str, default: Any) -> Any:
    if hasattr(card, key):
        v = getattr(card, key)
        if v is not None:
            return v
    if hasattr(card, "model_dump"):
        try:
            dd = card.model_dump()
            if dd.get(key) is not None:
                return dd[key]
            if isinstance(dd.get("select", None), dict) and dd["select"].get(key) is not None:
                return dd["select"][key]
        except Exception:
            pass
    return default


def select_claims(distilled: pd.DataFrame, card: SampleCard, *, k: int = 3) -> pd.DataFrame:
    """
    Deterministic selection (v1-minimal, audited-input aware):
      - eligibility gate (keep_term and optional tau on term_survival)
      - then term_survival desc (if present)
      - then context proxy score desc
      - then stat desc
      - then term_uid asc

    Output: flat table for downstream audit/report + embedded claim_json
    """
    required = {"term_id", "term_name", "source", "stat", "direction", "evidence_genes"}
    missing = sorted(required - set(distilled.columns))
    if missing:
        raise ValueError(f"select_claims: missing columns in distilled: {missing}")

    toks = _context_tokens(card)

    df = distilled.copy()

    df["term_id"] = df["term_id"].astype(str).str.strip()
    df["term_name"] = df["term_name"].astype(str).str.strip()
    df["source"] = df["source"].astype(str).str.strip()

    df["stat"] = pd.to_numeric(df["stat"], errors="coerce")
    if df["stat"].isna().any():
        i = int(df.index[df["stat"].isna()][0])
        raise ValueError(f"select_claims: non-numeric stat at row index={i}")

    df["term_uid"] = (
        df["term_uid"].astype(str).str.strip()
        if "term_uid" in df.columns
        else (df["source"] + ":" + df["term_id"])
    )

    df["context_score"] = df["term_name"].map(lambda s: _context_score(str(s), toks))

    # ---- Evidence hygiene gates ----
    if "keep_term" in df.columns:
        df["keep_term"] = df["keep_term"].fillna(True).astype(bool)
    else:
        df["keep_term"] = True

    # optional tau gate from card (distill may set keep_term already; this is extra safety)
    tau = _get_card_param(card, "tau", None)
    if tau is not None and "term_survival" in df.columns:
        try:
            tau_f = float(tau)
            df["term_survival"] = pd.to_numeric(df["term_survival"], errors="coerce")
            df["eligible_tau"] = df["term_survival"].ge(tau_f)
        except Exception:
            df["eligible_tau"] = True
    else:
        df["eligible_tau"] = True

    # prefer keep_term & eligible_tau, but do not hard-drop rows (keep deterministic artifacts)
    df["eligible"] = (df["keep_term"]) & (df["eligible_tau"])

    # survival sort key (missing -> -inf)
    if "term_survival" in df.columns:
        df["term_survival"] = pd.to_numeric(df["term_survival"], errors="coerce")
        df["term_survival_sort"] = df["term_survival"].fillna(-1.0)
    else:
        df["term_survival_sort"] = -1.0

    # allow overriding k from card (optional)
    k_card = _get_card_param(card, "k_claims", None)
    if k_card is not None:
        try:
            k = int(k_card)
        except Exception:
            pass

    df = (
        df.sort_values(
            ["eligible", "term_survival_sort", "context_score", "stat", "term_uid"],
            ascending=[False, False, False, False, True],
        )
        .head(int(k))
        .copy()
    )

    rows: list[dict[str, Any]] = []
    has_module = "module_id" in df.columns

    ctx_keys = _context_keys(card)
    ctx_vals = [
        str(getattr(card, "disease", "")) or "",
        str(getattr(card, "tissue", "")) or "",
        str(getattr(card, "perturbation", "")) or "",
        str(getattr(card, "comparison", "")) or "",
    ]

    for _, r in df.iterrows():
        term_id = str(r["term_id"]).strip()
        term_name = str(r["term_name"]).strip()
        direction = _norm_direction(r.get("direction", "na"))
        genes = _as_gene_list(r.get("evidence_genes"))
        source = str(r["source"]).strip()
        term_uid = str(r.get("term_uid") or f"{source}:{term_id}").strip()

        module_id = ""
        module_reason = ""
        if has_module and (not _is_na_scalar(r.get("module_id"))):
            module_id = str(r.get("module_id")).strip()

        if (not module_id) or (module_id.lower() in {t.lower() for t in _NA_TOKENS}):
            module_id = f"M_fallback_{_make_id(term_uid)}"
            module_reason = "missing_module_id"

        ctx_key = "|".join([term_uid] + ctx_vals)

        claim = Claim(
            claim_id=f"c_{_make_id(ctx_key)}",
            entity=term_name,
            direction=direction,
            context_keys=ctx_keys,
            evidence_ref=EvidenceRef(
                module_id=module_id,
                gene_ids=genes[:10],
                term_ids=[term_uid],
            ),
        )

        rows.append(
            {
                "claim_id": claim.claim_id,
                "entity": claim.entity,
                "direction": claim.direction,
                "context_keys": ",".join(claim.context_keys),
                "term_uid": term_uid,
                "source": source,
                "term_name": term_name,
                "module_id": claim.evidence_ref.module_id,
                "module_reason": module_reason,  # NEW: explain fallback
                "gene_ids": ",".join(claim.evidence_ref.gene_ids),
                "term_ids": ",".join(claim.evidence_ref.term_ids),
                "gene_set_hash": _hash_gene_set_audit(genes),  # CHANGED: set-stable
                "context_score": int(r.get("context_score", 0)),
                "eligible": bool(r.get("eligible", True)),
                "term_survival": r.get("term_survival", pd.NA),
                "keep_term": bool(r.get("keep_term", True)),
                "keep_reason": str(r.get("keep_reason", "ok")),
                "claim_json": claim.model_dump_json(),
            }
        )

    return pd.DataFrame(rows)
