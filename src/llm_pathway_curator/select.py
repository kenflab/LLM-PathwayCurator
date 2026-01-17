# LLM-PathwayCurator/src/llm_pathway_curator/select.py
from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from .claim_schema import Claim, EvidenceRef
from .sample_card import SampleCard

# Canonical claim direction (auditable)
_ALLOWED_DIRECTIONS = {"up", "down", "na"}
_NA_TOKENS = {"na", "nan", "none", "", "NA"}


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


def _make_id(s: str, *, n: int = 12) -> str:
    # Deterministic, low-collision ID for reproducible artifacts
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
    """
    Canonical gene list:
      - robust to list-like and strings
      - de-dup preserving order (do NOT sort; sorting erases evidence order)
    """
    if _is_na_scalar(x):
        return []
    if isinstance(x, (list, tuple, set)):
        genes = [str(g).strip() for g in x if str(g).strip()]
    else:
        s = str(x).strip().replace(";", ",").replace("|", ",")
        if not s or s.lower() in _NA_TOKENS:
            return []
        genes = [g.strip() for g in s.split(",") if g.strip()]

    return _dedup_preserve_order(genes)


def _hash_gene_set(genes: list[str]) -> str:
    """
    Stable short hash for evidence genes (v0).
    IMPORTANT:
      - hash full evidence gene list (not top10)
      - keep order produced by _dedup_preserve_order for determinism
    """
    payload = ",".join([str(g).strip() for g in genes if str(g).strip()])
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
    if s in _NA_TOKENS:
        return "na"
    return "na"


def _context_tokens(card: SampleCard) -> list[str]:
    """
    Deterministic tokens for v0 "context-conditioned" ordering.
    Drop NA-like values and very short tokens (avoid substring false positives).
    """
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
        if not s or s in _NA_TOKENS:
            continue
        # keep tokens that are not too short (reduce accidental matches)
        if len(s) < 3:
            continue
        toks.append(s)
    return toks


def _context_score(term_name: str, toks: list[str]) -> int:
    """
    Cheap v0 proxy for context alignment.
    Use substring match but require token length >= 3 (handled upstream).
    """
    name = str(term_name).lower()
    return sum(1 for t in toks if t and t in name)


def _context_keys(card: SampleCard) -> list[str]:
    """
    Report which context keys are actually specified (exclude NA).
    """
    keys: list[str] = []
    for k in ["disease", "tissue", "perturbation", "comparison"]:
        v = getattr(card, k, None)
        if v is None:
            continue
        s = str(v).strip().lower()
        if not s or s in _NA_TOKENS:
            continue
        keys.append(k)
    return keys


def select_claims(distilled: pd.DataFrame, card: SampleCard, *, k: int = 3) -> pd.DataFrame:
    """
    Deterministic v0 selection:
      - context proxy score (string match) desc
      - then stat desc
      - then term_uid asc

    Output: flat table for downstream audit/report + embedded claim_json

    NOTE:
      - v0 chooses one term_uid per claim (term_ids=[term_uid]).
      - gene_ids are representative (top10) but gene_set_hash fingerprints full evidence genes.
    """
    required = {"term_id", "term_name", "source", "stat", "direction", "evidence_genes"}
    missing = sorted(required - set(distilled.columns))
    if missing:
        raise ValueError(f"select_claims: missing columns in distilled: {missing}")

    has_term_uid = "term_uid" in distilled.columns
    toks = _context_tokens(card)

    df = distilled.copy()
    df["term_id"] = df["term_id"].astype(str).str.strip()
    df["term_name"] = df["term_name"].astype(str).str.strip()
    df["stat"] = pd.to_numeric(df["stat"], errors="coerce")
    if df["stat"].isna().any():
        i = int(df.index[df["stat"].isna()][0])
        raise ValueError(f"select_claims: non-numeric stat at row index={i}")

    df["context_score"] = df["term_name"].map(lambda s: _context_score(str(s), toks))

    # tie-break key (stable join key)
    df["term_uid"] = (
        df["term_uid"].astype(str).str.strip()
        if has_term_uid
        else (df["source"].astype(str).str.strip() + ":" + df["term_id"].astype(str).str.strip())
    )

    df = (
        df.sort_values(["context_score", "stat", "term_uid"], ascending=[False, False, True])
        .head(int(k))
        .copy()
    )

    rows: list[dict[str, Any]] = []
    has_module = "module_id" in df.columns  # use df (post-copy), not distilled

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
        if has_module and (not _is_na_scalar(r.get("module_id"))):
            module_id = str(r.get("module_id")).strip()

        if (not module_id) or (module_id in _NA_TOKENS) or (module_id.lower() in _NA_TOKENS):
            module_id = f"M_fallback_{_make_id(term_uid)}"

        # stable claim_id key (MUST use term_uid, not term_id)
        ctx_key = "|".join([term_uid] + ctx_vals)

        claim = Claim(
            claim_id=f"c_{_make_id(ctx_key)}",
            entity=term_name,  # v0: term; later: module-typed entity
            direction=direction,
            context_keys=ctx_keys,  # exclude NA
            evidence_ref=EvidenceRef(
                module_id=module_id,
                gene_ids=genes[:10],  # representative only
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
                "gene_ids": ",".join(claim.evidence_ref.gene_ids),
                "term_ids": ",".join(claim.evidence_ref.term_ids),
                "gene_set_hash": _hash_gene_set(genes),  # NEW: audit-grade fingerprint
                "context_score": int(r.get("context_score", 0)),
                "claim_json": claim.model_dump_json(),
            }
        )

    return pd.DataFrame(rows)
