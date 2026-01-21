# LLM-PathwayCurator/src/llm_pathway_curator/select.py
from __future__ import annotations

import hashlib
import os
from typing import Any

import pandas as pd

from .backends import BaseLLMBackend
from .claim_schema import Claim, EvidenceRef
from .llm_claims import claims_to_proposed_tsv, propose_claims_llm
from .sample_card import SampleCard

_ALLOWED_DIRECTIONS = {"up", "down", "na"}
_NA_TOKENS = {"na", "nan", "none", "", "NA"}


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


def _make_id(s: str, *, n: int = 12) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


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


def _norm_gene_id(g: str) -> str:
    # align with audit.py to avoid spurious drift
    return str(g).strip().upper()


def _hash_gene_set_audit(genes: list[str]) -> str:
    """
    Audit-grade fingerprint should be SET-stable:
      - same gene set -> same hash, regardless of order
      - normalize IDs to reduce casing drift (align with audit.py)
    """
    payload = ",".join(sorted({_norm_gene_id(g) for g in genes if str(g).strip()}))
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
    for v in [card.disease, card.tissue, card.perturbation, card.comparison]:
        s = str(v).strip().lower()
        if not s or s in {t.lower() for t in _NA_TOKENS}:
            continue
        if len(s) < 3:
            continue
        toks.append(s)
    return toks


def _context_score(term_name: str, toks: list[str]) -> int:
    # v1 proxy only; must be explicitly enabled by SampleCard knob
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


def _resolve_mode(card: SampleCard, mode: str | None) -> str:
    # Priority: explicit arg > env > SampleCard getter > default
    if mode is not None:
        s = str(mode).strip().lower()
        return s if s in {"deterministic", "llm"} else "deterministic"

    env = str(os.environ.get("LLMPATH_CLAIM_MODE", "")).strip().lower()
    if env:
        return env if env in {"deterministic", "llm"} else "deterministic"

    try:
        return card.claim_mode(default="deterministic")
    except Exception:
        return "deterministic"


def _resolve_k(card: SampleCard, k_default: int) -> int:
    # Priority: env > SampleCard getter(default=k_default)
    env = str(os.environ.get("LLMPATH_K_CLAIMS", "")).strip()
    if env:
        try:
            return max(1, int(env))
        except Exception:
            pass
    try:
        return int(card.k_claims(default=int(k_default)))
    except Exception:
        return max(1, int(k_default))


def _select_claims_deterministic(
    distilled: pd.DataFrame, card: SampleCard, *, k: int = 3
) -> pd.DataFrame:
    required = {"term_id", "term_name", "source", "stat", "direction", "evidence_genes"}
    missing = sorted(required - set(distilled.columns))
    if missing:
        raise ValueError(f"select_claims: missing columns in distilled: {missing}")

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

    # ---- context proxy: default OFF (must be enabled explicitly) ----
    try:
        enable_ctx_proxy = bool(card.enable_context_score_proxy(default=False))
    except Exception:
        enable_ctx_proxy = False

    if enable_ctx_proxy:
        toks = _context_tokens(card)
        df["context_score"] = df["term_name"].map(lambda s: _context_score(str(s), toks))
    else:
        df["context_score"] = 0

    # ---- Evidence hygiene gates ----
    if "keep_term" in df.columns:
        df["keep_term"] = df["keep_term"].fillna(True).astype(bool)
    else:
        df["keep_term"] = True

    try:
        preselect_tau_gate = bool(card.preselect_tau_gate(default=False))
    except Exception:
        preselect_tau_gate = False

    try:
        tau_f = float(card.audit_tau(default=0.8))
    except Exception:
        tau_f = 0.8

    if "term_survival" in df.columns:
        df["term_survival"] = pd.to_numeric(df["term_survival"], errors="coerce")
    else:
        df["term_survival"] = pd.NA

    if preselect_tau_gate:
        df["eligible_tau"] = df["term_survival"].ge(float(tau_f))
        df["eligible"] = (df["keep_term"]) & (df["eligible_tau"])
    else:
        df["eligible_tau"] = True
        df["eligible"] = df["keep_term"]

    # survival sort key (missing -> -inf)
    if "term_survival" in df.columns:
        df["term_survival_sort"] = df["term_survival"].fillna(-1.0)
    else:
        df["term_survival_sort"] = -1.0

    # module diversity default: 1 per module (SampleCard getter)
    try:
        max_per_module = int(card.max_per_module(default=1))
    except Exception:
        max_per_module = 1
    max_per_module = max(1, max_per_module)

    # ---- rank all candidates first ----
    df_ranked = df.sort_values(
        ["eligible", "term_survival_sort", "stat", "context_score", "term_uid"],
        ascending=[False, False, False, False, True],
    ).copy()

    has_module = "module_id" in df_ranked.columns

    # ---- pick with module diversity (deterministic scan) ----
    picked_idx: list[int] = []
    per_module_count: dict[str, int] = {}

    for idx, r in df_ranked.iterrows():
        if len(picked_idx) >= int(k):
            break

        mid = ""
        if has_module and (not _is_na_scalar(r.get("module_id"))):
            mid = str(r.get("module_id")).strip()
        if (not mid) or (mid.lower() in {t.lower() for t in _NA_TOKENS}):
            mid = f"M_missing::{str(r.get('term_uid'))}"

        c = per_module_count.get(mid, 0)
        if c >= max_per_module:
            continue

        per_module_count[mid] = c + 1
        picked_idx.append(idx)

    df_pick = df_ranked.loc[picked_idx].copy()

    rows: list[dict[str, Any]] = []
    ctx_keys = _context_keys(card)
    ctx_vals = [
        str(card.disease or ""),
        str(card.tissue or ""),
        str(card.perturbation or ""),
        str(card.comparison or ""),
    ]

    for _, r in df_pick.iterrows():
        term_id = str(r["term_id"]).strip()
        term_name = str(r["term_name"]).strip()
        direction = _norm_direction(r.get("direction", "na"))
        genes_full = [_norm_gene_id(g) for g in _as_gene_list(r.get("evidence_genes"))]
        source = str(r["source"]).strip()
        term_uid = str(r.get("term_uid") or f"{source}:{term_id}").strip()

        module_id = ""
        module_reason = ""
        if has_module and (not _is_na_scalar(r.get("module_id"))):
            module_id = str(r.get("module_id")).strip()

        if (not module_id) or (module_id.lower() in {t.lower() for t in _NA_TOKENS}):
            module_id = f"M_fallback_{_make_id(term_uid)}"
            module_reason = "missing_module_id"

        gene_set_hash = _hash_gene_set_audit(genes_full)
        ctx_key = "|".join([term_uid] + ctx_vals)

        claim = Claim(
            claim_id=f"c_{_make_id(ctx_key)}",
            entity=term_id,
            direction=direction,
            context_keys=ctx_keys,
            evidence_ref=EvidenceRef(
                module_id=module_id,
                gene_ids=genes_full[:10],  # compact ref (top10) for readability
                term_ids=[term_uid],
                gene_set_hash=gene_set_hash,
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
                "term_id": term_id,
                "term_name": term_name,
                "module_id": claim.evidence_ref.module_id,
                "module_reason": module_reason,
                "gene_ids": ",".join(claim.evidence_ref.gene_ids),
                "term_ids": ",".join(claim.evidence_ref.term_ids),
                "gene_set_hash": gene_set_hash,
                "context_score": int(r.get("context_score", 0)),
                "eligible": bool(r.get("eligible", True)),
                "term_survival": r.get("term_survival", pd.NA),
                "keep_term": bool(r.get("keep_term", True)),
                "keep_reason": str(r.get("keep_reason", "ok")),
                "claim_json": claim.model_dump_json(),
                "preselect_tau_gate": bool(preselect_tau_gate),
                "context_score_proxy": bool(enable_ctx_proxy),
            }
        )

    return pd.DataFrame(rows)


def select_claims(
    distilled: pd.DataFrame,
    card: SampleCard,
    *,
    k: int = 3,
    mode: str | None = None,
    backend: BaseLLMBackend | None = None,
    seed: int | None = None,
    outdir: str | None = None,
) -> pd.DataFrame:
    """
    C1: Claim proposal (schema-locked). The mechanical decider is audit_claims().

    - mode="deterministic": stable ranking + module diversity gate, emits Claim JSON.
    - mode="llm": LLM selects from top candidates but is post-validated against
      candidates (term_uid/entity/gene_set_hash) and MUST emit JSON; otherwise
      we fall back to deterministic output.

    Contract:
      - knobs are read ONLY via SampleCard getters (plus env override).
      - env has priority over card; card has priority over function defaults.
    """
    mode_eff = _resolve_mode(card, mode)
    k_eff = _resolve_k(card, k_default=int(k))

    if mode_eff == "llm":
        if backend is None:
            mode_eff = "deterministic"
        else:
            res = propose_claims_llm(
                distilled_with_modules=distilled,
                card=card,
                backend=backend,
                k=int(k_eff),
                seed=seed,
                outdir=outdir,
            )

            if (not res.used_fallback) and res.claims:
                out_llm = claims_to_proposed_tsv(
                    claims=res.claims,
                    distilled_with_modules=distilled,
                    card=card,
                )
                out_llm["claim_mode"] = "llm"
                out_llm["llm_notes"] = res.notes
                return out_llm

            out_det = _select_claims_deterministic(distilled, card, k=int(k_eff))
            out_det["claim_mode"] = "deterministic_fallback"
            out_det["llm_notes"] = res.notes
            return out_det

    out_det = _select_claims_deterministic(distilled, card, k=int(k_eff))
    out_det["claim_mode"] = "deterministic"
    out_det["llm_notes"] = ""
    return out_det
