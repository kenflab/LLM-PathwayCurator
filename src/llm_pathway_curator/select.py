# LLM-PathwayCurator/src/llm_pathway_curator/select.py
from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np
import pandas as pd

from .backends import BaseLLMBackend
from .claim_schema import Claim, EvidenceRef
from .llm_claims import claims_to_proposed_tsv, propose_claims_llm
from .sample_card import SampleCard

_ALLOWED_DIRECTIONS = {"up", "down", "na"}
_NA_TOKENS = {"na", "nan", "none", "", "NA"}
_NA_TOKENS_L = {t.lower() for t in _NA_TOKENS}


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
    """
    Tolerant parsing for evidence genes (align with schema/distill):
      - list/tuple/set -> preserve order (dedup)
      - string -> split on , ; | (fallback: whitespace if no commas)
      - NA -> []
    """
    if _is_na_scalar(x):
        return []
    if isinstance(x, (list, tuple, set)):
        genes = [str(g).strip() for g in x if str(g).strip()]
        return _dedup_preserve_order(genes)

    s = str(x).strip()
    if not s or s.lower() in _NA_TOKENS_L:
        return []

    s = s.replace(";", ",").replace("|", ",")
    s = s.replace("\n", " ").replace("\t", " ")
    s = " ".join(s.split()).strip()
    if not s or s.lower() in _NA_TOKENS_L:
        return []

    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
    else:
        parts = [p.strip() for p in s.split(" ") if p.strip()]

    return _dedup_preserve_order(parts)


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


def _hash_term_set_fallback(term_uids: list[str]) -> str:
    """
    Fallback when evidence genes are missing/empty.
    Keep it deterministic and set-stable across ordering.
    This is term-driven (not gene-driven).
    """
    canon = sorted({str(t).strip() for t in term_uids if str(t).strip()})
    payload = ",".join(canon)
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
    if s in _NA_TOKENS_L:
        return "na"
    return "na"


def _context_tokens(card: SampleCard) -> list[str]:
    toks: list[str] = []
    for v in [card.disease, card.tissue, card.perturbation, card.comparison]:
        s = str(v).strip().lower()
        if not s or s in _NA_TOKENS_L:
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
        if not s or s in _NA_TOKENS_L:
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


def _validate_k(k: Any) -> int:
    try:
        kk = int(k)
    except Exception as e:
        raise ValueError(f"select_claims: invalid k={k!r} (must be int>=1)") from e
    if kk < 1:
        raise ValueError(f"select_claims: invalid k={k!r} (must be int>=1)")
    return kk


# ===========================
# Stress suite (evidence identity collapse)
# ===========================
def _get_extra(card: SampleCard) -> dict[str, Any]:
    ex = getattr(card, "extra", {}) or {}
    return ex if isinstance(ex, dict) else {}


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


def _as_int(x: Any, default: int) -> int:
    try:
        if x is None:
            return int(default)
        return int(x)
    except Exception:
        return int(default)


def _as_float(x: Any, default: float) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _stress_enabled(card: SampleCard) -> bool:
    # default OFF (paper-aligned). Enable via env or SampleCard.extra.
    env = str(os.environ.get("LLMPATH_STRESS_GENERATE", "")).strip().lower()
    if env:
        return env in {"1", "true", "t", "yes", "y", "on"}
    ex = _get_extra(card)
    return _as_bool(ex.get("stress_generate", None), False)


def _module_prefix(card: SampleCard) -> str:
    """
    Ensure select's module_id fallback matches modules.py prefix behavior.
    Priority: env > card.extra > default "M"
    """
    env = str(os.environ.get("LLMPATH_MODULE_PREFIX", "")).strip()
    if env:
        return env
    ex = _get_extra(card)
    p = str(ex.get("module_prefix", "")).strip()
    return p or "M"


def _seed_for_claim(seed: int | None, claim_id: str) -> int:
    base = 0 if seed is None else int(seed)
    h = hashlib.blake2b(digest_size=8)
    h.update(str(base).encode("utf-8"))
    h.update(b"|")
    h.update(str(claim_id).encode("utf-8"))
    return int.from_bytes(h.digest(), byteorder="little", signed=False)


def _similarity(orig: set[str], pert: set[str]) -> tuple[float, float, float]:
    # (jaccard, recall, precision)
    if not orig and not pert:
        return (1.0, 1.0, 1.0)
    if not orig:
        prec = 0.0 if pert else 1.0
        return (0.0, 1.0, prec)
    inter = len(orig & pert)
    union = len(orig | pert)
    j = float(inter / union) if union else 0.0
    recall = float(inter / len(orig)) if orig else 1.0
    precision = float(inter / len(pert)) if pert else 0.0
    return (j, recall, precision)


def _perturb_gene_set(
    genes: list[str],
    gene_pool: np.ndarray,
    rng: np.random.Generator,
    *,
    p_drop: float,
    p_add: float,
) -> set[str]:
    if not genes:
        return set()

    g_arr = np.array(genes, dtype=object)

    keep_mask = rng.random(len(g_arr)) >= float(p_drop)
    kept = g_arr[keep_mask].tolist()
    out = set([_norm_gene_id(g) for g in kept if str(g).strip()])

    if float(p_add) > 0.0 and gene_pool.size > 0:
        k_add = int(round(float(p_add) * max(1, len(genes))))
        if k_add > 0:
            orig = set([_norm_gene_id(g) for g in genes if str(g).strip()])
            candidates = np.array(
                [g for g in gene_pool.tolist() if _norm_gene_id(g) not in orig], dtype=object
            )
            if candidates.size > 0:
                add = rng.choice(candidates, size=min(k_add, candidates.size), replace=False)
                out.update([_norm_gene_id(g) for g in add.tolist() if str(g).strip()])

    return out


def _module_hash_like_modules_py(terms: list[str], genes: list[str]) -> str:
    """
    Match modules.py: sha256("T:...\\nG:...")[:12]
    IMPORTANT: gene IDs are normalized (upper) to avoid spurious drift.
    """
    t = sorted([str(x).strip() for x in terms if str(x).strip()])
    g = sorted([_norm_gene_id(x) for x in genes if str(x).strip()])
    payload = "T:" + "|".join(t) + "\n" + "G:" + "|".join(g)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _build_term_gene_map(distilled: pd.DataFrame) -> dict[str, set[str]]:
    m: dict[str, set[str]] = {}
    if "term_uid" not in distilled.columns:
        return m
    if "evidence_genes" not in distilled.columns:
        return m

    for tu, xs in zip(
        distilled["term_uid"].astype(str).tolist(),
        distilled["evidence_genes"].tolist(),
        strict=True,
    ):
        key = str(tu).strip()
        if not key or key.lower() in _NA_TOKENS_L:
            continue
        genes = _as_gene_list(xs)
        gs = set([_norm_gene_id(g) for g in genes if str(g).strip()])
        if not gs:
            continue
        m.setdefault(key, set()).update(gs)
    return m


def _build_module_terms_genes(
    distilled: pd.DataFrame, term_to_genes: dict[str, set[str]]
) -> dict[str, tuple[list[str], set[str]]]:
    """
    module_id -> (sorted terms, union genes)
    Only works if distilled has module_id.
    """
    out: dict[str, tuple[list[str], set[str]]] = {}
    if "module_id" not in distilled.columns or "term_uid" not in distilled.columns:
        return out

    mod_to_terms: dict[str, set[str]] = {}
    for tu, mid in zip(
        distilled["term_uid"].astype(str).tolist(),
        distilled["module_id"].astype(str).tolist(),
        strict=True,
    ):
        t = str(tu).strip()
        m0 = str(mid).strip()
        if not t or not m0 or m0.lower() in _NA_TOKENS_L:
            continue
        mod_to_terms.setdefault(m0, set()).add(t)

    for m0, terms_set in mod_to_terms.items():
        terms = sorted(list(terms_set))
        genes_u: set[str] = set()
        for t in terms:
            genes_u |= set(term_to_genes.get(t, set()))
        out[m0] = (terms, genes_u)

    return out


def _evaluate_stress_for_claim(
    *,
    claim_id: str,
    term_ids: list[str],
    module_id: str,
    gene_set_hash: str,
    term_to_genes: dict[str, set[str]],
    module_to_terms_genes: dict[str, tuple[list[str], set[str]]],
    gene_pool: np.ndarray,
    seed: int | None,
    card: SampleCard,
) -> dict[str, Any]:
    """
    Stress suite (identity collapse) for a single claim.

    IMPORTANT:
      - This is a PROBE (measurement), not the mechanical DECIDER.
      - We therefore avoid "FAIL" vocabulary here; audit.py may turn WARN into FAIL/ABSTAIN.

    Emits:
      stress_status: PASS | WARN | ABSTAIN
      stress_ok: bool
      stress_reason: str
      stress_notes: str
    """
    ex = _get_extra(card)

    n = max(8, min(256, _as_int(ex.get("stress_n", None), 64)))
    p_drop = min(max(_as_float(ex.get("stress_gene_dropout_p", None), 0.20), 0.0), 0.95)
    p_add = min(max(_as_float(ex.get("stress_gene_jitter_p", None), 0.10), 0.0), 0.95)

    j_min = min(max(_as_float(ex.get("stress_jaccard_min", None), 0.60), 0.0), 1.0)
    r_min = min(max(_as_float(ex.get("stress_recall_min", None), 0.80), 0.0), 1.0)
    p_min = min(max(_as_float(ex.get("stress_precision_min", None), 0.80), 0.0), 1.0)

    surv_thr = min(max(_as_float(ex.get("stress_survival_thr", None), 0.80), 0.0), 1.0)

    base_genes: set[str] = set()
    for t in term_ids:
        base_genes |= set(term_to_genes.get(str(t).strip(), set()))

    if not base_genes:
        return {
            "stress_status": "ABSTAIN",
            "stress_ok": False,
            "stress_reason": "stress_missing_baseline_genes",
            "stress_notes": "no baseline evidence genes for referenced term_ids",
        }

    base_hash_claim = str(gene_set_hash).strip()
    if not base_hash_claim or base_hash_claim.lower() in _NA_TOKENS_L:
        return {
            "stress_status": "ABSTAIN",
            "stress_ok": False,
            "stress_reason": "stress_missing_gene_set_hash",
            "stress_notes": "claim gene_set_hash missing",
        }

    # Consistency check: claim hash should match baseline genes hash (set-stable)
    base_hash_calc = _hash_gene_set_audit(sorted(list(base_genes)))
    if base_hash_calc != base_hash_claim:
        return {
            "stress_status": "WARN",
            "stress_ok": False,
            "stress_reason": "stress_gene_set_hash_mismatch",
            "stress_notes": f"claim_hash={base_hash_claim} != baseline_hash={base_hash_calc}",
        }

    # Separate RNG streams to reduce unwanted correlation.
    rng_term = np.random.default_rng(_seed_for_claim(seed, claim_id))
    rng_mod = np.random.default_rng(_seed_for_claim(seed, claim_id) + 1)

    ok = 0
    js: list[float] = []
    rs: list[float] = []
    ps: list[float] = []

    base_list = sorted(list(base_genes))
    for _ in range(int(n)):
        pert = _perturb_gene_set(
            base_list, gene_pool=gene_pool, rng=rng_term, p_drop=p_drop, p_add=p_add
        )
        j, rr, pp = _similarity(base_genes, pert)
        js.append(j)
        rs.append(rr)
        ps.append(pp)
        if (j >= j_min) and (rr >= r_min) and (pp >= p_min):
            ok += 1

    surv = ok / float(n) if n > 0 else 0.0

    module_ok = True
    module_note = ""
    if module_id and module_id.lower() not in _NA_TOKENS_L:
        mtg = module_to_terms_genes.get(module_id)
        if mtg is not None:
            terms_m, genes_m = mtg
            if terms_m and genes_m:
                genes_m_list = sorted(list(genes_m))
                pert_m = _perturb_gene_set(
                    genes_m_list,
                    gene_pool=gene_pool,
                    rng=rng_mod,
                    p_drop=p_drop,
                    p_add=p_add,
                )
                h0 = _module_hash_like_modules_py(terms_m, genes_m_list)
                h1 = _module_hash_like_modules_py(terms_m, sorted(list(pert_m)))
                if h0 != h1 and surv < float(surv_thr):
                    module_ok = False
                    module_note = f"module_id_drift_proxy: hash {h0}->{h1}"

    if surv >= float(surv_thr) and module_ok:
        return {
            "stress_status": "PASS",
            "stress_ok": True,
            "stress_reason": "",
            "stress_notes": (
                f"identity_survival={surv:.3f} (n_ok={ok}/{n}); "
                f"mean_j={float(np.mean(js)):.3f} "
                f"mean_r={float(np.mean(rs)):.3f} "
                f"mean_p={float(np.mean(ps)):.3f}"
            ),
        }

    reason = "stress_identity_collapse"
    if not module_ok:
        reason = "stress_module_id_drift"

    note = (
        f"identity_survival={surv:.3f} < thr={float(surv_thr):.2f} "
        f"(n_ok={ok}/{n}); "
        f"mean_j={float(np.mean(js)):.3f} "
        f"mean_r={float(np.mean(rs)):.3f} "
        f"mean_p={float(np.mean(ps)):.3f}"
    )
    if module_note:
        note += f"; {module_note}"

    return {
        "stress_status": "WARN",
        "stress_ok": False,
        "stress_reason": reason,
        "stress_notes": note,
    }


def _select_claims_deterministic(
    distilled: pd.DataFrame, card: SampleCard, *, k: int = 3, seed: int | None = None
) -> pd.DataFrame:
    # direction is NOT required (we can default to "na")
    required = {"term_id", "term_name", "source", "stat", "evidence_genes"}
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

    # Stable term_uid (single source of truth)
    if "term_uid" in df.columns:
        df["term_uid"] = df["term_uid"].astype(str).str.strip()
    else:
        df["term_uid"] = (df["source"] + ":" + df["term_id"]).astype(str).str.strip()

    # ---- context proxy: default OFF (must be enabled explicitly) ----
    try:
        enable_ctx_proxy = bool(card.enable_context_score_proxy(default=False))
    except Exception:
        enable_ctx_proxy = False

    if enable_ctx_proxy:
        toks = _context_tokens(card)
        df["context_score"] = df["term_name"].map(lambda s: _context_score(str(s), toks))
        df["context_evaluated"] = True
    else:
        # IMPORTANT: OFF => NOT evaluated. Use NA (not 0) to avoid hard-gate collapse downstream.
        df["context_score"] = pd.NA
        df["context_evaluated"] = False

    # sorting helper (stable rank even with NA)
    df["context_score_sort"] = pd.to_numeric(df["context_score"], errors="coerce").fillna(-1)

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

    df["term_survival_sort"] = (
        df["term_survival"].fillna(-1.0) if "term_survival" in df.columns else -1.0
    )

    # module diversity default: 1 per module (SampleCard getter)
    try:
        max_per_module = int(card.max_per_module(default=1))
    except Exception:
        max_per_module = 1
    max_per_module = max(1, max_per_module)

    # ---- rank all candidates first ----
    df_ranked = df.sort_values(
        ["eligible", "term_survival_sort", "stat", "context_score_sort", "term_uid"],
        ascending=[False, False, False, False, True],
    ).copy()

    has_module = "module_id" in df_ranked.columns

    # ---- pick with module diversity (deterministic scan) ----
    picked_idx: list[int] = []
    per_module_count: dict[str, int] = {}
    blocked_by_module_cap = 0

    for idx, r in df_ranked.iterrows():
        if len(picked_idx) >= int(k):
            break

        mid = ""
        if has_module and (not _is_na_scalar(r.get("module_id"))):
            mid = str(r.get("module_id")).strip()

        if (not mid) or (mid.lower() in _NA_TOKENS_L):
            # treat "missing module" as its own bucket per term_uid to avoid collapsing
            mid = f"M_missing::{str(r.get('term_uid'))}"

        c = per_module_count.get(mid, 0)
        if c >= max_per_module:
            blocked_by_module_cap += 1
            continue

        per_module_count[mid] = c + 1
        picked_idx.append(idx)

    df_pick = df_ranked.loc[picked_idx].copy()

    # ---- prepare stress lookups (optional) ----
    do_stress = _stress_enabled(card)
    term_to_genes = _build_term_gene_map(df_ranked) if do_stress else {}
    module_to_terms_genes = _build_module_terms_genes(df_ranked, term_to_genes) if do_stress else {}

    gene_pool = np.array([], dtype=object)
    if do_stress:
        all_genes: list[str] = []
        for gs in term_to_genes.values():
            all_genes.extend(list(gs))
        gene_pool = np.array(sorted(set(all_genes)), dtype=object)

    rows: list[dict[str, Any]] = []
    ctx_keys = _context_keys(card)

    # selection diagnostics (helps explain "k=100 but only 3")
    n_total = int(df_ranked.shape[0])
    n_eligible = int(df_ranked["eligible"].sum()) if "eligible" in df_ranked.columns else n_total
    n_ineligible = int(n_total - n_eligible)
    n_picked = int(len(picked_idx))

    notes_common = (
        f"ranked={n_total}; eligible={n_eligible}; ineligible={n_ineligible}; "
        f"picked={n_picked}; k={int(k)}; max_per_module={int(max_per_module)}; "
        f"blocked_by_module_cap={int(blocked_by_module_cap)}"
    )

    mprefix = _module_prefix(card)

    for _, r in df_pick.iterrows():
        term_id = str(r["term_id"]).strip()
        term_name = str(r["term_name"]).strip()
        direction = _norm_direction(r.get("direction", "na"))
        source = str(r["source"]).strip()

        term_uid = str(r.get("term_uid") or f"{source}:{term_id}").strip()

        genes_full = [_norm_gene_id(g) for g in _as_gene_list(r.get("evidence_genes"))]
        genes_full = [g for g in genes_full if str(g).strip()]

        module_id = ""
        module_reason = ""
        module_missing = False

        if has_module and (not _is_na_scalar(r.get("module_id"))):
            module_id = str(r.get("module_id")).strip()

        if (not module_id) or (module_id.lower() in _NA_TOKENS_L):
            # Deterministic fallback that matches modules.py identity shape:
            # {prefix}{content_hash12}
            content_hash = _module_hash_like_modules_py([term_uid], genes_full)
            module_id = f"{mprefix}{content_hash}"
            module_reason = "missing_module_id"
            module_missing = True

        if genes_full:
            gene_set_hash = _hash_gene_set_audit(genes_full)
        else:
            gene_set_hash = _hash_term_set_fallback([term_uid])

        claim = Claim(
            entity=term_id,
            direction=direction,
            context_keys=ctx_keys,
            evidence_ref=EvidenceRef(
                module_id=module_id,
                gene_ids=genes_full[:10],
                term_ids=[term_uid],
                gene_set_hash=gene_set_hash,
            ),
        )

        rec: dict[str, Any] = {
            "claim_id": claim.claim_id,
            "entity": claim.entity,
            "direction": claim.direction,
            "context_keys": ",".join(claim.context_keys),
            "term_uid": term_uid,
            "source": source,
            "term_id": term_id,
            "term_name": term_name,
            "module_id": claim.evidence_ref.module_id,
            "module_missing": bool(module_missing),
            "module_reason": module_reason,
            "module_prefix_effective": mprefix,
            "gene_ids": ",".join(claim.evidence_ref.gene_ids),
            "term_ids": ",".join(claim.evidence_ref.term_ids),
            "gene_set_hash": claim.evidence_ref.gene_set_hash,
            # context fields (evaluation-aware; used by audit.py)
            "context_score": r.get("context_score", pd.NA),
            "context_evaluated": bool(r.get("context_evaluated", False)),
            "eligible": bool(r.get("eligible", True)),
            "term_survival": r.get("term_survival", pd.NA),
            "keep_term": bool(r.get("keep_term", True)),
            "keep_reason": str(r.get("keep_reason", "ok")),
            "claim_json": claim.model_dump_json(),
            "preselect_tau_gate": bool(preselect_tau_gate),
            "context_score_proxy": bool(enable_ctx_proxy),
            "select_notes": notes_common,
            # machine-readable diagnostics
            "select_diag_n_total": n_total,
            "select_diag_n_eligible": n_eligible,
            "select_diag_n_ineligible": n_ineligible,
            "select_diag_n_picked": n_picked,
            "select_diag_blocked_by_module_cap": int(blocked_by_module_cap),
            "select_diag_k": int(k),
            "select_diag_max_per_module": int(max_per_module),
        }

        if do_stress:
            st = _evaluate_stress_for_claim(
                claim_id=claim.claim_id,
                term_ids=[term_uid],
                module_id=module_id,
                gene_set_hash=claim.evidence_ref.gene_set_hash,
                term_to_genes=term_to_genes,
                module_to_terms_genes=module_to_terms_genes,
                gene_pool=gene_pool,
                seed=seed,
                card=card,
            )
            rec.update(st)

        rows.append(rec)

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

    IMPORTANT: k is treated as an explicit, caller-owned parameter.
    - The pipeline (or CLI/paper runner) should resolve precedence and pass k here.
    - This function validates k but does NOT override it via env/card, to avoid
      hard-to-debug "k=100 but only 3" regressions.

    - mode="deterministic": stable ranking + module diversity gate, emits Claim JSON.
    - mode="llm": LLM selects from top candidates but is post-validated against
      candidates (term_uid/entity/gene_set_hash) and MUST emit JSON; otherwise
      we fall back to deterministic output.

    NOTE: stress_* columns emitted by deterministic mode are PROBES (measurements),
    not final decisions. audit.py may convert WARN->ABSTAIN/FAIL depending on policy.
    """
    mode_eff = _resolve_mode(card, mode)
    k_eff = _validate_k(k)

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

            out_det = _select_claims_deterministic(distilled, card, k=int(k_eff), seed=seed)
            out_det["claim_mode"] = "deterministic_fallback"
            out_det["llm_notes"] = res.notes
            return out_det

    out_det = _select_claims_deterministic(distilled, card, k=int(k_eff), seed=seed)
    out_det["claim_mode"] = "deterministic"
    out_det["llm_notes"] = ""
    return out_det
