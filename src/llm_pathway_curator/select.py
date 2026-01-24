# LLM-PathwayCurator/src/llm_pathway_curator/select.py
from __future__ import annotations

import hashlib
import inspect
import os
import re
import sys
from collections.abc import Callable
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

_BRACKETED_LIST_RE = re.compile(r"^\s*[\[\(\{].*[\]\)\}]\s*$")

# Keep claim JSON payload bounded to avoid pathological TSV/JSON bloat.
# IMPORTANT: gene_set_hash MUST be computed from the SAME gene list stored in claim_json,
# otherwise downstream post-validate / stress scoring may delete good selections.
_DEFAULT_MAX_GENE_IDS_IN_CLAIM = 512


# -------------------------
# Small utilities
# -------------------------
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


def _validate_k(k: Any) -> int:
    try:
        kk = int(k)
    except Exception as e:
        raise ValueError(f"select_claims: invalid k={k!r} (must be int>=1)") from e
    if kk < 1:
        raise ValueError(f"select_claims: invalid k={k!r} (must be int>=1)")
    return kk


def _norm_gene_id(g: str) -> str:
    # align with audit.py/pipeline.py to avoid spurious drift
    return str(g).strip().upper()


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


def _as_gene_list(x: Any) -> list[str]:
    """
    Tolerant parsing for evidence genes (align with schema/distill):
      - list/tuple/set -> preserve order (dedup)
      - string -> supports:
          "A,B" / "A;B" / "A|B" / "A/B"
          "['A','B']" / '["A","B"]' / "{A,B}"
        fallback: whitespace split if no commas and looks tokenized
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

    # Strip wrapper if it looks like a bracketed list; do NOT eval.
    if _BRACKETED_LIST_RE.match(s):
        s = s.strip().lstrip("[({").rstrip("])}").strip()

    # normalize separators
    s = s.replace(";", ",").replace("|", ",").replace("/", ",")
    s = s.replace("\n", " ").replace("\t", " ")
    s = " ".join(s.split()).strip()
    if not s or s.lower() in _NA_TOKENS_L:
        return []

    if "," in s:
        parts = [p.strip().strip('"').strip("'") for p in s.split(",") if p.strip()]
    else:
        parts = [p.strip().strip('"').strip("'") for p in s.split(" ") if p.strip()]

    parts = [p.strip("[](){}").strip() for p in parts]
    parts = [p for p in parts if p and p.lower() not in _NA_TOKENS_L]
    return _dedup_preserve_order(parts)


def _hash_gene_set_audit(genes: list[str]) -> str:
    """
    Audit-grade fingerprint should be SET-stable:
      - same gene set -> same hash, regardless of order
      - normalize IDs to reduce casing drift
    """
    payload = ",".join(sorted({_norm_gene_id(g) for g in genes if str(g).strip()}))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _hash_term_set_fallback(term_uids: list[str]) -> str:
    """
    Fallback when evidence genes are missing/empty.
    Deterministic and set-stable.
    """
    canon = sorted({str(t).strip() for t in term_uids if str(t).strip()})
    payload = ",".join(canon)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _debug_enabled() -> bool:
    s = str(os.environ.get("LLMPATH_DEBUG", "")).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "on"}


def _dlog(msg: str) -> None:
    if _debug_enabled():
        print(msg, file=sys.stderr)


def _card_text(card: SampleCard, primary: str, fallback: str | None = None) -> str:
    """
    Backward-compatible SampleCard field access.
    Example: primary="condition", fallback="disease".
    """
    v = getattr(card, primary, None)
    if (v is None or (str(v).strip() == "")) and fallback:
        v = getattr(card, fallback, None)
    s = "" if v is None else str(v).strip()
    return s


def _max_gene_ids_in_claim(card: SampleCard) -> int:
    """
    Bound the number of evidence genes embedded in claim_json.
    Must remain >=1 to keep EvidenceRef meaningful.
    """
    ex = _get_extra(card)
    env = str(os.environ.get("LLMPATH_MAX_GENE_IDS_IN_CLAIM", "")).strip()
    if env:
        try:
            v = int(env)
            return max(1, min(5000, v))
        except Exception:
            pass
    v2 = _as_int(ex.get("max_gene_ids_in_claim", None), _DEFAULT_MAX_GENE_IDS_IN_CLAIM)
    return max(1, min(5000, int(v2)))


# -------------------------
# Context proxy (deterministic) - selection-time only
# (NOTE: context REVIEW belongs to pipeline.py; this is only for ranking/tiebreak.)
# -------------------------
def _context_tokens(card: SampleCard) -> list[str]:
    toks: list[str] = []
    for v in [
        _card_text(card, "condition", "disease"),
        _card_text(card, "tissue", None),
        _card_text(card, "perturbation", None),
        _card_text(card, "comparison", None),
    ]:
        s = str(v).strip().lower()
        if not s or s in _NA_TOKENS_L:
            continue
        parts = re.split(r"[^a-z0-9]+", s)
        parts = [p for p in parts if len(p) >= 3 and p not in _NA_TOKENS_L]
        toks.extend(parts)
    return _dedup_preserve_order(toks)


def _context_score(term_name: str, toks: list[str]) -> int:
    name = str(term_name).lower()
    return sum(1 for t in toks if t and t in name)


def _context_keys(card: SampleCard) -> list[str]:
    """
    Tool-facing context keys (stable vocabulary).
    Backward compat: if disease exists but condition missing, still emit "condition".
    """
    keys: list[str] = []
    cond = _card_text(card, "condition", "disease").strip().lower()
    if cond and cond not in _NA_TOKENS_L:
        keys.append("condition")

    for k in ["tissue", "perturbation", "comparison"]:
        v = getattr(card, k, None)
        if v is None:
            continue
        s = str(v).strip().lower()
        if not s or s in _NA_TOKENS_L:
            continue
        keys.append(k)

    return keys


def _is_context_swap(card: SampleCard) -> bool:
    """
    Detect paper ablation cards (context_swap/shuffled_context).
    """
    ex = _get_extra(card)
    for k in ["context_swap_to", "context_swap_from", "context_swap_to_cancer"]:
        v = ex.get(k, None)
        if isinstance(v, str) and v.strip():
            return True
        if v not in (None, "", [], {}):
            return True
    v2 = ex.get("shuffled_context", None)
    if isinstance(v2, bool) and v2:
        return True
    if isinstance(v2, str) and v2.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    return False


# -------------------------
# Context selection mode (deterministic, explicit)
# -------------------------
_SELECT_CONTEXT_ENV = "LLMPATH_SELECT_CONTEXT_MODE"  # off|proxy (default off)

# IMPORTANT:
# Avoid collision with pipeline's LLMPATH_CONTEXT_GATE_MODE semantics.
# select.py uses a separate env name for selection-time context gating.
_SELECT_CONTEXT_GATE_ENV = "LLMPATH_SELECT_CONTEXT_GATE_MODE"  # off|note|hard (default off)

# Legacy fallback ONLY if values match this file's vocabulary.
_LEGACY_CONTEXT_GATE_ENV = "LLMPATH_CONTEXT_GATE_MODE"


def _select_context_mode(card: SampleCard) -> str:
    """
    Selection-time context usage (deterministic proxy).
    Priority: env > auto(context_swap) > card.extra > legacy knob.
    """
    env = str(os.environ.get(_SELECT_CONTEXT_ENV, "")).strip().lower()
    if env:
        if env in {"proxy", "on", "true", "yes", "y", "1"}:
            return "proxy"
        return "off"

    if _is_context_swap(card):
        return "proxy"

    ex = _get_extra(card)
    v = str(ex.get("select_context_mode", "")).strip().lower()
    if v in {"proxy", "on", "true", "yes", "y", "1"}:
        return "proxy"
    if v in {"off", "disable", "disabled", "none"}:
        return "off"

    try:
        legacy = bool(card.enable_context_score_proxy(default=False))
        return "proxy" if legacy else "off"
    except Exception:
        return "off"


def _context_gate_mode(card: SampleCard) -> str:
    """
    Selection-time gating based on context proxy score.
    Values:
      - off: no gate
      - note: annotate only
      - hard: treat context_score==0 as ineligible (ONLY when proxy is enabled)
    Priority: new env > card.extra > legacy env (limited) > default off
    """
    env = str(os.environ.get(_SELECT_CONTEXT_GATE_ENV, "")).strip().lower()
    if env:
        return env if env in {"off", "note", "hard"} else "off"

    ex = _get_extra(card)
    v = str(ex.get("context_gate_mode", "")).strip().lower()
    if v in {"off", "note", "hard"}:
        return v

    # Legacy env fallback only if it matches off|note|hard (ignore "soft" etc.)
    env2 = str(os.environ.get(_LEGACY_CONTEXT_GATE_ENV, "")).strip().lower()
    if env2 in {"off", "note", "hard"}:
        return env2

    return "off"


def _context_tiebreak_int(card: SampleCard, term_uid: str) -> int:
    """
    Deterministic tie-breaker that MUST change when condition changes.
    """
    cond = _card_text(card, "condition", "disease").strip().lower()
    tu = str(term_uid).strip()
    payload = f"{cond}|{tu}".encode()
    h = hashlib.sha256(payload).hexdigest()[:8]
    return int(h, 16)


def _context_signature(card: SampleCard) -> str:
    """
    Compact signature of the context used for deterministic proxy.
    """
    cond = _card_text(card, "condition", "disease").strip().lower()
    tissue = _card_text(card, "tissue", None).strip().lower()
    pert = _card_text(card, "perturbation", None).strip().lower()
    comp = _card_text(card, "comparison", None).strip().lower()
    payload = f"{cond}|{tissue}|{pert}|{comp}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# -------------------------
# Mode resolution
# -------------------------
def _resolve_mode(card: SampleCard, mode: str | None) -> str:
    # Priority: explicit arg > env > SampleCard getter > default
    if mode is not None:
        s = str(mode).strip().lower()
        return s if s in {"deterministic", "llm"} else "deterministic"

    env = str(os.environ.get("LLMPATH_CLAIM_MODE", "")).strip().lower()
    if env:
        return env if env in {"deterministic", "llm"} else "deterministic"

    try:
        s = str(card.claim_mode(default="deterministic")).strip().lower()
        return s if s in {"deterministic", "llm"} else "deterministic"
    except Exception:
        return "deterministic"


# ===========================
# Stress suite (optional dev probe)
# NOTE: pipeline.py owns user-facing stress (dropout/contradiction).
# This remains opt-in and should stay OFF by default.
# ===========================
def _stress_enabled(card: SampleCard) -> bool:
    env = str(os.environ.get("LLMPATH_STRESS_GENERATE", "")).strip().lower()
    if env:
        return env in {"1", "true", "t", "yes", "y", "on"}
    ex = _get_extra(card)
    return _as_bool(ex.get("stress_generate", None), False)


def _seed_for_claim(seed: int | None, claim_id: str) -> int:
    base = 0 if seed is None else int(seed)
    h = hashlib.blake2b(digest_size=8)
    h.update(str(base).encode("utf-8"))
    h.update(b"|")
    h.update(str(claim_id).encode("utf-8"))
    return int.from_bytes(h.digest(), byteorder="little", signed=False)


def _similarity(orig: set[str], pert: set[str]) -> tuple[float, float, float]:
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


def _evaluate_stress_for_claim(
    *,
    claim_id: str,
    term_uids: list[str],
    gene_set_hash: str,
    term_to_genes: dict[str, set[str]],
    gene_pool: np.ndarray,
    seed: int | None,
    card: SampleCard,
) -> dict[str, Any]:
    ex = _get_extra(card)

    n = max(8, min(256, _as_int(ex.get("stress_n", None), 64)))
    p_drop = min(max(_as_float(ex.get("stress_gene_dropout_p", None), 0.20), 0.0), 0.95)
    p_add = min(max(_as_float(ex.get("stress_gene_jitter_p", None), 0.10), 0.0), 0.95)

    j_min = min(max(_as_float(ex.get("stress_jaccard_min", None), 0.60), 0.0), 1.0)
    r_min = min(max(_as_float(ex.get("stress_recall_min", None), 0.80), 0.0), 1.0)
    p_min = min(max(_as_float(ex.get("stress_precision_min", None), 0.80), 0.0), 1.0)

    surv_thr = min(max(_as_float(ex.get("stress_survival_thr", None), 0.80), 0.0), 1.0)

    base_genes: set[str] = set()
    for t in term_uids:
        base_genes |= set(term_to_genes.get(str(t).strip(), set()))

    if not base_genes:
        return {
            "stress_status": "ABSTAIN",
            "stress_ok": False,
            "stress_reason": "stress_missing_baseline_genes",
            "stress_notes": "no baseline evidence genes for referenced term_uids",
        }

    base_hash_claim = str(gene_set_hash).strip()
    if not base_hash_claim or base_hash_claim.lower() in _NA_TOKENS_L:
        return {
            "stress_status": "ABSTAIN",
            "stress_ok": False,
            "stress_reason": "stress_missing_gene_set_hash",
            "stress_notes": "claim gene_set_hash missing",
        }

    # IMPORTANT: baseline hash must match the embedded claim hash
    base_hash_calc = _hash_gene_set_audit(sorted(list(base_genes)))
    if base_hash_calc != base_hash_claim:
        return {
            "stress_status": "WARN",
            "stress_ok": False,
            "stress_reason": "stress_gene_set_hash_mismatch",
            "stress_notes": f"claim_hash={base_hash_claim} != baseline_hash={base_hash_calc}",
        }

    rng = np.random.default_rng(_seed_for_claim(seed, claim_id))

    ok = 0
    js: list[float] = []
    rs: list[float] = []
    ps: list[float] = []

    base_list = sorted(list(base_genes))
    for _ in range(int(n)):
        pert = _perturb_gene_set(
            base_list, gene_pool=gene_pool, rng=rng, p_drop=p_drop, p_add=p_add
        )
        j, rr, pp = _similarity(base_genes, pert)
        js.append(j)
        rs.append(rr)
        ps.append(pp)
        if (j >= j_min) and (rr >= r_min) and (pp >= p_min):
            ok += 1

    surv = ok / float(n) if n > 0 else 0.0

    if surv >= float(surv_thr):
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

    return {
        "stress_status": "WARN",
        "stress_ok": False,
        "stress_reason": "stress_identity_collapse",
        "stress_notes": (
            f"identity_survival={surv:.3f} < thr={float(surv_thr):.2f} "
            f"(n_ok={ok}/{n}); "
            f"mean_j={float(np.mean(js)):.3f} "
            f"mean_r={float(np.mean(rs)):.3f} "
            f"mean_p={float(np.mean(ps)):.3f}"
        ),
    }


# ===========================
# Deterministic selection
# ===========================
def _select_claims_deterministic(
    distilled: pd.DataFrame,
    card: SampleCard,
    *,
    k: int = 3,
    seed: int | None = None,
) -> pd.DataFrame:
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

    # ---- context selection: OFF/PROXY (auto-enabled for context_swap) ----
    sel_ctx_mode = _select_context_mode(card)
    ctx_gate_mode = _context_gate_mode(card)
    ctx_swap_active = _is_context_swap(card)
    ctx_sig = _context_signature(card)

    # Preserve pipeline-owned context_score if present; never overwrite provenance.
    # select.py proxy scores are stored in select_context_* columns.
    if "context_score" in df.columns:
        df["context_score"] = pd.to_numeric(df["context_score"], errors="coerce")
    else:
        df["context_score"] = pd.NA

    if "context_evaluated" in df.columns:
        # keep if already present (pipeline may set it)
        df["context_evaluated"] = df["context_evaluated"].fillna(False).astype(bool)
    else:
        df["context_evaluated"] = False

    if sel_ctx_mode == "proxy":
        toks = _context_tokens(card)
        df["select_context_tokens_n"] = int(len(toks))
        df["select_context_score"] = df["term_name"].map(lambda s: _context_score(str(s), toks))
        df["select_context_evaluated"] = True

        df["select_context_tiebreak"] = df["term_uid"].map(
            lambda tu: _context_tiebreak_int(card, str(tu))
        )
        df["select_context_tiebreak_sort"] = pd.to_numeric(
            df["select_context_tiebreak"], errors="coerce"
        ).fillna(2**31 - 1)
    else:
        df["select_context_tokens_n"] = 0
        df["select_context_score"] = pd.NA
        df["select_context_evaluated"] = False
        df["select_context_tiebreak"] = pd.NA
        df["select_context_tiebreak_sort"] = 2**31 - 1

    df["select_context_score_sort"] = pd.to_numeric(
        df["select_context_score"], errors="coerce"
    ).fillna(-1)

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

    # ---- optional context gate (selection-time; proxy only) ----
    df["context_gate_mode"] = str(ctx_gate_mode)
    df["context_gate_hit"] = False
    df["eligible_context"] = True

    # Gate must be defined on selection-time proxy score only (never pipeline score).
    if sel_ctx_mode == "proxy" and ctx_gate_mode in {"note", "hard"}:
        hit = df["select_context_score_sort"].ge(1)
        df["context_gate_hit"] = hit
        df["eligible_context"] = hit if ctx_gate_mode == "hard" else True
        if ctx_gate_mode == "hard":
            df["eligible"] = df["eligible"] & df["eligible_context"]

    df["term_survival_sort"] = df["term_survival"].fillna(-1.0)

    try:
        max_per_module = int(card.max_per_module(default=1))
    except Exception:
        max_per_module = 1
    max_per_module = max(1, max_per_module)

    # ---- Ranking policy ----
    if sel_ctx_mode == "proxy" and ctx_swap_active:
        sort_cols = [
            "eligible",
            "term_survival_sort",
            "select_context_score_sort",
            "select_context_tiebreak_sort",
            "stat",
            "term_uid",
        ]
        ascending = [False, False, False, True, False, True]
        df["select_context_policy"] = "proxy_swap_strong"
    else:
        sort_cols = [
            "eligible",
            "term_survival_sort",
            "select_context_score_sort",
            "stat",
            "select_context_tiebreak_sort",
            "term_uid",
        ]
        ascending = [False, False, False, False, True, True]
        df["select_context_policy"] = "default"

    df_ranked = df.sort_values(sort_cols, ascending=ascending, kind="mergesort").copy()

    has_module = "module_id" in df_ranked.columns

    def _effective_mid(r: pd.Series) -> str:
        mid = ""
        if has_module and (not _is_na_scalar(r.get("module_id"))):
            mid = str(r.get("module_id")).strip()
        if (not mid) or (mid.lower() in _NA_TOKENS_L):
            mid = f"M_missing::{str(r.get('term_uid'))}"
        return mid

    # ---- pass1: enforce module diversity cap ----
    picked_idx: list[int] = []
    per_module_count: dict[str, int] = {}
    blocked_by_module_cap = 0

    for idx, r in df_ranked.iterrows():
        if len(picked_idx) >= int(k):
            break
        mid = _effective_mid(r)
        c = per_module_count.get(mid, 0)
        if c >= max_per_module:
            blocked_by_module_cap += 1
            continue
        per_module_count[mid] = c + 1
        picked_idx.append(idx)

    # ---- pass2: if still short, relax cap to honor k ----
    relaxed = False
    relax_passes = 0
    if len(picked_idx) < int(k):
        remaining = [i for i in df_ranked.index.tolist() if i not in set(picked_idx)]
        if remaining:
            relaxed = True
            cap = max_per_module
            while len(picked_idx) < int(k) and remaining and cap < 10:
                cap += 1
                relax_passes += 1
                for idx in remaining:
                    if len(picked_idx) >= int(k):
                        break
                    r = df_ranked.loc[idx]
                    mid = _effective_mid(r)
                    c = per_module_count.get(mid, 0)
                    if c >= cap:
                        continue
                    per_module_count[mid] = c + 1
                    picked_idx.append(idx)
                remaining = [i for i in remaining if i not in set(picked_idx)]

            if len(picked_idx) < int(k) and remaining:
                need = int(k) - len(picked_idx)
                picked_idx.extend(remaining[:need])

    df_pick = df_ranked.loc[picked_idx].copy()

    # ---- optional stress lookups (dev-only; opt-in) ----
    do_stress = _stress_enabled(card)
    term_to_genes = _build_term_gene_map(df_ranked) if do_stress else {}
    gene_pool = np.array([], dtype=object)
    if do_stress:
        all_genes: list[str] = []
        for gs in term_to_genes.values():
            all_genes.extend(list(gs))
        gene_pool = np.array(sorted(set(all_genes)), dtype=object)

    rows: list[dict[str, Any]] = []
    ctx_keys = _context_keys(card)

    n_total = int(df_ranked.shape[0])
    n_eligible = int(df_ranked["eligible"].sum()) if "eligible" in df_ranked.columns else n_total
    n_ineligible = int(n_total - n_eligible)
    n_picked = int(len(picked_idx))

    n_ctx_hit = (
        int(df_ranked["context_gate_hit"].sum()) if "context_gate_hit" in df_ranked.columns else 0
    )

    notes_common = (
        f"ranked={n_total}; eligible={n_eligible}; ineligible={n_ineligible}; "
        f"picked={n_picked}; max_per_module={int(max_per_module)}; "
        f"blocked_by_module_cap={int(blocked_by_module_cap)}; "
        f"relaxed={bool(relaxed)}; relax_passes={int(relax_passes)}; "
        f"ctx_mode={sel_ctx_mode}; ctx_gate={ctx_gate_mode}; ctx_swap={bool(ctx_swap_active)}; "
        f"ctx_hit_terms={n_ctx_hit}"
    )

    max_gene_ids = _max_gene_ids_in_claim(card)

    for _, r in df_pick.iterrows():
        term_id = str(r["term_id"]).strip()
        term_name = str(r["term_name"]).strip()
        direction = _norm_direction(r.get("direction", "na"))
        source = str(r["source"]).strip()

        term_uid = str(r.get("term_uid") or f"{source}:{term_id}").strip()

        genes_full_raw = _as_gene_list(r.get("evidence_genes"))
        genes_full_norm = [_norm_gene_id(g) for g in genes_full_raw]
        genes_full_norm = [g for g in genes_full_norm if str(g).strip()]
        genes_full_norm = _dedup_preserve_order(genes_full_norm)

        genes_claim = genes_full_norm[:max_gene_ids]
        genes_suggest = genes_claim[:10]

        module_id = ""
        module_reason = ""
        module_missing = False

        if has_module and (not _is_na_scalar(r.get("module_id"))):
            module_id = str(r.get("module_id")).strip()

        if (not module_id) or (module_id.lower() in _NA_TOKENS_L):
            # Stable fallback module id (content-derived)
            payload = "T:" + term_uid + "\n" + "G:" + "|".join(sorted(genes_claim))
            content_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
            module_id = f"M{content_hash}"
            module_reason = "missing_module_id"
            module_missing = True

        if genes_claim:
            gene_set_hash = _hash_gene_set_audit(genes_claim)
        else:
            gene_set_hash = _hash_term_set_fallback([term_uid])

        claim = Claim(
            entity=term_name if term_name else term_id,
            direction=direction,
            context_keys=ctx_keys,
            evidence_ref=EvidenceRef(
                module_id=module_id,
                gene_ids=genes_claim,
                term_ids=[term_uid],
                gene_set_hash=gene_set_hash,
            ),
        )

        gene_snippet = ",".join([str(g) for g in genes_suggest])

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
            "gene_ids": gene_snippet,  # legacy compact snippet
            "gene_ids_suggest": gene_snippet,
            "gene_ids_snippet": gene_snippet,
            "gene_ids_n_total": int(len(genes_full_norm)),
            "gene_ids_n_in_claim": int(len(genes_claim)),
            "term_ids": ",".join(claim.evidence_ref.term_ids),
            "gene_set_hash": claim.evidence_ref.gene_set_hash,
            # Pipeline-owned (review) context signals (if present in distilled). Never overwritten.
            "context_score": r.get("context_score", pd.NA),
            "context_evaluated": bool(r.get("context_evaluated", False)),
            # Selection-time proxy context signals (deterministic; used for ranking/tiebreak/gate)
            "select_context_score": r.get("select_context_score", pd.NA),
            "select_context_evaluated": bool(r.get("select_context_evaluated", False)),
            "select_context_tokens_n": int(r.get("select_context_tokens_n", 0) or 0),
            "select_context_tiebreak": r.get("select_context_tiebreak", pd.NA),
            "context_signature": str(ctx_sig),
            "context_swap_active": bool(ctx_swap_active),
            "context_gate_mode": str(ctx_gate_mode),
            "context_gate_hit": bool(r.get("context_gate_hit", False)),
            "eligible_context": bool(r.get("eligible_context", True)),
            "eligible": bool(r.get("eligible", True)),
            "term_survival": r.get("term_survival", pd.NA),
            "keep_term": bool(r.get("keep_term", True)),
            "keep_reason": str(r.get("keep_reason", "ok")),
            "claim_json": claim.model_dump_json(),
            "preselect_tau_gate": bool(preselect_tau_gate),
            "context_score_proxy": bool(sel_ctx_mode == "proxy"),
            "select_context_mode": str(sel_ctx_mode),
            "select_context_policy": str(r.get("select_context_policy", "default")),
            "select_notes": notes_common,
            "select_diag_n_total": n_total,
            "select_diag_n_eligible": n_eligible,
            "select_diag_n_ineligible": n_ineligible,
            "select_diag_n_picked": n_picked,
            "select_diag_blocked_by_module_cap": int(blocked_by_module_cap),
            "select_diag_relaxed": bool(relaxed),
            "select_diag_relax_passes": int(relax_passes),
            "select_diag_max_per_module": int(max_per_module),
            "select_diag_ctx_hit_terms": int(n_ctx_hit),
        }

        if do_stress:
            st = _evaluate_stress_for_claim(
                claim_id=claim.claim_id,
                term_uids=[term_uid],
                gene_set_hash=claim.evidence_ref.gene_set_hash,
                term_to_genes=term_to_genes,
                gene_pool=gene_pool,
                seed=seed,
                card=card,
            )
            rec.update(st)

        rows.append(rec)

    return pd.DataFrame(rows)


# ===========================
# LLM mode compatibility wrapper
# ===========================
def _call_compat(fn: Callable[..., Any], **kwargs: Any) -> Any:
    """
    Call fn(**kwargs) but drop kwargs not accepted by fn signature.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(**kwargs)

    allowed = set(sig.parameters.keys())
    filt = {k: v for k, v in kwargs.items() if k in allowed}
    return fn(**filt)


# ===========================
# Public API
# ===========================
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
    - mode="llm": LLM selects from candidates but MUST emit schema-valid claims;
      otherwise we fall back to deterministic output.

    IMPORTANT:
      - claim_json embeds evidence_ref.gene_ids and gene_set_hash; they MUST be consistent.
      - user-facing columns may show a compact gene snippet (gene_ids_suggest) for readability.

    NOTE:
      - Context REVIEW (proxy/llm/off) is owned by pipeline.py.
      - select.py only provides selection-time context proxy for ranking/tiebreak.
    """
    mode_eff = _resolve_mode(card, mode)
    k_eff = _validate_k(k)
    _dlog(f"[select] mode_eff={mode_eff} backend={'yes' if backend else 'no'} k={k_eff}")

    # ---- LLM mode ----
    if mode_eff == "llm":
        if backend is None:
            out_det = _select_claims_deterministic(distilled, card, k=int(k_eff), seed=seed)
            out_det["claim_mode"] = "deterministic"
            out_det["llm_notes"] = "llm requested but backend=None; deterministic used"
            return out_det

        res = None
        notes = ""
        try:
            res = _call_compat(
                propose_claims_llm,
                distilled_with_modules=distilled,
                card=card,
                backend=backend,
                k=int(k_eff),
                seed=seed,
                outdir=outdir,
            )
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e)[:200]}"
            notes = f"LLM propose failed: {msg}"
            res = None

        out_llm: pd.DataFrame | None = None
        used_fallback = True

        if isinstance(res, pd.DataFrame):
            out_llm = res
            used_fallback = False
        elif res is not None:
            used_fallback = bool(getattr(res, "used_fallback", True))
            notes2 = str(getattr(res, "notes", "")).strip()
            notes = notes2 or notes
            claims_obj = getattr(res, "claims", None)

            if claims_obj is not None:
                try:
                    out_llm = _call_compat(
                        claims_to_proposed_tsv,
                        claims=claims_obj,
                        distilled_with_modules=distilled,
                        card=card,
                    )
                    used_fallback = False
                except Exception as e:
                    msg = f"{type(e).__name__}: {str(e)[:200]}"
                    notes = notes or f"claims_to_proposed_tsv failed: {msg}"
                    out_llm = None
                    used_fallback = True

        if out_llm is not None and (not used_fallback):
            out_llm = out_llm.copy()
            out_llm["claim_mode"] = "llm"
            out_llm["llm_notes"] = notes
            return out_llm

        out_det = _select_claims_deterministic(distilled, card, k=int(k_eff), seed=seed)
        out_det["claim_mode"] = "deterministic_fallback"
        out_det["llm_notes"] = notes or "llm returned fallback/empty; deterministic used"
        return out_det

    # ---- Deterministic ----
    out_det = _select_claims_deterministic(distilled, card, k=int(k_eff), seed=seed)
    out_det["claim_mode"] = "deterministic"
    out_det["llm_notes"] = ""
    return out_det
