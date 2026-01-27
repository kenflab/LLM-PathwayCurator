# LLM-PathwayCurator/src/llm_pathway_curator/select.py
from __future__ import annotations

import inspect
import json
import os
import re
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from . import _shared
from .backends import BaseLLMBackend
from .claim_schema import Claim, EvidenceRef
from .llm_claims import claims_to_proposed_tsv, propose_claims_llm
from .sample_card import SampleCard

_ALLOWED_DIRECTIONS = {"up", "down", "na"}
_NA_TOKENS_L = _shared.NA_TOKENS_L

_BRACKETED_LIST_RE = re.compile(r"^\s*[\[\(\{].*[\]\)\}]\s*$")

# Keep claim JSON payload bounded to avoid pathological TSV/JSON bloat.
# IMPORTANT: gene_set_hash MUST be computed from the SAME gene list stored in claim_json,
# otherwise downstream post-validate / stress scoring may delete good selections.
_DEFAULT_MAX_GENE_IDS_IN_CLAIM = 512


# -------------------------
# Small utilities
# -------------------------
def _is_na_scalar(x: Any) -> bool:
    """Single source of truth: _shared.is_na_scalar."""
    return _shared.is_na_scalar(x)


def _dedup_preserve_order(items: list[str]) -> list[str]:
    return _shared.dedup_preserve_order([str(x).strip() for x in items])


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


def _clean_gene_token(g: Any) -> str:
    return _shared.clean_gene_token(g)


def _norm_gene_id(g: Any) -> str:
    """
    Canonical gene id normalization.

    CONTRACT (must match schema.py / modules.py / audit.py):
      - trim quotes/whitespace/punctuation
      - DO NOT force uppercase (species/ID-system dependent)
    """
    return _clean_gene_token(g)


def _norm_direction(x: Any) -> str:
    """Single source of truth: _shared.normalize_direction()."""
    return _shared.normalize_direction(x)


def _as_gene_list(x: Any) -> list[str]:
    """
    Single source of truth: delegate to _shared.parse_genes().
    - conservative split
    - no uppercasing
    - dedup preserve order
    """
    return _shared.parse_genes(x)


def _hash_gene_set_claim(genes: list[str]) -> str:
    """
    Spec-level: must match Claim/EvidenceRef semantics.
    IMPORTANT: genes must be the SAME list embedded into claim_json.
    """
    return _shared.hash_gene_set_12hex(list(genes or []))


def _hash_term_set_fallback(term_uids: list[str]) -> str:
    """
    Fallback when evidence genes are missing/empty.
    Deterministic and set-stable.
    """
    return _shared.hash_set_12hex(list(term_uids or []))


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
# Context swap-aware "effective context" (CRITICAL)
# -------------------------
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


def _effective_context_value(card: SampleCard, field: str) -> str:
    """
    Single source of truth for selection-time context proxy inputs.

    Key idea:
      - If context_swap is active, card.extra may carry the swapped condition/cancer.
      - Many cards may have disease=None; relying on card.condition/disease alone collapses proxy.
    """
    ex = _get_extra(card)

    # For "condition", prefer swapped-to condition if present.
    if field == "condition":
        # common pattern: swap_to is the *new condition / cancer label*
        v = ex.get("context_swap_to", None)
        if isinstance(v, str) and v.strip():
            return str(v).strip()

        # sometimes only cancer swap is stored
        v2 = ex.get("context_swap_to_cancer", None)
        if isinstance(v2, str) and v2.strip():
            return str(v2).strip()

        # fallback to card.condition then card.disease
        base = _card_text(card, "condition", "disease")
        if base.strip():
            return base.strip()

        # last resort: if a plain "cancer" field exists on card
        c = getattr(card, "cancer", None)
        if c is not None and str(c).strip():
            return str(c).strip()

        return ""

    # For cancer/disease explicit fields, allow swap_to_cancer to influence.
    if field in {"cancer", "disease"}:
        v = ex.get("context_swap_to_cancer", None)
        if isinstance(v, str) and v.strip():
            return str(v).strip()
        base = _card_text(card, field, None)
        if base.strip():
            return base.strip()
        return ""

    # passthrough fields (tissue/perturbation/comparison)
    if field in {"tissue", "perturbation", "comparison"}:
        base = _card_text(card, field, None)
        return base.strip()

    return _card_text(card, field, None).strip()


def _context_signature(card: SampleCard) -> str:
    """
    Compact signature of the context used for deterministic proxy.

    MUST change when swap_to/from changes.
    """
    ex = _get_extra(card)

    cond = _effective_context_value(card, "condition").strip().lower()
    tissue = _effective_context_value(card, "tissue").strip().lower()
    pert = _effective_context_value(card, "perturbation").strip().lower()
    comp = _effective_context_value(card, "comparison").strip().lower()

    # Explicitly include swap markers if present (even if cond is empty).
    sw_from = str(ex.get("context_swap_from", "") or "").strip().lower()
    sw_to = str(ex.get("context_swap_to", "") or "").strip().lower()
    sw_to_c = str(ex.get("context_swap_to_cancer", "") or "").strip().lower()

    payload = (
        f"{cond}|{tissue}|{pert}|{comp}|"
        f"swap_from={sw_from}|swap_to={sw_to}|swap_to_cancer={sw_to_c}"
    )

    # Spec-level 12-hex helper lives in _shared
    return _shared.sha256_12hex(payload)


# -------------------------
# Context proxy (deterministic) - selection-time only
# -------------------------
def _context_tokens(card: SampleCard) -> list[str]:
    """
    Deterministic tokenization for selection-time proxy.

    CRITICAL:
      - Must use swap-aware effective context, otherwise swap has no effect.
    """
    toks: list[str] = []
    for v in [
        _effective_context_value(card, "condition"),
        _effective_context_value(card, "tissue"),
        _effective_context_value(card, "perturbation"),
        _effective_context_value(card, "comparison"),
        _effective_context_value(card, "cancer"),
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
    cond = _effective_context_value(card, "condition").strip().lower()
    if cond and cond not in _NA_TOKENS_L:
        keys.append("condition")

    for k in ["tissue", "perturbation", "comparison"]:
        v = _effective_context_value(card, k)
        s = str(v).strip().lower()
        if not s or s in _NA_TOKENS_L:
            continue
        keys.append(k)

    return keys


# -------------------------
# Context selection mode (deterministic, explicit)
# -------------------------
_SELECT_CONTEXT_ENV = "LLMPATH_SELECT_CONTEXT_MODE"  # off|proxy|review (default off)

# IMPORTANT:
# Avoid collision with pipeline's LLMPATH_CONTEXT_GATE_MODE semantics.
# select.py uses a separate env name for selection-time context gating.
_SELECT_CONTEXT_GATE_ENV = "LLMPATH_SELECT_CONTEXT_GATE_MODE"  # off|note|hard (default off)

# Legacy fallback ONLY if values match this file's vocabulary.
_LEGACY_CONTEXT_GATE_ENV = "LLMPATH_CONTEXT_GATE_MODE"


def _context_review_mode_card(card: SampleCard) -> str:
    """
    Pipeline/sample_card controlled context review mode.

    Values (paper-facing):
      - off
      - proxy
      - llm

    We map these to selection-time context usage:
      - llm   -> review   (use pipeline-produced context_status/context_evaluated)
      - proxy -> proxy
      - off   -> off
    """
    ex = _get_extra(card)
    v = str(ex.get("context_review_mode", "")).strip().lower()
    if v in {"off", "none", "disable", "disabled"}:
        return "off"
    if v in {"proxy"}:
        return "proxy"
    if v in {"llm"}:
        return "llm"
    return ""


def _select_context_mode(card: SampleCard) -> str:
    """
    Selection-time context usage.

    Values:
      - off: do not use context in C1 ranking/gate
      - proxy: compute deterministic proxy (swap-aware)
      - review: use pipeline-produced context_status/context_evaluated if present;
               fallback to proxy if missing

    Priority:
      1) env (LLMPATH_SELECT_CONTEXT_MODE)
      2) auto(context_swap)
      3) card.extra.select_context_mode
      4) card.extra.context_review_mode  (NEW: llm/proxy/off -> review/proxy/off)
      5) legacy SampleCard getter
    """
    env = str(os.environ.get(_SELECT_CONTEXT_ENV, "")).strip().lower()
    if env:
        if env in {"review"}:
            return "review"
        if env in {"proxy", "on", "true", "yes", "y", "1"}:
            return "proxy"
        return "off"

    # auto-enable proxy for swap ablation (paper-facing)
    if _is_context_swap(card):
        return "proxy"

    ex = _get_extra(card)
    v = str(ex.get("select_context_mode", "")).strip().lower()
    if v in {"review"}:
        return "review"
    if v in {"proxy", "on", "true", "yes", "y", "1"}:
        return "proxy"
    if v in {"off", "disable", "disabled", "none"}:
        return "off"

    crm = _context_review_mode_card(card)
    if crm == "llm":
        return "review"
    if crm == "proxy":
        return "proxy"
    if crm == "off":
        return "off"

    # legacy getter
    try:
        legacy = bool(card.enable_context_score_proxy(default=False))
        return "proxy" if legacy else "off"
    except Exception:
        return "off"


def _context_gate_mode(card: SampleCard, override: str | None = None) -> str:
    """
    Selection-time gating based on context signal.

    Values:
      - off: no gate
      - note: annotate only
      - hard:
          * in proxy mode: treat select_context_score==0 as ineligible
          * in review mode: treat context_status!=PASS as ineligible
            (unless missing, then fallback proxy)

    Priority:
      0) explicit override (this call only; strict vocab off|note|hard)
      1) new env
      2) card.extra
      3) legacy env (limited)
      4) default off
    """
    if override is not None:
        s0 = str(override).strip().lower()
        if s0 in {"off", "note", "hard"}:
            return s0

    env = str(os.environ.get(_SELECT_CONTEXT_GATE_ENV, "")).strip().lower()
    if env:
        return env if env in {"off", "note", "hard"} else "off"

    ex = _get_extra(card)
    v = str(ex.get("select_context_gate_mode", "")).strip().lower()
    if v in {"off", "note", "hard"}:
        return v

    # Backward compat: allow context_gate_mode in extra if it matches this vocabulary.
    v2 = str(ex.get("context_gate_mode", "")).strip().lower()
    if v2 in {"off", "note", "hard"}:
        return v2

    # Legacy env fallback only if it matches off|note|hard (ignore "soft" etc.)
    env2 = str(os.environ.get(_LEGACY_CONTEXT_GATE_ENV, "")).strip().lower()
    if env2 in {"off", "note", "hard"}:
        return env2

    return "off"


def _context_tiebreak_int(card: SampleCard, term_uid: str, ctx_sig: str) -> int:
    """
    Deterministic tie-breaker that MUST change when context changes (including swap).

    Use ctx_sig explicitly so disease=None doesn't collapse.
    """
    tu = str(term_uid).strip()
    payload = {"ctx_sig": str(ctx_sig), "term_uid": tu}

    # Stable integer in [0, 2**31-2] (matches sort usage downstream)
    return _shared.seed_int_from_payload(payload, mod=2**31 - 1)


def _context_status_score(row: pd.Series) -> int:
    """
    Convert pipeline context review outputs to an ordering score.

    Expected columns (if present):
      - context_evaluated: bool
      - context_status: PASS|WARN|FAIL|...
    """
    ev = False
    if "context_evaluated" in row.index and (not _is_na_scalar(row.get("context_evaluated"))):
        try:
            ev = bool(row.get("context_evaluated"))
        except Exception:
            ev = False

    st = ""
    if "context_status" in row.index:
        st = (
            ""
            if _is_na_scalar(row.get("context_status"))
            else str(row.get("context_status")).strip().upper()
        )

    if (not ev) and (not st):
        return -1

    if st == "PASS":
        return 2
    if st == "WARN":
        return 1
    if st == "FAIL":
        return -2

    # Unknown but evaluated: treat as weak negative (do not reward)
    return 0 if ev else -1


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
    payload = {"seed": base, "claim_id": str(claim_id)}
    # Keep wide range; numpy RNG accepts large ints
    return _shared.seed_int_from_payload(payload, mod=2**63 - 1)


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
    base_hash_calc = _hash_gene_set_claim(sorted(list(base_genes)))
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
# Context review (LLM) - minimal, shortlist-only
# ===========================
def _env_wants_llm_context_review() -> bool:
    s = str(os.environ.get("LLMPATH_CONTEXT_REVIEW_MODE", "")).strip().lower()
    return s in {"llm", "on", "true", "yes", "y", "1"}


def _backend_invoke_text(backend: BaseLLMBackend, prompt: str) -> str:
    """
    Best-effort backend invocation without assuming a single interface.
    Tries common method names used across backends.
    """
    # Prefer JSON-capable methods if they exist.
    for meth in ["chat_json", "complete_json", "generate_json", "json"]:
        fn = getattr(backend, meth, None)
        if callable(fn):
            try:
                out = _call_compat(
                    fn, prompt=prompt, text=prompt, messages=[{"role": "user", "content": prompt}]
                )
                if isinstance(out, (dict, list)):
                    return json.dumps(out, ensure_ascii=False)
                if out is None:
                    continue
                return str(out)
            except Exception:
                pass

    for meth in ["chat", "complete", "generate", "invoke", "call"]:
        fn = getattr(backend, meth, None)
        if callable(fn):
            out = _call_compat(
                fn, prompt=prompt, text=prompt, messages=[{"role": "user", "content": prompt}]
            )
            if out is None:
                continue
            return str(out)

    raise RuntimeError("backend has no callable chat/complete/generate interface")


def _parse_first_json_obj(text: str) -> dict[str, Any]:
    """
    Robust-ish JSON extraction from LLM output.
    Accepts either a raw JSON object or a text containing one.
    """
    s = str(text).strip()
    if not s:
        return {}
    # Fast path
    if s.startswith("{") and s.endswith("}"):
        try:
            return dict(json.loads(s))
        except Exception:
            pass

    # Find first {...} span (greedy but bounded by last brace)
    i = s.find("{")
    j = s.rfind("}")
    if i >= 0 and j > i:
        cand = s[i : j + 1]
        try:
            obj = json.loads(cand)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _llm_context_review_prompt(
    *,
    card: SampleCard,
    term_uid: str,
    term_name: str,
    source: str,
    evidence_genes: list[str],
) -> str:
    ctx = {
        "condition": _effective_context_value(card, "condition"),
        "tissue": _effective_context_value(card, "tissue"),
        "perturbation": _effective_context_value(card, "perturbation"),
        "comparison": _effective_context_value(card, "comparison"),
        "cancer": _effective_context_value(card, "cancer"),
        "context_signature": _context_signature(card),
        "context_swap_active": _is_context_swap(card),
    }

    genes_snip = ",".join([g for g in evidence_genes[:16] if str(g).strip()])
    # Keep prompt short for llama3.1:8b
    return (
        "You are auditing whether a pathway/gene-set term matches the sample context.\n"
        "Return ONLY a JSON object with keys: status, confidence, reason.\n"
        'status must be one of ["PASS","WARN","FAIL"]. confidence must be 0..1.\n'
        "Be strict: PASS only if clearly context-consistent.\n\n"
        f"CONTEXT: {json.dumps(ctx, ensure_ascii=False)}\n"
        "TERM: {"
        f'"term_uid": "{str(term_uid)}", '
        f'"term_name": "{str(term_name)}", '
        f'"source": "{str(source)}"'
        "}\n"
        f'EVIDENCE_GENES_SNIPPET: "{genes_snip}"\n'
    )


def _maybe_apply_llm_context_review(
    distilled: pd.DataFrame,
    card: SampleCard,
    *,
    review_backend: BaseLLMBackend | None,
    k: int,
    seed: int | None,
    outdir: str | None,
    context_review_mode: str,
) -> pd.DataFrame:
    """
    Fill pipeline-owned context review columns in distilled *only when missing*:
      - context_evaluated (bool)
      - context_status (PASS|WARN|FAIL)
      - context_method (llm)
      - context_confidence (float)
      - context_reason (str)

    Strategy:
      - shortlist-only to avoid O(N_terms) LLM calls
      - do NOT overwrite rows already evaluated by pipeline
    """
    want_llm = (
        str(context_review_mode or "").strip().lower() == "llm" or _env_wants_llm_context_review()
    )
    if not want_llm:
        return distilled
    if review_backend is None:
        return distilled

    if not isinstance(distilled, pd.DataFrame) or distilled.empty:
        return distilled

    df = distilled.copy()

    # Ensure term_uid exists for stable cache
    if "term_uid" in df.columns:
        df["term_uid"] = df["term_uid"].astype(str).str.strip()
    else:
        if ("source" in df.columns) and ("term_id" in df.columns):
            df["term_uid"] = (
                df["source"].astype(str).str.strip() + ":" + df["term_id"].astype(str).str.strip()
            )
        else:
            df["term_uid"] = df.index.astype(str)

    # Create/normalize columns
    if "context_evaluated" in df.columns:
        try:
            df["context_evaluated"] = df["context_evaluated"].fillna(False).astype(bool)
        except Exception:
            df["context_evaluated"] = False
    else:
        df["context_evaluated"] = False

    if "context_status" in df.columns:
        df["context_status"] = df["context_status"].fillna("").astype(str).str.strip()
    else:
        df["context_status"] = ""

    if "context_method" in df.columns:
        df["context_method"] = df["context_method"].fillna("").astype(str).str.strip()
    else:
        df["context_method"] = ""

    if "context_confidence" not in df.columns:
        df["context_confidence"] = pd.NA
    if "context_reason" not in df.columns:
        df["context_reason"] = ""

    # --- Only evaluate rows that are not already LLM-reviewed ---
    # Problem: upstream proxy may set context_evaluated=True / context_status=PASS,
    # which blocks LLM review under the old "missing only" mask.
    # Policy:
    #   - If context_review_mode=llm: (re)review rows unless they are already method=llm.
    #   - Never overwrite existing LLM results.
    m_method = df["context_method"].astype(str).str.strip().str.lower()
    m_status = df["context_status"].astype(str).str.strip()

    already_llm = m_method.eq("llm")

    # Treat proxy-derived evaluations as "needs LLM"
    # (including method empty/none/proxy). We prefer to overwrite these.
    proxy_like = m_method.isin({"", "none", "proxy"})

    # If status is empty, it's definitely missing.
    status_missing = m_status.eq("")

    # Rows we will LLM-review:
    #  - not already LLM, AND
    #  - either status missing OR proxy-like method
    missing_mask = (~already_llm) & (status_missing | proxy_like)

    n_missing = int(missing_mask.sum())
    if n_missing <= 0:
        return df

    # Shortlist size: default 10*k (cap 500, min 50)
    # ex = _get_extra(card)
    envL = str(os.environ.get("LLMPATH_CONTEXT_REVIEW_SHORTLIST", "")).strip()
    if envL:
        try:
            L = int(envL)
        except Exception:
            L = 0
    else:
        L = 0
    if L <= 0:
        L = max(50, min(500, int(10 * int(k))))

    # Rank for shortlist using existing signals (cheap):
    # prefer high |stat| (more "important") while keeping stable ordering.
    if "stat" in df.columns:
        stat_num = pd.to_numeric(df["stat"], errors="coerce").fillna(0.0)
        score = stat_num.abs()
    else:
        score = pd.Series(np.ones(df.shape[0], dtype=float), index=df.index)

    # Add proxy context as a weak prioritizer (optional, deterministic)
    toks = _context_tokens(card)
    if "term_name" in df.columns and toks:
        try:
            proxy = df["term_name"].astype(str).map(lambda s: _context_score(str(s), toks))
            proxy = pd.to_numeric(proxy, errors="coerce").fillna(0.0)
        except Exception:
            proxy = pd.Series(np.zeros(df.shape[0], dtype=float), index=df.index)
    else:
        proxy = pd.Series(np.zeros(df.shape[0], dtype=float), index=df.index)

    # Shortlist candidates: missing only
    cand = df.loc[missing_mask].copy()
    cand["_short_stat"] = score.loc[cand.index]
    cand["_short_proxy"] = proxy.loc[cand.index]
    # Higher proxy first, then higher |stat|
    cand = cand.sort_values(
        ["_short_proxy", "_short_stat"], ascending=[False, False], kind="mergesort"
    )
    cand = cand.head(int(L)).copy()

    # Cache file (optional; very small and safe)
    cache: dict[str, Any] = {}
    cache_path = None
    if outdir:
        try:
            os.makedirs(str(outdir), exist_ok=True)
            ctx_sig = _context_signature(card)
            cache_path = os.path.join(str(outdir), f"context_review_cache.{ctx_sig}.json")
            if os.path.exists(cache_path):
                cache = json.load(open(cache_path))
                if not isinstance(cache, dict):
                    cache = {}
        except Exception:
            cache = {}
            cache_path = None

    # Deterministic-ish seed: just used to jitter sleep/backoff if you want later
    _ = seed

    updated = 0
    for idx, r in cand.iterrows():
        term_uid = str(r.get("term_uid") or "").strip()
        if not term_uid:
            continue

        key = f"{_context_signature(card)}::{term_uid}"
        if key in cache and isinstance(cache.get(key), dict):
            obj = cache.get(key) or {}
        else:
            term_name = str(r.get("term_name") or r.get("term_id") or "").strip()
            source = str(r.get("source") or "").strip()
            genes = _as_gene_list(r.get("evidence_genes"))
            genes = [_norm_gene_id(g) for g in genes if str(g).strip()]
            genes = _dedup_preserve_order([g for g in genes if g])

            prompt = _llm_context_review_prompt(
                card=card,
                term_uid=term_uid,
                term_name=term_name,
                source=source,
                evidence_genes=genes,
            )

            t0 = time.time()
            try:
                txt = _backend_invoke_text(review_backend, prompt)
            except Exception as e:
                _dlog(
                    f"[context_review][ERR] term_uid={term_uid} {type(e).__name__}: {str(e)[:160]}"
                )
                txt = ""

            obj = _parse_first_json_obj(txt)
            obj["_latency_s"] = float(max(0.0, time.time() - t0))
            cache[key] = obj

            # be a polite backend citizen
            sleep_ms = str(os.environ.get("LLMPATH_CONTEXT_REVIEW_SLEEP_MS", "")).strip()
            if sleep_ms:
                try:
                    ms = int(sleep_ms)
                    if ms > 0:
                        time.sleep(min(2.0, ms / 1000.0))
                except Exception:
                    pass

        status = str(obj.get("status") or "").strip().upper()
        if status not in {"PASS", "WARN", "FAIL"}:
            # strict fallback: if model returns garbage, do not hallucinate PASS
            status = "WARN"

        conf = obj.get("confidence", None)
        try:
            conf_f = float(conf) if conf is not None else float("nan")
        except Exception:
            conf_f = float("nan")
        if not (0.0 <= conf_f <= 1.0):
            conf_f = float("nan")

        reason = str(obj.get("reason") or "").strip()

        # Do NOT overwrite existing LLM results, but DO overwrite proxy/none.
        cur_method = str(df.at[idx, "context_method"] or "").strip().lower()
        if cur_method == "llm":
            continue

        df.at[idx, "context_evaluated"] = True
        df.at[idx, "context_status"] = status
        df.at[idx, "context_method"] = "llm"
        df.at[idx, "context_reason"] = reason
        df.at[idx, "context_confidence"] = conf_f if not np.isnan(conf_f) else pd.NA
        updated += 1

        if _debug_enabled():
            m = str(df.at[idx, "context_method"]).strip().lower()
            st_now = str(df.at[idx, "context_status"]).strip()
            if st_now and m and m != "llm":
                _dlog(f"[context_review][WARN] method!=llm term_uid={term_uid} method={m}")

    if cache_path:
        try:
            json.dump(cache, open(cache_path, "w"), ensure_ascii=False, indent=2)
        except Exception:
            pass

    _dlog(
        f"[context_review] updated={updated} shortlist={int(min(L, n_missing))} missing={n_missing}"
    )
    return df


# ===========================
# Deterministic selection
# ===========================
def _select_claims_deterministic(
    distilled: pd.DataFrame,
    card: SampleCard,
    *,
    k: int = 3,
    seed: int | None = None,
    context_gate_mode_override: str | None = None,
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
        df["term_uid"] = (
            df.apply(
                lambda r: _shared.make_term_uid(r.get("source"), r.get("term_id")),
                axis=1,
            )
            .astype(str)
            .str.strip()
        )

    # ---- context selection: OFF/PROXY/REVIEW ----
    sel_ctx_mode = _select_context_mode(card)
    ctx_gate_mode = _context_gate_mode(card, override=context_gate_mode_override)
    ctx_swap_active = _is_context_swap(card)
    ctx_sig = _context_signature(card)

    # Preserve pipeline-owned context fields if present; never overwrite provenance.
    if "context_score" in df.columns:
        df["context_score"] = pd.to_numeric(df["context_score"], errors="coerce")
    else:
        df["context_score"] = pd.NA

    if "context_evaluated" in df.columns:
        df["context_evaluated"] = df["context_evaluated"].fillna(False).astype(bool)
    else:
        df["context_evaluated"] = False

    if "context_status" in df.columns:
        df["context_status"] = df["context_status"].astype(str).str.strip()
    else:
        df["context_status"] = ""

    if "context_method" in df.columns:
        df["context_method"] = df["context_method"].fillna("").astype(str).str.strip()
    else:
        df["context_method"] = ""

    # selection-time proxy scores are stored in select_context_* columns
    if sel_ctx_mode in {"proxy", "review"}:
        toks = _context_tokens(card)
        df["select_context_tokens_n"] = int(len(toks))
        df["select_context_score"] = df["term_name"].map(lambda s: _context_score(str(s), toks))
        df["select_context_evaluated"] = True
        df["select_context_tiebreak"] = df["term_uid"].map(
            lambda tu: _context_tiebreak_int(card, str(tu), ctx_sig=str(ctx_sig))
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

    # REVIEW mode: if pipeline produced context_status, use it to rank.
    # (Fallback: if not present/evaluated, treat as -1 and rely on proxy score/tiebreak.)
    if sel_ctx_mode == "review":
        df["select_context_status_score"] = df.apply(_context_status_score, axis=1)
    else:
        df["select_context_status_score"] = -1

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

    # ---- optional context gate (selection-time) ----
    # Semantics (IMPORTANT):
    #   - eligible_context: passes the context check (True = allowed)
    #   - context_gate_hit: fails the context check (True = blocked)  [diagnostic]
    df["context_gate_mode"] = str(ctx_gate_mode)
    df["eligible_context"] = True
    df["context_gate_hit"] = False

    if sel_ctx_mode != "off" and ctx_gate_mode in {"note", "hard"}:
        # Compute "pass" first, then derive hit = ~pass.
        if sel_ctx_mode == "review":
            # PASS requires evaluated + PASS. If review is missing, fall back to proxy>=1.
            ev = df["context_evaluated"].astype(bool)
            st = df["context_status"].astype(str).str.strip().str.upper()

            pass_review = ev & (st == "PASS")
            missing_review = (~ev) & (st == "")
            pass_proxy = df["select_context_score_sort"].ge(1)

            pass_ctx = pass_review | (missing_review & pass_proxy)
        else:
            pass_ctx = df["select_context_score_sort"].ge(1)

        df["eligible_context"] = pass_ctx.astype(bool)
        df["context_gate_hit"] = (~pass_ctx).astype(bool)

        # Apply to eligibility only in hard mode; note mode is diagnostic only.
        if ctx_gate_mode == "hard":
            df["eligible"] = df["eligible"] & df["eligible_context"]

    df["term_survival_sort"] = df["term_survival"].fillna(-1.0)

    try:
        max_per_module = int(card.max_per_module(default=1))
    except Exception:
        max_per_module = 1
    max_per_module = max(1, max_per_module)

    # ---- Ranking policy ----
    # Primary: eligible -> stability -> (review score if any) -> proxy score -> tiebreak -> stat
    if sel_ctx_mode == "review":
        sort_cols = [
            "eligible",
            "term_survival_sort",
            "select_context_status_score",
            "select_context_score_sort",
            "select_context_tiebreak_sort",
            "stat",
            "term_uid",
        ]
        ascending = [False, False, False, False, True, False, True]
        df["select_context_policy"] = "review_then_proxy"
    elif sel_ctx_mode == "proxy" and ctx_swap_active:
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
        f"ctx_sig={ctx_sig}; ctx_hit_terms={n_ctx_hit}"
    )

    max_gene_ids = _max_gene_ids_in_claim(card)

    for _, r in df_pick.iterrows():
        term_id = str(r["term_id"]).strip()
        term_name = str(r["term_name"]).strip()
        direction = _norm_direction(r.get("direction", "na"))
        source = str(r["source"]).strip()

        term_uid = str(r.get("term_uid") or _shared.make_term_uid(source, term_id)).strip()

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
            content_hash = _shared.module_hash_content12(terms=[term_uid], genes=genes_claim)
            module_id = f"M{content_hash}"
            module_reason = "missing_module_id"
            module_missing = True

        # Gene-set hash MUST match gene_ids embedded in claim_json (bounded by max_gene_ids).
        if genes_claim:
            gene_set_hash = _hash_gene_set_claim(genes_claim)
        else:
            gene_set_hash = _hash_term_set_fallback([term_uid])

        # IMPORTANT (reproducibility):
        # module_id fallback should NOT depend on max_gene_ids_in_claim, otherwise module diversity
        # and ranking can change just by changing display/embedding limits.
        if module_missing:
            content_hash = _shared.module_hash_content12(terms=[term_uid], genes=genes_full_norm)
            module_id = f"M{content_hash}"
            module_reason = "missing_module_id"
            module_missing = True

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
            "term_ids": _shared.join_id_list_tsv(
                claim.evidence_ref.term_ids, delim=_shared.ID_JOIN_DELIM
            ),
            "gene_set_hash": claim.evidence_ref.gene_set_hash,
            # Pipeline-owned (review) context signals (if present in distilled). Never overwritten.
            "context_score": r.get("context_score", pd.NA),
            "context_evaluated": bool(r.get("context_evaluated", False)),
            "context_status": str(r.get("context_status", "") or ""),
            "context_method": str(r.get("context_method", "") or ""),
            "context_confidence": r.get("context_confidence", pd.NA),
            "context_reason": str(r.get("context_reason", "") or ""),
            # Selection-time context signals (deterministic; used for ranking/tiebreak/gate)
            "select_context_score": r.get("select_context_score", pd.NA),
            "select_context_evaluated": bool(r.get("select_context_evaluated", False)),
            "select_context_tokens_n": int(r.get("select_context_tokens_n", 0) or 0),
            "select_context_status_score": int(r.get("select_context_status_score", -1) or -1),
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
            "context_score_proxy": bool(sel_ctx_mode in {"proxy", "review"}),
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
def _call_compat(fn: Any, **kwargs: Any) -> Any:
    """
    Call fn(**kwargs) but drop kwargs not accepted by fn signature.
    Keeps pipeline resilient across incremental refactors.
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
    k: int = 50,
    mode: str | None = None,
    backend: BaseLLMBackend | None = None,
    claim_backend: BaseLLMBackend | None = None,
    review_backend: BaseLLMBackend | None = None,
    context_gate_mode: str = "soft",
    context_review_mode: str = "off",
    seed: int | None = None,
    outdir: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    C1: Claim proposal (schema-locked). The mechanical decider is audit_claims().

    - mode="deterministic": stable ranking + module diversity gate, emits Claim JSON.
    - mode="llm": LLM selects from candidates but MUST emit schema-valid claims;
      otherwise we fall back to deterministic output.

    IMPORTANT:
      - claim_json embeds evidence_ref.gene_ids and gene_set_hash; they MUST be consistent.
      - user-facing columns may show a compact gene snippet (gene_ids_suggest) for readability.

    NOTE (C1 contract):
      - selection-time context usage is controlled by:
          * LLMPATH_SELECT_CONTEXT_MODE = off|proxy|review
          * LLMPATH_SELECT_CONTEXT_GATE_MODE = off|note|hard
      - pipeline-owned review results (context_status/context_evaluated) may be used in
        review mode for ranking/gating, but are never overwritten here.
    """
    mode_eff = _resolve_mode(card, mode)
    k_eff = _validate_k(k)

    # Pick a single backend for the "review-only" path.
    # Pipeline may attach backend as role=review via `backend`,
    # or pass explicitly via `review_backend`.
    review_be = review_backend if review_backend is not None else backend

    _dlog(
        f"[select] mode_eff={mode_eff} backend={'yes' if backend else 'no'} "
        f"review_backend={'yes' if review_be else 'no'} k={k_eff} "
        f"ctx_review_mode={context_review_mode}"
    )

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
    # If the run asks for LLM context review, we must populate pipeline-owned context fields
    # BEFORE deterministic ranking/gating, otherwise review mode never triggers.
    distilled2 = _maybe_apply_llm_context_review(
        distilled,
        card,
        review_backend=review_be,
        k=int(k_eff),
        seed=seed,
        outdir=outdir,
        context_review_mode=str(context_review_mode or "off"),
    )

    # Make provenance explicit for downstream merge/debug (do not overwrite meaningful values)
    try:
        if isinstance(distilled2, pd.DataFrame) and (not distilled2.empty):
            if "context_review_mode" not in distilled2.columns:
                distilled2 = distilled2.copy()
                distilled2["context_review_mode"] = str(context_review_mode or "off")
            else:
                s = distilled2["context_review_mode"].astype(str).fillna("").str.strip()
                if (s == "").all():
                    distilled2 = distilled2.copy()
                    distilled2["context_review_mode"] = str(context_review_mode or "off")
    except Exception:
        pass

    gate_override = None
    try:
        s = str(context_gate_mode or "").strip().lower()
        # Only accept this file's vocab; ignore "soft" to avoid semantic collision.
        if s in {"off", "note", "hard"}:
            gate_override = s
        elif s and s != "soft":
            _dlog(f"[select][WARN] context_gate_mode arg ignored: {s!r} (allowed: off|note|hard)")
    except Exception:
        gate_override = None

    out_det = _select_claims_deterministic(
        distilled2,
        card,
        k=int(k_eff),
        seed=seed,
        context_gate_mode_override=gate_override,
    )

    out_det["claim_mode"] = "deterministic"
    out_det["llm_notes"] = ""
    return out_det
