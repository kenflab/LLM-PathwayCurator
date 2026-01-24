# LLM-PathwayCurator/src/llm_pathway_curator/select.py
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
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
    # align with audit.py to avoid spurious drift
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


def _debug_enabled() -> bool:
    s = str(os.environ.get("LLMPATH_DEBUG", "")).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "on"}


def _dlog(msg: str) -> None:
    if _debug_enabled():
        print(msg, file=sys.stderr)


def _emit_dev_meta(card: SampleCard) -> bool:
    """
    Developer-only metadata emission switch.
    Default OFF to avoid leaking internal versions/keys into user-facing outputs.
    Enable via:
      - env: LLMPATH_DEV_META=1
      - card.extra: {"emit_developer_meta": true}
    """
    env = str(os.environ.get("LLMPATH_DEV_META", "")).strip().lower()
    if env:
        return env in {"1", "true", "t", "yes", "y", "on"}
    ex = _get_extra(card)
    return _as_bool(ex.get("emit_developer_meta", None), False)


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
    This must remain >=1 to keep EvidenceRef meaningful.
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
# Context proxy (deterministic)
# -------------------------
def _context_tokens(card: SampleCard) -> list[str]:
    """
    Deterministic proxy tokens from SampleCard core context.
    NOTE: This is a weak heuristic; must be explicitly enabled or auto-enabled for context_swap.
    Backward compat: condition falls back to disease.
    """
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
    # proxy only; must be enabled by policy
    name = str(term_name).lower()
    return sum(1 for t in toks if t and t in name)


def _context_keys(card: SampleCard) -> list[str]:
    """
    Tool-facing context keys (stable vocabulary).
    Backward compat: if disease exists but condition missing, still emit "condition".
    """
    keys: list[str] = []

    # condition (primary) / disease (fallback)
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
    If runner wrote swap metadata, we treat it as a strong signal that
    selection SHOULD depend on context even without LLM.
    """
    ex = _get_extra(card)
    for k in ["context_swap_to", "context_swap_from", "context_swap_to_cancer"]:
        v = ex.get(k, None)
        if isinstance(v, str) and v.strip():
            return True
        if v not in (None, "", [], {}):
            return True
    # Backward compat key (if you used a different name earlier)
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
_CONTEXT_GATE_ENV = "LLMPATH_CONTEXT_GATE_MODE"  # off|note|hard (default off)


def _context_gate_mode(card: SampleCard) -> str:
    """
    Selection-time gating based on context proxy score.
    Values:
      - off: no gate
      - note: annotate only
      - hard: treat context_score==0 as ineligible (ONLY when proxy is enabled)
    Priority: env > card.extra > default off
    """
    env = str(os.environ.get(_CONTEXT_GATE_ENV, "")).strip().lower()
    if env:
        return env if env in {"off", "note", "hard"} else "off"
    ex = _get_extra(card)
    v = str(ex.get("context_gate_mode", "")).strip().lower()
    return v if v in {"off", "note", "hard"} else "off"


def _select_context_mode(card: SampleCard) -> str:
    """
    Selection-time context usage (deterministic proxy).
    Priority: env > auto(context_swap) > card.extra > legacy knob.

    Accepted values:
      - off
      - proxy
      - on/true/yes (treated as proxy)

    Key design:
      - Default remains OFF for normal runs.
      - For context_swap cards, auto-enable proxy so the ablation is meaningful.
    """
    env = str(os.environ.get(_SELECT_CONTEXT_ENV, "")).strip().lower()
    if env:
        if env in {"proxy", "on", "true", "yes", "y", "1"}:
            return "proxy"
        return "off"

    # Auto-enable for paper ablation cards
    if _is_context_swap(card):
        return "proxy"

    ex = _get_extra(card)
    v = str(ex.get("select_context_mode", "")).strip().lower()
    if v in {"proxy", "on", "true", "yes", "y", "1"}:
        return "proxy"
    if v in {"off", "disable", "disabled", "none"}:
        return "off"

    # legacy knob: enable_context_score_proxy(default=False)
    try:
        legacy = bool(card.enable_context_score_proxy(default=False))
        return "proxy" if legacy else "off"
    except Exception:
        return "off"


def _context_tiebreak_int(card: SampleCard, term_uid: str) -> int:
    """
    Deterministic tie-breaker that MUST change when condition changes.

    NOTE:
      - Deterministic (no RNG)
      - Uses condition (fallback disease) only, so context_swap reliably changes ordering
    """
    cond = _card_text(card, "condition", "disease").strip().lower()
    tu = str(term_uid).strip()
    payload = f"{cond}|{tu}".encode()
    h = hashlib.sha256(payload).hexdigest()[:8]  # 32-bit is enough
    return int(h, 16)


def _context_signature(card: SampleCard) -> str:
    """
    Compact signature of the context used for deterministic proxy.
    Useful to prove context_swap actually changed what selection saw.
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
# Context relevance review (LLM probe + cache)
# ===========================
_CONTEXT_ALLOWED = {"PASS", "WARN", "FAIL"}
_CONTEXT_REVIEW_ENV = "LLMPATH_CONTEXT_REVIEW_MODE"  # off|llm (default off)
_CONTEXT_REVIEW_VERSION = "context_review_v1"  # internal only (do not emit by default)


def _context_review_mode(card: SampleCard) -> str:
    # Priority: env > card.extra > default off
    env = str(os.environ.get(_CONTEXT_REVIEW_ENV, "")).strip().lower()
    if env:
        return env if env in {"off", "llm"} else "off"
    ex = _get_extra(card)
    v = str(ex.get("context_review_mode", "")).strip().lower()
    return v if v in {"off", "llm"} else "off"


def _context_review_cache_path(outdir: str | None) -> Path | None:
    # IMPORTANT: avoid any filesystem work unless explicitly enabled by ctx_review==llm
    if not outdir:
        return None
    p = Path(str(outdir)).resolve() / "context_review_cache.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _json_sanitize(obj: Any) -> Any:
    """
    Convert obj into a JSON-serializable structure with deterministic handling.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            ks = str(k)
            out[ks] = _json_sanitize(v)
        return out

    if isinstance(obj, set):
        items = [_json_sanitize(x) for x in obj]
        return sorted(items, key=lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True))

    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(x) for x in obj]

    if isinstance(obj, pd.Series):
        return _json_sanitize(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize(obj.to_dict(orient="records"))

    return str(obj)


def _stable_json_dumps(obj: Any) -> str:
    safe = _json_sanitize(obj)
    return json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _context_review_key(
    *,
    card: SampleCard,
    term_uid: str,
    term_name: str,
    source: str,
    gene_ids: list[str],
    direction: str = "na",
) -> str:
    payload = {
        "condition": _card_text(card, "condition", "disease").strip().lower(),
        "tissue": _card_text(card, "tissue", None).strip().lower(),
        "perturbation": _card_text(card, "perturbation", None).strip().lower(),
        "comparison": _card_text(card, "comparison", None).strip().lower(),
        "term_uid": str(term_uid).strip(),
        "term_name": str(term_name).strip().lower(),
        "source": str(source).strip().lower(),
        "direction": str(direction).strip().lower(),
        "gene_ids": [str(g).strip().upper() for g in gene_ids if str(g).strip()],
        "version": _CONTEXT_REVIEW_VERSION,
    }
    return hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()[:16]


def _load_context_cache(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or (not path.exists()):
        return {}
    out: dict[str, dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                k = str(rec.get("key", "")).strip()
                v = rec.get("value", None)
                if k and isinstance(v, dict):
                    out[k] = v
    except Exception:
        return {}
    return out


def _append_context_cache(path: Path | None, *, key: str, value: dict[str, Any]) -> None:
    if path is None:
        return
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(_stable_json_dumps({"key": key, "value": value}) + "\n")
    except Exception:
        return


def _extract_json_obj_from_text(text: str) -> Any:
    s = (text or "").strip()
    if not s:
        raise ValueError("empty model output")

    try:
        return json.loads(s)
    except Exception:
        pass

    start = -1
    open_ch = ""
    for i, ch in enumerate(s):
        if ch == "{":
            start = i
            open_ch = "{"
            break
        if ch == "[":
            start = i
            open_ch = "["
            break
    if start < 0:
        raise ValueError(f"no JSON object/array start found (first 200 chars): {s[:200]!r}")

    close_ch = "}" if open_ch == "{" else "]"

    depth = 0
    in_str = False
    esc = False
    for j in range(start, len(s)):
        ch = s[j]
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch == open_ch:
            depth += 1
            continue
        if ch == close_ch:
            depth -= 1
            if depth == 0:
                cand = s[start : j + 1].strip()
                return json.loads(cand)

    raise ValueError("unterminated JSON (no matching closing brace/bracket found)")


def _call_backend_text(
    fn: Any, *, prompt: str, seed: int | None, json_mode: bool | None = None
) -> str:
    if json_mode is not None:
        try:
            return str(fn(prompt=prompt, seed=seed, json_mode=bool(json_mode)))
        except TypeError:
            pass
        try:
            return str(fn(prompt=prompt, json_mode=bool(json_mode)))
        except TypeError:
            pass

    try:
        return str(fn(prompt=prompt, seed=seed))
    except TypeError:
        pass
    try:
        return str(fn(prompt=prompt))
    except TypeError:
        pass

    if json_mode is not None:
        try:
            return str(fn(prompt, bool(json_mode)))
        except TypeError:
            pass
    try:
        return str(fn(prompt))
    except TypeError as e:
        raise e


def _is_soft_error_obj(obj: Any) -> tuple[bool, str]:
    if not isinstance(obj, dict):
        return (False, "")
    err = obj.get("error")
    if isinstance(err, dict):
        msg = err.get("message", "")
        if isinstance(msg, str) and msg.strip():
            return (True, msg.strip())
    return (False, "")


def _backend_call_json(backend: BaseLLMBackend, *, prompt: str, seed: int | None) -> dict[str, Any]:
    last_err: Exception | None = None

    gen = getattr(backend, "generate", None)
    if callable(gen):
        try:
            raw = _call_backend_text(gen, prompt=prompt, seed=seed, json_mode=True)
            obj0 = _extract_json_obj_from_text(raw)
            if isinstance(obj0, dict):
                is_err, msg = _is_soft_error_obj(obj0)
                if is_err:
                    raise RuntimeError(f"backend soft-error JSON: {msg}")
                return obj0
            raise ValueError(
                f"backend generate(json_mode=True) returned non-object type={type(obj0).__name__}"
            )
        except Exception as e:
            last_err = e

    candidates_json = [
        "complete_json",
        "chat_json",
        "generate_json",
        "call_json",
        "run_json",
        "json",
    ]
    for name in candidates_json:
        fn = getattr(backend, name, None)
        if fn is None or (not callable(fn)):
            continue
        try:
            out = fn(prompt=prompt, seed=seed)  # type: ignore[misc]
            if isinstance(out, dict):
                is_err, msg = _is_soft_error_obj(out)
                if is_err:
                    raise RuntimeError(f"backend soft-error JSON: {msg}")
                return out
            if isinstance(out, str):
                obj = _extract_json_obj_from_text(out)
                if isinstance(obj, dict):
                    is_err, msg = _is_soft_error_obj(obj)
                    if is_err:
                        raise RuntimeError(f"backend soft-error JSON: {msg}")
                    return obj
                raise ValueError(f"backend JSON returned non-object type={type(obj).__name__}")
            obj2 = _extract_json_obj_from_text(str(out))
            if isinstance(obj2, dict):
                is_err, msg = _is_soft_error_obj(obj2)
                if is_err:
                    raise RuntimeError(f"backend soft-error JSON: {msg}")
                return obj2
            raise ValueError(f"backend JSON returned non-object type={type(obj2).__name__}")
        except Exception as e:
            last_err = e
            continue

    candidates_text = ["chat", "complete", "call", "run", "__call__"]
    for name in candidates_text:
        fn = getattr(backend, name, None)
        if fn is None or (not callable(fn)):
            continue
        try:
            raw = _call_backend_text(fn, prompt=prompt, seed=seed, json_mode=True)
            obj = _extract_json_obj_from_text(raw)
            if isinstance(obj, dict):
                is_err, msg = _is_soft_error_obj(obj)
                if is_err:
                    raise RuntimeError(f"backend soft-error JSON: {msg}")
                return obj
            raise ValueError(f"backend text parsed non-object type={type(obj).__name__}")
        except Exception as e:
            last_err = e
            continue

    err_msg = ""
    if last_err is not None:
        err_msg = f"{type(last_err).__name__}: {str(last_err)[:200]}"
    raise RuntimeError(
        "context_review: backend JSON call failed "
        f"(tried generate(json_mode=True), json={candidates_json}, text={candidates_text}). "
        f"last_error={err_msg}"
    ) from last_err


def _context_review_prompt(
    *,
    card: SampleCard,
    term_name: str,
    term_id: str,
    source: str,
    gene_ids: list[str],
    direction: str = "na",
    comparison: str = "",
) -> str:
    ctx = {
        "condition": _card_text(card, "condition", "disease"),
        "tissue": _card_text(card, "tissue", None),
        "perturbation": _card_text(card, "perturbation", None),
        "comparison": _card_text(card, "comparison", None),
    }
    g = [str(x).strip().upper() for x in gene_ids if str(x).strip()]
    payload = {
        "context": ctx,
        "pathway_term": {
            "term_name": str(term_name).strip(),
            "term_id": str(term_id).strip(),
            "source": str(source).strip(),
        },
        "claim": {
            "direction": str(direction).strip().lower(),
            "comparison": str(comparison).strip(),
        },
        "evidence_gene_ids": g[:10],
        "task": (
            "Judge whether this pathway term is context-relevant AND whether the claimed "
            "direction (up/down/na) seems consistent with the evidence genes and the comparison. "
            "Do NOT use external knowledge. If direction cannot be assessed from the given inputs, "
            "use WARN (not FAIL). Return JSON ONLY with keys: context_status (PASS|WARN|FAIL), "
            "context_reason (short code), context_notes (<=200 chars), confidence (0-1)."
        ),
        "output_schema": {
            "context_status": "PASS|WARN|FAIL",
            "context_reason": "string",
            "context_notes": "string",
            "confidence": "number",
        },
    }
    return (
        "You are a careful biomedical pathway reviewer. "
        "Do not invent evidence. If uncertain, use WARN.\n"
        "Return JSON only.\n\n" + _stable_json_dumps(payload)
    )


def _normalize_context_review(obj: dict[str, Any]) -> dict[str, Any]:
    st = str(obj.get("context_status", "")).strip().upper()
    if st not in _CONTEXT_ALLOWED:
        st = "WARN"
    reason = str(obj.get("context_reason", "")).strip()
    notes = str(obj.get("context_notes", "")).strip()
    try:
        conf = float(obj.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    conf = min(max(conf, 0.0), 1.0)
    return {
        "context_status": st,
        "context_reason": reason or ("ok" if st == "PASS" else "uncertain"),
        "context_notes": notes[:200],
        "context_confidence": conf,
    }


def _patch_claim_json_with_context(
    claim_json: Any,
    *,
    status: str,
    reason: str,
    notes: str,
    confidence: float,
    method: str = "llm",
    dev_meta: dict[str, Any] | None = None,
) -> str:
    try:
        obj = json.loads(claim_json) if isinstance(claim_json, str) else dict(claim_json)
    except Exception:
        obj = {}

    obj["context_evaluated"] = True
    obj["context_method"] = str(method)
    st = str(status).strip().upper()
    obj["context_status"] = st if st in _CONTEXT_ALLOWED else "WARN"
    obj["context_reason"] = str(reason).strip() if reason is not None else None
    obj["context_notes"] = str(notes).strip()[:200] if notes is not None else None

    try:
        obj["context_confidence"] = float(confidence)
    except Exception:
        obj["context_confidence"] = 0.0

    if dev_meta and isinstance(dev_meta, dict):
        for k, v in dev_meta.items():
            obj[k] = v

    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _apply_context_review_llm(
    df_claims: pd.DataFrame,
    *,
    card: SampleCard,
    backend: BaseLLMBackend,
    seed: int | None,
    outdir: str | None,
) -> pd.DataFrame:
    cache_path = _context_review_cache_path(outdir)
    cache = _load_context_cache(cache_path)
    emit_dev = _emit_dev_meta(card)

    out = df_claims.copy()

    for col, default in [
        ("context_review_evaluated", False),
        ("context_review_status", pd.NA),
        ("context_review_reason", pd.NA),
        ("context_review_notes", pd.NA),
        ("context_review_confidence", pd.NA),
        ("context_review_mode", "off"),
        ("context_review_method", pd.NA),
    ]:
        if col not in out.columns:
            out[col] = default

    if emit_dev:
        for col, default in [
            ("context_review_key", pd.NA),
            ("context_review_version", _CONTEXT_REVIEW_VERSION),
            ("context_review_error", pd.NA),
        ]:
            if col not in out.columns:
                out[col] = default

    for idx, r in out.iterrows():
        term_uid = str(r.get("term_uid", "")).strip()
        term_name = str(r.get("term_name", "")).strip()
        term_id = str(r.get("term_id", "")).strip()
        source = str(r.get("source", "")).strip()

        genes = _as_gene_list(r.get("gene_ids", ""))
        genes = [str(g).strip().upper() for g in genes if str(g).strip()]
        direction = str(r.get("direction", "na")).strip().lower()

        key = _context_review_key(
            card=card,
            term_uid=term_uid,
            term_name=term_name,
            source=source,
            gene_ids=genes[:10],
            direction=direction,
        )

        out.at[idx, "context_review_mode"] = "llm"
        out.at[idx, "context_review_method"] = "llm"
        if emit_dev:
            out.at[idx, "context_review_key"] = key
            out.at[idx, "context_review_version"] = _CONTEXT_REVIEW_VERSION
            out.at[idx, "context_review_error"] = pd.NA

        norm: dict[str, Any] | None = None
        from_cache = False
        if key in cache and isinstance(cache.get(key), dict):
            norm = cache[key]
            from_cache = True

        if norm is None:
            prompt = _context_review_prompt(
                card=card,
                term_name=term_name,
                term_id=term_id,
                source=source,
                gene_ids=genes[:10],
                direction=direction,
                comparison=_card_text(card, "comparison", None),
            )
            try:
                raw = _backend_call_json(backend, prompt=prompt, seed=seed)
                norm = _normalize_context_review(raw)
                cache[key] = norm
                _append_context_cache(cache_path, key=key, value=norm)
            except Exception as e:
                msg = f"{type(e).__name__}: {str(e)[:180]}"
                norm = {
                    "context_status": "WARN",
                    "context_reason": "backend_error",
                    "context_notes": msg,
                    "context_confidence": 0.0,
                }
                if emit_dev:
                    out.at[idx, "context_review_error"] = msg

        st = str(norm.get("context_status", "WARN")).strip().upper()
        rsn = str(norm.get("context_reason", "uncertain")).strip()
        nts = str(norm.get("context_notes", "")).strip()
        try:
            conf = float(norm.get("context_confidence", 0.0))
        except Exception:
            conf = 0.0

        out.at[idx, "context_review_evaluated"] = True
        out.at[idx, "context_review_status"] = st
        out.at[idx, "context_review_reason"] = rsn
        out.at[idx, "context_review_notes"] = nts
        out.at[idx, "context_review_confidence"] = conf

        if "claim_json" in out.columns:
            cj = out.at[idx, "claim_json"]
            if not _is_na_scalar(cj) and str(cj).strip():
                dev_meta = None
                if emit_dev:
                    dev_meta = {
                        "context_review_key": str(key),
                        "context_review_version": _CONTEXT_REVIEW_VERSION,
                        "context_review_cache_hit": bool(from_cache),
                    }
                out.at[idx, "claim_json"] = _patch_claim_json_with_context(
                    cj,
                    status=st,
                    reason=rsn,
                    notes=nts,
                    confidence=conf,
                    method="llm",
                    dev_meta=dev_meta,
                )

    return out


# ===========================
# Stress suite (identity collapse) - PROBE
# ===========================
def _stress_enabled(card: SampleCard) -> bool:
    env = str(os.environ.get("LLMPATH_STRESS_GENERATE", "")).strip().lower()
    if env:
        return env in {"1", "true", "t", "yes", "y", "on"}
    ex = _get_extra(card)
    return _as_bool(ex.get("stress_generate", None), False)


def _module_prefix(card: SampleCard) -> str:
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

    base_hash_calc = _hash_gene_set_audit(sorted(list(base_genes)))
    if base_hash_calc != base_hash_claim:
        return {
            "stress_status": "WARN",
            "stress_ok": False,
            "stress_reason": "stress_gene_set_hash_mismatch",
            "stress_notes": f"claim_hash={base_hash_claim} != baseline_hash={base_hash_calc}",
        }

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

    if sel_ctx_mode == "proxy":
        toks = _context_tokens(card)
        df["context_tokens_n"] = int(len(toks))
        df["context_score"] = df["term_name"].map(lambda s: _context_score(str(s), toks))
        df["context_evaluated"] = True

        # Always compute condition-dependent deterministic tiebreak when proxy is enabled
        df["context_tiebreak"] = df["term_uid"].map(lambda tu: _context_tiebreak_int(card, str(tu)))
        df["context_tiebreak_sort"] = pd.to_numeric(df["context_tiebreak"], errors="coerce").fillna(
            2**31 - 1
        )
    else:
        df["context_tokens_n"] = 0
        df["context_score"] = pd.NA
        df["context_evaluated"] = False
        df["context_tiebreak"] = pd.NA
        df["context_tiebreak_sort"] = 2**31 - 1  # neutral (worst)

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

    # ---- optional context gate (selection-time) ----
    # Only meaningful when proxy is enabled.
    df["context_gate_mode"] = str(ctx_gate_mode)
    df["context_gate_hit"] = False
    df["eligible_context"] = True
    if sel_ctx_mode == "proxy" and ctx_gate_mode in {"note", "hard"}:
        # gate criterion: at least one token overlap in term_name
        hit = df["context_score_sort"].ge(1)
        df["context_gate_hit"] = hit
        df["eligible_context"] = hit if ctx_gate_mode == "hard" else True
        if ctx_gate_mode == "hard":
            df["eligible"] = df["eligible"] & df["eligible_context"]

    df["term_survival_sort"] = (
        df["term_survival"].fillna(-1.0) if "term_survival" in df.columns else -1.0
    )

    try:
        max_per_module = int(card.max_per_module(default=1))
    except Exception:
        max_per_module = 1
    max_per_module = max(1, max_per_module)

    # ---- Ranking policy ----
    # Default (non-swap): stability-first
    #   eligible -> term_survival -> context_score -> stat -> tiebreak -> term_uid
    #
    # context_swap (swap card detected): enforce context sensitivity
    # even when context_score ties at 0
    #   eligible -> term_survival -> context_score -> tiebreak -> stat -> term_uid
    #
    # Rationale:
    #   - This is an ablation; the whole point is that changing context changes
    #     which representatives you pick.
    #   - The tiebreak depends on condition and will therefore change under
    #     context_swap deterministically.
    if sel_ctx_mode == "proxy" and ctx_swap_active:
        sort_cols = [
            "eligible",
            "term_survival_sort",
            "context_score_sort",
            "context_tiebreak_sort",
            "stat",
            "term_uid",
        ]
        ascending = [False, False, False, True, False, True]
        df["select_context_policy"] = "proxy_swap_strong"
    else:
        sort_cols = [
            "eligible",
            "term_survival_sort",
            "context_score_sort",
            "stat",
            "context_tiebreak_sort",
            "term_uid",
        ]
        ascending = [False, False, False, False, True, True]
        df["select_context_policy"] = "default"

    df_ranked = df.sort_values(sort_cols, ascending=ascending).copy()

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

    n_total = int(df_ranked.shape[0])
    n_eligible = int(df_ranked["eligible"].sum()) if "eligible" in df_ranked.columns else n_total
    n_ineligible = int(n_total - n_eligible)
    n_picked = int(len(picked_idx))

    # Context diag summary
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

    mprefix = _module_prefix(card)
    max_gene_ids = _max_gene_ids_in_claim(card)

    for _, r in df_pick.iterrows():
        term_id = str(r["term_id"]).strip()
        term_name = str(r["term_name"]).strip()
        direction = _norm_direction(r.get("direction", "na"))
        source = str(r["source"]).strip()

        term_uid = str(r.get("term_uid") or f"{source}:{term_id}").strip()

        # FULL genes for claim_json contract (bounded, deduped, normalized)
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
            content_hash = _module_hash_like_modules_py([term_uid], genes_claim)
            module_id = f"{mprefix}{content_hash}"
            module_reason = "missing_module_id"
            module_missing = True

        # IMPORTANT: gene_set_hash must match genes_claim stored in claim_json
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
            "gene_ids": ",".join([str(g) for g in genes_suggest]),
            "gene_ids_suggest": ",".join([str(g) for g in genes_suggest]),
            "gene_ids_n_total": int(len(genes_full_norm)),
            "gene_ids_n_in_claim": int(len(genes_claim)),
            "term_ids": ",".join(claim.evidence_ref.term_ids),
            "gene_set_hash": claim.evidence_ref.gene_set_hash,
            "context_score": r.get("context_score", pd.NA),
            "context_evaluated": bool(r.get("context_evaluated", False)),
            "context_tokens_n": int(r.get("context_tokens_n", 0) or 0),
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
            "context_tiebreak": r.get("context_tiebreak", pd.NA),
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

    NOTE: context review (LLM) is a PROBE and is opt-in (off by default).

    IMPORTANT:
      - claim_json embeds evidence_ref.gene_ids and gene_set_hash; they MUST be consistent.
      - user-facing columns may show a compact gene snippet (gene_ids_suggest) for readability.
    """
    mode_eff = _resolve_mode(card, mode)
    k_eff = _validate_k(k)
    ctx_review = _context_review_mode(card)
    _dlog(
        f"[select] ctx_review={ctx_review} mode_eff={mode_eff} backend={'yes' if backend else 'no'}"
    )

    # ---- LLM mode ----
    if mode_eff == "llm":
        if backend is None:
            out_det = _select_claims_deterministic(distilled, card, k=int(k_eff), seed=seed)
            out_det["claim_mode"] = "deterministic"
            out_det["llm_notes"] = "llm requested but backend=None; deterministic used"
            if ctx_review == "llm":
                out_det["context_review_mode"] = "off"
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
            if ctx_review == "llm":
                out_llm = _apply_context_review_llm(
                    out_llm, card=card, backend=backend, seed=seed, outdir=outdir
                )
            return out_llm

        out_det = _select_claims_deterministic(distilled, card, k=int(k_eff), seed=seed)
        out_det["claim_mode"] = "deterministic_fallback"
        out_det["llm_notes"] = notes or "llm returned fallback/empty; deterministic used"
        if ctx_review == "llm":
            out_det = _apply_context_review_llm(
                out_det, card=card, backend=backend, seed=seed, outdir=outdir
            )
        return out_det

    # ---- Deterministic ----
    out_det = _select_claims_deterministic(distilled, card, k=int(k_eff), seed=seed)
    out_det["claim_mode"] = "deterministic"
    out_det["llm_notes"] = ""

    if ctx_review == "llm" and backend is not None:
        out_det = _apply_context_review_llm(
            out_det, card=card, backend=backend, seed=seed, outdir=outdir
        )

    return out_det
