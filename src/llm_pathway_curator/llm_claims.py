# LLM-PathwayCurator/src/llm_pathway_curator/llm_claims.py

"""
LLM-based claim proposal for LLM-PathwayCurator.

This module proposes structured `Claim` objects from distilled evidence using
an LLM backend. It is designed to be:
- contract-driven (stable IDs, deterministic evidence linking),
- robust across heterogeneous backends (OpenAI/Gemini/Ollama/local),
- audit-grade (persist prompt/candidates/raw/meta artifacts).

Key ideas
---------
- Evidence identity is tool-owned (term_uid + gene_set_hash).
- Context VALUES are prompt-facing; context KEYS are contract-facing.
- FAIL decisions are never "promoted" by thresholding; gating affects non-FAIL.

Notes
-----
This file contains many private helpers. Public entrypoints:
- propose_claims_llm
- claims_to_proposed_tsv
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from . import _shared
from .backends import BaseLLMBackend
from .claim_schema import Claim
from .sample_card import SampleCard

_NA_TOKENS = set(_shared.NA_TOKENS)
_NA_TOKENS_L = _shared.NA_TOKENS_L


def _is_na_scalar(x: Any) -> bool:
    """
    Check scalar NA safely (single source of truth).

    Parameters
    ----------
    x : Any
        Input value.

    Returns
    -------
    bool
        True if `x` should be treated as scalar NA.

    Notes
    -----
    Delegates to `_shared.is_na_scalar` to avoid behavior drift across layers.
    """
    return _shared.is_na_scalar(x)


def _strip_na(s: Any) -> str:
    """
    Convert an arbitrary value to a non-NA trimmed string.

    Parameters
    ----------
    s : Any
        Input value.

    Returns
    -------
    str
        Trimmed string, or "" if NA-like.

    Notes
    -----
    NA vocabulary follows `_shared.NA_TOKENS_L`.
    """
    if _is_na_scalar(s):
        return ""
    t = str(s).strip()
    if not t or t.lower() in _NA_TOKENS_L:
        return ""
    return t


def _looks_like_12hex(s: Any) -> bool:
    """
    Check whether a value is exactly 12 lowercase hex characters.

    Parameters
    ----------
    s : Any
        Input value.

    Returns
    -------
    bool
        True if `s` is a 12-hex token; otherwise False.

    Notes
    -----
    This is a local mirror used for lightweight checks. The spec-level
    equivalent is `_shared.looks_like_12hex`.
    """
    if _is_na_scalar(s):
        return False
    x = str(s).strip().lower()
    if len(x) != 12:
        return False
    return all(ch in "0123456789abcdef" for ch in x)


def _sha256_text(s: str) -> str:
    """
    Compute a SHA-256 hex digest of a text payload.

    Parameters
    ----------
    s : str
        Input text.

    Returns
    -------
    str
        64-character lowercase hex digest.
    """
    h = hashlib.sha256()
    h.update((s or "").encode("utf-8"))
    return h.hexdigest()


def _stable_json_dumps(obj: Any) -> str:
    """
    Serialize an object to deterministic JSON.

    Parameters
    ----------
    obj : Any
        JSON-serializable object.

    Returns
    -------
    str
        Deterministic JSON string.

    Notes
    -----
    Uses:
    - sort_keys=True
    - separators=(",", ":")
    - ensure_ascii=False
    """
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _df_records_sha256(df: pd.DataFrame) -> str:
    """
    Compute a stable SHA-256 fingerprint for a DataFrame (records payload).

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    str
        SHA-256 hex digest of `df.to_dict(orient="records")` serialized via
        deterministic JSON.

    Notes
    -----
    Intended for provenance/debugging (not a spec-level ID).
    """
    recs = df.to_dict(orient="records")
    return _sha256_text(_stable_json_dumps(recs))


def _parse_soft_error_json(s: str) -> dict[str, Any] | None:
    """
    Parse standardized soft-error JSON emitted by some backends.

    Parameters
    ----------
    s : str
        Input text.

    Returns
    -------
    dict[str, Any] or None
        Parsed object if it matches {"error": {...}}, otherwise None.
    """
    try:
        obj = json.loads(s)
    except Exception:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("error"), dict):
        return obj
    return None


def _safe_write_json(path: Path, obj: Any) -> None:
    """
    Atomically write JSON to disk.

    Parameters
    ----------
    path : pathlib.Path
        Target file path.
    obj : Any
        JSON-serializable payload.

    Notes
    -----
    Writes to a temporary file (suffix ".tmp") and then replaces the target
    to reduce partial-write risk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def _context_dict(card: SampleCard) -> dict[str, str]:
    """
    Build prompt-facing context dictionary from a SampleCard.

    Parameters
    ----------
    card : SampleCard
        Input sample card.

    Returns
    -------
    dict[str, str]
        Mapping of context keys to cleaned string values.

    Notes
    -----
    Values are for prompting only and MUST NOT be used for claim identity
    (claim_id / gene_set_hash).
    """
    d = card.context_dict()
    return {k: _strip_na(v) for k, v in d.items()}


def _context_keys(card: SampleCard) -> list[str]:
    """
    Return contract-facing context keys in stable order.

    Parameters
    ----------
    card : SampleCard
        Input sample card.

    Returns
    -------
    list[str]
        Ordered subset of ["condition", "tissue", "perturbation", "comparison"].

    Notes
    -----
    Stable key order is treated as part of the contract.
    """
    keys: list[str] = []
    d = card.context_dict()
    for k in ["condition", "tissue", "perturbation", "comparison"]:
        if _strip_na(d.get(k, "")):
            keys.append(k)
    return keys


def _norm_gene_id(g: Any) -> str:
    """
    Normalize a gene token using the spec policy.

    Parameters
    ----------
    g : Any
        Gene-like token.

    Returns
    -------
    str
        Cleaned token.

    Notes
    -----
    Delegates to `_shared.clean_gene_token` (trim-only; no forced uppercase).
    """
    return _shared.clean_gene_token(g)


_BRACKETED_LIST_RE = re.compile(r"^\s*[\[\(\{].*[\]\)\}]\s*$")


def _parse_gene_list(x: Any) -> list[str]:
    """
    Parse a gene field into a cleaned list of gene tokens.

    Parameters
    ----------
    x : Any
        Scalar or list-like gene field.

    Returns
    -------
    list[str]
        Cleaned gene tokens.

    Notes
    -----
    Delegates to `_shared.parse_genes` for conservative splitting and
    deterministic de-duplication.
    """
    return _shared.parse_genes(x)


def _hash_gene_set_12hex(genes: list[str]) -> str:
    """
    Compute a 12-hex fingerprint for a gene set.

    Parameters
    ----------
    genes : list[str]
        Gene tokens (order-insensitive).

    Returns
    -------
    str
        12-character lowercase hex fingerprint.

    Notes
    -----
    Delegates to `_shared.hash_gene_set_12hex` for deterministic hashing.
    """
    return _shared.hash_gene_set_12hex(list(genes or []))


def _hash_term_set_fallback_12hex(term_uids: list[str]) -> str:
    """
    Fallback fingerprint when gene IDs are unavailable.

    Parameters
    ----------
    term_uids : list[str]
        Term UID tokens.

    Returns
    -------
    str
        12-character lowercase hex fingerprint.

    Notes
    -----
    Delegates to `_shared.hash_set_12hex`. This is term-driven, not gene-driven.
    """
    return _shared.hash_set_12hex(list(term_uids or []))


def _soft_error_json(message: str, *, err_type: str = "llm_error", retryable: bool = False) -> str:
    """
    Create standardized soft-error JSON text.

    Parameters
    ----------
    message : str
        Error message.
    err_type : str, optional
        Error type token.
    retryable : bool, optional
        Whether the failure is considered retryable.

    Returns
    -------
    str
        JSON string formatted as {"error": {"message": ..., "type": ..., "retryable": ...}}.
    """
    return json.dumps({"error": {"message": message, "type": err_type, "retryable": retryable}})


def _as_bool_env(name: str, default: bool) -> bool:
    """
    Parse a boolean environment variable.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : bool
        Default value if unset/empty.

    Returns
    -------
    bool
        Parsed boolean value.

    Notes
    -----
    Truthy values: {"1","true","t","yes","y","on"}.
    """
    v = str(os.environ.get(name, "")).strip().lower()
    if not v:
        return bool(default)
    return v in {"1", "true", "t", "yes", "y", "on"}


def _as_int_env(name: str, default: int) -> int:
    """
    Parse an integer environment variable with fallback.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : int
        Default value if unset/empty/malformed.

    Returns
    -------
    int
        Parsed integer value, or `default` on failure.
    """
    try:
        v = str(os.environ.get(name, "")).strip()
        return int(v) if v else int(default)
    except Exception:
        return int(default)


def _module_prefix(card: SampleCard) -> str:
    """
    Resolve module_id prefix for fallback module IDs.

    Parameters
    ----------
    card : SampleCard
        Input sample card (may provide extra["module_prefix"]).

    Returns
    -------
    str
        Module prefix string.

    Notes
    -----
    Precedence:
    1) env LLMPATH_MODULE_PREFIX
    2) card.extra["module_prefix"]
    3) "M"
    """
    env = str(os.environ.get("LLMPATH_MODULE_PREFIX", "")).strip()
    if env:
        return env
    ex = getattr(card, "extra", {}) or {}
    if isinstance(ex, dict):
        p = str(ex.get("module_prefix", "")).strip()
        if p:
            return p
    return "M"


def _max_gene_ids_in_claim(card: SampleCard) -> int:
    """
    Determine maximum number of gene IDs to embed in a claim payload.

    Parameters
    ----------
    card : SampleCard
        Input sample card (may provide extra["max_gene_ids_in_claim"]).

    Returns
    -------
    int
        Maximum gene ID count (clamped to [1, 5000]).

    Notes
    -----
    Precedence:
    1) env LLMPATH_MAX_GENE_IDS_IN_CLAIM (default 512)
    2) card.extra["max_gene_ids_in_claim"]
    3) 512
    """
    env = _as_int_env("LLMPATH_MAX_GENE_IDS_IN_CLAIM", 512)
    if env:
        return max(1, min(5000, int(env)))
    ex = getattr(card, "extra", {}) or {}
    if isinstance(ex, dict):
        try:
            v = int(ex.get("max_gene_ids_in_claim", 512))
            return max(1, min(5000, v))
        except Exception:
            pass
    return 512


def _module_hash_like_modules_py_12hex(term_uids: list[str], genes: list[str]) -> str:
    """
    Compute module content hash consistent with modules.py.

    Parameters
    ----------
    term_uids : list[str]
        Term UID tokens.
    genes : list[str]
        Gene tokens.

    Returns
    -------
    str
        12-character lowercase hex fingerprint.

    Notes
    -----
    Delegates to `_shared.module_hash_content12` for cross-module consistency.
    """
    return _shared.module_hash_content12(terms=list(term_uids or []), genes=list(genes or []))


def _strict_k_enabled() -> bool:
    """
    Check whether strict-k output is required from the LLM.

    Returns
    -------
    bool
        True if strict-k is enabled.

    Notes
    -----
    Controlled by env LLMPATH_LLM_STRICT_K.
    Default is False for backward compatibility.
    """
    return _as_bool_env("LLMPATH_LLM_STRICT_K", False)


def _scalar_int(x: Any, default: int | None = None) -> int | None:
    """
    Convert a value to int if possible, with a tuple-unpacking guard.

    Parameters
    ----------
    x : Any
        Input value.
    default : int or None, optional
        Fallback if conversion fails or input is None.

    Returns
    -------
    int or None
        Parsed integer, or `default`.

    Notes
    -----
    Special-case: (v,) -> v to mitigate common trailing-comma packing mistakes.
    """
    if x is None:
        return default
    if isinstance(x, tuple) and len(x) == 1:
        x = x[0]
    try:
        return int(x)
    except Exception:
        return default


def _clamp_int(x: int, lo: int, hi: int) -> int:
    """
    Clamp an integer to an inclusive range.

    Parameters
    ----------
    x : int
        Input value.
    lo : int
        Lower bound.
    hi : int
        Upper bound.

    Returns
    -------
    int
        Clamped value.
    """
    return max(int(lo), min(int(hi), int(x)))


def _backend_model_hint(backend: BaseLLMBackend) -> str:
    """
    Get a best-effort model name hint from a backend.

    Parameters
    ----------
    backend : BaseLLMBackend
        LLM backend instance.

    Returns
    -------
    str
        Model hint string, or "" if unavailable.
    """
    for attr in ("model_name", "model", "model_id"):
        v = getattr(backend, attr, None)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def _is_small_local_model(backend: BaseLLMBackend) -> bool:
    """
    Heuristically detect "small" local models to reduce prompt budget.

    Parameters
    ----------
    backend : BaseLLMBackend
        LLM backend instance.

    Returns
    -------
    bool
        True if treated as small (e.g., 7B/8B).

    Notes
    -----
    Override:
    - env LLMPATH_LLM_ASSUME_SMALL_MODEL=1 forces True.
    """
    if _as_bool_env("LLMPATH_LLM_ASSUME_SMALL_MODEL", False):
        return True

    hint = _backend_model_hint(backend).lower()
    if not hint:
        return False
    # common patterns: llama3.1:8b, 8b, 7b, q4_0 etc (keep conservative)
    return ("8b" in hint) or ("7b" in hint)


def _auto_topn_for_k(*, k: int, backend: BaseLLMBackend) -> tuple[int, dict[str, Any]]:
    """
    Auto-tune candidate pool size (top_n) from k.

    Parameters
    ----------
    k : int
        Target number of claims to return.
    backend : BaseLLMBackend
        Backend instance used to infer small-model heuristics.

    Returns
    -------
    (int, dict[str, Any])
        - top_n : int
          Auto-tuned candidate budget.
        - meta : dict[str, Any]
          Explainable parameters and model hint used for tuning.

    Notes
    -----
    Rule:
    - top_n = clamp(alpha*k, min_n, max_n)
    - if small model, use a smaller effective max
    """
    k0 = max(1, int(k))

    alpha = _as_int_env("LLMPATH_LLM_TOPN_ALPHA", 12)
    min_n = _as_int_env("LLMPATH_LLM_TOPN_MIN", 15)
    max_n = _as_int_env("LLMPATH_LLM_TOPN_MAX", 80)
    max_small = _as_int_env("LLMPATH_LLM_TOPN_MAX_SMALL", 50)

    small = _is_small_local_model(backend)
    eff_max = min(max_n, max_small) if small else max_n

    n = _clamp_int(alpha * k0, min_n, eff_max)
    n = max(n, k0)

    meta = {
        "top_n_mode": "auto",
        "top_n_alpha": int(alpha),
        "top_n_min": int(min_n),
        "top_n_max": int(max_n),
        "top_n_max_small": int(max_small),
        "top_n_small_model": bool(small),
        "top_n_model_hint": _backend_model_hint(backend),
    }
    return int(n), meta


def _llm_required_by_contract() -> bool:
    """
    Determine whether LLM failure should be treated as fatal.

    Returns
    -------
    bool
        True if LLM is required by the run contract.

    Notes
    -----
    Contract rules:
    - If LLMPATH_CLAIM_MODE=llm => required.
    - Overrides:
      * LLMPATH_LLM_FAIL_FAST=1 forces required
      * LLMPATH_LLM_FAIL_FAST=0 forces not required
    """
    v = str(os.environ.get("LLMPATH_LLM_FAIL_FAST", "")).strip().lower()
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True

    claim_mode = str(os.environ.get("LLMPATH_CLAIM_MODE", "")).strip().lower()
    return claim_mode == "llm"


def _raise_if_required(notes: str, *, meta: dict[str, Any]) -> None:
    """
    Raise a compact error if the run contract requires an LLM.

    Parameters
    ----------
    notes : str
        Short failure note.
    meta : dict[str, Any]
        Metadata for compact error context (backend, k, top_n, etc.).

    Raises
    ------
    RuntimeError
        If LLM is required and a failure occurred.

    Notes
    -----
    Intended to be called after artifacts are written.
    """
    if not _llm_required_by_contract():
        return
    backend = str(meta.get("backend_class", ""))
    top_n = meta.get("top_n", "")
    k = meta.get("k", "")
    msg = f"LLM required but failed: {notes} (backend={backend} k={k} top_n={top_n})"
    raise RuntimeError(msg)


# -------------------------
# Backend calling (robust across Ollama/OpenAI/Gemini)
# -------------------------
def _try_call(fn: Callable[..., Any], **kwargs: Any) -> Any:
    """
    Call a backend function while avoiding brittle signature coupling.

    Parameters
    ----------
    fn : Callable[..., Any]
        Callable to invoke.
    **kwargs : Any
        Candidate keyword arguments.

    Returns
    -------
    Any
        Backend output.

    Raises
    ------
    RuntimeError
        If all fallback calling strategies fail.

    Notes
    -----
    Strategy:
    - Try kwargs as-is.
    - Then drop kwargs in a safe order (seed -> json_mode -> prompt).
    - Finally, fallback to a single positional prompt call when available.
    """
    try:
        return fn(**kwargs)
    except TypeError:
        pass

    # Drop in a safe order: seed -> json_mode -> prompt
    drop_orders = [
        ["seed"],
        ["json_mode"],
        ["prompt"],
    ]
    keys = dict(kwargs)
    for drop in drop_orders:
        for d in drop:
            keys.pop(d, None)
        try:
            return fn(**keys)
        except TypeError:
            continue

    # final positional fallback: ONLY prompt
    try:
        if "prompt" in kwargs:
            return fn(kwargs.get("prompt"))
    except Exception as e:
        raise RuntimeError(f"backend call failed for {getattr(fn, '__name__', 'fn')}") from e

    raise RuntimeError(f"backend call failed for {getattr(fn, '__name__', 'fn')}")


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json_fragment(text: str) -> str | None:
    """
    Extract a JSON object/array from free-form model output.

    Parameters
    ----------
    text : str
        Model output text.

    Returns
    -------
    str or None
        Extracted JSON fragment if found; otherwise None.

    Notes
    -----
    Handles:
    - fenced blocks: ```json ... ```
    - full-text JSON
    - best-effort scan from the first '{' or '[' using JSONDecoder.raw_decode
    """
    s = (text or "").strip()
    if not s:
        return None

    # Prefer fenced json blocks
    m = _JSON_FENCE_RE.search(s)
    if m:
        cand = (m.group(1) or "").strip()
        if cand:
            return cand

    # If the whole text is JSON, great
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        return s

    # Scan for first { or [ and try to decode from there
    starts = []
    for ch in ("{", "["):
        i = s.find(ch)
        if i >= 0:
            starts.append(i)
    if not starts:
        return None

    decoder = json.JSONDecoder()
    for i in sorted(starts):
        sub = s[i:].lstrip()
        try:
            obj, end = decoder.raw_decode(sub)
            frag = sub[:end].strip()
            if frag:
                return frag
        except Exception:
            continue

    return None


def _backend_call_json(backend: BaseLLMBackend, *, prompt: str, seed: int | None) -> dict[str, Any]:
    """
    Perform a best-effort JSON call across heterogeneous backends.

    Parameters
    ----------
    backend : BaseLLMBackend
        Backend instance.
    prompt : str
        Prompt text.
    seed : int or None
        Optional seed (best-effort; may be ignored by backends).

    Returns
    -------
    dict[str, Any]
        Parsed JSON object returned by the backend.

    Raises
    ------
    RuntimeError
        If no usable backend method is found or JSON decoding fails for all
        candidates.

    Notes
    -----
    Preferred path:
    - backend.generate(prompt=..., json_mode=True, seed=...) returns JSON text
      or a standardized {"error": {...}} object.
    """
    # 0) Prefer BaseLLMBackend contract first

    def _wrap(obj: Any) -> dict[str, Any]:
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list):
            return {"claims": obj}
        raise TypeError("backend JSON must be a dict or a list")

    try:
        if hasattr(backend, "generate") and callable(backend.generate):
            out = _try_call(backend.generate, prompt=prompt, json_mode=True, seed=seed)
            if isinstance(out, dict):
                return _wrap(out)
            if isinstance(out, list):
                return {"claims": out}
            s = out if isinstance(out, str) else str(out)
            frag = _extract_json_fragment(s) or s.strip()
            return json.loads(frag)
    except Exception:
        pass

    # A) explicit json-returning methods (legacy)
    json_candidates = [
        "complete_json",
        "chat_json",
        "generate_json",
        "call_json",
        "run_json",
        "json",
    ]
    last_err: Exception | None = None
    for name in json_candidates:
        fn = getattr(backend, name, None)
        if fn is None or (not callable(fn)):
            continue
        try:
            out = _try_call(fn, prompt=prompt, seed=seed)
            if isinstance(out, dict):
                return _wrap(out)
            if isinstance(out, list):
                return _wrap(out)
            if isinstance(out, str):
                frag = _extract_json_fragment(out) or out.strip()
                return _wrap(json.loads(frag))
        except Exception as e:
            last_err = e
            continue

    # B) text-returning methods (legacy)
    text_candidates = [
        "generate",
        "chat",
        "complete",
        "call",
        "run",
    ]
    for name in text_candidates:
        fn = getattr(backend, name, None)
        if fn is None or (not callable(fn)):
            continue
        try:
            out = _try_call(fn, prompt=prompt, json_mode=True, seed=seed)
            s = out if isinstance(out, str) else str(out)
            frag = _extract_json_fragment(s)
            if frag:
                return _wrap(json.loads(frag))
            return _wrap(json.loads(s.strip()))
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        f"propose_claims_llm: backend has no usable JSON/text method "
        f"(tried json={json_candidates} text={text_candidates})"
    ) from last_err


@dataclass(frozen=True)
class LLMClaimResult:
    """
    Container for LLM claim proposal results.

    Attributes
    ----------
    claims : list[Claim]
        Validated and post-processed claims. Empty if failure/fallback.
    raw_text : str
        Raw JSON text persisted for audit/debug.
    used_fallback : bool
        True if LLM output was unusable or a soft-error occurred.
    notes : str
        Compact status note (e.g., "ok", "post_validate_failed: ...").
    meta : dict[str, Any]
        Metadata used for reproducibility (k, top_n, hashes, backend class, etc.).
    """

    claims: list[Claim]
    raw_text: str
    used_fallback: bool
    notes: str
    meta: dict[str, Any]


def build_claim_prompt(*, card: SampleCard, candidates: pd.DataFrame, k: int) -> str:
    """
    Build a compact JSON-only prompt for proposing claims.

    Parameters
    ----------
    card : SampleCard
        Sample card providing context values and stable context keys.
    candidates : pandas.DataFrame
        Candidate evidence rows (top_n pool) used as the ONLY selectable source.
        Expected columns include term_uid, term_id, term_name, direction, and
        optionally term_survival and gene_ids_suggest/evidence_genes.
    k : int
        Target number of claims to request from the model.

    Returns
    -------
    str
        Prompt string instructing the model to return valid JSON only.

    Notes
    -----
    The prompt enforces copy-exact rules for:
    - entity == term_id
    - evidence_ref.term_ids == [term_uid]
    Context values are prompt-facing only; identity uses context KEYS.
    """
    ctx = _context_dict(card)
    ctx_keys = _context_keys(card)

    def _clean_field(s: Any) -> str:
        # Prevent delimiter injection / weird newlines that confuse models
        t = "" if _is_na_scalar(s) else str(s)
        t = t.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
        # Avoid using '|' in the payload lines since we use it as a delimiter
        t = t.replace("|", " ").replace("=", " ").strip()
        return " ".join(t.split())

    lines: list[str] = []
    for _, r in candidates.iterrows():
        term_uid = _clean_field(r.get("term_uid", ""))
        term_id = _clean_field(r.get("term_id", ""))
        term_name = _clean_field(r.get("term_name", ""))

        if not term_uid or not term_id:
            continue

        direction = _clean_field(r.get("direction", "na")).lower()
        if direction not in {"up", "down", "na"}:
            direction = "na"

        ts = ""
        try:
            v = r.get("term_survival")
            ts = "" if _is_na_scalar(v) else f"{float(v):.4f}"
        except Exception:
            ts = ""

        genes = _parse_gene_list(r.get("gene_ids_suggest", []))[:5]
        genes = [g.strip().upper() for g in genes if str(g).strip()]
        genes_s = ",".join(genes)

        # Compact, semantics-preserving candidate line
        lines.append(
            f"uid={term_uid} | id={term_id} | name={term_name} | "
            f"dir={direction} | surv={ts} | genes={genes_s}"
        )

    if not lines:
        lines = ["uid=NA | id=NA | name=NA | dir=na | surv= | genes="]

    # IMPORTANT:
    # Do NOT truncate candidates to k. The whole point of top_n is to give the model a pool
    # to choose from. Keep an absolute cap only.
    try:
        max_lines = int(str(os.environ.get("LLMPATH_LLM_MAX_CAND_LINES", "60")).strip())
    except Exception:
        max_lines = 60
    max_lines = max(5, min(max_lines, 300))
    max_lines = min(max_lines, len(lines))
    lines = lines[:max_lines]

    # IMPORTANT: Example must be JSON (not Python dict repr)
    example = {
        "claims": [
            {
                "entity": "COPY_id_FROM_SELECTED_CANDIDATE",
                "direction": "up",
                "context_keys": ctx_keys,
                "evidence_ref": {"term_ids": ["COPY_uid_FROM_SELECTED_CANDIDATE"]},
            }
        ]
    }

    # IMPORTANT:
    # For small local models, requesting "exactly k" can trigger long generations/timeouts.
    # We request "up to k" from the model; exact-k is enforced at the tool layer
    # (deterministic repair/backfill), not at the model layer.
    k_line = f"Return up to {int(k)} claims (prefer as many as possible, max {int(k)}).\n"

    header = (
        "Task: select representative pathway terms for a biomedical QA tool.\n"
        "OUTPUT: VALID JSON ONLY (no markdown, no prose). Do not wrap in ```.\n" + k_line + "\n"
        "\n"
        "HARD RULES (COPY EXACTLY):\n"
        "1) Choose ONLY from the Candidates lines.\n"
        "2) Do NOT repeat uid (uids must be unique).\n"
        "3) evidence_ref.term_ids MUST be [uid] (a list with EXACTLY ONE string).\n"
        "4) evidence_ref.term_ids[0] MUST equal the chosen uid EXACTLY.\n"
        "5) entity MUST equal the chosen id EXACTLY.\n"
        "6) Do NOT invent/abbreviate/modify uid or id (no ellipsis).\n"
        "\n"
        "OTHER RULES:\n"
        "- direction must be one of: up, down, na.\n"
        f"- context_keys MUST equal: {', '.join(ctx_keys)} (exact match, same order).\n"
        "\n"
        "SCHEMA EXAMPLE (JSON):\n"
        f"{_stable_json_dumps(example)}\n"
        "\n"
        "CONTEXT (JSON):\n"
        f"{_stable_json_dumps({'context': ctx})}\n"
        "\n"
        "CANDIDATES (each line has uid/id/name/dir/surv/genes):\n"
    )

    return header + "\n".join(lines) + "\n\nReturn JSON now:\n"


def _coerce_candidate_row_to_claim_dict(
    row: dict[str, Any], *, ctx_keys_resolved: list[str]
) -> dict[str, Any] | None:
    """
    Salvage a candidate-row JSON into a Claim-shaped dict.

    Parameters
    ----------
    row : dict[str, Any]
        Candidate-like payload (e.g., term_uid/term_id/module_id/...).
    ctx_keys_resolved : list[str]
        Resolved context keys to attach to the claim.

    Returns
    -------
    dict[str, Any] or None
        Claim-shaped dict if salvageable; otherwise None.

    Notes
    -----
    This handles a common failure mode of some models that return a single
    candidate row instead of the Claim schema.
    """
    if not isinstance(row, dict):
        return None

    term_uid = str(row.get("term_uid", "")).strip()
    term_id = str(row.get("term_id", "")).strip()
    module_id = str(row.get("module_id", "")).strip()
    gsh = str(row.get("gene_set_hash", "")).strip().lower()
    direction = str(row.get("direction", "na")).strip().lower()

    if not term_uid or not term_id:
        return None

    genes = _parse_gene_list(row.get("gene_ids_suggest", []))
    genes = [str(g).strip().upper() for g in genes if str(g).strip()]

    if direction not in {"up", "down", "na"}:
        direction = "na"

    return {
        "entity": term_id,
        "direction": direction,
        "context_keys": list(ctx_keys_resolved or []),
        "evidence_ref": {
            "module_id": module_id,
            "term_ids": [term_uid],
            "gene_ids": genes[:10],
            "gene_set_hash": gsh,
        },
    }


def _validate_claims_json(
    text: str,
    *,
    ctx_keys_resolved: list[str] | None = None,  # legacy callers may pass this
    **_ignored: Any,  # tolerate future/legacy kwargs without breaking
) -> list[Claim]:
    """
    Parse and validate LLM output into `list[Claim]`.

    Parameters
    ----------
    text : str
        JSON text emitted by the model.
    ctx_keys_resolved : list[str] or None, optional
        Context keys to attach when salvaging a candidate-row JSON.
        (Primary projection happens later in post-validation.)
    **_ignored : Any
        Ignored keyword arguments for forward/legacy compatibility.

    Returns
    -------
    list[Claim]
        Parsed Claim objects (schema-validated).

    Raises
    ------
    ValueError
        If the payload is not a supported shape.
    pydantic.ValidationError
        If Claim validation fails.

    Notes
    -----
    Accepts:
    - {"claims":[...]}
    - [...]
    - single-claim dict
    - candidate-row dict (salvaged) when keys look like a candidate
    """
    obj = json.loads(text)

    if isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict) and isinstance(obj.get("claims"), list):
        items = obj["claims"]
    elif isinstance(obj, dict) and ("entity" in obj) and ("evidence_ref" in obj):
        items = [obj]
    else:
        raise ValueError(
            "LLM output must be {'claims': [...]} or a list of claims (or a single claim dict)"
        )

    claims: list[Claim] = []
    for it in items:
        # extra safety: allow "candidate row json" salvage (optional)
        if (
            isinstance(it, dict)
            and ("term_uid" in it)
            and ("term_id" in it)
            and ("evidence_ref" not in it)
        ):
            cdict = _coerce_candidate_row_to_claim_dict(
                it, ctx_keys_resolved=list(ctx_keys_resolved or [])
            )
            if cdict is not None:
                claims.append(Claim.model_validate(cdict))
                continue

        claims.append(Claim.model_validate(it))
    return claims


def _post_validate_against_candidates(
    *,
    claims: list[Claim],
    cand: pd.DataFrame,
    require_gene_set_hash: bool,
    ctx_keys_resolved: list[str],
    card: SampleCard,
) -> tuple[bool, str, list[Claim]]:
    """
    Enforce contract guardrails against the candidate pool and normalize payloads.

    Parameters
    ----------
    claims : list[Claim]
        Schema-validated claims returned by the model.
    cand : pandas.DataFrame
        Candidate pool DataFrame used to constrain selection.
    require_gene_set_hash : bool
        If True, require a valid 12-hex gene_set_hash for each candidate.
    ctx_keys_resolved : list[str]
        Stable context keys to project onto each claim.
    card : SampleCard
        Sample card used for configuration (module prefix, gene cap, etc.).

    Returns
    -------
    (bool, str, list[Claim])
        ok : bool
            True if validation/normalization succeeded.
        reason : str
            "ok" or a short failure reason.
        claims_norm : list[Claim]
            Normalized claims with tool-owned evidence_ref fields filled.

    Notes
    -----
    Guarantees:
    - Exactly one term_uid per claim, and it must exist in candidates.
    - entity must match candidate term_id.
    - evidence_ref fields are tool-owned:
      module_id, gene_ids (capped), gene_set_hash (identity).
    - context_keys are projected to resolved stable keys.
    - claim_id is tool-owned by Claim schema and is not overridden here.
    """
    cand2 = cand.copy()

    if "evidence_genes" in cand2.columns and "gene_ids_suggest" in cand2.columns:
        # Prefer evidence_genes as "full truth" when present; keep gene_ids_suggest as fallback.
        cand2["gene_ids_truth"] = cand2["evidence_genes"].map(_parse_gene_list)
    else:
        cand2["gene_ids_truth"] = cand2.get("gene_ids_suggest", [[] for _ in range(len(cand2))])

    cand2["term_uid"] = cand2.get("term_uid", pd.Series([""] * len(cand2))).astype(str).str.strip()
    cand2["term_id"] = cand2.get("term_id", pd.Series([""] * len(cand2))).astype(str).str.strip()
    cand2 = cand2.drop_duplicates(subset=["term_uid"], keep="first")

    if "gene_set_hash" not in cand2.columns:
        cand2["gene_set_hash"] = ""
    cand2["gene_set_hash"] = cand2["gene_set_hash"].astype(str).str.strip().str.lower()

    if "gene_ids_suggest" not in cand2.columns:
        cand2["gene_ids_suggest"] = [[] for _ in range(len(cand2))]

    def _candidate_truth_hash(row: pd.Series) -> str:
        """
        Candidate-side source-of-truth identity hash.
        Priority:
          1) precomputed gene_set_hash if valid 12-hex
          2) compute from FULL gene_ids_suggest (NOT capped)
          3) fallback from term_uid
        """
        gsh0 = str(row.get("gene_set_hash", "") or "").strip().lower()
        if _looks_like_12hex(gsh0):
            return gsh0

        genes_full = _parse_gene_list(row.get("gene_ids_truth", []))
        if genes_full:
            return _hash_gene_set_12hex(genes_full)

        tu = str(row.get("term_uid", "") or "").strip()
        return _hash_term_set_fallback_12hex([tu]) if tu else ""

    by_uid = cand2.set_index("term_uid", drop=False)

    seen_uid: set[str] = set()
    normalized: list[Claim] = []

    cap = _max_gene_ids_in_claim(card)

    for c in claims:
        ck_proj = list(ctx_keys_resolved or [])

        term_ids = list(c.evidence_ref.term_ids or [])
        if len(term_ids) != 1:
            return (False, "term_ids must have exactly one term_uid", [])

        term_uid = str(term_ids[0]).strip()
        if not term_uid or term_uid not in by_uid.index:
            return (False, f"term_uid not in candidates: {term_uid}", [])

        if term_uid in seen_uid:
            return (False, f"duplicate term_uid selected: {term_uid}", [])
        seen_uid.add(term_uid)

        row = by_uid.loc[term_uid]

        term_id_expected = str(row.get("term_id", "")).strip()
        if term_id_expected and str(c.entity).strip() != term_id_expected:
            return (False, f"entity != term_id for term_uid={term_uid}", [])

        # identity hash (tool-owned)
        gsh_expected = _candidate_truth_hash(row)
        if require_gene_set_hash and (not _looks_like_12hex(gsh_expected)):
            return (False, f"candidate gene_set_hash missing/invalid for term_uid={term_uid}", [])

        # genes payload (tool-owned; may be capped)
        genes_full = _parse_gene_list(row.get("gene_ids_truth", []))
        genes_full = [_norm_gene_id(g) for g in genes_full if str(g).strip()]
        seen_g: set[str] = set()
        genes_full2: list[str] = []
        for g in genes_full:
            if g and g not in seen_g:
                seen_g.add(g)
                genes_full2.append(g)
        genes_claim = genes_full2[:cap]

        # module_id: prefer candidate module_id; else deterministic fallback
        mid = str(row.get("module_id", "") or "").strip()
        if (not mid) or (mid.lower() in _NA_TOKENS_L):
            mh = _module_hash_like_modules_py_12hex(
                [term_uid], genes_full2 if genes_full2 else genes_claim
            )
            mid = f"{_module_prefix(card)}{mh}"

        update: dict[str, Any] = {"context_keys": ck_proj}
        try:
            ev = c.evidence_ref.model_copy(
                update={
                    "module_id": mid,
                    "gene_ids": genes_claim,
                    "term_ids": [term_uid],
                    "gene_set_hash": gsh_expected,
                }
            )
            update["evidence_ref"] = ev
            c2 = c.model_copy(update=update)
        except Exception:
            d = c.model_dump()
            d["context_keys"] = ck_proj
            d_ev = d.get("evidence_ref", {}) if isinstance(d.get("evidence_ref"), dict) else {}
            if isinstance(d_ev, dict):
                d_ev["module_id"] = mid
                d_ev["gene_ids"] = genes_claim
                d_ev["term_ids"] = [term_uid]
                d_ev["gene_set_hash"] = gsh_expected
                d["evidence_ref"] = d_ev
            c2 = Claim.model_validate(d)

        normalized.append(c2)

    return (True, "ok", normalized)


def _term_uid_from_claim(c: Claim) -> str:
    """
    Return the single term_uid token from a Claim.

    The contract requires ``c.evidence_ref.term_ids`` to contain exactly one
    term_uid token. This helper returns that token (stripped). If the contract
    is not met, it returns an empty string.

    Notes
    -----
    Minimal salvage for small-model artifacts:
    - If the token starts with "uid=", strip that prefix.
    - If the token contains a '|' delimiter, keep only the left side.
    """
    try:
        term_ids = list(c.evidence_ref.term_ids or [])
    except Exception:
        return ""
    if len(term_ids) != 1:
        return ""

    uid = str(term_ids[0] or "").strip()
    if not uid:
        return ""

    # salvage common small-model formats: "uid=..." or "uid=... | ..."
    lo = uid.lower()
    if lo.startswith("uid="):
        uid = uid.split("=", 1)[1].strip()

    if "|" in uid:
        uid = uid.split("|", 1)[0].strip()

    uid = uid.strip(" ,;")
    return uid


def _repair_llm_claims_contract_minimal(
    *,
    claims: list[Claim],
    cand: pd.DataFrame,
    k: int,
    ctx_keys_resolved: list[str],
    enforce_exact_k: bool = False,
) -> tuple[list[Claim], dict[str, Any]]:
    """
    Deterministically repair common small-model contract violations.

    Repairs (conservative):
    - Drop claims with invalid/missing single-term uid.
    - Drop claims not present in the candidate pool (when available).
    - Drop duplicate uids (first occurrence wins).
    - Keep at most k_eff.
    - If any repair occurred, backfill from candidates (candidate order)
      until k_eff is reached.

    When enforce_exact_k=True (strict-k mode), "returning fewer than k_eff"
    is treated as a repair condition and triggers deterministic backfill.
    """
    meta: dict[str, Any] = {
        "changed": False,
        "enforce_exact_k": bool(enforce_exact_k),
        "input_n": int(len(claims or [])),
        "kept_n": 0,
        "k": int(max(1, int(k))),
        "k_eff": 0,
        "shortfall": 0,
        "dropped_dup": 0,
        "dropped_invalid_uid": 0,
        "dropped_not_in_candidates": 0,
        "truncated": 0,
        "backfilled": 0,
    }

    # Normalize candidate uid list (deterministic order, unique)
    cand_uids: list[str] = []
    cand_set: set[str] = set()
    cand_df: pd.DataFrame | None = None
    if isinstance(cand, pd.DataFrame) and ("term_uid" in cand.columns):
        cand_df = cand.copy()
        cand_df["term_uid"] = cand_df["term_uid"].astype(str).str.strip()
        for x in cand_df["term_uid"].tolist():
            if x and x not in cand_set:
                cand_set.add(x)
                cand_uids.append(x)

    # Effective k: respect candidate availability when known
    k_eff0 = int(min(meta["k"], len(cand_uids))) if cand_uids else int(meta["k"])
    k_eff = int(max(1, k_eff0))
    meta["k_eff"] = int(k_eff)

    kept: list[Claim] = []
    seen: set[str] = set()

    # If LLM returned nothing, strict-k mode should still produce a fixed-size output
    if not claims:
        if enforce_exact_k and cand_uids and (cand_df is not None):
            meta["changed"] = True
            by_uid = cand_df.set_index("term_uid", drop=False)
            for uid in cand_uids:
                if len(kept) >= k_eff:
                    break
                try:
                    r = by_uid.loc[uid]
                except Exception:
                    continue
                if isinstance(r, pd.DataFrame):
                    if r.empty:
                        continue
                    r = r.iloc[0]

                term_id = str(r.get("term_id", "") or "").strip()
                if not term_id:
                    continue

                direction = str(r.get("direction", "na") or "").strip().lower()
                if direction not in {"up", "down", "na"}:
                    direction = "na"

                payload = {
                    "entity": term_id,
                    "direction": direction,
                    "context_keys": list(ctx_keys_resolved or []),
                    "evidence_ref": {"term_ids": [uid]},
                }
                try:
                    kept.append(Claim.model_validate(payload))
                    seen.add(uid)
                    meta["backfilled"] += 1
                except Exception:
                    continue

        meta["kept_n"] = int(len(kept))
        meta["shortfall"] = int(max(0, k_eff - len(kept)))
        return kept, meta

    stop_idx: int | None = None

    for i, c in enumerate(claims):
        uid = _term_uid_from_claim(c)
        if not uid:
            meta["dropped_invalid_uid"] += 1
            meta["changed"] = True
            continue

        if cand_set and (uid not in cand_set):
            meta["dropped_not_in_candidates"] += 1
            meta["changed"] = True
            continue

        if uid in seen:
            meta["dropped_dup"] += 1
            meta["changed"] = True
            continue

        seen.add(uid)
        kept.append(c)

        if len(kept) >= k_eff:
            stop_idx = i
            break

    if stop_idx is not None and (stop_idx + 1) < len(claims):
        meta["truncated"] = int(len(claims) - (stop_idx + 1))
        meta["changed"] = True

    # In strict-k mode, "short output" is itself a repair condition
    if enforce_exact_k and (len(kept) < k_eff) and cand_uids and (cand_df is not None):
        meta["shortfall"] = int(k_eff - len(kept))
        meta["changed"] = True

    # Backfill ONLY if we had to repair something (or strict-k shortfall)
    if meta["changed"] and cand_uids and (len(kept) < k_eff) and (cand_df is not None):
        by_uid = cand_df.set_index("term_uid", drop=False)

        for uid in cand_uids:
            if len(kept) >= k_eff:
                break
            if uid in seen:
                continue

            try:
                r = by_uid.loc[uid]
            except Exception:
                continue

            if isinstance(r, pd.DataFrame):
                if r.empty:
                    continue
                r = r.iloc[0]

            term_id = str(r.get("term_id", "") or "").strip()
            if not term_id:
                continue

            direction = str(r.get("direction", "na") or "").strip().lower()
            if direction not in {"up", "down", "na"}:
                direction = "na"

            payload = {
                "entity": term_id,
                "direction": direction,
                "context_keys": list(ctx_keys_resolved or []),
                "evidence_ref": {"term_ids": [uid]},
            }
            try:
                kept.append(Claim.model_validate(payload))
                seen.add(uid)
                meta["backfilled"] += 1
            except Exception:
                continue

    meta["kept_n"] = int(len(kept))
    meta["shortfall"] = int(max(0, k_eff - len(kept)))
    return kept, meta


def propose_claims_llm(
    *,
    distilled_with_modules: pd.DataFrame,
    card: SampleCard,
    backend: BaseLLMBackend,
    k: int,
    seed: int | None = None,
    outdir: str | None = None,
    artifact_tag: str | None = None,
) -> LLMClaimResult:
    """
    Propose claims via an LLM and write audit-grade artifacts.

    Parameters
    ----------
    distilled_with_modules : pandas.DataFrame
        Distilled evidence table with module information (or sufficient columns
        to derive term_uid). Must contain:
        - term_uid OR (source, term_id)
        - term_id, term_name, source
        Optional:
        - module_id, gene_set_hash
        - evidence_genes / evidence_genes_str / gene_ids_suggest
        - keep_term, term_survival, stat, context_score
    card : SampleCard
        Sample card providing prompt context and contract keys.
    backend : BaseLLMBackend
        LLM backend adapter.
    k : int
        Target number of claims.
    seed : int or None, optional
        Optional seed (best-effort; may be ignored).
    outdir : str or None, optional
        Output directory for artifacts. If None, no artifacts are written.
    artifact_tag : str or None, optional
        Optional tag to avoid overwriting per-call artifacts.

    Returns
    -------
    LLMClaimResult
        Claims and metadata. On failure, `claims` may be empty and
        `used_fallback` True.

    Raises
    ------
    ValueError
        If required columns are missing.
    RuntimeError
        If LLM is required by contract and call/validation fails.

    Notes
    -----
    Artifacts (when outdir is set):
    - llm_claims.prompt.json
    - llm_claims.candidates.json
    - llm_claims.raw.json
    - llm_claims.meta.json
    Plus tagged variants when `artifact_tag` is provided.
    """
    k = _scalar_int(k, 1) or 1
    seed = _scalar_int(seed, None)
    df = distilled_with_modules.copy()

    if "term_uid" not in df.columns:
        if not {"source", "term_id"}.issubset(set(df.columns)):
            raise ValueError("propose_claims_llm: requires term_uid OR (source, term_id)")
        df["term_uid"] = (
            df.apply(
                lambda r: _shared.make_term_uid(r.get("source"), r.get("term_id")),
                axis=1,
            )
            .astype(str)
            .str.strip()
        )

    required = {"term_uid", "term_id", "term_name", "source"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"propose_claims_llm: missing columns: {missing}")

    if "module_id" not in df.columns:
        df["module_id"] = ""

    # gene_ids_suggest: canonical list[str]
    if "evidence_genes" in df.columns:
        df["gene_ids_suggest"] = df["evidence_genes"].map(_parse_gene_list)
    elif "evidence_genes_str" in df.columns:
        df["gene_ids_suggest"] = df["evidence_genes_str"].map(_parse_gene_list)
    else:
        df["gene_ids_suggest"] = [[] for _ in range(len(df))]

    if "gene_set_hash" not in df.columns:
        df["gene_set_hash"] = ""

    def _fill_gsh(row: pd.Series) -> str:
        gsh = str(row.get("gene_set_hash", "") or "").strip().lower()
        if _looks_like_12hex(gsh):
            return gsh
        genes = _parse_gene_list(row.get("gene_ids_suggest", []))
        if genes:
            return _hash_gene_set_12hex(genes)
        term_uid = str(row.get("term_uid", "") or "").strip()
        return _hash_term_set_fallback_12hex([term_uid]) if term_uid else ""

    df["gene_set_hash"] = df.apply(_fill_gsh, axis=1)

    if "keep_term" in df.columns:
        df["keep_term"] = df["keep_term"].fillna(True).astype(bool)
    else:
        df["keep_term"] = True

    if "term_survival" in df.columns:
        df["term_survival_sort"] = pd.to_numeric(df["term_survival"], errors="coerce").fillna(-1.0)
    else:
        df["term_survival_sort"] = -1.0

    df["stat_sort"] = pd.to_numeric(
        df.get("stat", pd.Series(0, index=df.index)), errors="coerce"
    ).fillna(0.0)

    # Optional context proxy: if enabled and context_score exists, use it as a tie-breaker.
    try:
        ctx_proxy_on = bool(card.enable_context_score_proxy(default=False))
    except Exception:
        ctx_proxy_on = False

    if ctx_proxy_on:
        df["context_score_sort"] = pd.to_numeric(
            df.get("context_score", pd.Series(0, index=df.index)),
            errors="coerce",
        ).fillna(0.0)
    else:
        df["context_score_sort"] = 0.0

    # ---- Candidate budget (top_n) ----
    # Priority:
    #   1) explicit env LLMPATH_LLM_TOPN
    #   2) auto rule: top_n = clamp(alpha*k, min, max) with smaller max for small local models
    topn_meta: dict[str, Any] = {}
    topn_env = str(os.environ.get("LLMPATH_LLM_TOPN", "")).strip()
    if topn_env:
        try:
            top_n = int(topn_env)
        except Exception:
            top_n = 50
        hard_max = _as_int_env("LLMPATH_LLM_TOPN_HARD_MAX", 200)
        hard_max_small = _as_int_env("LLMPATH_LLM_TOPN_HARD_MAX_SMALL", 80)
        eff_hard_max = min(hard_max, hard_max_small) if _is_small_local_model(backend) else hard_max
        top_n = _clamp_int(top_n, 3, eff_hard_max)
        topn_meta = {
            "top_n_mode": "env",
            "top_n_env": topn_env,
            "top_n_model_hint": _backend_model_hint(backend),
            "top_n_small_model": bool(_is_small_local_model(backend)),
            "top_n_hard_max": int(eff_hard_max),
            "top_n_soft_max": int(
                _as_int_env("LLMPATH_LLM_TOPN_MAX_SMALL", 50)
                if _is_small_local_model(backend)
                else _as_int_env("LLMPATH_LLM_TOPN_MAX", 80)
            ),
        }
    else:
        top_n, topn_meta = _auto_topn_for_k(k=int(k), backend=backend)

    # always ensure top_n >= k and bounded by available rows later
    top_n = max(int(top_n), max(1, int(k)))

    df_rank = df.sort_values(
        ["keep_term", "term_survival_sort", "context_score_sort", "stat_sort", "term_uid"],
        ascending=[False, False, False, False, True],
    ).head(top_n)

    # ensure candidates are not duplicated by term_uid (important for post-validate)
    df_rank = df_rank.drop_duplicates(subset=["term_uid"], keep="first")

    strict_k = bool(_strict_k_enabled())
    require_gsh = os.environ.get("LLMPATH_LLM_REQUIRE_GENESET_HASH", "1").strip() != "0"
    prompt = build_claim_prompt(card=card, candidates=df_rank, k=int(k))

    meta: dict[str, Any] = {
        "k": int(k),
        "seed": None if seed is None else int(seed),
        "top_n": int(top_n),
        "require_gene_set_hash": bool(require_gsh),
        "context_keys_resolved": _context_keys(card),
        "context_proxy_enabled": bool(ctx_proxy_on),
        "candidates_sha256": _df_records_sha256(df_rank),
        "notes": "",
        "backend_class": type(backend).__name__,
        "artifact_tag": (artifact_tag or ""),
        "strict_k": bool(strict_k),
        "top_n_mode": str(topn_meta.get("top_n_mode", "")),
        "top_n_env": str(topn_meta.get("top_n_env", "")),
        "top_n_alpha": topn_meta.get("top_n_alpha", None),
        "top_n_min": topn_meta.get("top_n_min", None),
        "top_n_max": topn_meta.get("top_n_max", None),
        "top_n_max_small": topn_meta.get("top_n_max_small", None),
        "top_n_small_model": bool(topn_meta.get("top_n_small_model", False)),
        "top_n_model_hint": str(topn_meta.get("top_n_model_hint", "")),
        "top_n_hard_max": topn_meta.get("top_n_hard_max", None),
        "top_n_soft_max": topn_meta.get("top_n_soft_max", None),
    }

    def _write_artifacts(*, raw_text: str, meta_obj: dict[str, Any]) -> None:
        if not outdir:
            return
        od = Path(outdir)

        _safe_write_json(od / "llm_claims.prompt.json", {"prompt": prompt})
        _safe_write_json(
            od / "llm_claims.candidates.json", {"candidates": df_rank.to_dict(orient="records")}
        )

        # legacy (backward compatible)
        _safe_write_json(od / "llm_claims.raw.json", {"raw": raw_text})
        _safe_write_json(od / "llm_claims.meta.json", meta_obj)

        # tagged (non-overwriting, if requested)
        if artifact_tag:
            _safe_write_json(od / f"llm_claims.{artifact_tag}.raw.json", {"raw": raw_text})
            _safe_write_json(od / f"llm_claims.{artifact_tag}.meta.json", meta_obj)

    raw_text = ""
    obj: dict[str, Any] | None = None
    try:
        obj = _backend_call_json(backend, prompt=prompt, seed=_scalar_int(seed, None))
        raw_text = _stable_json_dumps(obj)
    except Exception as e:
        msg = str(e)
        meta["notes"] = f"backend_call_failed: {type(e).__name__}"
        meta["exception_type"] = type(e).__name__
        meta["exception_msg"] = msg[:200]
        raw_text = _soft_error_json(msg, err_type="backend_call_failed", retryable=False)
        _write_artifacts(raw_text=raw_text, meta_obj=meta)
        _raise_if_required(meta["notes"], meta=meta)
        return LLMClaimResult(
            claims=[],
            raw_text=raw_text,
            used_fallback=True,
            notes=meta["notes"],
            meta=meta,
        )

    _write_artifacts(raw_text=raw_text, meta_obj=meta)

    # If backend produced a soft error JSON (standard contract), treat as fallback immediately.
    # Note: obj is dict at this point; raw_text is stable JSON text.
    if isinstance(obj, dict) and isinstance(obj.get("error"), dict):
        msg = str((obj.get("error") or {}).get("message", ""))
        meta["notes"] = f"soft_error: {msg[:200]}"
        _write_artifacts(raw_text=raw_text, meta_obj=meta)
        _raise_if_required(meta["notes"], meta=meta)
        return LLMClaimResult(
            claims=[],
            raw_text=raw_text,
            used_fallback=True,
            notes=meta["notes"],
            meta=meta,
        )

    # Also keep the legacy safety net in case raw_text was modified externally
    soft = _parse_soft_error_json(raw_text)
    if soft is not None:
        msg = str((soft.get("error") or {}).get("message", ""))
        meta["notes"] = f"soft_error: {msg[:200]}"
        _write_artifacts(raw_text=raw_text, meta_obj=meta)
        _raise_if_required(meta["notes"], meta=meta)
        return LLMClaimResult(
            claims=[], raw_text=raw_text, used_fallback=True, notes=meta["notes"], meta=meta
        )

    ctx_keys_resolved = _context_keys(card)

    try:
        claims = _validate_claims_json(raw_text, ctx_keys_resolved=ctx_keys_resolved)
    except Exception as e:
        meta["notes"] = f"llm_output_invalid: {type(e).__name__}"
        meta["exception_type"] = type(e).__name__
        meta["exception_msg"] = str(e)[:200]
        _write_artifacts(raw_text=raw_text, meta_obj=meta)
        _raise_if_required(meta["notes"], meta=meta)
        return LLMClaimResult(
            claims=[],
            raw_text=raw_text,
            used_fallback=True,
            notes=meta["notes"],
            meta=meta,
        )

    claims, repair_meta = _repair_llm_claims_contract_minimal(
        claims=claims,
        cand=df_rank,
        k=int(k),
        ctx_keys_resolved=ctx_keys_resolved,
        enforce_exact_k=bool(strict_k),
    )

    if bool(repair_meta.get("changed", False)):
        meta["llm_repair"] = repair_meta

    # Optional strict-k guardrail (paper/figure reproducibility)
    # NOTE: strict-k is enforced AFTER tool-side repair/backfill, not by the LLM itself.
    if bool(strict_k):
        # allow fewer only if candidates fewer
        k_eff = min(int(k), int(len(df_rank)))
        if len(claims) != k_eff:
            meta["notes"] = f"strict_k_failed: expected={k_eff} got={len(claims)}"
            _write_artifacts(raw_text=raw_text, meta_obj=meta)
            _raise_if_required(meta["notes"], meta=meta)
            return LLMClaimResult(
                claims=[],
                raw_text=raw_text,
                used_fallback=True,
                notes=meta["notes"],
                meta=meta,
            )

    ok, why, claims_norm = _post_validate_against_candidates(
        claims=claims,
        cand=df_rank,
        require_gene_set_hash=bool(require_gsh),
        ctx_keys_resolved=ctx_keys_resolved,
        card=card,
    )
    if not ok:
        meta["notes"] = f"post_validate_failed: {why}"
        _write_artifacts(raw_text=raw_text, meta_obj=meta)
        _raise_if_required(meta["notes"], meta=meta)
        return LLMClaimResult(
            claims=[],
            raw_text=raw_text,
            used_fallback=True,
            notes=meta["notes"],
            meta=meta,
        )

    meta["notes"] = "ok"
    _write_artifacts(raw_text=raw_text, meta_obj=meta)
    return LLMClaimResult(
        claims=claims_norm,
        raw_text=raw_text,
        used_fallback=False,
        notes="ok",
        meta=meta,
    )


def claims_to_proposed_tsv(
    *,
    claims: list[Claim],
    distilled_with_modules: pd.DataFrame,
    card: SampleCard,
) -> pd.DataFrame:
    """
    Convert proposed claims into a flat TSV-like DataFrame for export.

    Parameters
    ----------
    claims : list[Claim]
        Proposed claims (typically from `propose_claims_llm`).
    distilled_with_modules : pandas.DataFrame
        Distilled evidence table used to enrich exported rows with term metadata.
    card : SampleCard
        Sample card providing context values (export columns).

    Returns
    -------
    pandas.DataFrame
        Row-wise export with fields including:
        claim_id, entity, direction, context_keys, term_uid, module_id,
        gene_ids, term_ids, gene_set_hash, and serialized claim_json.

    Notes
    -----
    Context VALUES are exported as columns for convenience, but MUST NOT be
    baked into identity (claim_id / gene_set_hash).
    """
    ctx = card.context_dict()
    df = distilled_with_modules.copy()

    if "term_uid" not in df.columns:
        df["term_uid"] = (
            df["source"].astype(str).str.strip() + ":" + df["term_id"].astype(str).str.strip()
        )

    df["term_uid"] = df["term_uid"].astype(str).str.strip()
    df["term_id"] = df["term_id"].astype(str).str.strip()
    df["term_name"] = df["term_name"].astype(str).str.strip()
    df["source"] = df["source"].astype(str).str.strip()

    # enforce uniqueness to avoid surprises
    df = df.drop_duplicates(subset=["term_uid"], keep="first")
    by_uid = df.set_index("term_uid", drop=False)

    rows: list[dict[str, Any]] = []
    for c in claims:
        term_uid = str(c.evidence_ref.term_ids[0]).strip() if c.evidence_ref.term_ids else ""
        term_id = str(c.entity).strip()
        term_name = ""
        source = ""
        term_survival = pd.NA
        keep_term = True
        keep_reason = "ok"
        context_score = pd.NA

        if term_uid and term_uid in by_uid.index:
            r = by_uid.loc[term_uid]
            term_id = str(r.get("term_id", term_id)).strip() or term_id
            term_name = str(r.get("term_name", "")).strip()
            source = str(r.get("source", "")).strip()
            term_survival = r.get("term_survival", pd.NA)
            keep_term = bool(r.get("keep_term", True))
            keep_reason = str(r.get("keep_reason", "ok"))
            try:
                context_score = pd.to_numeric(r.get("context_score", pd.NA), errors="coerce")
            except Exception:
                context_score = pd.NA

        claim_json = c.model_dump_json()

        gene_ids = list(c.evidence_ref.gene_ids or [])
        term_ids = list(c.evidence_ref.term_ids or [])

        rows.append(
            {
                "condition": ctx.get("condition", "NA"),
                "tissue": ctx.get("tissue", "NA"),
                "perturbation": ctx.get("perturbation", "NA"),
                "comparison": ctx.get("comparison", "NA"),
                "context_key": card.context_key(),
                "claim_id": c.claim_id,
                "entity": term_id,
                "direction": c.direction,
                "context_keys": ",".join(list(c.context_keys or [])),
                "term_uid": term_uid,
                "source": source,
                "term_id": term_id,
                "term_name": term_name,
                "module_id": c.evidence_ref.module_id,
                "module_reason": "",
                "gene_ids": ",".join([str(x) for x in gene_ids]),
                "term_ids": ",".join([str(x) for x in term_ids]),
                "gene_set_hash": str(c.evidence_ref.gene_set_hash or "").strip().lower(),
                "context_score": context_score,
                "eligible": True,
                "term_survival": term_survival,
                "keep_term": keep_term,
                "keep_reason": keep_reason,
                "claim_json": claim_json,
            }
        )

    return pd.DataFrame(rows)
