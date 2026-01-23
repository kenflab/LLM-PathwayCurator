# LLM-PathwayCurator/src/llm_pathway_curator/llm_claims.py
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

from .backends import BaseLLMBackend
from .claim_schema import Claim
from .sample_card import SampleCard

_NA_TOKENS = {"", "na", "nan", "none", "NA"}
_NA_TOKENS_L = {t.lower() for t in _NA_TOKENS}


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


def _strip_na(s: Any) -> str:
    if _is_na_scalar(s):
        return ""
    t = str(s).strip()
    if not t or t.lower() in _NA_TOKENS_L:
        return ""
    return t


def _looks_like_12hex(s: Any) -> bool:
    if _is_na_scalar(s):
        return False
    x = str(s).strip().lower()
    if len(x) != 12:
        return False
    return all(ch in "0123456789abcdef" for ch in x)


def _sha256_text(s: str) -> str:
    h = hashlib.sha256()
    h.update((s or "").encode("utf-8"))
    return h.hexdigest()


def _stable_json_dumps(obj: Any) -> str:
    # stable across runs
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _df_records_sha256(df: pd.DataFrame) -> str:
    # stable hash for provenance/debugging
    recs = df.to_dict(orient="records")
    return _sha256_text(_stable_json_dumps(recs))


def _parse_soft_error_json(s: str) -> dict[str, Any] | None:
    """
    Some backends return {"error": {...}} as JSON. Treat it as soft error.
    """
    try:
        obj = json.loads(s)
    except Exception:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("error"), dict):
        return obj
    return None


def _safe_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def _context_dict(card: SampleCard) -> dict[str, str]:
    """
    Prompt-facing context. Values are for prompting only; NOT used for claim_id identity.
    Tool-facing neutral key is "condition".
    """
    d = card.context_dict()
    return {k: _strip_na(v) for k, v in d.items()}


def _context_keys(card: SampleCard) -> list[str]:
    """
    Stable order is part of the contract.
    Use tool-facing CORE keys: condition/tissue/perturbation/comparison.
    """
    keys: list[str] = []
    d = card.context_dict()
    for k in ["condition", "tissue", "perturbation", "comparison"]:
        if _strip_na(d.get(k, "")):
            keys.append(k)
    return keys


def _norm_gene_id(g: Any) -> str:
    return str(g).strip().upper()


_BRACKETED_LIST_RE = re.compile(r"^\s*[\[\(\{].*[\]\)\}]\s*$")


def _parse_gene_list(x: Any) -> list[str]:
    """
    Parse genes into canonical list[str] (upper, de-dup, preserve order).

    Accept:
      - list-like
      - csv-ish strings with separators: , ; | /
      - bracket-ish strings: "['A','B']" / '["A","B"]' / "{A,B}" (NO eval)
      - whitespace split as fallback
    """
    if _is_na_scalar(x):
        return []

    if isinstance(x, (list, tuple, set)):
        items = [str(t).strip() for t in x if str(t).strip()]
    else:
        s = str(x).strip()
        if not s or s.lower() in _NA_TOKENS_L:
            return []

        # strip bracket wrappers if it looks like a list literal; do NOT eval
        s0 = s.strip()
        if _BRACKETED_LIST_RE.match(s0):
            s0 = s0.strip().lstrip("[({").rstrip("])}").strip()

        # normalize separators
        s0 = s0.replace(";", ",").replace("|", ",").replace("/", ",")
        s0 = s0.replace("\n", " ").replace("\t", " ")
        s0 = " ".join(s0.split()).strip()
        if not s0 or s0.lower() in _NA_TOKENS_L:
            return []

        if "," in s0:
            items = [t.strip().strip('"').strip("'") for t in s0.split(",") if t.strip()]
        else:
            items = [t.strip().strip('"').strip("'") for t in s0.split(" ") if t.strip()]

        items = [t.strip("[](){}").strip() for t in items]
        items = [t for t in items if t and t.lower() not in _NA_TOKENS_L]

    seen: set[str] = set()
    out: list[str] = []
    for t in items:
        u = _norm_gene_id(t)
        if u and (u not in seen):
            seen.add(u)
            out.append(u)
    return out


def _hash_gene_set_12hex(genes: list[str]) -> str:
    """Audit-grade: set-stable, normalized (align with audit.py), 12-hex."""
    uniq = sorted({_norm_gene_id(g) for g in genes if str(g).strip()})
    payload = ",".join(uniq)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _hash_term_set_fallback_12hex(term_uids: list[str]) -> str:
    """
    Fallback when genes are missing/empty.
    Deterministic 12-hex from term_uid set (order-invariant).
    """
    canon = sorted({str(t).strip() for t in term_uids if str(t).strip()})
    payload = ",".join(canon)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _soft_error_json(message: str, *, err_type: str = "llm_error", retryable: bool = False) -> str:
    return json.dumps({"error": {"message": message, "type": err_type, "retryable": retryable}})


def _as_bool_env(name: str, default: bool) -> bool:
    v = str(os.environ.get(name, "")).strip().lower()
    if not v:
        return bool(default)
    return v in {"1", "true", "t", "yes", "y", "on"}


def _as_int_env(name: str, default: int) -> int:
    try:
        v = str(os.environ.get(name, "")).strip()
        return int(v) if v else int(default)
    except Exception:
        return int(default)


def _module_prefix(card: SampleCard) -> str:
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
    t = sorted([str(x).strip() for x in term_uids if str(x).strip()])
    g = sorted([_norm_gene_id(x) for x in genes if str(x).strip()])
    payload = "T:" + "|".join(t) + "\n" + "G:" + "|".join(g)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# -------------------------
# Backend calling (robust across Ollama/OpenAI/Gemini)
# -------------------------
def _try_call(fn: Callable[..., Any], **kwargs: Any) -> Any:
    """
    Try calling a function with a subset of kwargs.
    This avoids brittle signature coupling across backends/refactors.

    IMPORTANT:
      - Preserve json_mode whenever possible.
      - Drop seed first (many backends do not accept it).
      - Avoid positional fallback with seed (2nd positional can be misinterpreted as json_mode).
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
    Best-effort extraction of a JSON object/array from free-form text.
    Handles:
      - ```json ... ```
      - leading/trailing commentary
      - returns the first valid JSON object/array found
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
    Best-effort JSON call:
      1) Try explicit JSON methods (complete_json/chat_json/...)
      2) Try text methods and extract JSON fragment
    """
    # A) explicit json-returning methods (if any)
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
                return out
            if isinstance(out, str):
                frag = _extract_json_fragment(out) or out.strip()
                return json.loads(frag)
        except Exception as e:
            last_err = e
            continue

    # B) text-returning methods (most common)
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
                return json.loads(frag)
            return json.loads(s.strip())
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        f"propose_claims_llm: backend has no usable JSON/text method "
        f"(tried json={json_candidates} text={text_candidates})"
    ) from last_err


@dataclass(frozen=True)
class LLMClaimResult:
    claims: list[Claim]
    raw_text: str
    used_fallback: bool
    notes: str
    meta: dict[str, Any]


def build_claim_prompt(*, card: SampleCard, candidates: pd.DataFrame, k: int) -> str:
    """
    Lightweight prompt while preserving semantics:
      - keep term_name (semantic)
      - keep small gene snippet (<=5)
      - keep direction + term_survival as hints (optional)
      - use compact line format (not verbose JSON)
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

    lines = lines[: min(max(1, int(k)), 100)]
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

    header = (
        "You select representative pathway terms for a biomedical analysis tool.\n"
        "Return VALID JSON ONLY. No markdown. No commentary.\n"
        f"Return exactly {int(k)} claims (or fewer only if candidates fewer).\n"
        "\n"
        "CRITICAL COPY RULES:\n"
        "- You MUST choose ONLY from the candidate lines.\n"
        "- evidence_ref.term_ids MUST be a list with EXACTLY ONE string, and it MUST equal "
        "the chosen uid EXACTLY.\n"
        "- entity MUST equal the chosen id EXACTLY.\n"
        "- Do NOT invent, abbreviate, or modify uid/id (no ellipsis, no '...').\n"
        "\n"
        "Other rules:\n"
        "- direction must be one of: up, down, na.\n"
        f"- context_keys MUST equal: {', '.join(ctx_keys)} (exact match, same order).\n"
        "\n"
        "Output schema example (JSON):\n"
        f"{_stable_json_dumps(example)}\n"
        "\n"
        "Context (JSON):\n"
        f"{_stable_json_dumps({'context': ctx})}\n"
        "\n"
        "Candidates (each line has uid/id/name/dir/surv/genes):\n"
    )

    return header + "\n".join(lines) + "\n\nReturn JSON now:\n"


def _coerce_candidate_row_to_claim_dict(
    row: dict[str, Any], *, ctx_keys_resolved: list[str]
) -> dict[str, Any] | None:
    """
    Some models (esp. local) return a single candidate row JSON instead of Claim JSON.
    Salvage it into a Claim-shaped dict.

    Expected candidate-ish keys:
      term_uid, term_id, module_id, gene_set_hash, gene_ids_suggest, direction
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
    Parse and validate LLM output into list[Claim].

    Acceptable inputs:
      - {"claims":[...]}
      - [...]
      - single-claim dict (common model failure mode)

    NOTE:
      Some older call sites pass ctx_keys_resolved=...; we accept and ignore it here.
      Context key projection is handled later by _post_validate_against_candidates().
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
            # This path is conservative; only salvage if it clearly looks like candidate row JSON
            # and the caller provided ctx_keys_resolved (or we fall back later).
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
    Guardrails beyond schema:
      - term_ids exactly one term_uid existing in candidates
      - entity matches candidate term_id
      - tool fills evidence_ref.gene_set_hash from candidates (LLM output may omit or be wrong)
      - gene_ids (if provided) subset of candidate gene_ids_suggest
      - unique term_uid across returned claims
      - context_keys projected to resolved keys (stable)
      - claim_id is TOOL-OWNED by Claim schema; do NOT generate or overwrite here

    IMPORTANT (NbTx-friendly, reproducible):
      - LLM is allowed to omit gene_set_hash.
      - Even if LLM returns gene_set_hash, the TOOL treats candidates (distilled-derived)
        as source-of-truth and OVERWRITES it deterministically.
      - Therefore gene_set_hash mismatch MUST NOT fail the run.
    """
    cand2 = cand.copy()

    cand2["term_uid"] = cand2.get("term_uid", pd.Series([""] * len(cand2))).astype(str).str.strip()
    cand2["term_id"] = cand2.get("term_id", pd.Series([""] * len(cand2))).astype(str).str.strip()

    cand2 = cand2.drop_duplicates(subset=["term_uid"], keep="first")

    if "gene_set_hash" not in cand2.columns:
        cand2["gene_set_hash"] = ""
    cand2["gene_set_hash"] = cand2["gene_set_hash"].astype(str).str.strip().str.lower()

    if "gene_ids_suggest" not in cand2.columns:
        cand2["gene_ids_suggest"] = [[] for _ in range(len(cand2))]

    def _as_suggest_set(v: Any) -> set[str]:
        genes = _parse_gene_list(v)
        return {g.strip().upper() for g in genes if g.strip()}

    def _candidate_truth_hash(row: pd.Series) -> str:
        """
        Candidate-side source-of-truth.
        If precomputed gene_set_hash is present/valid, use it;
        else compute from candidate gene_ids_suggest (full evidence genes),
        else term_uid fallback.
        """
        gsh0 = str(row.get("gene_set_hash", "") or "").strip().lower()
        if _looks_like_12hex(gsh0):
            return gsh0

        genes = _parse_gene_list(row.get("gene_ids_suggest", []))
        if genes:
            return _hash_gene_set_12hex(genes)

        tu = str(row.get("term_uid", "") or "").strip()
        return _hash_term_set_fallback_12hex([tu]) if tu else ""

    by_uid = cand2.set_index("term_uid", drop=False)

    seen_uid: set[str] = set()
    normalized: list[Claim] = []

    for c in claims:
        # TOOL-OWNED: context_keys are deterministic and MUST match resolved keys
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

        # ---- TOOL-OWNED evidence_ref (module_id / gene_ids / gene_set_hash) ----
        genes_full = _parse_gene_list(row.get("gene_ids_suggest", []))
        genes_full = [_norm_gene_id(g) for g in genes_full if str(g).strip()]
        # dedup preserve order
        seen_g: set[str] = set()
        genes_full2: list[str] = []
        for g in genes_full:
            if g and g not in seen_g:
                seen_g.add(g)
                genes_full2.append(g)

        cap = _max_gene_ids_in_claim(card)
        genes_claim = genes_full2[:cap]

        mid = str(row.get("module_id", "") or "").strip()
        if (not mid) or (mid.lower() in _NA_TOKENS_L):
            mh = _module_hash_like_modules_py_12hex([term_uid], genes_claim)
            mid = f"{_module_prefix(card)}{mh}"

        # IMPORTANT: gene_set_hash must match genes_claim stored in claim_json
        gsh_tool = (
            _hash_gene_set_12hex(genes_claim)
            if genes_claim
            else _hash_term_set_fallback_12hex([term_uid])
        )

        term_id_expected = str(row.get("term_id", "")).strip()
        if term_id_expected and str(c.entity).strip() != term_id_expected:
            return (False, f"entity != term_id for term_uid={term_uid}", [])

        # ---- tool-owned gene_set_hash (source-of-truth is candidates) ----
        gsh_expected = _candidate_truth_hash(row)
        if require_gene_set_hash:
            if (not gsh_expected) or (not _looks_like_12hex(gsh_expected)):
                return (
                    False,
                    f"candidate gene_set_hash missing/invalid for term_uid={term_uid}",
                    [],
                )

        # If LLM provided a hash, we do NOT fail on mismatch; we ignore it.
        # (Optional: we could later record a warning in meta; here we only normalize.)
        genes = [str(x).strip().upper() for x in (c.evidence_ref.gene_ids or []) if str(x).strip()]
        if genes:
            sug_set = _as_suggest_set(row.get("gene_ids_suggest", []))
            bad = [g for g in genes if g not in sug_set]
            if bad:
                return (
                    False,
                    f"gene_ids not in gene_ids_suggest for term_uid={term_uid}: {bad[:3]}",
                    [],
                )

        # gsh_final = gsh_expected or str(c.evidence_ref.gene_set_hash or "").strip().lower() or ""

        update: dict[str, Any] = {"context_keys": ck_proj}
        try:
            ev = c.evidence_ref.model_copy(
                update={
                    "module_id": mid,
                    "gene_ids": genes_claim,
                    "term_ids": [term_uid],
                    "gene_set_hash": gsh_tool,
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
                d_ev["gene_set_hash"] = gsh_tool
                d["evidence_ref"] = d_ev
            c2 = Claim.model_validate(d)

        normalized.append(c2)

    return (True, "ok", normalized)


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
    Propose typed claims via LLM (JSON only) and persist audit-grade artifacts.

    artifact_tag:
      - Optional tag to avoid overwriting artifacts when multiple LLM calls happen in a run.
      - When provided, writes BOTH:
          llm_claims.<tag>.raw.json / llm_claims.<tag>.meta.json  (stable per-call)
        and also the legacy:
          llm_claims.raw.json / llm_claims.meta.json              (backward compatible)
    """
    df = distilled_with_modules.copy()

    if "term_uid" not in df.columns:
        if not {"source", "term_id"}.issubset(set(df.columns)):
            raise ValueError("propose_claims_llm: requires term_uid OR (source, term_id)")
        df["term_uid"] = (
            df["source"].astype(str).str.strip() + ":" + df["term_id"].astype(str).str.strip()
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
        df.get("stat", pd.Series([0] * len(df))), errors="coerce"
    ).fillna(0.0)

    # Optional context proxy: if enabled and context_score exists, use it as a tie-breaker.
    try:
        ctx_proxy_on = bool(card.enable_context_score_proxy(default=False))
    except Exception:
        ctx_proxy_on = False

    if ctx_proxy_on:
        df["context_score_sort"] = pd.to_numeric(
            df.get("context_score", 0), errors="coerce"
        ).fillna(0.0)
    else:
        df["context_score_sort"] = 0.0

    # ---- LLM mode should be LIGHT by default ----
    # default 12, clamp 5..60
    try:
        top_n = int(str(os.environ.get("LLMPATH_LLM_TOPN", "12")).strip())
    except Exception:
        top_n = 12
    top_n = max(5, min(top_n, 60))

    df_rank = df.sort_values(
        ["keep_term", "term_survival_sort", "context_score_sort", "stat_sort", "term_uid"],
        ascending=[False, False, False, False, True],
    ).head(top_n)

    # ensure candidates are not duplicated by term_uid (important for post-validate)
    df_rank = df_rank.drop_duplicates(subset=["term_uid"], keep="first")

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
    try:
        obj = _backend_call_json(backend, prompt=prompt, seed=seed)
        raw_text = _stable_json_dumps(obj)
    except Exception as e:
        # Never write empty raw: persist a machine-readable error payload.
        msg = str(e)
        meta["notes"] = f"backend_call_failed: {type(e).__name__}"
        meta["exception_type"] = type(e).__name__
        meta["exception_msg"] = msg[:200]
        raw_text = _soft_error_json(msg, err_type="backend_call_failed", retryable=False)
        _write_artifacts(raw_text=raw_text, meta_obj=meta)
        return LLMClaimResult(
            claims=[],
            raw_text=raw_text,
            used_fallback=True,
            notes=meta["notes"],
            meta=meta,
        )

    _write_artifacts(raw_text=raw_text, meta_obj=meta)

    soft = _parse_soft_error_json(raw_text)
    if soft is not None:
        msg = str((soft.get("error") or {}).get("message", ""))
        meta["notes"] = f"soft_error: {msg[:200]}"
        _write_artifacts(raw_text=raw_text, meta_obj=meta)
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
        return LLMClaimResult(
            claims=[], raw_text=raw_text, used_fallback=True, notes=meta["notes"], meta=meta
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
    # Do NOT bake context values into IDs (tool contract), but do export them as columns.
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
                context_score = int(r.get("context_score", 0))
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
