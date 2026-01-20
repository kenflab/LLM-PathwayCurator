# src/llm_pathway_curator/llm_claims.py
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .backends import BaseLLMBackend
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
        v = pd.isna(x)
        return bool(v) if isinstance(v, bool) else False
    except Exception:
        return False


def _strip_na(s: Any) -> str:
    if _is_na_scalar(s):
        return ""
    t = str(s).strip()
    if not t or t.lower() in {x.lower() for x in _NA_TOKENS}:
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


def _parse_soft_error_json(s: str) -> dict[str, Any] | None:
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
    return {
        "disease": _strip_na(getattr(card, "disease", "")),
        "tissue": _strip_na(getattr(card, "tissue", "")),
        "perturbation": _strip_na(getattr(card, "perturbation", "")),
        "comparison": _strip_na(getattr(card, "comparison", "")),
    }


def _context_keys(card: SampleCard) -> list[str]:
    keys: list[str] = []
    for k in ["disease", "tissue", "perturbation", "comparison"]:
        v = getattr(card, k, None)
        s = _strip_na(v)
        if s:
            keys.append(k)
    return keys


def _norm_gene_id(g: Any) -> str:
    return str(g).strip().upper()


def _parse_gene_list(x: Any) -> list[str]:
    if _is_na_scalar(x):
        return []
    if isinstance(x, (list, tuple, set)):
        items = [str(t).strip() for t in x if str(t).strip()]
    else:
        s = str(x).strip().replace(";", ",").replace("|", ",")
        if not s or s.lower() in {t.lower() for t in _NA_TOKENS}:
            return []
        items = [t.strip() for t in s.split(",") if t.strip()]

    seen: set[str] = set()
    out: list[str] = []
    for t in items:
        u = _norm_gene_id(t)
        if u and (u not in seen):
            seen.add(u)
            out.append(u)
    return out


def _hash_gene_set_12hex(genes: list[str]) -> str:
    """
    Audit-grade: set-stable, normalized (align with audit.py), 12-hex.
    """
    uniq = sorted({_norm_gene_id(g) for g in genes if str(g).strip()})
    payload = ",".join(uniq)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class LLMClaimResult:
    claims: list[Claim]
    raw_text: str
    used_fallback: bool
    notes: str
    meta: dict[str, Any]


def build_claim_prompt(*, card: SampleCard, candidates: pd.DataFrame, k: int) -> str:
    """
    JSON-only prompt: select + type claims. LLM must COPY fields from candidates.

    candidates expected columns (minimal):
      term_uid, term_id, term_name, source, module_id, gene_set_hash, gene_ids_suggest
    """
    ctx = _context_dict(card)

    cand_rows: list[dict[str, Any]] = []
    for _, r in candidates.iterrows():
        # gene_ids_suggest: provide BOTH list and csv string for robustness
        genes_s = str(r.get("gene_ids_suggest", "") or "").strip()
        genes_list = [
            t.strip().upper()
            for t in genes_s.replace(";", ",").replace("|", ",").split(",")
            if t.strip()
        ]

        ts = None
        try:
            v = r.get("term_survival")
            ts = None if _is_na_scalar(v) else float(v)
        except Exception:
            ts = None

        cand_rows.append(
            {
                "term_uid": str(r.get("term_uid", "")).strip(),
                "term_id": str(r.get("term_id", "")).strip(),
                "term_name": str(r.get("term_name", "")).strip(),
                "source": str(r.get("source", "")).strip(),
                "direction": str(r.get("direction", "na")).strip(),
                "term_survival": ts,
                "module_id": str(r.get("module_id", "")).strip(),
                "gene_set_hash": str(r.get("gene_set_hash", "")).strip().lower(),
                "gene_ids_suggest": genes_list,  # <-- canonical
                "gene_ids_suggest_csv": genes_s,  # <-- back-compat / readability
            }
        )

    instructions = {
        "task": "Select representative pathway terms and return typed Claim JSON objects.",
        "hard_rules": [
            "OUTPUT MUST BE VALID JSON ONLY. No markdown. No commentary.",
            "Return exactly k claims unless fewer candidates exist.",
            "Use candidates only: copy term_uid, term_id, module_id, gene_set_hash.",
            "Each claim MUST include evidence_ref.term_ids with EXACTLY ONE term_uid string.",
            "Each claim MUST include evidence_ref.gene_set_hash (12-hex) copied from candidate.",
            "gene_set_hash must be an exact match.",
            "If you include evidence_ref.gene_ids, pick from gene_ids_suggest only.",
            "Do not invent new identifiers. Do not change term_uid strings.",
            "direction must be one of: up, down, na.",
            "entity must be term_id (not term_name).",
            "If you include evidence_ref.gene_ids, pick from gene_ids_suggest only "
            "(case-insensitive).",
        ],
        "output_schema": {
            "claims": [
                {
                    "claim_id": "string unique",
                    "entity": "term_id",
                    "direction": "up|down|na",
                    "context_keys": ["disease|tissue|perturbation|comparison"],
                    "evidence_ref": {
                        "module_id": "string",
                        "term_ids": ["term_uid"],
                        "gene_ids": ["GENE1", "GENE2"],
                        "gene_set_hash": "12-hex string",
                    },
                }
            ]
        },
    }

    payload = {"context": ctx, "k": int(k), "candidates": cand_rows}

    return (
        "You are a strict JSON generator for a biomedical analysis tool.\n"
        "INSTRUCTIONS:\n"
        f"{json.dumps(instructions, ensure_ascii=False)}\n"
        "PAYLOAD:\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n"
        "RETURN_JSON_ONLY:\n"
    )


def _validate_claims_json(text: str) -> list[Claim]:
    obj = json.loads(text)
    if isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict) and isinstance(obj.get("claims"), list):
        items = obj["claims"]
    else:
        raise ValueError("LLM output must be {'claims': [...]} or a list of claims")

    claims: list[Claim] = []
    for it in items:
        claims.append(Claim.model_validate(it))
    return claims


def _post_validate_against_candidates(
    *,
    claims: list[Claim],
    cand: pd.DataFrame,
    require_gene_set_hash: bool,
    ctx_keys_resolved: list[str],
) -> tuple[bool, str, list[Claim]]:
    """
    Extra guardrail beyond Claim schema:
      - term_ids must have exactly one term_uid
      - term_uid must exist in candidates
      - entity must match candidate term_id
      - gene_set_hash must match candidate gene_set_hash (and be 12-hex if required)
      - gene_ids (if provided) must be subset of candidate gene_ids_suggest
      - term_uid must be unique across returned claims
      - context_keys are projected onto resolved keys (does not fail the claim)
    Returns (ok, message, normalized_claims).
    """
    cand2 = cand.copy()
    cand2["term_uid"] = cand2.get("term_uid", pd.Series([""] * len(cand2))).astype(str).str.strip()
    cand2["term_id"] = cand2.get("term_id", pd.Series([""] * len(cand2))).astype(str).str.strip()

    if "gene_set_hash" not in cand2.columns:
        cand2["gene_set_hash"] = ""
    cand2["gene_set_hash"] = cand2["gene_set_hash"].astype(str).str.strip().str.lower()

    if "gene_ids_suggest" not in cand2.columns:
        cand2["gene_ids_suggest"] = ""
    cand2["gene_ids_suggest"] = cand2["gene_ids_suggest"].astype(str)

    by_uid = cand2.set_index("term_uid", drop=False)

    seen_uid: set[str] = set()
    normalized: list[Claim] = []

    ctx_allowed = set(ctx_keys_resolved or [])

    for c in claims:
        # --- project context keys without mutating input object ---
        ck_in = list(c.context_keys or [])
        ck_proj = [k for k in ck_in if (k in ctx_allowed)]
        if not ck_proj:
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

        gsh_expected = str(row.get("gene_set_hash", "")).strip().lower()
        gsh = str(c.evidence_ref.gene_set_hash or "").strip().lower()

        if require_gene_set_hash:
            if (not _looks_like_12hex(gsh)) or (not gsh_expected):
                return (
                    False,
                    f"gene_set_hash required but missing/invalid for term_uid={term_uid}",
                    [],
                )
            if gsh != gsh_expected:
                return (
                    False,
                    f"gene_set_hash mismatch for term_uid={term_uid}: {gsh} != {gsh_expected}",
                    [],
                )
        else:
            if gsh and gsh_expected and gsh != gsh_expected:
                return (
                    False,
                    f"gene_set_hash mismatch for term_uid={term_uid}: {gsh} != {gsh_expected}",
                    [],
                )

        # gene_ids subset check (if present)
        genes = [str(x).strip().upper() for x in (c.evidence_ref.gene_ids or []) if str(x).strip()]
        if genes:
            sug = str(row.get("gene_ids_suggest", "") or "")
            sug_set = {
                t.strip().upper()
                for t in sug.replace(";", ",").replace("|", ",").split(",")
                if t.strip()
            }
            bad = [g for g in genes if g not in sug_set]
            if bad:
                return (
                    False,
                    f"gene_ids not in gene_ids_suggest for term_uid={term_uid}: {bad[:3]}",
                    [],
                )

        # Create normalized claim (pydantic v2 safe path)
        try:
            c2 = c.model_copy(update={"context_keys": ck_proj})
        except Exception:
            # fallback: rebuild via dict
            d = c.model_dump()
            d["context_keys"] = ck_proj
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
) -> LLMClaimResult:
    """
    LLM-based claim proposal (C1-LLM).
    - JSON-only
    - strict post-validation against candidates
    - writes prompt/candidates/raw/meta for auditability
    """
    _ = seed  # seed is handled by backend (OpenAIBackend supports it)

    df = distilled_with_modules.copy()

    # Accept missing term_uid: create stable one
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

    # Provide gene_ids_suggest for overlap (audit min_overlap=1 by default)
    if "evidence_genes_str" in df.columns:
        df["gene_ids_suggest"] = df["evidence_genes_str"].astype(str)
    elif "evidence_genes" in df.columns:
        df["gene_ids_suggest"] = df["evidence_genes"].map(
            lambda xs: ",".join([str(x) for x in xs]) if isinstance(xs, list) else str(xs)
        )
    else:
        df["gene_ids_suggest"] = ""

    # Ensure gene_set_hash exists and is filled deterministically from evidence_genes when possible
    if "gene_set_hash" not in df.columns:
        df["gene_set_hash"] = ""

    # Fill missing/empty gene_set_hash using evidence_genes
    if "evidence_genes" in df.columns:

        def _fill_gsh(row: pd.Series) -> str:
            gsh = str(row.get("gene_set_hash", "") or "").strip().lower()
            if _looks_like_12hex(gsh):
                return gsh
            genes = _parse_gene_list(row.get("evidence_genes"))
            return _hash_gene_set_12hex(genes) if genes else ""

        df["gene_set_hash"] = df.apply(_fill_gsh, axis=1)
    else:
        df["gene_set_hash"] = df["gene_set_hash"].astype(str).str.strip().str.lower()

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

    try:
        top_n = int(str(os.environ.get("LLMPATH_LLM_TOPN", "30")).strip())
    except Exception:
        top_n = 30

    top_n = max(5, min(top_n, 200))

    # Deterministic ranking for candidates
    df_rank = df.sort_values(
        ["keep_term", "term_survival_sort", "stat_sort", "term_uid"],
        ascending=[False, False, False, True],
    ).head(top_n)

    require_gsh = os.environ.get("LLMPATH_LLM_REQUIRE_GENESET_HASH", "1").strip() != "0"

    prompt = build_claim_prompt(card=card, candidates=df_rank, k=int(k))

    raw = backend.generate(prompt, json_mode=True) or ""
    raw_text = str(raw).strip()

    meta: dict[str, Any] = {
        "k": int(k),
        "top_n": int(top_n),
        "require_gene_set_hash": bool(require_gsh),
        "candidates_sha256": _sha256_text(df_rank.to_csv(index=False, sep="\t")),
        "notes": "",
    }

    # optional artifact output
    if outdir:
        od = Path(outdir)
        _safe_write_json(od / "llm_claims.prompt.json", {"prompt": prompt})
        _safe_write_json(
            od / "llm_claims.candidates.json",
            {"candidates": df_rank.to_dict(orient="records")},
        )
        _safe_write_json(od / "llm_claims.raw.json", {"raw": raw_text})

    soft = _parse_soft_error_json(raw_text)
    if soft is not None:
        msg = str((soft.get("error") or {}).get("message", ""))
        meta["notes"] = f"soft_error: {msg}"
        if outdir:
            _safe_write_json(Path(outdir) / "llm_claims.meta.json", meta)
        return LLMClaimResult(
            claims=[], raw_text=raw_text, used_fallback=True, notes=meta["notes"], meta=meta
        )

    # Validate JSON + schema
    try:
        claims = _validate_claims_json(raw_text)
    except Exception as e:
        meta["notes"] = f"llm_output_invalid: {type(e).__name__}"
        if outdir:
            _safe_write_json(Path(outdir) / "llm_claims.meta.json", meta)
        return LLMClaimResult(
            claims=[], raw_text=raw_text, used_fallback=True, notes=meta["notes"], meta=meta
        )

    # Post-validate against candidate set (critical for paper/tool quality)
    ctx_keys_resolved = _context_keys(card)
    ok, why, claims_norm = _post_validate_against_candidates(
        claims=claims,
        cand=df_rank,
        require_gene_set_hash=bool(require_gsh),
        ctx_keys_resolved=ctx_keys_resolved,
    )
    if not ok:
        meta["notes"] = f"post_validate_failed: {why}"
        if outdir:
            _safe_write_json(Path(outdir) / "llm_claims.meta.json", meta)
        return LLMClaimResult(
            claims=[], raw_text=raw_text, used_fallback=True, notes=meta["notes"], meta=meta
        )

    claims = claims_norm

    meta["notes"] = "ok"
    if outdir:
        _safe_write_json(Path(outdir) / "llm_claims.meta.json", meta)

    return LLMClaimResult(
        claims=claims, raw_text=raw_text, used_fallback=False, notes="ok", meta=meta
    )


def claims_to_proposed_tsv(
    *,
    claims: list[Claim],
    distilled_with_modules: pd.DataFrame,
    card: SampleCard,
) -> pd.DataFrame:
    df = distilled_with_modules.copy()

    # accept missing term_uid
    if "term_uid" not in df.columns:
        df["term_uid"] = (
            df["source"].astype(str).str.strip() + ":" + df["term_id"].astype(str).str.strip()
        )

    df["term_uid"] = df["term_uid"].astype(str).str.strip()
    df["term_id"] = df["term_id"].astype(str).str.strip()
    df["term_name"] = df["term_name"].astype(str).str.strip()
    df["source"] = df["source"].astype(str).str.strip()
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
        context_score = 0

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
                context_score = 0

        claim_json = c.model_dump_json()

        rows.append(
            {
                "claim_id": c.claim_id,
                "entity": term_id,
                "direction": c.direction,
                "context_keys": ",".join(list(c.context_keys)),
                "term_uid": term_uid,
                "source": source,
                "term_id": term_id,
                "term_name": term_name,
                "module_id": c.evidence_ref.module_id,
                "module_reason": "",
                "gene_ids": ",".join(list(c.evidence_ref.gene_ids)),
                "term_ids": ",".join(list(c.evidence_ref.term_ids)),
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
