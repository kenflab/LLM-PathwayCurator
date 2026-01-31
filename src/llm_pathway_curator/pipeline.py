# LLM-PathwayCurator/src/llm_pathway_curator/pipeline.py
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import platform
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from . import _shared
from .audit import audit_claims
from .backends import BaseLLMBackend, get_backend_from_env
from .distill import distill_evidence
from .modules import attach_module_ids, factorize_modules_connected_components
from .report import write_report, write_report_jsonl
from .sample_card import SampleCard
from .schema import EvidenceTable
from .select import select_claims
from .utils import build_id_to_symbol_from_distilled, load_id_map_tsv, map_ids_to_symbols


@dataclass(frozen=True)
class RunConfig:
    evidence_table: str
    sample_card: str
    outdir: str
    force: bool = False
    seed: int | None = None
    run_meta_name: str = "run_meta.json"
    tau: float | None = None
    k_claims: int | None = None
    stress_evidence_dropout_p: float | None = None
    stress_evidence_dropout_min_keep: int | None = None
    stress_contradictory_p: float | None = None
    stress_contradictory_max_extra: int | None = None


@dataclass(frozen=True)
class RunResult:
    run_id: str
    outdir: str
    artifacts: dict[str, str]
    meta_path: str


# -------------------------
# Mode normalization (pipeline-owned; single source of truth)
# -------------------------
_ALLOWED_GATE_MODES = {"soft", "hard"}
_ALLOWED_REVIEW_MODES = {"off", "proxy", "llm"}


def _norm_gate_mode(x: Any, *, default: str = "soft") -> str:
    s = ("" if x is None else str(x)).strip().lower()
    if s in _ALLOWED_GATE_MODES:
        return s
    return str(default).strip().lower()


def _norm_review_mode(x: Any, *, default: str = "proxy") -> str:
    s = ("" if x is None else str(x)).strip().lower()
    if s in _ALLOWED_REVIEW_MODES:
        return s
    # tolerate common falsy values
    if s in {"none", "disabled", "false", "0"}:
        return "off"
    return str(default).strip().lower()


def _require_file(path: str, label: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not p.is_file():
        raise IsADirectoryError(f"{label} is not a file: {path}")
    return p


def _file_fingerprint(path: str) -> dict[str, Any]:
    p = Path(path)
    st = p.stat()
    return {"path": str(p), "size_bytes": int(st.st_size), "mtime_epoch": float(st.st_mtime)}


def _safe_mkdir_outdir(outdir: str, force: bool) -> None:
    p = Path(outdir)
    if p.exists():
        if not p.is_dir():
            raise NotADirectoryError(f"outdir exists but is not a directory: {outdir}")
        if not force and any(p.iterdir()):
            raise FileExistsError(f"outdir is not empty: {outdir} (use --force)")
    p.mkdir(parents=True, exist_ok=True)


def _dump_model(obj: Any) -> Any:
    """
    pydantic v2/v1 + 素のdict を吸収して JSON-able にする。
    """
    if obj is None:
        return None
    # pydantic v2
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        try:
            return obj.model_dump()
        except Exception:
            pass
    # pydantic v1
    if hasattr(obj, "dict") and callable(obj.dict):
        try:
            return obj.dict()
        except Exception:
            pass
    # already dict-like
    if isinstance(obj, dict):
        return obj
    # best-effort fallback
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return str(obj)


def _write_json(path: str | Path, obj: Any) -> None:
    """Atomic-ish JSON write: write temp then replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def _write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def _sha256_text(s: str) -> str:
    h = hashlib.sha256()
    h.update(s.encode("utf-8"))
    return h.hexdigest()


def _sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Streaming sha256 for large files."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _tool_version() -> str:
    """
    Best-effort tool version string for run_meta.
    Avoid hard dependency.
    """
    try:  # pragma: no cover
        import importlib.metadata as _ilm

        return str(_ilm.version("llm-pathway-curator"))
    except Exception:  # pragma: no cover
        return "unknown"


def _env_fingerprint() -> dict[str, Any]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "pandas": getattr(pd, "__version__", "unknown"),
        "tool_version": _tool_version(),
    }


def _make_run_id(cfg: RunConfig) -> str:
    payload = json.dumps(asdict(cfg), sort_keys=True, ensure_ascii=False)
    return f"{int(time.time())}_{_sha256_text(payload)[:10]}"


def _resolve_tau(cfg_tau: float | None, card: SampleCard) -> float:
    """
    Single source-of-truth for tau used by mechanical audit + report contract.
    Priority:
      1) cfg.tau if provided (explicit run override)
      2) card.audit_tau() (contract boundary)
    """
    if cfg_tau is not None:
        try:
            return float(cfg_tau)
        except Exception:
            pass
    return float(card.audit_tau())


def _env_int_strict(name: str) -> int | None:
    """
    Parse env var as int.
    Returns None if unset/empty. Raises ValueError if malformed.
    """
    s = (os.environ.get(name, "") or "").strip()
    if not s:
        return None
    return int(s)


def _resolve_k_claims(cfg: RunConfig, card: SampleCard) -> tuple[int, dict[str, Any]]:
    """
    Single source-of-truth for k_claims used by claim selection.
    Priority:
      1) cfg.k_claims (explicit run override; Fig2/CLI should set this)
      2) env LLMPATH_K_CLAIMS
      3) card.k_claims() if available
      4) fallback 3

    Returns:
      (k_effective, meta_dict)
    """
    if cfg.k_claims is not None:
        k = int(cfg.k_claims)
        if k < 1:
            raise ValueError(f"k_claims must be >= 1 (cfg.k_claims={cfg.k_claims})")
        return k, {"k_source": "cfg", "k_cfg": cfg.k_claims, "k_env": "", "k_card": ""}

    try:
        k_env = _env_int_strict("LLMPATH_K_CLAIMS")
    except Exception as e:
        raise ValueError(
            f"invalid env LLMPATH_K_CLAIMS={os.environ.get('LLMPATH_K_CLAIMS', '')!r}"
        ) from e

    if k_env is not None:
        if int(k_env) < 1:
            raise ValueError(f"LLMPATH_K_CLAIMS must be >= 1 (got {k_env})")
        return int(k_env), {"k_source": "env", "k_cfg": None, "k_env": str(k_env), "k_card": ""}

    k_card = None
    if hasattr(card, "k_claims") and callable(card.k_claims):
        try:
            k_card = int(card.k_claims())
        except Exception:
            k_card = None

    if k_card is not None:
        if int(k_card) < 1:
            raise ValueError(f"card.k_claims() must be >= 1 (got {k_card})")
        return int(k_card), {"k_source": "card", "k_cfg": None, "k_env": "", "k_card": str(k_card)}

    return 3, {"k_source": "fallback", "k_cfg": None, "k_env": "", "k_card": ""}


# -------------------------
# Tool-wide NA parsing helpers (pipeline-owned)
# -------------------------
# Shared NA tokens / scalar NA check (single source of truth)
_NA_TOKENS_L = _shared.NA_TOKENS_L


def _is_na_scalar(x: Any) -> bool:
    return _shared.is_na_scalar(x)


def _parse_ids(x: Any) -> list[str]:
    return _shared.parse_id_list(x)


def _norm_gene_id(g: Any) -> str:
    return _shared.norm_gene_id_upper(g)


def _hash_gene_set_12(genes: list[str]) -> str:
    return _shared.hash_gene_set_12hex_upper(genes)


def _extract_claim_evidence(cj: str) -> tuple[list[str], str, str]:
    """
    Returns (term_ids, gene_set_hash, direction) from claim_json.
    Backward-tolerant but expects schema-ish structure:
      - obj.evidence_ref or obj.claim.evidence_ref
      - obj.direction or obj.claim.direction
    """
    try:
        obj = json.loads(cj)
    except Exception:
        return ([], "", "na")

    ev = None
    direction = "na"

    if isinstance(obj, dict):
        direction = str(obj.get("direction", "na") or "na").strip().lower()
        ev = obj.get("evidence_ref") or obj.get("evidence_refs")
        if isinstance(obj.get("claim"), dict):
            if direction == "na":
                direction = str(obj["claim"].get("direction", "na") or "na").strip().lower()
            if ev is None:
                ev = obj["claim"].get("evidence_ref") or obj["claim"].get("evidence_refs")

    if direction not in {"up", "down"}:
        direction = "na"

    if not isinstance(ev, dict):
        return ([], "", direction)

    term_ids = _parse_ids(ev.get("term_ids") or ev.get("term_id") or "")
    gsh = ev.get("gene_set_hash")
    gsh = "" if _is_na_scalar(gsh) else str(gsh).strip().lower()
    return (term_ids, gsh, direction)


def _flip_claim_direction_json(cj: str) -> str:
    """
    Flip direction in claim_json (up<->down). If missing/na, leave as-is.
    Writes back without pretty-print to keep stable-ish.
    """
    try:
        obj = json.loads(cj)
    except Exception:
        return cj

    def _flip(d: str) -> str:
        d = str(d).strip().lower()
        if d == "up":
            return "down"
        if d == "down":
            return "up"
        return d

    if isinstance(obj, dict):
        if "direction" in obj:
            obj["direction"] = _flip(obj.get("direction"))
        if isinstance(obj.get("claim"), dict):
            if "direction" in obj["claim"]:
                obj["claim"]["direction"] = _flip(obj["claim"].get("direction"))

    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return cj


def _env_float(name: str, default: float) -> float:
    s = (os.environ.get(name, "") or "").strip()
    if not s:
        return float(default)
    try:
        return float(s)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    s = (os.environ.get(name, "") or "").strip()
    if not s:
        return int(default)
    try:
        return int(s)
    except Exception:
        return int(default)


def _env_str(name: str, default: str = "") -> str:
    return (os.environ.get(name, default) or "").strip()


def _get_card_extra(card: SampleCard) -> dict[str, Any]:
    try:
        ex = getattr(card, "extra", {}) or {}
        return ex if isinstance(ex, dict) else {}
    except Exception:
        return {}


def _set_card_extra(card: SampleCard, key: str, value: Any) -> None:
    try:
        ex = getattr(card, "extra", None)
        if isinstance(ex, dict):
            ex[key] = value
    except Exception:
        pass


def _card_condition(card: SampleCard) -> str:
    """
    Tool-facing neutral label used as the first field of context identity.

    IMPORTANT:
      - This function must return the *intrinsic/original* condition.
      - Do NOT read context-swap keys here; swap is handled separately
        by _context_swap_info() / _ctx_ids_from_card().
    """

    def _s(x: Any) -> str:
        try:
            return str(x or "").strip()
        except Exception:
            return ""

    # 1) explicit field
    try:
        if hasattr(card, "condition"):
            c = _s(getattr(card, "condition", ""))
            if c:
                return c
    except Exception:
        pass

    # 2) extra hints (non-swap)
    ex = _get_card_extra(card)
    for k in ("condition", "cancer"):
        c = _s(ex.get(k, ""))
        if c:
            return c

    # 3) fallback to disease
    try:
        c = _s(getattr(card, "disease", ""))
        if c:
            return c
    except Exception:
        pass

    return ""


# -------------------------
# Gate/review mode resolution (pipeline-owned; normalized)
# -------------------------
def _resolve_context_gate_mode(card: SampleCard, *, default: str = "soft") -> str:
    env_v = _env_str("LLMPATH_CONTEXT_GATE_MODE", "").strip().lower()
    if env_v:
        return _norm_gate_mode(env_v, default=default)
    ex = _get_card_extra(card)
    v = str(ex.get("context_gate_mode", "") or "").strip().lower()
    return _norm_gate_mode(v, default=default)


def _resolve_context_review_mode(card: SampleCard, *, default: str = "proxy") -> str:
    env_v = _env_str("LLMPATH_CONTEXT_REVIEW_MODE", "").strip().lower()
    if env_v:
        return _norm_review_mode(env_v, default=default)
    ex = _get_card_extra(card)
    v = str(ex.get("context_review_mode", "") or "").strip().lower()
    return _norm_review_mode(v, default=default)


def _enforce_gate_review_compat_effective(card, gate_mode: str, review_mode: str) -> dict:
    """
    Enforce safe compatibility between gate/review modes.

    Contract (paper + tool):
      - If gate_mode is "hard", we MUST have a deterministic review signal
        unless the user explicitly accepts that everything becomes UNEVALUATED.
      - Therefore, (gate=hard, review=off) is disallowed and we force review=proxy.
      - We NEVER silently relax hard->soft, because that breaks user intent and Fig2 comparability.
    """
    gate_mode_in = gate_mode
    review_mode_in = review_mode

    extra = getattr(card, "extra", {}) or {}
    swap_active = bool(extra.get("context_swap_from")) or bool(extra.get("context_swap_to"))

    updates: dict[str, str] = {}
    reasons: list[str] = []

    if gate_mode == "hard" and review_mode == "off":
        review_mode = "proxy"
        updates["review_mode"] = "proxy"
        reasons.append(
            "disallow (gate=hard, review=off); force review=proxy to keep hard-gate meaningful"
        )
        if swap_active:
            reasons.append(
                "swap_active: forced proxy prevents silent all-ABSTAIN/UNEVALUATED collapse"
            )

    if review_mode == "llm":
        reasons.append("review=llm: compatible with gate soft/hard")

    return {
        "gate_mode": gate_mode,
        "review_mode": review_mode,
        "changed": (gate_mode != gate_mode_in) or (review_mode != review_mode_in),
        "updates": updates,
        "reasons": reasons,
        "inputs": {
            "gate_mode": gate_mode_in,
            "review_mode": review_mode_in,
            "swap_active": swap_active,
        },
    }


# -------------------------
# Claim payload contract enforcement (pipeline-owned)
# -------------------------
def _pick_claim_payload_col(df: pd.DataFrame) -> str | None:
    for c in ["claim_json", "claim_json_str", "claim_json_raw"]:
        if c in df.columns:
            return c
    return None


def _stable_claim_id_from_fields(
    *,
    entity: str,
    direction: str,
    module_id: str,
    gene_set_hash: str,
    term_ids: list[str],
) -> str:
    payload = json.dumps(
        {
            "entity": entity,
            "direction": direction,
            "module_id": module_id,
            "gene_set_hash": gene_set_hash,
            "term_ids": sorted([t for t in term_ids if str(t).strip()]),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _synthesize_claim_json_row(row: pd.Series, card: SampleCard) -> str:
    """
    Synthesize a minimal claim JSON payload if upstream didn't provide one.
    This is tool robustness (contract enforcement), not paper logic.
    """
    entity = str(row.get("entity", "") or "").strip()
    direction = str(row.get("direction", "") or "").strip().lower()
    if direction not in {"up", "down", "na", ""}:
        direction = "na"
    if not direction:
        direction = "na"

    module_id = str(row.get("module_id_effective", row.get("module_id", "")) or "").strip()
    gene_set_hash = (
        str(row.get("gene_set_hash_effective", row.get("gene_set_hash", "")) or "").strip().lower()
    )

    gene_ids = _parse_ids(row.get("gene_ids", row.get("gene_ids_str", "")))
    term_ids = _parse_ids(row.get("term_ids", row.get("term_ids_str", "")))

    context_keys = ["condition", "tissue", "perturbation", "comparison"]

    ctx_review_cols = [
        "context_evaluated",
        "context_method",
        "context_status",
        "context_reason",
        "context_notes",
        "context_confidence",
        "context_gate_mode",
        "context_review_mode",
    ]
    ctx_review: dict[str, Any] = {}
    for c in ctx_review_cols:
        if c in row.index and not _is_na_scalar(row.get(c)):
            ctx_review[c] = row.get(c)

    ctx = {
        "condition": _card_condition(card),
        "tissue": str(getattr(card, "tissue", "") or "").strip(),
        "perturbation": str(getattr(card, "perturbation", "") or "").strip(),
        "comparison": str(getattr(card, "comparison", "") or "").strip(),
    }

    claim_id = str(row.get("claim_id", "") or "").strip()
    if not claim_id:
        claim_id = _stable_claim_id_from_fields(
            entity=entity,
            direction=direction,
            module_id=module_id,
            gene_set_hash=gene_set_hash,
            term_ids=term_ids,
        )

    obj: dict[str, Any] = {
        "claim_id": claim_id,
        "entity": entity,
        "direction": direction,
        "context_keys": context_keys,
        "evidence_ref": {
            "module_id": module_id,
            "gene_ids": gene_ids,
            "term_ids": term_ids,
            "gene_set_hash": gene_set_hash,
        },
        "context": ctx,
        **ctx_review,
    }
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _canonicalize_claim_json_value(s: str) -> str:
    """
    Make claim_json TSV-safe and deterministic:
      - parse JSON dict
      - dump compact single-line canonical JSON
    If parsing fails, return original.
    """
    try:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            return s
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return s


def _canonicalize_claim_json_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce canonical single-line claim_json across the dataframe.
    """
    out = df.copy()
    if "claim_json" not in out.columns:
        return out
    out["claim_json"] = out["claim_json"].map(
        lambda x: "" if _is_na_scalar(x) else _canonicalize_claim_json_value(str(x))
    )
    return out


def _ensure_claim_payload_and_id(proposed: pd.DataFrame, card: SampleCard) -> pd.DataFrame:
    """
    Enforce the pipeline contract:
      - proposed MUST have a string payload column among claim_json/claim_json_str/claim_json_raw
      - proposed MUST have claim_id
      - normalize to canonical claim_json (string) for downstream stress/audit/report

    CRITICAL:
      - If a payload column exists but claim_json is empty for some rows,
        synthesize minimal claim_json for those rows.
        (Otherwise context review cannot write back into claim_json.)
    """
    out = proposed.copy()

    if "claim_id" not in out.columns:
        out["claim_id"] = ""

    payload_col = _pick_claim_payload_col(out)

    # -------------------------
    # Case A: payload column exists (but may be empty/NA)
    # -------------------------
    if payload_col is not None:
        if payload_col != "claim_json":
            out["claim_json"] = out[payload_col]
        out["claim_json"] = out["claim_json"].map(lambda x: "" if _is_na_scalar(x) else str(x))

        def _maybe_claim_id_from_json(s: str) -> str:
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    cid = str(obj.get("claim_id", "") or "").strip()
                    return cid
            except Exception:
                return ""
            return ""

        # Fill claim_id from JSON if possible
        missing_id = out["claim_id"].map(lambda x: _is_na_scalar(x) or (not str(x).strip()))
        if missing_id.any():
            out.loc[missing_id, "claim_id"] = out.loc[missing_id, "claim_json"].map(
                lambda s: _maybe_claim_id_from_json(str(s))
            )

        # If still missing, synthesize a stable claim_id
        missing_id2 = out["claim_id"].map(lambda x: _is_na_scalar(x) or (not str(x).strip()))
        if missing_id2.any():

            def _mk(row: pd.Series) -> str:
                entity = str(row.get("entity", "") or "").strip()
                direction = str(row.get("direction", "") or "").strip().lower()
                module_id = str(
                    row.get("module_id_effective", row.get("module_id", "")) or ""
                ).strip()
                gsh = (
                    str(row.get("gene_set_hash_effective", row.get("gene_set_hash", "")) or "")
                    .strip()
                    .lower()
                )
                term_ids = _parse_ids(row.get("term_ids", row.get("term_ids_str", "")))
                return _stable_claim_id_from_fields(
                    entity=entity,
                    direction=direction,
                    module_id=module_id,
                    gene_set_hash=gsh,
                    term_ids=term_ids,
                )

            out.loc[missing_id2, "claim_id"] = out.loc[missing_id2].apply(_mk, axis=1)

        # >>> CRITICAL: if claim_json is empty, synthesize minimal payload
        empty_payload = out["claim_json"].map(lambda x: _is_na_scalar(x) or (not str(x).strip()))
        if empty_payload.any():
            out.loc[empty_payload, "claim_json"] = out.loc[empty_payload].apply(
                lambda r: _synthesize_claim_json_row(r, card),
                axis=1,
            )

        out = _canonicalize_claim_json_column(out)
        return out

    # -------------------------
    # Case B: no payload column -> always synthesize
    # -------------------------
    out["claim_json_raw"] = out.apply(lambda r: _synthesize_claim_json_row(r, card), axis=1)
    out["claim_json"] = out["claim_json_raw"]

    def _extract_id(s: str) -> str:
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return str(obj.get("claim_id", "") or "").strip()
        except Exception:
            return ""
        return ""

    out["claim_id"] = out["claim_json"].map(_extract_id).map(lambda x: str(x).strip())
    out = _canonicalize_claim_json_column(out)
    return out


def _restore_claim_payload_into_audit_log(
    audited: pd.DataFrame, proposed: pd.DataFrame
) -> pd.DataFrame:
    """
    If audit_claims dropped claim_json payload columns, restore them from proposed.
    Preference: merge on claim_id. Fallback: restore by row order if necessary.
    """
    out = audited.copy()
    if _pick_claim_payload_col(out) is not None:
        return out

    if "claim_id" in out.columns and "claim_id" in proposed.columns:
        p_cols = [
            c for c in ["claim_json", "claim_json_str", "claim_json_raw"] if c in proposed.columns
        ]
        if p_cols:
            p = proposed[["claim_id"] + p_cols].copy()
            p = p.drop_duplicates(subset=["claim_id"], keep="first")
            out2 = out.merge(p, on="claim_id", how="left", suffixes=("", "_from_proposed"))
            if "claim_json" not in out2.columns:
                col = _pick_claim_payload_col(out2)
                if col is not None and col != "claim_json":
                    out2["claim_json"] = out2[col]
            if "claim_json" in out2.columns:
                out2["claim_json"] = out2["claim_json"].map(
                    lambda x: "" if _is_na_scalar(x) else str(x)
                )
            return out2

    pcol = _pick_claim_payload_col(proposed)
    if pcol is None:
        return out

    a0 = out.reset_index(drop=True)
    p0 = proposed.reset_index(drop=True)
    n = min(len(a0), len(p0))
    a0.loc[: n - 1, "claim_json"] = p0.loc[: n - 1, pcol].map(
        lambda x: "" if _is_na_scalar(x) else str(x)
    )
    return a0


# -------------------------
# Stress helpers (pipeline-owned; user-facing key name is "stress")
# -------------------------
def _apply_evidence_gene_dropout(
    distilled: pd.DataFrame,
    *,
    p_drop: float,
    seed: int,
    genes_col: str = "evidence_genes",
    min_keep: int = 1,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Drop genes from evidence lists to induce identity collapse.
    Deterministic given seed.
    """
    p = float(p_drop)
    if p <= 0.0:
        return distilled.copy(), {"enabled": False, "p_drop": p, "min_keep": int(min_keep)}

    if genes_col not in distilled.columns:
        return distilled.copy(), {
            "enabled": True,
            "p_drop": p,
            "min_keep": int(min_keep),
            "warning": f"missing column {genes_col}",
        }

    rng = __import__("random").Random(int(seed))
    out = distilled.copy()

    dropped_total = 0
    kept_total = 0
    n_rows = 0
    n_rows_changed = 0

    new_lists: list[list[str]] = []
    drop_counts: list[int] = []

    for xs in out[genes_col].tolist():
        n_rows += 1
        if _is_na_scalar(xs):
            genes = []
        elif isinstance(xs, (list, tuple, set)):
            genes = [str(g).strip() for g in xs if str(g).strip()]
        else:
            genes = [g.strip() for g in str(xs).replace(";", ",").split(",") if g.strip()]

        if not genes:
            new_lists.append([])
            drop_counts.append(0)
            continue

        kept: list[str] = []
        dropped: list[str] = []
        for g in genes:
            if rng.random() < p:
                dropped.append(g)
            else:
                kept.append(g)

        if len(kept) < int(min_keep):
            need = int(min_keep) - len(kept)
            if dropped:
                rescue = sorted(dropped)[:need]
                kept.extend(rescue)
                dropped = [g for g in dropped if g not in set(rescue)]

        kept_total += len(kept)
        dropped_total += len(dropped)
        if dropped:
            n_rows_changed += 1

        new_lists.append([_norm_gene_id(g) for g in kept if str(g).strip()])
        drop_counts.append(len(dropped))

    out[genes_col] = new_lists
    out["stress_drop_count"] = drop_counts
    out["evidence_genes_str"] = out[genes_col].map(lambda ys: ",".join([str(y) for y in ys]))

    meta = {
        "enabled": True,
        "p_drop": float(p),
        "min_keep": int(min_keep),
        "n_rows": int(n_rows),
        "n_rows_changed": int(n_rows_changed),
        "dropped_total": int(dropped_total),
        "kept_total": int(kept_total),
    }
    return out, meta


def _build_term_uid_maps(dist: pd.DataFrame) -> tuple[set[str], dict[str, str], set[str]]:
    """
    Build mapping for resolving term_ids to term_uids.

    Safer policy:
      - Prefer using explicit term_id/source columns when available.
      - Avoid parsing term_uid by splitting on ':' (term_id may contain ':').
    """
    # Known term_uids
    if "term_uid" not in dist.columns:
        return set(), {}, set()

    term_uids = dist["term_uid"].astype(str).str.strip().tolist()
    known = {t for t in term_uids if t and t.lower() not in _NA_TOKENS_L}

    raw_to_uids: dict[str, set[str]] = {}

    # Preferred: if term_id exists, map raw term_id
    # -> term_uid (may still be ambiguous across sources)
    if "term_id" in dist.columns:
        term_ids = dist["term_id"].astype(str).str.strip().tolist()
        # If source exists, we can reduce false ambiguity for raw matches
        sources = (
            dist["source"].astype(str).str.strip().tolist() if "source" in dist.columns else None
        )

        for tu, tid in zip(term_uids, term_ids, strict=False):
            if not tu or tu.lower() in _NA_TOKENS_L:
                continue
            if not tid or tid.lower() in _NA_TOKENS_L:
                continue
            raw_to_uids.setdefault(tid, set()).add(tu)

        # Also allow source-qualified raw keys when source exists: "source:term_id"
        if sources is not None:
            for tu, tid, src in zip(term_uids, term_ids, sources, strict=False):
                if not tu or tu.lower() in _NA_TOKENS_L:
                    continue
                if not tid or tid.lower() in _NA_TOKENS_L:
                    continue
                if not src or src.lower() in _NA_TOKENS_L:
                    continue
                raw_to_uids.setdefault(f"{src}:{tid}", set()).add(tu)
    else:
        # Fallback (legacy): use exact term_uid itself as "raw"
        for tu in known:
            raw_to_uids.setdefault(tu, set()).add(tu)

    raw_unique = {raw: next(iter(u)) for raw, u in raw_to_uids.items() if len(u) == 1}
    raw_amb = {raw for raw, u in raw_to_uids.items() if len(u) > 1}
    return known, raw_unique, raw_amb


def _resolve_term_ids_to_uids(
    term_ids: list[str],
    *,
    known: set[str],
    raw_unique: dict[str, str],
    raw_ambiguous: set[str],
) -> tuple[list[str], list[str], list[str]]:
    """
    Resolve input term identifiers to canonical term_uid strings.

    Policy:
      - Accept exact term_uid matches.
      - Accept raw term_id matches (if uniquely mappable).
      - Accept source-qualified "source:term_id" matches when uniquely mappable.
      - If ambiguous, ABSTAIN with explicit reason.
    """
    resolved: list[str] = []
    unknown: list[str] = []
    ambiguous: list[str] = []

    for t in term_ids:
        tt = str(t).strip()
        if not tt:
            continue

        # exact term_uid
        if tt in known:
            resolved.append(tt)
            continue

        # try raw_unique by exact string first (supports "source:term_id" too)
        if tt in raw_unique:
            resolved.append(raw_unique[tt])
            continue
        if tt in raw_ambiguous:
            ambiguous.append(tt)
            continue

        # fallback: try raw term_id (last component) ONLY if present in mapping
        # (This is intentionally conservative; do not split term_uid generally.)
        raw = tt.split(":", 1)[-1].strip()
        if raw in raw_unique:
            resolved.append(raw_unique[raw])
        elif raw in raw_ambiguous:
            ambiguous.append(tt)
        else:
            unknown.append(tt)

    return resolved, unknown, ambiguous


def _score_dropout_stress_on_claims(
    proposed: pd.DataFrame,
    *,
    stressed_distilled: pd.DataFrame,
    genes_col: str = "evidence_genes",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Add stress_* columns based on gene_set_hash drift under stressed_distilled.
    """
    out = proposed.copy()

    for c in ["stress_status", "stress_reason", "stress_notes"]:
        if c not in out.columns:
            out[c] = ""

    dist = stressed_distilled.copy()
    if "term_uid" not in dist.columns:
        if {"source", "term_id"}.issubset(set(dist.columns)):
            dist["term_uid"] = (
                dist["source"].astype(str).str.strip()
                + ":"
                + dist["term_id"].astype(str).str.strip()
            )
        elif "term_id" in dist.columns:
            dist["term_uid"] = dist["term_id"].astype(str).str.strip()
        else:
            out["stress_status"] = "ABSTAIN"
            out["stress_reason"] = "MISSING_TERM_UID"
            out["stress_notes"] = "stressed_distilled missing term identifiers"
            return out, {
                "evaluated": True,
                "enabled": True,
                "n_eval": 0,
                "n_pass": 0,
                "n_fail": 0,
                "n_abstain": int(len(out)),
            }

    known, raw_unique, raw_amb = _build_term_uid_maps(dist)

    term_to_genes: dict[str, set[str]] = {}
    if genes_col in dist.columns:
        term_uids = dist["term_uid"].astype(str).str.strip().tolist()
        genes_vals = dist[genes_col].tolist()
        for tk, xs in zip(term_uids, genes_vals, strict=True):
            if not tk or tk.lower() in _NA_TOKENS_L:
                continue
            if _is_na_scalar(xs):
                genes = []
            elif isinstance(xs, (list, tuple, set)):
                genes = [_norm_gene_id(g) for g in xs if str(g).strip()]
            else:
                genes = [_norm_gene_id(g) for g in _parse_ids(xs)]
            gs = {g for g in genes if g}
            if gs:
                term_to_genes.setdefault(tk, set()).update(gs)

    n_eval = 0
    n_fail = 0
    n_abstain = 0
    n_pass = 0

    if "claim_json" not in out.columns:
        out["claim_json"] = ""

    for i, row in out.iterrows():
        cj = row.get("claim_json", None)
        if cj is None or _is_na_scalar(cj) or (not str(cj).strip()):
            out.at[i, "stress_status"] = "ABSTAIN"
            out.at[i, "stress_reason"] = "MISSING_CLAIM_JSON"
            out.at[i, "stress_notes"] = "cannot evaluate stress without claim_json"
            n_abstain += 1
            continue

        term_ids, gsh_ref, _dir = _extract_claim_evidence(str(cj))
        if not term_ids or not gsh_ref:
            out.at[i, "stress_status"] = "ABSTAIN"
            out.at[i, "stress_reason"] = "MISSING_EVIDENCE_REF"
            out.at[i, "stress_notes"] = "term_ids or gene_set_hash missing"
            n_abstain += 1
            continue

        resolved, unknown, ambiguous = _resolve_term_ids_to_uids(
            term_ids, known=known, raw_unique=raw_unique, raw_ambiguous=raw_amb
        )
        if ambiguous:
            out.at[i, "stress_status"] = "ABSTAIN"
            out.at[i, "stress_reason"] = "AMBIGUOUS_TERM_ID"
            out.at[i, "stress_notes"] = f"ambiguous term_ids={ambiguous} (use term_uid)"
            n_abstain += 1
            continue
        if not resolved:
            out.at[i, "stress_status"] = "ABSTAIN"
            out.at[i, "stress_reason"] = "UNKNOWN_TERM_ID"
            out.at[i, "stress_notes"] = f"unknown term_ids={unknown}"
            n_abstain += 1
            continue

        ev_seen: set[str] = set()
        union: list[str] = []
        for t in resolved:
            for g in sorted(term_to_genes.get(str(t).strip(), set())):
                if g not in ev_seen:
                    ev_seen.add(g)
                    union.append(g)

        if not union:
            out.at[i, "stress_status"] = "ABSTAIN"
            out.at[i, "stress_reason"] = "EMPTY_EVIDENCE_AFTER_STRESS"
            out.at[i, "stress_notes"] = "no union evidence genes under stress"
            n_abstain += 1
            continue

        # --- Compute stress hash under BOTH policies (preserve-case and UPPER) ---
        # Rationale:
        #   claim_json.evidence_ref.gene_set_hash might have been produced by either legacy (UPPER)
        #   or newer preserve-case hashing. To avoid false FAIL, accept either match and record it.
        gsh_ref_norm = str(gsh_ref).strip().lower()

        try:
            gsh_stress_upper = _shared.hash_gene_set_12hex_upper(union).lower()
        except Exception:
            gsh_stress_upper = ""

        try:
            gsh_stress_preserve = _shared.hash_gene_set_12hex(union).lower()
        except Exception:
            gsh_stress_preserve = ""

        n_eval += 1

        matched_policy = ""
        if gsh_stress_upper and (gsh_stress_upper == gsh_ref_norm):
            matched_policy = "upper"
        elif gsh_stress_preserve and (gsh_stress_preserve == gsh_ref_norm):
            matched_policy = "preserve"
        else:
            matched_policy = ""

        if not matched_policy:
            out.at[i, "stress_status"] = "FAIL"
            out.at[i, "stress_reason"] = "EVIDENCE_DROPOUT_GENESET_HASH_DRIFT"
            out.at[i, "stress_notes"] = (
                f"hash_ref={gsh_ref_norm} "
                f"hash_stress_upper={gsh_stress_upper or 'NA'} "
                f"hash_stress_preserve={gsh_stress_preserve or 'NA'}"
            )
            n_fail += 1
        else:
            out.at[i, "stress_status"] = "PASS"
            out.at[i, "stress_reason"] = ""
            out.at[i, "stress_notes"] = (
                f"hash_ref={gsh_ref_norm} "
                f"hash_stress={gsh_ref_norm} (match_policy={matched_policy})"
            )
            n_pass += 1

    stats = {
        "evaluated": True,
        "enabled": True,
        "n_eval": int(n_eval),
        "n_pass": int(n_pass),
        "n_fail": int(n_fail),
        "n_abstain": int(n_abstain),
    }
    return out, stats


def _inject_contradictory_direction(
    proposed: pd.DataFrame,
    *,
    p_inject: float,
    seed: int,
    max_extra: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Duplicate a subset of rows and flip direction in claim_json to force intra-run contradiction.
    Keeps evidence_ref identical.
    """
    p = float(p_inject)
    if p <= 0.0 or proposed.empty:
        return proposed.copy(), {
            "evaluated": True,
            "enabled": False,
            "p_inject": p,
            "n_injected": 0,
        }

    rng = __import__("random").Random(int(seed) + 1337)
    out = proposed.copy()
    out["is_contradict_injected"] = False

    injected_rows: list[pd.Series] = []
    n_injected = 0

    if "claim_json" not in out.columns:
        out["claim_json"] = ""

    for _, row in proposed.iterrows():
        if max_extra > 0 and n_injected >= int(max_extra):
            break
        if rng.random() >= p:
            continue

        cj = row.get("claim_json", None)
        if cj is None or _is_na_scalar(cj) or (not str(cj).strip()):
            continue

        term_ids, gsh_ref, direction = _extract_claim_evidence(str(cj))
        if not term_ids or not gsh_ref:
            continue
        if direction not in {"up", "down"}:
            continue

        row2 = row.copy()
        row2["claim_json"] = _flip_claim_direction_json(str(cj))

        if "direction" in out.columns:
            d = str(row.get("direction", "")).strip().lower()
            if d == "up":
                row2["direction"] = "down"
            elif d == "down":
                row2["direction"] = "up"

        row2["is_contradict_injected"] = True
        injected_rows.append(row2)
        n_injected += 1

    if injected_rows:
        out = pd.concat([out, pd.DataFrame(injected_rows)], ignore_index=True)

    meta = {"evaluated": True, "enabled": True, "p_inject": float(p), "n_injected": int(n_injected)}
    return out, meta


# -------------------------
# Context weights (module×context) learning + context_score_proxy (pipeline-owned)
# -------------------------
def _norm_ctx_piece(x: Any) -> str:
    s = "" if _is_na_scalar(x) else str(x)
    s = s.strip()
    if not s or s.lower() in _NA_TOKENS_L:
        return "NA"
    return re.sub(r"\s+", " ", s)


def _context_swap_info(card: SampleCard) -> dict[str, str]:
    """
    Read swap hints from card.extra in a backward-tolerant way.
    Keys supported:
      - context_swap_from
      - context_swap_to
      - context_swap_to_cancer (legacy-ish)
    """
    ex = _get_card_extra(card)

    def _s(x: Any) -> str:
        try:
            return str(x or "").strip()
        except Exception:
            return ""

    swap_from = _s(ex.get("context_swap_from", ""))
    swap_to = _s(ex.get("context_swap_to", ""))
    swap_to_cancer = _s(ex.get("context_swap_to_cancer", ""))
    if (not swap_to) and swap_to_cancer:
        swap_to = swap_to_cancer

    return {"swap_from": swap_from, "swap_to": swap_to}


def _ctx_id_from_fields(*, condition: str, tissue: Any, perturbation: Any, comparison: Any) -> str:
    parts = [
        _norm_ctx_piece(condition),
        _norm_ctx_piece(tissue),
        _norm_ctx_piece(perturbation),
        _norm_ctx_piece(comparison),
    ]
    return "|".join(parts)


def _ctx_ids_from_card(card: SampleCard) -> dict[str, str]:
    """
    Paper contract:
      - ctx_id_original: derived from "original" condition
        (swap_from if present else card condition)
      - ctx_id_effective: derived from "effective" condition
        (swap_to if present else original)
    We keep tissue/perturbation/comparison unchanged.
    """
    swap = _context_swap_info(card)
    base_condition = str(_card_condition(card) or "").strip()
    cond_original = swap["swap_from"] or base_condition
    cond_effective = swap["swap_to"] or cond_original

    tissue = getattr(card, "tissue", "")
    perturbation = getattr(card, "perturbation", "")
    comparison = getattr(card, "comparison", "")

    ctx_id_original = _ctx_id_from_fields(
        condition=cond_original, tissue=tissue, perturbation=perturbation, comparison=comparison
    )
    ctx_id_effective = _ctx_id_from_fields(
        condition=cond_effective, tissue=tissue, perturbation=perturbation, comparison=comparison
    )

    return {
        "ctx_id_original": ctx_id_original,
        "ctx_id_effective": ctx_id_effective,
        "swap_from": cond_original if swap["swap_from"] else "",
        "swap_to": cond_effective if swap["swap_to"] else "",
    }


def _read_json_quiet(path: str | Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _infer_ctx_id_from_run_meta(meta: dict[str, Any]) -> str:
    sample = meta.get("inputs", {}).get("sample", {})
    if not isinstance(sample, dict):
        sample = {}
    condition = _norm_ctx_piece(sample.get("condition", "NA"))
    tissue = _norm_ctx_piece(sample.get("tissue", "NA"))
    perturbation = _norm_ctx_piece(sample.get("perturbation", "NA"))
    comparison = _norm_ctx_piece(sample.get("comparison", "NA"))
    return "|".join([condition, tissue, perturbation, comparison])


def _load_context_corpus_index() -> pd.DataFrame:
    """
    Load corpus index either from explicit TSV or from meta_glob/dist_glob.

    Returns df with columns: path, ctx_id

    Priority:
      1) LLMPATH_CONTEXT_CORPUS_INDEX_TSV (explicit)
      2) LLMPATH_CONTEXT_CORPUS_META_GLOB (run_meta.json files; preferred)
      3) LLMPATH_CONTEXT_CORPUS_GLOB (distilled.with_modules.tsv files)
         + sibling run_meta.json inference

    Notes:
      - We do NOT invent ctx_id if it cannot be inferred.
      - For dist_glob-only mode, we infer ctx_id from a sibling run_meta.json when available.
    """
    idx_path = _env_str("LLMPATH_CONTEXT_CORPUS_INDEX_TSV", "")
    if idx_path:
        p = Path(idx_path)
        if p.exists() and p.is_file():
            df = pd.read_csv(p, sep="\t")
            if {"path", "ctx_id"}.issubset(set(df.columns)):
                out = df[["path", "ctx_id"]].copy()
                out["path"] = out["path"].astype(str)
                out["ctx_id"] = out["ctx_id"].astype(str)
                return out

    dist_glob = _env_str("LLMPATH_CONTEXT_CORPUS_GLOB", "")
    meta_glob = _env_str("LLMPATH_CONTEXT_CORPUS_META_GLOB", "")

    dist_paths: list[Path] = []
    meta_paths: list[Path] = []
    if dist_glob:
        import glob as _glob

        dist_paths = [Path(x) for x in sorted(_glob.glob(dist_glob)) if Path(x).is_file()]
    if meta_glob:
        import glob as _glob

        meta_paths = [Path(x) for x in sorted(_glob.glob(meta_glob)) if Path(x).is_file()]

    rows: list[dict[str, str]] = []

    # --- Preferred: meta_glob ---
    # run_meta.json → infer ctx_id + locate distilled.with_modules.tsv
    if meta_paths:
        for mp in meta_paths:
            meta = _read_json_quiet(mp)
            ctx_id = _infer_ctx_id_from_run_meta(meta)
            art = meta.get("artifacts", {}) if isinstance(meta.get("artifacts", {}), dict) else {}
            cand = art.get("distilled_with_modules_tsv", "") or art.get(
                "distilled.with_modules.tsv", ""
            )
            dp = Path(cand) if cand else (mp.parent / "distilled.with_modules.tsv")
            if dp.exists() and dp.is_file():
                rows.append({"path": str(dp), "ctx_id": str(ctx_id)})
        if rows:
            return pd.DataFrame(rows)

    # --- Fallback: dist_glob only ---
    # Try to infer ctx_id from sibling run_meta.json near each distilled file.
    if dist_paths:
        for dp in dist_paths:
            # common layouts:
            #   <outdir>/distilled.with_modules.tsv
            #   <outdir>/run_meta.json
            mp = dp.parent / "run_meta.json"
            if not mp.exists() or not mp.is_file():
                # allow custom meta name in corpus; best-effort search
                # (keep it tight: only immediate parent)
                alt = list(dp.parent.glob("*meta*.json"))
                mp = alt[0] if alt else mp

            if mp.exists() and mp.is_file():
                meta = _read_json_quiet(mp)
                ctx_id = _infer_ctx_id_from_run_meta(meta)
                if str(ctx_id).strip():
                    rows.append({"path": str(dp), "ctx_id": str(ctx_id)})

        if rows:
            return pd.DataFrame(rows)

    return pd.DataFrame(columns=["path", "ctx_id"])


def _entropy_from_probs(ps: list[float]) -> float:
    h = 0.0
    for p in ps:
        if p > 0:
            h -= p * math.log(p)
    return h


def _learn_module_context_weights(
    corpus_index: pd.DataFrame,
    *,
    module_col: str = "module_id",
    eps: float = 1e-9,
    min_module_n: int = 3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Learn weights from corpus with RUN-based probabilities.

    Treat each file as one run.
    For each run:
      - read distilled.with_modules.tsv
      - collect unique module_ids present in that run
      - record (run_key, module_id, ctx_id)
    """
    meta: dict[str, Any] = {
        "evaluated": True,
        "n_index_rows": int(getattr(corpus_index, "shape", [0, 0])[0]),
        "n_runs_read": 0,
        "min_module_n": int(min_module_n),
        "eps": float(eps),
    }

    if corpus_index is None or corpus_index.empty:
        return pd.DataFrame(), {**meta, "reason": "EMPTY_CORPUS_INDEX"}

    rows: list[tuple[str, str, str]] = []  # (run_key, module_id, ctx_id)
    n_runs = 0

    for _, r in corpus_index.iterrows():
        p = str(r.get("path", "") or "").strip()
        ctx_id = str(r.get("ctx_id", "") or "").strip()
        if not p or not ctx_id:
            continue
        fp = Path(p)
        if not fp.exists() or not fp.is_file():
            continue
        try:
            df = pd.read_csv(fp, sep="\t")
        except Exception:
            continue

        mc = (
            module_col
            if module_col in df.columns
            else ("module_id_effective" if "module_id_effective" in df.columns else "")
        )
        if not mc:
            continue

        mods = (
            df[mc]
            .map(lambda x: "" if _is_na_scalar(x) else str(x).strip())
            .replace("", pd.NA)
            .dropna()
            .unique()
            .tolist()
        )
        if not mods:
            continue

        n_runs += 1
        run_key = str(fp.resolve())
        for m in mods:
            rows.append((run_key, str(m), str(ctx_id)))

    meta["n_runs_read"] = int(n_runs)
    if not rows:
        return pd.DataFrame(), {**meta, "reason": "NO_MODULES_FOUND_IN_CORPUS"}

    pairs = pd.DataFrame(rows, columns=["run_key", "module_id", "ctx_id"])

    n_total_runs = int(pairs["run_key"].nunique())
    if n_total_runs <= 0:
        return pd.DataFrame(), {**meta, "reason": "N_RUNS_ZERO_AFTER_PARSING"}

    run_mod = pairs.drop_duplicates(subset=["run_key", "module_id"])
    run_ctx = pairs.drop_duplicates(subset=["run_key", "ctx_id"])
    run_modctx = pairs.drop_duplicates(subset=["run_key", "module_id", "ctx_id"])

    c_mc = run_modctx.value_counts(["module_id", "ctx_id"]).reset_index(name="n_runs")
    c_m = run_mod.value_counts(["module_id"]).reset_index(name="n_runs_module")
    c_c = run_ctx.value_counts(["ctx_id"]).reset_index(name="n_runs_ctx")

    out = c_mc.merge(c_m, on="module_id", how="left").merge(c_c, on="ctx_id", how="left")

    out = out.loc[out["n_runs_module"] >= int(min_module_n)].copy()
    if out.empty:
        return pd.DataFrame(), {**meta, "reason": "ALL_MODULES_FILTERED_BY_MIN_N"}

    out["p_mc"] = out["n_runs"] / float(n_total_runs)
    out["p_m"] = out["n_runs_module"] / float(n_total_runs)
    out["p_c"] = out["n_runs_ctx"] / float(n_total_runs)
    out["p_c_given_m"] = out["n_runs"] / out["n_runs_module"].astype(float)

    out["weight_log_odds"] = (out["p_c_given_m"] + eps) / (out["p_c"] + eps)
    out["weight_log_odds"] = out["weight_log_odds"].map(
        lambda x: math.log(float(x)) if float(x) > 0 else float("-inf")
    )

    out["weight_mi"] = (out["p_mc"] + eps) / ((out["p_m"] * out["p_c"]) + eps)
    out["weight_mi"] = out["p_mc"] * out["weight_mi"].map(
        lambda x: math.log(float(x)) if float(x) > 0 else float("-inf")
    )

    ent = (
        out.groupby("module_id")["p_c_given_m"]
        .apply(lambda s: _entropy_from_probs([float(x) for x in s.tolist()]))
        .reset_index(name="module_entropy")
    )
    out = out.merge(ent, on="module_id", how="left")
    k_ctx = out.groupby("module_id")["ctx_id"].nunique().reset_index(name="k_ctx")
    out = out.merge(k_ctx, on="module_id", how="left")
    out["module_entropy_max"] = out["k_ctx"].map(
        lambda k: math.log(float(k)) if float(k) > 1 else 0.0
    )
    out["module_spec"] = out.apply(
        lambda r: 1.0 - (float(r["module_entropy"]) / float(r["module_entropy_max"]))
        if float(r["module_entropy_max"]) > 0
        else 0.0,
        axis=1,
    )

    keep = [
        "module_id",
        "ctx_id",
        "n_runs",
        "n_runs_module",
        "n_runs_ctx",
        "p_c_given_m",
        "p_c",
        "weight_log_odds",
        "weight_mi",
        "module_entropy",
        "module_spec",
        "k_ctx",
    ]
    out = out[keep].sort_values(["module_id", "ctx_id"], kind="mergesort").reset_index(drop=True)

    meta.update(
        {
            "reason": "OK",
            "n_total_runs": int(n_total_runs),
            "n_modules": int(out["module_id"].nunique()),
            "n_contexts": int(out["ctx_id"].nunique()),
        }
    )
    return out, meta


def _attach_context_score_proxy(
    proposed: pd.DataFrame,
    *,
    ctx_id: str,
    weights: pd.DataFrame,
    weight_col: str = "weight_log_odds",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Attach context_score_proxy (module×context weight) to proposed claims.

    IMPORTANT:
      - This proxy is *not* the u01 hash proxy.
      - Do NOT overwrite it later with another proxy.
    """
    out = proposed.copy()
    meta = {"evaluated": True, "ctx_id": str(ctx_id), "weight_col": str(weight_col)}

    if out is None or out.empty:
        return out, {**meta, "reason": "EMPTY_PROPOSED"}

    claim_mod_col = None
    for c in ["module_id_effective", "module_id"]:
        if c in out.columns:
            claim_mod_col = c
            break

    out["context_ctx_id"] = str(ctx_id)

    if claim_mod_col is None:
        out["context_score_proxy"] = 0.0
        out["context_score_proxy_norm"] = 0.0
        out["module_context_weight"] = 0.0
        out["context_proxy_method"] = "module_weight"
        return out, {**meta, "reason": "MISSING_MODULE_ID_IN_PROPOSED"}

    out["context_proxy_method"] = "module_weight"

    if weights is None or weights.empty:
        out["context_score_proxy"] = 0.0
        out["context_score_proxy_norm"] = 0.0
        out["module_context_weight"] = 0.0
        return out, {**meta, "reason": "EMPTY_WEIGHTS"}

    w = weights.loc[weights["ctx_id"].astype(str) == str(ctx_id)].copy()
    if w.empty:
        out["context_score_proxy"] = 0.0
        out["context_score_proxy_norm"] = 0.0
        out["module_context_weight"] = 0.0
        return out, {**meta, "reason": "CTX_ID_NOT_IN_WEIGHTS"}

    if weight_col not in w.columns:
        if "weight_log_odds" in w.columns:
            weight_col = "weight_log_odds"
        elif "weight_mi" in w.columns:
            weight_col = "weight_mi"
        else:
            out["context_score_proxy"] = 0.0
            out["context_score_proxy_norm"] = 0.0
            out["module_context_weight"] = 0.0
            return out, {**meta, "reason": f"MISSING_WEIGHT_COL:{meta['weight_col']}"}

    w2 = w[["module_id", weight_col]].copy()
    w2 = w2.drop_duplicates(subset=["module_id"], keep="first")
    w2 = w2.rename(
        columns={"module_id": "module_id_for_context", weight_col: "module_context_weight"}
    )

    out["module_id_for_context"] = out[claim_mod_col].map(
        lambda x: "" if _is_na_scalar(x) else str(x).strip()
    )
    out = out.merge(w2, on="module_id_for_context", how="left")

    out["module_context_weight"] = pd.to_numeric(
        out["module_context_weight"], errors="coerce"
    ).fillna(0.0)
    out["context_score_proxy"] = out["module_context_weight"].astype(float)

    def _sigmoid(x: float) -> float:
        x = float(x)
        if x >= 30:
            return 1.0
        if x <= -30:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    out["context_score_proxy_norm"] = out["context_score_proxy"].map(_sigmoid)

    meta.update(
        {
            "reason": "OK",
            "claim_mod_col": claim_mod_col,
            "n_scored": int(out.shape[0]),
            "n_nonzero": int((out["context_score_proxy"] != 0.0).sum()),
        }
    )
    return out, meta


def _maybe_rerank_with_context_proxy(
    proposed: pd.DataFrame,
    *,
    lam: float,
) -> pd.DataFrame:
    """
    Conservative deterministic re-rank using module×context weight proxy.
    """
    if proposed is None or proposed.empty:
        return proposed
    if "context_score_proxy_norm" not in proposed.columns:
        return proposed
    if "survival_score" not in proposed.columns:
        return proposed

    out = proposed.copy()
    lam = float(lam)

    s = pd.to_numeric(out["survival_score"], errors="coerce").fillna(0.0).astype(float)
    c = pd.to_numeric(out["context_score_proxy_norm"], errors="coerce").fillna(0.0).astype(float)
    out["_rank_key_context"] = s * (1.0 + lam * c)

    out["_orig_i"] = range(len(out))
    out = out.sort_values(
        ["_rank_key_context", "_orig_i"], ascending=[False, True], kind="mergesort"
    )
    out = out.drop(columns=["_rank_key_context", "_orig_i"]).reset_index(drop=True)
    return out


def _maybe_rerank_with_context_score(
    proposed: pd.DataFrame,
    *,
    lam: float,
) -> pd.DataFrame:
    """
    Conservative deterministic re-rank using FINAL context_score.

    Contract:
      - Use context_score (0..1) which already reflects:
          review confidence > module-weight proxy > hash-u01 proxy
      - Must be executed AFTER context_score is materialized.

    Ranking key:
      survival_score * (1 + lam * context_score)
    """
    if proposed is None or proposed.empty:
        return proposed
    if "context_score" not in proposed.columns:
        return proposed
    if "survival_score" not in proposed.columns:
        return proposed

    out = proposed.copy()
    lam = float(lam)

    s = pd.to_numeric(out["survival_score"], errors="coerce").fillna(0.0).astype(float)
    c = (
        pd.to_numeric(out["context_score"], errors="coerce")
        .fillna(0.0)
        .astype(float)
        .clip(0.0, 1.0)
    )

    out["_rank_key_context"] = s * (1.0 + lam * c)
    out["_orig_i"] = range(len(out))

    out = out.sort_values(
        ["_rank_key_context", "_orig_i"], ascending=[False, True], kind="mergesort"
    )
    out = out.drop(columns=["_rank_key_context", "_orig_i"]).reset_index(drop=True)
    return out


# -------------------------
# Context review (pipeline-owned)
# -------------------------
def _extract_context_fields_from_claim_json(claim_json: Any) -> dict[str, Any]:
    out = {
        "context_status": pd.NA,
        "context_reason": pd.NA,
        "context_notes": pd.NA,
        "context_confidence": pd.NA,
        "context_method": pd.NA,
        "context_gate_mode": pd.NA,
        "context_review_mode": pd.NA,
        "context_evaluated": pd.NA,
    }

    try:
        obj = json.loads(claim_json) if isinstance(claim_json, str) else dict(claim_json)
        if not isinstance(obj, dict):
            return out
    except Exception:
        return out

    if "context_status" in obj:
        out["context_status"] = str(obj.get("context_status") or "").strip() or pd.NA
    if "context_reason" in obj:
        out["context_reason"] = str(obj.get("context_reason") or "").strip() or pd.NA
    if "context_notes" in obj:
        out["context_notes"] = str(obj.get("context_notes") or "").strip() or pd.NA
    if "context_confidence" in obj:
        try:
            out["context_confidence"] = float(obj.get("context_confidence"))
        except Exception:
            out["context_confidence"] = pd.NA

    if "context_method" in obj:
        out["context_method"] = str(obj.get("context_method") or "").strip() or pd.NA
    if "context_gate_mode" in obj:
        out["context_gate_mode"] = str(obj.get("context_gate_mode") or "").strip() or pd.NA
    if "context_review_mode" in obj:
        out["context_review_mode"] = str(obj.get("context_review_mode") or "").strip() or pd.NA
    if "context_evaluated" in obj:
        v = obj.get("context_evaluated")
        out["context_evaluated"] = bool(v) if isinstance(v, bool) else pd.NA

    return out


def _normalize_context_confidence_for_gating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing context_confidence only.
    IMPORTANT: do NOT overwrite existing numeric confidences.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    if "context_status" not in out.columns:
        return out
    if "context_confidence" not in out.columns:
        out["context_confidence"] = pd.NA

    st = out["context_status"].astype(str).str.strip().str.upper()
    conf = pd.to_numeric(out["context_confidence"], errors="coerce")

    # only fill where NA
    pass_mask = st.eq("PASS") & conf.isna()
    failish_mask = ~st.eq("PASS") & conf.isna()

    conf.loc[pass_mask] = 1.0
    conf.loc[failish_mask] = 0.0

    # clamp
    conf = conf.clip(lower=0.0, upper=1.0)

    out["context_confidence"] = conf
    return out


def _sync_context_columns_from_claim_json(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "claim_json" not in df.columns:
        return df

    out = df.copy()
    extracted = out["claim_json"].apply(_extract_context_fields_from_claim_json)
    extracted_df = pd.DataFrame(list(extracted))

    for col in extracted_df.columns:
        if col not in out.columns:
            out[col] = pd.Series([pd.NA] * len(out), index=out.index, dtype="object")
        else:
            if out[col].dtype != "object":
                out[col] = out[col].astype("object")

        if extracted_df[col].dtype != "object":
            extracted_df[col] = extracted_df[col].astype("object")

        mask = ~extracted_df[col].map(_is_na_scalar)
        if bool(mask.any()):
            out.loc[mask, col] = extracted_df.loc[mask, col].values

    return out


def _ensure_context_review_fields(proposed: pd.DataFrame, *, gate_mode: str) -> pd.DataFrame:
    out = proposed.copy()
    gate_mode = _norm_gate_mode(gate_mode, default="soft")

    if "context_evaluated" not in out.columns:
        out["context_evaluated"] = False
    else:
        out["context_evaluated"] = out["context_evaluated"].map(
            lambda x: bool(x) if not _is_na_scalar(x) else False
        )

    if "context_method" not in out.columns:
        out["context_method"] = ""
    if "context_status" not in out.columns:
        out["context_status"] = ""
    if "context_reason" not in out.columns:
        out["context_reason"] = ""
    if "context_notes" not in out.columns:
        out["context_notes"] = ""
    if "context_confidence" not in out.columns:
        out["context_confidence"] = pd.NA
    else:
        out["context_confidence"] = pd.to_numeric(out["context_confidence"], errors="coerce")

    not_eval = ~out["context_evaluated"].astype(bool)
    if not_eval.any():
        out.loc[
            not_eval & (out["context_status"].astype(str).str.strip() == ""), "context_status"
        ] = "UNEVALUATED"
        out.loc[
            not_eval & (out["context_reason"].astype(str).str.strip() == ""), "context_reason"
        ] = "NOT_EVALUATED"

    if "context_gate_mode" not in out.columns:
        out["context_gate_mode"] = gate_mode

    if "context_review_mode" not in out.columns:
        out["context_review_mode"] = ""

    return out


def _write_context_review_into_claim_json(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "claim_json" not in out.columns:
        return out

    fields = [
        "context_evaluated",
        "context_method",
        "context_status",
        "context_reason",
        "context_notes",
        "context_confidence",
        "context_gate_mode",
        "context_review_mode",
    ]

    def _merge_row(s: str, row: pd.Series) -> str:
        s = str(s or "").strip()
        if not s:
            return s
        try:
            obj = json.loads(s)
        except Exception:
            return s
        if not isinstance(obj, dict):
            return s

        for k in fields:
            if k in row.index and not _is_na_scalar(row.get(k)):
                v = row.get(k)

                if k == "context_confidence":
                    try:
                        v = float(v)
                    except Exception:
                        continue

                if isinstance(v, (pd.Timestamp,)):
                    v = str(v)
                if isinstance(v, (bytes, bytearray)):
                    v = v.decode("utf-8", errors="replace")
                if isinstance(v, (pd.Series, pd.DataFrame)):
                    v = str(v)
                obj[k] = v

        try:
            return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return s

    out = out.reset_index(drop=True)
    out["claim_json"] = [_merge_row(out.at[idx, "claim_json"], out.loc[idx]) for idx in out.index]
    out = _canonicalize_claim_json_column(out)
    return out


def _tokenize_context_text(s: str) -> list[str]:
    """
    Deterministic tokenization for context proxy.
    """
    s = str(s or "").strip().lower()
    if not s or s in _NA_TOKENS_L:
        return []
    parts = re.split(r"[^a-z0-9]+", s)
    toks = [p for p in parts if p and len(p) >= 3]
    stop = {
        "and",
        "the",
        "with",
        "from",
        "into",
        "over",
        "under",
        "cell",
        "cells",
        "human",
        "mouse",
        "patient",
        "patients",
        "sample",
        "samples",
        "dataset",
        "data",
        "cancer",
        "tumor",
        "tumour",
        "disease",
        "tissue",
    }
    out: list[str] = []
    seen: set[str] = set()
    for t in toks:
        if t in stop:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _proxy_context_v2_u01(*, ctx_id: str, context_keys: list[str], term_uid: str) -> float:
    """
    Deterministic proxy context score u in [0,1).
    """
    ctx_id = str(ctx_id or "").strip()
    term_uid = str(term_uid or "").strip()
    keys = [str(k).strip() for k in (context_keys or []) if str(k).strip()]
    payload = json.dumps(
        {"ctx_id": ctx_id, "context_keys": keys, "term_uid": term_uid},
        sort_keys=True,
        ensure_ascii=False,
    )
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    x = int(h[:13], 16)  # 52 bits
    return float(x) / float(1 << 52)


def _ensure_proxy_context_score_columns(
    proposed: pd.DataFrame, card: SampleCard, *, ctx_id_override: str | None = None
) -> pd.DataFrame:
    """
    Ensure proxy context scores are materialized deterministically.

    IMPORTANT:
      - If module×context weights already produced context_score_proxy, keep it.
      - Always add hash-u01 proxy into separate columns:
          context_score_proxy_u01 / context_score_proxy_u01_norm
    """
    if proposed is None or proposed.empty:
        return proposed

    out = proposed.copy()

    def _split_ctx_id(ctx_id: str) -> tuple[str, str, str, str]:
        # ctx_id is "condition|tissue|perturbation|comparison" but be tolerant.
        parts = [p.strip() for p in str(ctx_id or "").split("|")]
        while len(parts) < 4:
            parts.append("")
        return parts[0], parts[1], parts[2], parts[3]

    # Decide ctx_id (effective scoring context)
    if ctx_id_override is not None and str(ctx_id_override).strip():
        ctx_id = str(ctx_id_override).strip()
        condition, tissue, perturb, comp = _split_ctx_id(ctx_id)
    else:
        condition = _card_condition(card)
        tissue = str(getattr(card, "tissue", "") or "").strip()
        perturb = str(getattr(card, "perturbation", "") or "").strip()
        comp = str(getattr(card, "comparison", "") or "").strip()
        ctx_id = f"{condition}|{tissue}|{perturb}|{comp}"

    # If already attached by earlier steps, follow that ctx_id for consistency
    existing_ctx_id = None
    if "context_ctx_id" in out.columns and len(out):
        v = out["context_ctx_id"].iloc[0]
        if not _is_na_scalar(v) and str(v).strip():
            existing_ctx_id = str(v).strip()

    if existing_ctx_id:
        ctx_id = existing_ctx_id
        condition, tissue, perturb, comp = _split_ctx_id(ctx_id)

    if "context_ctx_id" not in out.columns:
        out["context_ctx_id"] = ctx_id
    else:
        out["context_ctx_id"] = out["context_ctx_id"].map(
            lambda x: ctx_id if _is_na_scalar(x) or (not str(x).strip()) else str(x)
        )

    # deterministic signature + token count
    toks: list[str] = []
    try:
        for s in (condition, tissue, perturb, comp):
            toks.extend(_tokenize_context_text(s))
    except Exception:
        toks = []

    tok_n = int(len(toks))
    sig_payload = (ctx_id + "|" + ",".join(toks)).encode("utf-8")
    sig12 = hashlib.sha256(sig_payload).hexdigest()[:12]

    if "context_tokens_n" not in out.columns:
        out["context_tokens_n"] = tok_n
    else:
        out["context_tokens_n"] = (
            pd.to_numeric(out["context_tokens_n"], errors="coerce").fillna(tok_n).astype(int)
        )

    if "context_signature" not in out.columns:
        out["context_signature"] = sig12
    else:
        out["context_signature"] = out["context_signature"].map(
            lambda x: sig12 if _is_na_scalar(x) or (not str(x).strip()) else str(x)
        )

    # term_uid for u01
    if "term_uid" not in out.columns:
        if {"source", "term_id"}.issubset(set(out.columns)):
            out["term_uid"] = (
                out["source"].astype(str).str.strip() + ":" + out["term_id"].astype(str).str.strip()
            )
        elif "term_id" in out.columns:
            out["term_uid"] = out["term_id"].astype(str).str.strip()
        else:
            if "context_score_proxy_u01" not in out.columns:
                out["context_score_proxy_u01"] = 0.0
            if "context_score_proxy_u01_norm" not in out.columns:
                out["context_score_proxy_u01_norm"] = out["context_score_proxy_u01"]
            # backward-compat
            if "context_score_proxy" not in out.columns:
                out["context_score_proxy"] = out["context_score_proxy_u01"]
            if "context_score_proxy_norm" not in out.columns:
                out["context_score_proxy_norm"] = out["context_score_proxy_u01_norm"]
            return out

    context_keys = ["condition", "tissue", "perturbation", "comparison"]
    u = out["term_uid"].map(
        lambda tu: _proxy_context_v2_u01(ctx_id=ctx_id, context_keys=context_keys, term_uid=str(tu))
    )

    out["context_score_proxy_u01"] = pd.to_numeric(u, errors="coerce").fillna(0.0).astype(float)
    out["context_score_proxy_u01_norm"] = out["context_score_proxy_u01"]

    # backward-compat: if no other proxy exists, populate legacy columns too
    if "context_score_proxy" not in out.columns:
        out["context_score_proxy"] = out["context_score_proxy_u01"]
    if "context_score_proxy_norm" not in out.columns:
        out["context_score_proxy_norm"] = out["context_score_proxy_u01_norm"]

    return out


def _ensure_context_score_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Contract:
      - context_score is ALWAYS materialized (float in [0,1]) for downstream audit/report.
      - Prefer context_confidence (review output).
      - Fallback to context_score_proxy_norm (module×context proxy) if confidence missing.
      - Final fallback to context_score_proxy_u01_norm (hash-u01 proxy).
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    if "context_confidence" in out.columns:
        conf = pd.to_numeric(out["context_confidence"], errors="coerce")
    else:
        conf = pd.Series([pd.NA] * len(out), index=out.index, dtype="float")

    proxy_norm = (
        pd.to_numeric(out["context_score_proxy_norm"], errors="coerce")
        if "context_score_proxy_norm" in out.columns
        else pd.Series([pd.NA] * len(out), index=out.index, dtype="float")
    )
    u01_norm = (
        pd.to_numeric(out["context_score_proxy_u01_norm"], errors="coerce")
        if "context_score_proxy_u01_norm" in out.columns
        else pd.Series([pd.NA] * len(out), index=out.index, dtype="float")
    )

    if "context_score" in out.columns:
        cs = pd.to_numeric(out["context_score"], errors="coerce")
    else:
        cs = pd.Series([pd.NA] * len(out), index=out.index, dtype="float")

    cs = cs.where(~cs.isna(), conf)
    cs = cs.where(~cs.isna(), proxy_norm)
    cs = cs.where(~cs.isna(), u01_norm)
    cs = cs.fillna(0.0).clip(lower=0.0, upper=1.0)

    out["context_score"] = cs.astype(float)
    return out


def _build_anchor_modules_from_distilled(
    distilled: pd.DataFrame, card: SampleCard
) -> dict[str, Any]:
    dist = distilled.copy()
    if dist is None or dist.empty:
        return {"ok": False, "reason": "EMPTY_DISTILLED", "tokens": [], "n_anchor_terms": 0}

    ctx_fields = [
        _card_condition(card),
        str(getattr(card, "condition", "") or ""),
        str(getattr(card, "tissue", "") or ""),
        str(getattr(card, "perturbation", "") or ""),
        str(getattr(card, "comparison", "") or ""),
    ]
    tokens: list[str] = []
    for s in ctx_fields:
        tokens.extend(_tokenize_context_text(s))

    tok_seen: set[str] = set()
    tok_uniq: list[str] = []
    for t in tokens:
        if t not in tok_seen:
            tok_seen.add(t)
            tok_uniq.append(t)
    tokens = tok_uniq

    mod_col = None
    for c in ["module_id_effective", "module_id"]:
        if c in dist.columns:
            mod_col = c
            break

    if mod_col is None:
        return {
            "ok": False,
            "reason": "MISSING_MODULE_ID_IN_DISTILLED",
            "tokens": tokens,
            "n_anchor_terms": 0,
        }

    cols = []
    for c in ["term_name", "term_id", "term_uid", "source"]:
        if c in dist.columns:
            cols.append(c)

    if not cols:
        return {
            "ok": False,
            "reason": "MISSING_TERM_TEXT_COLS",
            "tokens": tokens,
            "n_anchor_terms": 0,
        }

    term_text = dist[cols].astype(str).agg(" ".join, axis=1).str.lower()

    if not tokens:
        return {"ok": False, "reason": "EMPTY_CONTEXT_TOKENS", "tokens": [], "n_anchor_terms": 0}

    anchor_mask = pd.Series(False, index=dist.index)
    for t in tokens:
        try:
            anchor_mask = anchor_mask | term_text.str.contains(re.escape(t), na=False)
        except Exception:
            continue

    anchor_rows = dist.loc[anchor_mask].copy()
    n_anchor_terms = int(anchor_rows.shape[0])

    if n_anchor_terms == 0:
        return {
            "ok": False,
            "reason": "NO_ANCHOR_TERMS_FOUND",
            "tokens": tokens,
            "n_anchor_terms": 0,
        }

    anchor_modules = (
        anchor_rows[mod_col]
        .map(lambda x: "" if _is_na_scalar(x) else str(x).strip())
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    anchor_modules = [m for m in anchor_modules if str(m).strip()]

    preview_terms: list[str] = []
    if "term_uid" in anchor_rows.columns:
        preview_terms = (
            anchor_rows["term_uid"]
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .head(10)
            .tolist()
        )

    return {
        "ok": True,
        "reason": "OK",
        "tokens": tokens,
        "module_col": mod_col,
        "n_anchor_terms": int(n_anchor_terms),
        "n_anchor_modules": int(len(anchor_modules)),
        "anchor_modules": sorted(anchor_modules)[:50],
        "anchor_terms_preview": preview_terms,
    }


def _proxy_context_review(
    proposed: pd.DataFrame,
    card: SampleCard,
    *,
    gate_mode: str,
    review_mode: str,
    distilled_for_proxy: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Proxy context review (review-grade; deterministic).

    - Primary: hash-u01 proxy_context_v2 (always defined if term_uid exists)
    - Optional: anchor-module sanity check (if anchors found)
        - If u01 says PASS but module is not anchored, downgrade PASS -> ABSTAIN (conservative).
    - Writes: context_status/reason/notes/confidence as *consistent* values.

    Env knobs:
      - LLMPATH_PROXY_CONTEXT_P_FAIL (default 0.05)
      - LLMPATH_PROXY_CONTEXT_P_WARN (default 0.20)
        (Interpretation: u < p_fail => FAIL; p_fail <= u < p_warn => ABSTAIN; u >= p_warn => PASS)
    """
    out = proposed.copy()
    gm = _norm_gate_mode(gate_mode, default="soft")
    rm = _norm_review_mode(review_mode, default="proxy")

    card_ctx = {
        "condition": str(_card_condition(card) or "").strip(),
        "tissue": str(getattr(card, "tissue", "") or "").strip(),
        "perturbation": str(getattr(card, "perturbation", "") or "").strip(),
        "comparison": str(getattr(card, "comparison", "") or "").strip(),
    }

    if "claim_json" not in out.columns:
        out["claim_json"] = ""

    out = _ensure_context_review_fields(out, gate_mode=gm)
    out["context_review_mode"] = rm or "proxy"
    out["context_method"] = "proxy_context_v2"

    # Minimal required key for proxy review: condition only.
    # Other fields can be empty; they will be treated as "NA" in ctx_id/signature.
    if not str(card_ctx.get("condition", "")).strip():
        out["context_evaluated"] = True
        out["context_status"] = "FAIL"
        out["context_reason"] = "MISSING_CONDITION"
        out["context_notes"] = "condition is required for context review"
        out["context_confidence"] = 1.0
        out = _write_context_review_into_claim_json(out)
        meta = {
            "evaluated": True,
            "mode": "proxy_context_v2",
            "gate_mode": gm,
            "reason": "MISSING_CONDITION",
            "n_eval": int(len(out)),
            "n_pass": 0,
            "n_fail": int(len(out)),
            "n_abstain": 0,
        }
        return out, meta

    ctx_id_override = None
    if "context_ctx_id" in out.columns and len(out):
        v = out["context_ctx_id"].iloc[0]
        if not _is_na_scalar(v) and str(v).strip():
            ctx_id_override = str(v).strip()
    out = _ensure_proxy_context_score_columns(out, card, ctx_id_override=ctx_id_override)

    p_fail = float(_env_float("LLMPATH_PROXY_CONTEXT_P_FAIL", 0.05))
    p_warn = float(_env_float("LLMPATH_PROXY_CONTEXT_P_WARN", 0.20))
    p_fail = max(0.0, min(1.0, p_fail))
    p_warn = max(0.0, min(1.0, p_warn))
    if p_warn < p_fail:
        p_warn = p_fail

    anchor_info = (
        _build_anchor_modules_from_distilled(distilled_for_proxy, card)
        if distilled_for_proxy is not None
        else {"ok": False, "reason": "NO_DISTILLED_FOR_PROXY", "tokens": [], "n_anchor_terms": 0}
    )
    anchor_modules_raw = (
        set(anchor_info.get("anchor_modules", []) or []) if anchor_info.get("ok", False) else set()
    )
    # IMPORTANT: anchor-based downgrade is HARD-gate only (soft must remain advisory).
    anchor_modules = anchor_modules_raw if gm == "hard" else set()

    claim_mod_col = None
    for c in ["module_id_effective", "module_id"]:
        if c in out.columns:
            claim_mod_col = c
            break

    n_eval = int(len(out))
    n_pass = 0
    n_fail = 0
    n_abstain = 0
    n_downgrade_anchor = 0

    ctx_id = (
        str(out["context_ctx_id"].iloc[0]) if "context_ctx_id" in out.columns and len(out) else ""
    )
    key_list = ["condition", "tissue", "perturbation", "comparison"]

    out = out.reset_index(drop=True)
    for i in out.index:
        u = (
            float(out.at[i, "context_score_proxy_u01"])
            if "context_score_proxy_u01" in out.columns
            else 0.0
        )
        u = max(0.0, min(1.0, u))

        out.at[i, "context_evaluated"] = True
        out.at[i, "context_confidence"] = u

        if u < p_fail:
            st = "FAIL"
            rsn = "PROXY_U01_LOW_FAIL"
        elif u < p_warn:
            st = "ABSTAIN"
            rsn = "PROXY_U01_LOW_ABSTAIN"
        else:
            st = "PASS"
            rsn = "PROXY_U01_OK"

        mid = ""
        if claim_mod_col is not None:
            mid = (
                ""
                if _is_na_scalar(out.at[i, claim_mod_col])
                else str(out.at[i, claim_mod_col]).strip()
            )

        anchor_note = ""
        if anchor_modules:
            if not mid:
                if st == "PASS":
                    st = "ABSTAIN"
                    rsn = "MISSING_MODULE_ID_FOR_ANCHOR"
                    n_downgrade_anchor += 1
                anchor_note = "anchor=on;module_id_missing"
            else:
                if (mid not in anchor_modules) and (st == "PASS"):
                    st = "ABSTAIN"
                    rsn = "NOT_IN_ANCHOR_MODULE"
                    n_downgrade_anchor += 1
                    anchor_note = f"anchor=on;module={mid};anchored=0"
                else:
                    anchor_note = (
                        f"anchor=on;module={mid};anchored={1 if mid in anchor_modules else 0}"
                    )

        out.at[i, "context_status"] = st
        out.at[i, "context_reason"] = rsn

        ctx_sig = ""
        if "context_signature" in out.columns:
            ctx_sig = str(out.at[i, "context_signature"])

        notes = (
            f"proxy_context_v2: "
            f"u={u:.3f} p_fail={p_fail:.3f} p_warn={p_warn:.3f} "
            f"key={key_list} ctx_sig={ctx_sig}"
        )
        if anchor_note:
            notes = notes + " " + anchor_note
        out.at[i, "context_notes"] = notes

        if st == "PASS":
            n_pass += 1
        elif st == "FAIL":
            n_fail += 1
        else:
            n_abstain += 1

    out = _write_context_review_into_claim_json(out)

    meta = {
        "evaluated": True,
        "mode": "proxy_context_v2",
        "gate_mode": gm,
        "ctx_id": ctx_id,
        "thresholds": {"p_fail": float(p_fail), "p_warn": float(p_warn)},
        "anchor_used": bool(anchor_modules),
        "anchor_info": {k: v for k, v in anchor_info.items() if k != "anchor_modules"},
        "n_eval": int(n_eval),
        "n_pass": int(n_pass),
        "n_fail": int(n_fail),
        "n_abstain": int(n_abstain),
        "n_anchor_downgrade": int(n_downgrade_anchor),
    }
    return out, meta


def _apply_context_review(
    proposed: pd.DataFrame,
    card: SampleCard,
    *,
    gate_mode: str,
    review_mode: str,
    backend: BaseLLMBackend | None,
    seed: int,
    distilled_for_proxy: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Apply context review based on selected mode.

    CONTRACT:
      - If review_mode != off, we must NOT leave context evaluation "empty".
      - If review_mode=llm is requested but no evaluated results are present
        (or backend missing), we MUST fall back to proxy and record it.

    Notes:
      - Pipeline itself does not run LLM prompts here.
      - If select.py already wrote evaluated context into claim_json/columns, we keep it.
      - Otherwise: proxy fallback (never leave unevaluated).
    """
    rm = _norm_review_mode(review_mode, default="proxy")
    gm = _norm_gate_mode(gate_mode, default="soft")

    # --- OFF: explicit no review ---
    if rm == "off":
        out = _ensure_context_review_fields(proposed, gate_mode=gm)

        # Make "off" explicit and stable for downstream gating/logging
        if "context_status" in out.columns:
            empty = out["context_status"].astype(str).str.strip().eq("")
            out.loc[empty, "context_status"] = "UNEVALUATED"
        if "context_reason" in out.columns:
            empty = out["context_reason"].astype(str).str.strip().eq("")
            out.loc[empty, "context_reason"] = "REVIEW_OFF"

        out["context_review_mode"] = "off"
        out["context_method"] = "off"
        out = _write_context_review_into_claim_json(out)
        return out, {
            "evaluated": False,
            "mode": "off",
            "gate_mode": gm,
            "note": "context review disabled",
        }

    # Helper: detect whether any evaluated result exists already.
    def _has_any_evaluated(df: pd.DataFrame) -> bool:
        if df is None or df.empty:
            return False
        if "context_evaluated" in df.columns:
            try:
                s = df["context_evaluated"].map(
                    lambda x: bool(x) if not _is_na_scalar(x) else False
                )
                return bool(s.any())
            except Exception:
                pass
        if "claim_json" in df.columns:
            try:
                ex = df["claim_json"].apply(_extract_context_fields_from_claim_json)
                ex_df = pd.DataFrame(list(ex))
                if "context_evaluated" in ex_df.columns:
                    v = (
                        ex_df["context_evaluated"]
                        .map(lambda x: bool(x) if isinstance(x, bool) else False)
                        .fillna(False)
                    )
                    return bool(v.any())
            except Exception:
                return False
        return False

    already = _has_any_evaluated(proposed)

    # --- LLM requested ---
    if rm == "llm":
        # If backend missing => proxy fallback (and record honestly).
        if backend is None:
            out, meta = _proxy_context_review(
                proposed,
                card,
                gate_mode=gm,
                review_mode="proxy",
                distilled_for_proxy=distilled_for_proxy,
            )
            meta.update(
                {
                    "requested_mode": "llm",
                    "fallback": "proxy",
                    "warning": "context_review_mode=llm but backend is None; fell back to proxy",
                    "seed": int(seed),
                }
            )
            return out, meta

        # Backend exists, but pipeline does not run prompts.
        # If select.py already produced evaluated results, keep them.
        if already:
            # IMPORTANT: pipeline.py does not execute prompts here.
            # If LLM-evaluated fields already exist (likely from select.py),
            # we must not mislabel this step as having run the LLM itself.
            out = _ensure_context_review_fields(proposed, gate_mode=gm)
            out["context_review_mode"] = "llm"
            out["context_method"] = "llm_from_select"
            out = _write_context_review_into_claim_json(out)
            return out, {
                "evaluated": True,
                "mode": "llm",
                "gate_mode": gm,
                "note": (
                    "LLM-evaluated fields were already present "
                    "(likely produced upstream in select.py)."
                ),
                "backend_enabled": True,
                "seed": int(seed),
            }

        # Otherwise: MUST NOT leave it empty => proxy fallback, but record requested llm.
        out, meta = _proxy_context_review(
            proposed,
            card,
            gate_mode=gm,
            review_mode="proxy",
            distilled_for_proxy=distilled_for_proxy,
        )
        meta.update(
            {
                "requested_mode": "llm",
                "fallback": "proxy",
                "warning": "context_review_mode=llm but no evaluated results were present; "
                "fell back to proxy to satisfy contract (no-empty).",
                "backend_enabled": True,
                "seed": int(seed),
            }
        )
        return out, meta

    # --- Proxy (default) ---
    return _proxy_context_review(
        proposed,
        card,
        gate_mode=gm,
        review_mode=rm or "proxy",
        distilled_for_proxy=distilled_for_proxy,
    )


def _restore_context_scores_into_audit_log(
    audited: pd.DataFrame, proposed: pd.DataFrame
) -> pd.DataFrame:
    """
    Ensure audit_log.tsv always contains context_score
    (and key proxy fields) if they exist in proposed.
    """
    out = audited.copy()
    if out is None or out.empty:
        return out
    if proposed is None or proposed.empty:
        return out
    if "claim_id" not in out.columns or "claim_id" not in proposed.columns:
        return out

    keep_cols = [
        "context_score",
        "context_confidence",
        "context_status",
        "context_reason",
        "context_review_mode",
        "context_method",
        "context_gate_mode",
        "context_ctx_id",
        "context_ctx_id_original",
        "context_ctx_id_effective",
        "context_swap_from",
        "context_swap_to",
        "context_signature",
        "context_tokens_n",
        "context_score_proxy",
        "context_score_proxy_norm",
        "context_score_proxy_u01",
        "context_score_proxy_u01_norm",
        "module_context_weight",
        "context_proxy_method",
    ]
    take = [c for c in keep_cols if c in proposed.columns]
    if not take:
        return out

    p = proposed[["claim_id"] + take].copy()
    p = p.drop_duplicates(subset=["claim_id"], keep="first")
    out2 = out.merge(p, on="claim_id", how="left", suffixes=("", "_from_proposed"))

    if "context_score" in out2.columns:
        out2["context_score"] = (
            pd.to_numeric(out2["context_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        )
    if "context_confidence" in out2.columns:
        out2["context_confidence"] = pd.to_numeric(out2["context_confidence"], errors="coerce")

    return out2


def _apply_context_gate_to_proposed(
    proposed: pd.DataFrame,
    *,
    gate_mode: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Pre-audit context gate.

    Goal (paper-facing):
      - hard gate must affect the *input* to audit_claims, not only post-hoc logs.
      - single source of truth is context_status + context_evaluated.

    Policy:
      - if gate_mode != hard: no change
      - if hard:
          * if context_status != PASS -> mark as ineligible (when eligible column exists)
          * also set preselect_tau_gate to 0 if such column exists (defensive)
          * never delete rows (keep for transparency); audit can still log them

    Returns:
      (updated_df, meta)
    """
    gm = _norm_gate_mode(gate_mode, default="soft")
    meta: dict[str, Any] = {
        "evaluated": True,
        "gate_mode": gm,
        "reason": "NOOP",
        "n_rows": int(getattr(proposed, "shape", [0, 0])[0]),
        "n_blocked": 0,
    }

    if proposed is None or proposed.empty:
        meta["reason"] = "EMPTY_PROPOSED"
        return proposed, meta

    if gm != "hard":
        return proposed, meta

    out = proposed.copy()

    if "context_status" not in out.columns:
        meta["reason"] = "MISSING_CONTEXT_STATUS"
        return out, meta

    ctx = out["context_status"].astype(str).str.strip().str.upper()
    blocked = ~ctx.eq("PASS")
    n_blocked = int(blocked.sum())
    meta["n_blocked"] = n_blocked
    meta["reason"] = "OK"

    # Mark eligibility if the column exists; do not invent semantics otherwise.
    if "eligible" in out.columns:
        try:
            out["eligible"] = out["eligible"].map(
                lambda x: bool(x) if not _is_na_scalar(x) else True
            )
            out.loc[blocked, "eligible"] = False
            meta["eligible_written"] = True
        except Exception:
            meta["eligible_written"] = False

    # Defensive: if a preselect gate scalar exists, force it to "off" for blocked rows.
    if "preselect_tau_gate" in out.columns:
        try:
            col = out["preselect_tau_gate"]

            # If it's boolean-like, write False (NOT 0.0).
            if (
                (str(col.dtype).lower() == "bool")
                or isinstance(getattr(col.dtype, "type", None), type)
                and getattr(col.dtype, "type", None) is bool
            ):
                out.loc[blocked, "preselect_tau_gate"] = False
                meta["preselect_tau_gate_written"] = True
            else:
                # Otherwise treat as numeric gate.
                out["preselect_tau_gate"] = pd.to_numeric(
                    out["preselect_tau_gate"], errors="coerce"
                )
                out.loc[blocked, "preselect_tau_gate"] = 0.0
                meta["preselect_tau_gate_written"] = True
        except Exception:
            meta["preselect_tau_gate_written"] = False

    # Paper-friendly traceability
    if "context_gate_applied_pre_audit" not in out.columns:
        out["context_gate_applied_pre_audit"] = False
    out.loc[blocked, "context_gate_applied_pre_audit"] = True

    return out, meta


def _apply_context_gate_to_audited(
    audited: pd.DataFrame,
    proposed: pd.DataFrame,
    *,
    gate_mode: str,
) -> pd.DataFrame:
    gm = _norm_gate_mode(gate_mode, default="soft")
    if gm != "hard":
        return audited

    out = audited.copy()
    if out is None or out.empty:
        return out

    if "claim_id" not in out.columns or "claim_id" not in proposed.columns:
        return out

    # --- Build proposed status table (source of truth for hard gating) ---
    p = proposed[["claim_id"]].copy()
    if "context_status" in proposed.columns:
        p["context_status_from_proposed"] = (
            proposed["context_status"].astype(str).fillna("").map(lambda s: str(s).strip())
        )
    else:
        p["context_status_from_proposed"] = ""

    p = p.drop_duplicates(subset=["claim_id"], keep="first")

    # If audited already has this column, drop it to avoid merge suffixing.
    if "context_status_from_proposed" in out.columns:
        out = out.drop(columns=["context_status_from_proposed"])

    out2 = out.merge(p, on="claim_id", how="left", suffixes=("", "_p"))

    # If merge still suffixed (extremely defensive), recover the column.
    if "context_status_from_proposed" not in out2.columns:
        for cand in (
            "context_status_from_proposed_p",
            "context_status_from_proposed_x",
            "context_status_from_proposed_y",
        ):
            if cand in out2.columns:
                out2["context_status_from_proposed"] = out2[cand]
                break
        else:
            # Nothing to gate with
            return out2

    ctx = out2["context_status_from_proposed"].astype(str).str.strip().str.upper()
    force = ~ctx.eq("PASS")

    if "status" not in out2.columns:
        out2["status"] = ""
    if "abstain_reason" not in out2.columns:
        out2["abstain_reason"] = ""
    if "audit_notes" not in out2.columns:
        out2["audit_notes"] = ""

    st = out2["status"].astype(str).str.strip().str.upper()
    downgrade = force & st.eq("PASS")

    if downgrade.any():
        out2.loc[downgrade, "status"] = "ABSTAIN"

        ar = out2.loc[downgrade, "abstain_reason"].astype(str)
        out2.loc[downgrade, "abstain_reason"] = ar.mask(
            ar.str.len() > 0, ar + ";CONTEXT_GATE"
        ).mask(ar.str.len() == 0, "CONTEXT_GATE")

        an = out2.loc[downgrade, "audit_notes"].astype(str)
        out2.loc[downgrade, "audit_notes"] = an.mask(
            an.str.len() > 0, an + " | context_gate=hard"
        ).mask(an.str.len() == 0, "context_gate=hard")

    return out2


def _excel_safe_ids(x: Any, *, list_sep: str = ";") -> str:
    """
    Pipeline wrapper for spec-level Excel-safe ID serialization.

    IMPORTANT:
      - Delegate to _shared.excel_safe_ids to avoid contract drift.
      - Keep list_sep default aligned with _shared.ID_JOIN_DELIM (";").
    """
    try:
        return _shared.excel_safe_ids(x, list_sep=str(list_sep))
    except Exception:
        return ""


def _fail_if_empty(df: pd.DataFrame, *, step: str, outdir: Path) -> None:
    if df is None or getattr(df, "shape", (0,))[0] == 0:
        raise RuntimeError(f"[{step}] produced 0 rows (outdir={outdir}).")


def run_pipeline(cfg: RunConfig, *, run_id: str | None = None) -> RunResult:
    """
    Run the tool pipeline:
      distill → modules → claims → (context_review) → stress → audit → report (+report.jsonl).
    """
    _require_file(cfg.evidence_table, "evidence_table")
    _require_file(cfg.sample_card, "sample_card")

    _safe_mkdir_outdir(cfg.outdir, cfg.force)

    outdir = Path(cfg.outdir)
    meta_path = outdir / cfg.run_meta_name

    rid = str(run_id or _make_run_id(cfg))

    meta: dict[str, Any] = {
        "tool": "llm-pathway-curator",
        "cmd": "run",
        "run_id": rid,
        "status": "running",
        "started_epoch": time.time(),
        "config": asdict(cfg),
        "env": _env_fingerprint(),
        "inputs": {
            "evidence_table": _file_fingerprint(cfg.evidence_table),
            "sample_card": _file_fingerprint(cfg.sample_card),
            "sample_card_sha256": _sha256_file(cfg.sample_card),
        },
        "artifacts": {},
    }
    _write_json(meta_path, meta)

    def _mark_step(step: str) -> None:
        meta["step"] = step
        _write_json(meta_path, meta)

    def _call_compat(fn: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Call fn(*args, **kwargs) but drop kwargs not accepted by fn signature,
        UNLESS fn accepts **kwargs (VAR_KEYWORD), in which case pass all kwargs.
        """
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return fn(*args, **kwargs)

        for p in sig.parameters.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                return fn(*args, **kwargs)

        allowed = set(sig.parameters.keys())
        filt = {k: v for k, v in kwargs.items() if k in allowed}
        return fn(*args, **filt)

    def _need_llm_backend(*, claim_mode: str, context_review_mode: str) -> bool:
        cm = str(claim_mode).strip().lower()
        rm = str(context_review_mode).strip().lower()
        return (cm == "llm") or (rm == "llm")

    try:
        _mark_step("normalize_inputs")
        ev_tbl = EvidenceTable.read_tsv(cfg.evidence_table)
        meta["inputs"]["evidence_summary"] = ev_tbl.summarize()
        _write_json(meta_path, meta)

        ev_norm_path = outdir / "evidence.normalized.tsv"
        ev_tbl.write_tsv(str(ev_norm_path))
        meta["artifacts"]["evidence_normalized_tsv"] = str(ev_norm_path)
        meta["inputs"]["evidence_normalized_sha256"] = _sha256_file(ev_norm_path)

        ev = ev_tbl.df.copy()

        card_original = SampleCard.from_json(cfg.sample_card)
        try:
            card = card_original.model_copy(deep=True)  # pydantic v2
        except Exception:
            card = card_original  # best-effort fallback

        condition = _card_condition(card)

        species = ""
        try:
            extra = getattr(card, "extra", {}) or {}
            if isinstance(extra, dict):
                species = str(extra.get("species", "") or "").strip()
        except Exception:
            species = ""

        meta.setdefault("inputs", {}).setdefault("sample", {})
        meta["inputs"]["sample"].update(
            {
                "condition": condition,
                "tissue": str(getattr(card, "tissue", "") or "").strip(),
                "perturbation": str(getattr(card, "perturbation", "") or "").strip(),
                "comparison": str(getattr(card, "comparison", "") or "").strip(),
                "species": species,
            }
        )
        seed0 = int(cfg.seed or 42)

        effective_tau = _resolve_tau(cfg.tau, card)

        sc_raw_path = outdir / "sample_card.raw.json"
        _write_json(sc_raw_path, _dump_model(card_original))
        meta["artifacts"]["sample_card_raw_json"] = str(sc_raw_path)
        meta["inputs"]["sample_card_raw_sha256"] = _sha256_file(sc_raw_path)

        sc_norm_path = outdir / "sample_card.normalized.json"
        _write_json(sc_norm_path, _dump_model(card))
        meta["artifacts"]["sample_card_normalized_json"] = str(sc_norm_path)
        meta["inputs"]["sample_card_normalized_sha256"] = _sha256_file(sc_norm_path)

        try:
            tau_card = float(card.audit_tau())
        except Exception:
            tau_card = None

        meta["inputs"]["audit"] = {
            "tau_cfg": cfg.tau,
            "tau_card": tau_card,
            "tau_effective": effective_tau,
            "min_gene_overlap": card.audit_min_gene_overlap(),
        }
        _write_json(meta_path, meta)

        _mark_step("distill")
        distilled = distill_evidence(ev, card, seed=int(seed0))

        dist_path = outdir / "distilled.tsv"
        _write_tsv(distilled, dist_path)
        meta["artifacts"]["distilled_tsv"] = str(dist_path)

        _mark_step("modules")
        mod_out = factorize_modules_connected_components(distilled)

        meta["inputs"]["modules"] = getattr(mod_out.edges_df, "attrs", {}).get("modules", {})
        _write_json(meta_path, meta)

        if not mod_out.modules_df.empty:
            meta["inputs"]["modules_stats"] = {
                "n_modules": int(mod_out.modules_df.shape[0]),
                "n_terms_median": float(mod_out.modules_df["n_terms"].median()),
                "n_genes_median": float(mod_out.modules_df["n_genes"].median()),
                "n_terms_max": int(mod_out.modules_df["n_terms"].max()),
                "n_genes_max": int(mod_out.modules_df["n_genes"].max()),
            }
            _write_json(meta_path, meta)

        modules_path = outdir / "modules.tsv"
        term_modules_path = outdir / "term_modules.tsv"
        edges_path = outdir / "term_gene_edges.tsv"
        _write_tsv(mod_out.modules_df, modules_path)
        _write_tsv(mod_out.term_modules_df, term_modules_path)
        _write_tsv(mod_out.edges_df, edges_path)

        meta["artifacts"].update(
            {
                "modules_tsv": str(modules_path),
                "term_modules_tsv": str(term_modules_path),
                "term_gene_edges_tsv": str(edges_path),
            }
        )

        distilled2 = attach_module_ids(
            distilled, mod_out.term_modules_df, modules_df=mod_out.modules_df
        )
        dist2_path = outdir / "distilled.with_modules.tsv"
        _write_tsv(distilled2, dist2_path)
        meta["artifacts"]["distilled_with_modules_tsv"] = str(dist2_path)

        _mark_step("select_claims")

        # Resolve gate/review modes (env > card.extra), then enforce compatibility centrally.
        context_gate_mode = _norm_gate_mode(
            _resolve_context_gate_mode(card, default="soft"), default="soft"
        )
        context_review_mode = _norm_review_mode(
            _resolve_context_review_mode(card, default="proxy"), default="proxy"
        )

        compat_final = _enforce_gate_review_compat_effective(
            card, gate_mode=context_gate_mode, review_mode=context_review_mode
        )

        # Apply possibly-updated modes returned by the compat guard.
        context_gate_mode = _norm_gate_mode(
            compat_final.get("gate_mode", context_gate_mode), default="soft"
        )
        context_review_mode = _norm_review_mode(
            compat_final.get("review_mode", context_review_mode), default="proxy"
        )

        meta.setdefault("inputs", {}).setdefault("claims", {})
        meta["inputs"]["claims"]["context_gate_review_compat_final"] = compat_final
        _write_json(meta_path, meta)

        claim_mode_env = _env_str("LLMPATH_CLAIM_MODE", "").strip().lower()
        mode_arg = "llm" if claim_mode_env == "llm" else "deterministic"

        shared_backend: BaseLLMBackend | None = None
        shared_notes = ""
        if _need_llm_backend(claim_mode=claim_mode_env, context_review_mode=context_review_mode):
            try:
                shared_backend = get_backend_from_env(seed=int(seed0))
            except Exception as e:
                shared_backend = None
                shared_notes = f"backend_init_error:{type(e).__name__}"

        claim_backend = shared_backend if claim_mode_env == "llm" else None
        review_backend = shared_backend if context_review_mode == "llm" else None

        backend_for_select: BaseLLMBackend | None = claim_backend or review_backend
        backend_for_select_role = (
            "claim" if claim_backend is not None else ("review" if review_backend else "none")
        )

        meta["inputs"]["llm"] = {
            "claim": {
                "enabled": (claim_mode_env == "llm"),
                "claim_mode_env": claim_mode_env,
                "backend_env": _env_str("LLMPATH_BACKEND", "").lower(),
                "backend_enabled": bool(claim_backend is not None),
                "backend_notes": shared_notes if (claim_mode_env == "llm") else "",
            },
            "review": {
                "enabled": (context_review_mode == "llm"),
                "review_mode": context_review_mode,
                "backend_env": _env_str("LLMPATH_BACKEND", "").lower(),
                "backend_enabled": bool(review_backend is not None),
                "backend_notes": shared_notes if (context_review_mode == "llm") else "",
            },
            "select_entrypoint": {
                "backend_attached": bool(backend_for_select is not None),
                "backend_role": backend_for_select_role,
            },
        }

        k_eff, k_meta = _resolve_k_claims(cfg, card)
        meta.setdefault("inputs", {}).setdefault("claims", {})
        meta["inputs"]["claims"].update(
            {
                **k_meta,
                "k_effective": int(k_eff),
                "k_env_raw": os.environ.get("LLMPATH_K_CLAIMS", ""),
                "context_gate_mode": context_gate_mode,
                "context_review_mode": context_review_mode,
            }
        )
        _write_json(meta_path, meta)

        proposed = _call_compat(
            select_claims,
            distilled=distilled2,
            card=card,
            k=int(k_eff),
            mode=mode_arg,
            backend=backend_for_select,
            claim_backend=claim_backend,
            review_backend=review_backend,
            context_gate_mode=context_gate_mode,
            context_review_mode=context_review_mode,
            seed=int(seed0),
            outdir=str(outdir),
        )

        if proposed is None or proposed.empty:
            proposed = _call_compat(
                select_claims,
                distilled=distilled2,
                card=card,
                k=int(k_eff),
                mode="deterministic",
                backend=review_backend if context_review_mode == "llm" else None,
                claim_backend=None,
                review_backend=review_backend,
                context_gate_mode=context_gate_mode,
                context_review_mode=context_review_mode,
                seed=int(seed0),
                outdir=str(outdir),
            )

        _fail_if_empty(proposed, step="select_claims", outdir=outdir)

        proposed = _ensure_claim_payload_and_id(proposed, card)
        proposed = _ensure_context_review_fields(proposed, gate_mode=context_gate_mode)

        # --- Context IDs (original vs effective; supports context_swap) ---
        ctx_ids = _ctx_ids_from_card(card)
        ctx_id_original = ctx_ids["ctx_id_original"]
        ctx_id_effective = ctx_ids["ctx_id_effective"]
        meta.setdefault("inputs", {}).setdefault("claims", {})
        meta["inputs"]["claims"]["context_ctx_id_original"] = ctx_id_original
        meta["inputs"]["claims"]["context_ctx_id_effective"] = ctx_id_effective
        meta["inputs"]["claims"]["context_swap_from"] = ctx_ids.get("swap_from", "")
        meta["inputs"]["claims"]["context_swap_to"] = ctx_ids.get("swap_to", "")

        # --- module×context weights (optional) ---
        weights_df = pd.DataFrame()
        w_meta: dict[str, Any] = {"evaluated": False, "reason": "disabled"}

        has_corpus = bool(
            _env_str("LLMPATH_CONTEXT_CORPUS_INDEX_TSV", "")
            or _env_str("LLMPATH_CONTEXT_CORPUS_META_GLOB", "")
            or _env_str("LLMPATH_CONTEXT_CORPUS_GLOB", "")
        )

        if has_corpus:
            corpus_index = _load_context_corpus_index()
            weights_df, w_meta = _learn_module_context_weights(
                corpus_index,
                module_col="module_id",
                eps=float(_env_float("LLMPATH_CONTEXT_EPS", 1e-9)),
                min_module_n=int(_env_int("LLMPATH_CONTEXT_MIN_MODULE_N", 3)),
            )

            if weights_df is not None and (not weights_df.empty):
                w_path = outdir / "module_context_weights.tsv"
                _write_tsv(weights_df, w_path)
                meta["artifacts"]["module_context_weights_tsv"] = str(w_path)

        weight_choice = _env_str("LLMPATH_CONTEXT_WEIGHT", "log_odds").strip().lower()
        wcol = (
            "weight_log_odds" if weight_choice in {"logodds", "log_odds", "odds"} else "weight_mi"
        )

        proposed, ctx_score_meta = _attach_context_score_proxy(
            proposed,
            ctx_id=ctx_id_effective,
            weights=weights_df,
            weight_col=wcol,
        )

        rerank = _env_str("LLMPATH_CONTEXT_RERANK", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        lam = float(_env_float("LLMPATH_CONTEXT_RERANK_LAMBDA", 0.25))
        rerank_pending = bool(rerank)

        meta["inputs"]["claims"]["context_weight_learning"] = w_meta
        meta["inputs"]["claims"]["context_score_proxy"] = ctx_score_meta
        meta["inputs"]["claims"]["context_rerank"] = {
            "enabled": bool(rerank_pending),
            "lambda": float(lam),
            "timing": "post_review_context_score",
            "rank_key": "survival_score*(1+lambda*context_score)",
        }
        _write_json(meta_path, meta)

        # --- Context review (llm/proxy/off) ---
        proposed, ctx_review_meta = _apply_context_review(
            proposed,
            card,
            gate_mode=context_gate_mode,
            review_mode=context_review_mode,
            backend=review_backend,
            seed=int(seed0),
            distilled_for_proxy=distilled2,
        )

        # --- Ensure proxy columns (u01) are present ---
        proposed = _ensure_proxy_context_score_columns(
            proposed, card, ctx_id_override=ctx_id_effective
        )

        proposed["context_ctx_id_original"] = ctx_id_original
        proposed["context_ctx_id_effective"] = ctx_id_effective
        proposed["context_swap_from"] = ctx_ids.get("swap_from", "")
        proposed["context_swap_to"] = ctx_ids.get("swap_to", "")

        # >>> CRITICAL FIX: always materialize FINAL context_score (single source of truth)
        proposed = _ensure_context_score_columns(proposed)

        # --- Post-review rerank (final context_score) ---
        if "rerank_pending" in locals() and bool(rerank_pending):
            proposed = _maybe_rerank_with_context_score(proposed, lam=float(lam))

        # --- Determine score source (log-truthful) ---
        src = "unknown"
        evaluated_any = False
        try:
            if "context_evaluated" in proposed.columns:
                evaluated_any = bool(
                    proposed["context_evaluated"]
                    .map(lambda x: bool(x) if not _is_na_scalar(x) else False)
                    .any()
                )
        except Exception:
            evaluated_any = False

        if evaluated_any:
            # Prefer explicit method label over inference
            meth = ""
            try:
                if "context_method" in proposed.columns:
                    m = proposed["context_method"].astype(str).str.strip().str.lower()
                    meth = m.iloc[0] if len(m) else ""
            except Exception:
                meth = ""

            if meth == "llm":
                src = "llm_review_confidence"
            elif meth.startswith("proxy"):
                src = "proxy_review_confidence"
            elif meth == "off":
                src = "off"
            else:
                src = "review_confidence"
        else:
            used_module_weight = False
            try:
                if "context_proxy_method" in proposed.columns:
                    m = proposed["context_proxy_method"].astype(str).str.strip()
                    used_module_weight = bool((m == "module_weight").any())
                if used_module_weight and "module_context_weight" in proposed.columns:
                    w = pd.to_numeric(proposed["module_context_weight"], errors="coerce").fillna(
                        0.0
                    )
                    used_module_weight = bool((w != 0.0).any())
            except Exception:
                used_module_weight = False

            src = "module_weight_proxy" if used_module_weight else "hash_u01_proxy"

        meta.setdefault("inputs", {}).setdefault("claims", {})
        meta["inputs"]["claims"]["context_score_source"] = src
        _write_json(meta_path, meta)

        proposed = _canonicalize_claim_json_column(proposed)

        meta["inputs"]["claims"].update(
            {
                "n_proposed_rows": int(getattr(proposed, "shape", [0, 0])[0]),
                "n_proposed_cols": int(getattr(proposed, "shape", [0, 0])[1]),
                "claim_payload_col": _pick_claim_payload_col(proposed) or "",
                "claim_json_present": bool("claim_json" in proposed.columns),
                "context_gate_mode": context_gate_mode,
                "context_review_mode": context_review_mode,
                "context_review_runtime": ctx_review_meta,
            }
        )
        _write_json(meta_path, meta)

        _mark_step("stress")

        p_drop = float(
            cfg.stress_evidence_dropout_p
            if cfg.stress_evidence_dropout_p is not None
            else _env_float("LLMPATH_STRESS_EVIDENCE_DROPOUT_P", 0.0)
        )
        min_keep = int(
            cfg.stress_evidence_dropout_min_keep
            if cfg.stress_evidence_dropout_min_keep is not None
            else _env_int("LLMPATH_STRESS_EVIDENCE_DROPOUT_MIN_KEEP", 1)
        )
        p_contra = float(
            cfg.stress_contradictory_p
            if cfg.stress_contradictory_p is not None
            else _env_float("LLMPATH_STRESS_CONTRADICTORY_P", 0.0)
        )
        contra_cap = int(
            cfg.stress_contradictory_max_extra
            if cfg.stress_contradictory_max_extra is not None
            else _env_int("LLMPATH_STRESS_CONTRADICTORY_MAX_EXTRA", 0)
        )

        meta["inputs"]["stress"] = {
            "evidence_dropout_p": float(p_drop),
            "evidence_dropout_min_keep": int(min_keep),
            "contradictory_p": float(p_contra),
            "contradictory_max_extra": int(contra_cap),
            "seed": int(seed0),
        }
        _write_json(meta_path, meta)

        stressed_distilled = distilled2.copy()

        dropout_meta: dict[str, Any] = {
            "evaluated": True,
            "enabled": False,
            "p_drop": float(p_drop),
            "min_keep": int(min_keep),
        }
        stress_score_meta: dict[str, Any] = {
            "evaluated": True,
            "enabled": False,
            "n_eval": 0,
            "n_pass": 0,
            "n_fail": 0,
            "n_abstain": 0,
        }
        contra_meta: dict[str, Any] = {
            "evaluated": True,
            "enabled": False,
            "p_inject": float(p_contra),
            "n_injected": 0,
        }

        if float(p_drop) > 0.0:
            stressed_distilled, dropout_meta = _apply_evidence_gene_dropout(
                distilled2,
                p_drop=float(p_drop),
                seed=int(seed0),
                genes_col="evidence_genes",
                min_keep=int(min_keep),
            )

            stressed_path = outdir / "distilled.stressed_dropout.tsv"
            _write_tsv(stressed_distilled, stressed_path)
            meta["artifacts"]["distilled_stressed_dropout_tsv"] = str(stressed_path)

            proposed, stress_score_meta = _score_dropout_stress_on_claims(
                proposed, stressed_distilled=stressed_distilled, genes_col="evidence_genes"
            )
        else:
            stress_score_meta = {
                "evaluated": True,
                "enabled": False,
                "reason": "evidence_dropout_p==0",
            }

        if float(p_contra) > 0.0:
            proposed, contra_meta = _inject_contradictory_direction(
                proposed,
                p_inject=float(p_contra),
                seed=int(seed0),
                max_extra=int(contra_cap),
            )
            proposed = _canonicalize_claim_json_column(proposed)
        else:
            contra_meta = {
                "evaluated": True,
                "enabled": False,
                "reason": "contradictory_p==0",
                "n_injected": 0,
            }

        meta["inputs"]["stress_runtime"] = {
            "dropout": dropout_meta,
            "dropout_scoring": stress_score_meta,
            "contradictory": contra_meta,
        }
        _write_json(meta_path, meta)

        if "gene_ids" in proposed.columns:
            proposed["gene_ids_excel"] = proposed["gene_ids"].map(_excel_safe_ids)
        elif "gene_ids_str" in proposed.columns:
            proposed["gene_ids_excel"] = proposed["gene_ids_str"].map(_excel_safe_ids)

        if "term_ids" in proposed.columns:
            proposed["term_ids_excel"] = proposed["term_ids"].map(_excel_safe_ids)
        elif "term_ids_str" in proposed.columns:
            proposed["term_ids_excel"] = proposed["term_ids_str"].map(_excel_safe_ids)

        id2sym: dict[str, str] = {}
        try:
            id2sym = build_id_to_symbol_from_distilled(distilled2)
        except Exception:
            id2sym = {}

        try:
            map_path = _env_str("LLMPATH_GENE_ID_MAP_TSV", "")
            if map_path:
                extra_map = load_id_map_tsv(map_path)
                if isinstance(extra_map, dict) and extra_map:
                    id2sym = {**id2sym, **extra_map}
        except Exception:
            pass

        if ("gene_ids" in proposed.columns) and id2sym:
            raw = proposed["gene_ids"].map(lambda x: "" if _is_na_scalar(x) else str(x))
            syms = raw.map(lambda x: map_ids_to_symbols(x, id2sym))
            proposed["gene_symbols"] = syms
            proposed["gene_symbols_str"] = syms.map(
                lambda xs: ";".join(map(str, xs)) if isinstance(xs, list) else ""
            )

        proposed = _sync_context_columns_from_claim_json(proposed)
        proposed = _normalize_context_confidence_for_gating(proposed)

        # >>> CRITICAL FIX (defensive): keep context_score present even after sync/normalize
        proposed = _ensure_context_score_columns(proposed)

        proposed, pre_audit_gate_meta = _apply_context_gate_to_proposed(
            proposed,
            gate_mode=context_gate_mode,
        )
        meta.setdefault("inputs", {}).setdefault("claims", {})
        meta["inputs"]["claims"]["context_gate_pre_audit"] = pre_audit_gate_meta
        _write_json(meta_path, meta)

        proposed_path = outdir / "claims.proposed.tsv"
        _write_tsv(proposed, proposed_path)
        meta["artifacts"]["claims_proposed_tsv"] = str(proposed_path)

        _mark_step("audit")
        audited = _call_compat(audit_claims, proposed, distilled2, card, tau=effective_tau)

        audited = _restore_claim_payload_into_audit_log(audited, proposed)
        audited = _restore_context_scores_into_audit_log(audited, proposed)
        audited = _apply_context_gate_to_audited(audited, proposed, gate_mode=context_gate_mode)

        for c in ["abstain_reason", "fail_reason", "audit_notes", "module_reason"]:
            if c in audited.columns:
                audited[c] = audited[c].astype("string").fillna("")

        audit_path = outdir / "audit_log.tsv"
        _write_tsv(audited, audit_path)
        meta["artifacts"]["audit_log_tsv"] = str(audit_path)

        _mark_step("report")
        write_report(audited, distilled2, card, str(outdir))
        meta["artifacts"]["report_dir"] = str(outdir)

        _mark_step("report_jsonl")
        jsonl_path = _call_compat(
            write_report_jsonl,
            audit_log=audited,
            card=card,
            outdir=str(outdir),
            run_id=rid,
            tau=effective_tau,
            method="llm-pathway-curator",
            condition=condition,
            comparison=str(getattr(card, "comparison", "") or ""),
        )
        meta["artifacts"]["report_jsonl"] = str(jsonl_path)

        meta["status"] = "ok"
        meta["finished_epoch"] = time.time()
        _write_json(meta_path, meta)

        return RunResult(
            run_id=rid,
            outdir=str(outdir),
            artifacts={k: str(v) for k, v in meta["artifacts"].items()},
            meta_path=str(meta_path),
        )

    except KeyboardInterrupt:
        meta["status"] = "aborted"
        meta["finished_epoch"] = time.time()
        _write_json(meta_path, meta)
        raise
    except Exception as e:
        meta["status"] = "error"
        meta["finished_epoch"] = time.time()
        meta["error"] = {"type": type(e).__name__, "message": str(e), "step": meta.get("step")}
        _write_json(meta_path, meta)
        raise
