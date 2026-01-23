# LLM-PathwayCurator/src/llm_pathway_curator/pipeline.py
from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .audit import audit_claims
from .backends import BaseLLMBackend, GeminiBackend, OllamaBackend, OpenAIBackend
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


@dataclass(frozen=True)
class RunResult:
    run_id: str
    outdir: str
    artifacts: dict[str, str]
    meta_path: str


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


def _env_fingerprint() -> dict[str, Any]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "pandas": getattr(pd, "__version__", "unknown"),
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
    # 1) cfg
    if cfg.k_claims is not None:
        k = int(cfg.k_claims)
        if k < 1:
            raise ValueError(f"k_claims must be >= 1 (cfg.k_claims={cfg.k_claims})")
        return k, {"k_source": "cfg", "k_cfg": cfg.k_claims, "k_env": "", "k_card": ""}

    # 2) env
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

    # 3) card default (optional; tolerate absence)
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

    # 4) fallback
    return 3, {"k_source": "fallback", "k_cfg": None, "k_env": "", "k_card": ""}


# -------------------------
# Tool-wide NA parsing helpers (pipeline-owned)
# -------------------------
_NA_TOKENS = {"", "na", "nan", "none", "NA"}
_NA_TOKENS_L = {t.lower() for t in _NA_TOKENS}


def _is_na_scalar(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, (list, tuple, set, dict)):
        return False
    try:
        v = pd.isna(x)
        return bool(v) if isinstance(v, bool) else False
    except Exception:
        return False


def _parse_ids(x: Any) -> list[str]:
    """
    Backward-tolerant ID parser:
      - accepts list-like
      - accepts Excel-safe strings (may begin with a single quote)
      - accepts comma/semicolon/pipe/whitespace separated strings
    """
    if _is_na_scalar(x):
        return []

    if isinstance(x, (list, tuple, set)):
        items = [str(t).strip() for t in x if str(t).strip()]
    else:
        s = str(x).strip()
        if s.startswith("'"):
            s = s[1:].strip()
        if not s or s.lower() in _NA_TOKENS_L:
            return []
        s = (
            s.replace("|", ",")
            .replace(";", ",")
            .replace("\t", ",")
            .replace("\n", ",")
            .replace(" ", ",")
        )
        items = [t.strip() for t in s.split(",") if t.strip()]

    seen: set[str] = set()
    uniq: list[str] = []
    for t in items:
        tt = str(t).strip()
        if not tt or tt.lower() in _NA_TOKENS_L:
            continue
        if tt not in seen:
            seen.add(tt)
            uniq.append(tt)
    return uniq


def _norm_gene_id(g: str) -> str:
    return str(g).strip().upper()


def _hash_gene_set_12(genes: list[str]) -> str:
    uniq = sorted({_norm_gene_id(g) for g in genes if str(g).strip()})
    payload = ",".join(uniq)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


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


def _card_condition(card: SampleCard) -> str:
    """
    Tool-facing neutral label. Prefer card.condition if present; fall back to card.disease.
    This prevents the pipeline from looking cancer-specific.
    """
    c = ""
    if hasattr(card, "condition"):
        try:
            c = str(card.condition or "").strip()
        except Exception:
            c = ""
    if not c:
        try:
            c = str(getattr(card, "disease", "") or "").strip()
        except Exception:
            c = ""
    return c


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
    """
    out = proposed.copy()

    if "claim_id" not in out.columns:
        out["claim_id"] = ""

    payload_col = _pick_claim_payload_col(out)
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

        missing_id = out["claim_id"].map(lambda x: _is_na_scalar(x) or (not str(x).strip()))
        if missing_id.any():
            out.loc[missing_id, "claim_id"] = out.loc[missing_id, "claim_json"].map(
                lambda s: _maybe_claim_id_from_json(str(s))
            )

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

        # Canonicalize (single-line, deterministic)
        out = _canonicalize_claim_json_column(out)
        return out

    # No payload at all -> synthesize
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

    # Canonicalize (single-line, deterministic)
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

    Returns:
      - stressed_distilled (copy)
      - stress_meta (summary stats)
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


def _resolve_term_ids_to_uids(
    term_ids: list[str],
    *,
    known: set[str],
    raw_unique: dict[str, str],
    raw_ambiguous: set[str],
) -> tuple[list[str], list[str], list[str]]:
    """
    Resolve term_ids to term_uid:
      - if already known term_uid => keep
      - else if raw term_id uniquely maps => replace
      - else track unknown/ambiguous
    """
    resolved: list[str] = []
    unknown: list[str] = []
    ambiguous: list[str] = []

    for t in term_ids:
        tt = str(t).strip()
        if not tt:
            continue
        if tt in known:
            resolved.append(tt)
            continue
        raw = tt.split(":", 1)[-1].strip()
        if raw in raw_unique:
            resolved.append(raw_unique[raw])
            continue
        if raw in raw_ambiguous:
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

    Rule:
      - recompute gene_set_hash from stressed distilled union evidence genes over claim term_ids
      - compare to claim_json evidence_ref.gene_set_hash
      - mismatch => stress FAIL (identity collapse)
      - if cannot evaluate => stress ABSTAIN
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

        gsh_stress = _hash_gene_set_12(union).lower()
        n_eval += 1

        if gsh_stress != str(gsh_ref).strip().lower():
            out.at[i, "stress_status"] = "FAIL"
            out.at[i, "stress_reason"] = "EVIDENCE_DROPOUT_GENESET_HASH_DRIFT"
            out.at[i, "stress_notes"] = (
                f"hash_stress={gsh_stress} != hash_ref={str(gsh_ref).strip().lower()}"
            )
            n_fail += 1
        else:
            out.at[i, "stress_status"] = "PASS"
            out.at[i, "stress_reason"] = ""
            out.at[i, "stress_notes"] = f"hash_stress={gsh_stress} (ok)"
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

    max_extra:
      - 0 means no cap (inject based on probability)
      - >0 caps number of injected duplicates (useful for small runs)
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
# Context review (pipeline-owned)
# -------------------------
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


def _resolve_gate_review_compat(card: SampleCard) -> dict[str, Any]:
    """
    Guardrail:
      hard gate + review off => guaranteed all-ABSTAIN
    Policy:
      - If gate=hard and review is off/empty, auto-set review to 'proxy'
    """
    ex = _get_card_extra(card)
    gate = str(ex.get("context_gate_mode", "") or "").strip().lower()
    review = str(ex.get("context_review_mode", "") or "").strip().lower()

    if gate != "hard":
        return {"compat_action": "none", "gate_mode": gate, "review_mode": review}

    if not review or review in {"off", "false", "0", "none", "disabled"}:
        _set_card_extra(card, "context_review_mode", "proxy")
        return {
            "compat_action": "auto_enable_context_review_proxy",
            "gate_mode": "hard",
            "review_mode_before": review,
            "review_mode_after": "proxy",
        }

    return {"compat_action": "none", "gate_mode": "hard", "review_mode": review}


def _resolve_context_gate_mode(card: SampleCard, *, default: str = "soft") -> str:
    """
    Single source-of-truth for context gate mode.
    Priority:
      1) env LLMPATH_CONTEXT_GATE_MODE
      2) card.extra.context_gate_mode
      3) default
    """
    env_v = _env_str("LLMPATH_CONTEXT_GATE_MODE", "").strip().lower()
    if env_v:
        return env_v
    ex = _get_card_extra(card)
    v = str(ex.get("context_gate_mode", "") or "").strip().lower()
    return v if v else str(default).strip().lower()


def _resolve_context_review_mode(card: SampleCard, *, default: str = "proxy") -> str:
    """
    Single source-of-truth for context review mode.
    Priority:
      1) env LLMPATH_CONTEXT_REVIEW_MODE
      2) card.extra.context_review_mode
      3) default
    """
    env_v = _env_str("LLMPATH_CONTEXT_REVIEW_MODE", "").strip().lower()
    if env_v:
        return env_v
    ex = _get_card_extra(card)
    v = str(ex.get("context_review_mode", "") or "").strip().lower()
    return v if v else str(default).strip().lower()


def _ensure_context_review_fields(proposed: pd.DataFrame, *, gate_mode: str) -> pd.DataFrame:
    """
    Ensure explicit context review fields exist as COLUMNS (not only inside claim_json).
    This stage does NOT decide PASS/FAIL; it prevents "silent unevaluated".
    """
    out = proposed.copy()
    gate_mode = (gate_mode or "").strip().lower()

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
        out["context_confidence"] = ""

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

    return out


def _proxy_context_review(
    proposed: pd.DataFrame, card: SampleCard, *, gate_mode: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Minimal deterministic "context review" without LLM.

    Rule (intentionally conservative but useful):
      - Determine required context keys from claim_json if present; else use default keys.
      - PASS if required keys in card context are all non-empty (after strip)
      - FAIL otherwise (missing keys)
    This converts "UNEVALUATED" -> evaluated=True with an explicit decision,
    allowing hard gate to function without forcing an LLM dependency.

    Returns:
      (updated_df, meta)
    """
    out = proposed.copy()
    gate_mode = (gate_mode or "").strip().lower()

    # Card context
    card_ctx = {
        "condition": str(_card_condition(card) or "").strip(),
        "tissue": str(getattr(card, "tissue", "") or "").strip(),
        "perturbation": str(getattr(card, "perturbation", "") or "").strip(),
        "comparison": str(getattr(card, "comparison", "") or "").strip(),
    }

    if "claim_json" not in out.columns:
        out["claim_json"] = ""

    n_eval = 0
    n_pass = 0
    n_fail = 0

    # Ensure columns exist
    out = _ensure_context_review_fields(out, gate_mode=gate_mode)

    for i, row in out.iterrows():
        cj = str(row.get("claim_json", "") or "").strip()
        req_keys = ["condition", "tissue", "perturbation", "comparison"]

        if cj:
            try:
                obj = json.loads(cj)
                if isinstance(obj, dict):
                    ck = obj.get("context_keys", None)
                    if isinstance(ck, list) and ck:
                        req_keys = [str(x).strip() for x in ck if str(x).strip()]
            except Exception:
                pass

        missing = [k for k in req_keys if (k in card_ctx and not str(card_ctx.get(k, "")).strip())]

        # Evaluate
        n_eval += 1
        out.at[i, "context_evaluated"] = True
        out.at[i, "context_method"] = "proxy"

        if missing:
            out.at[i, "context_status"] = "FAIL"
            out.at[i, "context_reason"] = "MISSING_CONTEXT_KEYS"
            out.at[i, "context_notes"] = f"missing={missing}"
            out.at[i, "context_confidence"] = ""
            n_fail += 1
        else:
            out.at[i, "context_status"] = "PASS"
            out.at[i, "context_reason"] = "OK"
            out.at[i, "context_notes"] = ""
            out.at[i, "context_confidence"] = ""
            n_pass += 1

    # Write back into claim_json too (contract-friendly)
    out = _write_context_review_into_claim_json(out)

    meta = {
        "evaluated": True,
        "mode": "proxy",
        "gate_mode": gate_mode,
        "n_eval": int(n_eval),
        "n_pass": int(n_pass),
        "n_fail": int(n_fail),
    }
    return out, meta


def _write_context_review_into_claim_json(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirror context review columns into claim_json (single-line canonical).
    This makes downstream audit/report robust even if they only read claim_json.
    """
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

                # Normalize types to JSON-serializable python scalars
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


def _apply_context_review(
    proposed: pd.DataFrame,
    card: SampleCard,
    *,
    gate_mode: str,
    review_mode: str,
    backend: BaseLLMBackend | None,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Apply context review based on selected mode.
    Current implementation:
      - proxy: deterministic check over card context presence
      - llm: not implemented here (kept as a future extension), falls back to proxy
    """
    rm = (review_mode or "").strip().lower()
    gm = (gate_mode or "").strip().lower()

    if rm in {"off", "none", "disabled", "0", "false"}:
        # Explicitly leave unevaluated; hard gate will ABSTAIN by design
        out = _ensure_context_review_fields(proposed, gate_mode=gm)
        out = _write_context_review_into_claim_json(out)
        return out, {
            "evaluated": False,
            "mode": "off",
            "gate_mode": gm,
            "note": "context review disabled",
        }

    if rm == "llm":
        # Keep tool usable even when llm review isn't wired here
        # (You can later replace this with an actual backend-based reviewer.)
        out, meta = _proxy_context_review(proposed, card, gate_mode=gm)
        meta["note"] = "llm context review not implemented in pipeline; used proxy fallback"
        meta["backend_enabled"] = bool(backend is not None)
        meta["seed"] = int(seed)
        return out, meta

    # default: proxy
    return _proxy_context_review(proposed, card, gate_mode=gm)


def _excel_safe_ids(x: Any, *, list_sep: str = ";") -> str:
    """
    Make an ID field safe for Excel:
      - Accept list-like or scalar.
      - Treat ',', ';', '|', whitespace as separators.
      - Normalize to list_sep.
      - Prefix with a single quote to force Text in Excel.
    """
    if _is_na_scalar(x):
        return ""

    if isinstance(x, (list, tuple, set)):
        parts = [str(t).strip() for t in x if str(t).strip()]
        s = list_sep.join(parts)
        if not s:
            return ""
        return s if s.startswith("'") else ("'" + s)

    s0 = str(x).strip()
    if not s0 or s0.lower() in _NA_TOKENS_L:
        return ""

    s = (
        s0.replace("|", ";")
        .replace(",", ";")
        .replace("\t", ";")
        .replace("\n", ";")
        .replace(" ", ";")
    )
    parts = [p.strip() for p in s.split(";") if p.strip()]
    s_norm = list_sep.join(parts)

    if not s_norm:
        return ""
    return s_norm if s_norm.startswith("'") else ("'" + s_norm)


# -------------------------
# Stress helpers continue
# -------------------------
def _build_term_uid_maps(dist: pd.DataFrame) -> tuple[set[str], dict[str, str], set[str]]:
    term_uids = dist["term_uid"].astype(str).str.strip().tolist()
    known = {t for t in term_uids if t and t.lower() not in _NA_TOKENS_L}

    raw_to_uids: dict[str, set[str]] = {}
    for tu in known:
        raw = tu.split(":", 1)[-1].strip()
        if not raw:
            continue
        raw_to_uids.setdefault(raw, set()).add(tu)

    raw_unique = {raw: next(iter(u)) for raw, u in raw_to_uids.items() if len(u) == 1}
    raw_amb = {raw for raw, u in raw_to_uids.items() if len(u) > 1}
    return known, raw_unique, raw_amb


# (NOTE: _build_term_uid_maps is defined twice above in your original;
# kept once here by reusing the earlier definition.)


def run_pipeline(cfg: RunConfig, *, run_id: str | None = None) -> RunResult:
    """
    Run the tool pipeline:
      distill → modules → claims → (context_review) → stress → audit → report (+report.jsonl).

    This is a user-facing tool API and intentionally contains no paper/benchmark logic.
    For reproducibility, it writes run metadata (run_meta.json) and stable artifacts
    into cfg.outdir.
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
        },
        "artifacts": {},
    }
    _write_json(meta_path, meta)

    def _mark_step(step: str) -> None:
        meta["step"] = step
        _write_json(meta_path, meta)

    def _call_compat(fn, /, *args, **kwargs):
        """
        Call fn(*args, **kwargs) but drop kwargs not accepted by fn signature.
        IMPORTANT: do NOT swallow exceptions raised inside fn.
        """
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return fn(*args, **kwargs)

        allowed = set(sig.parameters.keys())
        filt = {k: v for k, v in kwargs.items() if k in allowed}
        return fn(*args, **filt)

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
        card = SampleCard.from_json(cfg.sample_card)

        compat = _resolve_gate_review_compat(card)
        meta.setdefault("inputs", {}).setdefault("claims", {})
        meta["inputs"]["claims"]["context_gate_review_compat"] = compat
        _write_json(meta_path, meta)

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

        effective_tau = _resolve_tau(cfg.tau, card)

        sc_path = outdir / "sample_card.resolved.json"
        _write_json(sc_path, card.model_dump())
        meta["artifacts"]["sample_card_resolved_json"] = str(sc_path)

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
        distilled = distill_evidence(ev, card, seed=cfg.seed)

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

        distilled2 = attach_module_ids(distilled, mod_out.term_modules_df)
        dist2_path = outdir / "distilled.with_modules.tsv"
        _write_tsv(distilled2, dist2_path)
        meta["artifacts"]["distilled_with_modules_tsv"] = str(dist2_path)

        _mark_step("select_claims")

        backend: BaseLLMBackend | None = None
        llm_backend_notes = ""

        claim_mode = _env_str("LLMPATH_CLAIM_MODE", "").lower()
        if claim_mode == "llm":
            b = _env_str("LLMPATH_BACKEND", "ollama").lower()
            try:
                if b == "openai":
                    api_key = _env_str("LLMPATH_OPENAI_API_KEY", "")
                    if api_key:
                        backend = OpenAIBackend(
                            api_key=api_key,
                            model_name=_env_str("LLMPATH_OPENAI_MODEL", "gpt-4o"),
                            temperature=float(_env_str("LLMPATH_TEMPERATURE", "0.0")),
                            seed=int(cfg.seed or 42),
                        )
                    else:
                        llm_backend_notes = "missing LLMPATH_OPENAI_API_KEY"
                elif b == "gemini":
                    api_key = _env_str("LLMPATH_GEMINI_API_KEY", "")
                    if api_key:
                        backend = GeminiBackend(
                            api_key=api_key,
                            model_name=_env_str("LLMPATH_GEMINI_MODEL", "models/gemini-2.0-flash"),
                            temperature=float(_env_str("LLMPATH_TEMPERATURE", "0.0")),
                        )
                    else:
                        llm_backend_notes = "missing LLMPATH_GEMINI_API_KEY"
                else:
                    backend = OllamaBackend(
                        host=os.environ.get("LLMPATH_OLLAMA_HOST", None),
                        model_name=os.environ.get("LLMPATH_OLLAMA_MODEL", None),
                        temperature=float(_env_str("LLMPATH_TEMPERATURE", "0.0")),
                        timeout=float(_env_str("LLMPATH_OLLAMA_TIMEOUT", "120")),
                    )
            except Exception as e:
                backend = None
                llm_backend_notes = f"backend_init_error:{type(e).__name__}"

        meta["inputs"]["llm"] = {
            "claim_mode_env": claim_mode,
            "backend_env": _env_str("LLMPATH_BACKEND", "ollama").lower(),
            "backend_enabled": bool(backend is not None),
            "backend_notes": llm_backend_notes,
        }

        k_eff, k_meta = _resolve_k_claims(cfg, card)
        meta.setdefault("inputs", {}).setdefault("claims", {})
        meta["inputs"]["claims"].update(
            {
                **k_meta,
                "k_effective": int(k_eff),
                "k_env_raw": os.environ.get("LLMPATH_K_CLAIMS", ""),
            }
        )

        proposed = select_claims(
            distilled2,
            card,
            k=int(k_eff),
            backend=backend,
            seed=cfg.seed,
            outdir=str(outdir),
        )

        # Enforce claim payload + claim_id contract BEFORE context/stress/audit/report
        proposed = _ensure_claim_payload_and_id(proposed, card)

        # Resolve gate/review modes (single truth)
        context_gate_mode = _resolve_context_gate_mode(card, default="soft")
        context_review_mode = _resolve_context_review_mode(card, default="proxy")

        # Ensure context columns exist
        proposed = _ensure_context_review_fields(proposed, gate_mode=context_gate_mode)

        # Apply context review (writes back into claim_json too)
        seed0 = int(cfg.seed or 42)
        proposed, ctx_review_meta = _apply_context_review(
            proposed,
            card,
            gate_mode=context_gate_mode,
            review_mode=context_review_mode,
            backend=backend,
            seed=seed0,
        )

        # Canonicalize after review write-back
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

        p_drop = _env_float("LLMPATH_STRESS_EVIDENCE_DROPOUT_P", 0.0)
        min_keep = _env_int("LLMPATH_STRESS_EVIDENCE_DROPOUT_MIN_KEEP", 1)
        p_contra = _env_float("LLMPATH_STRESS_CONTRADICTORY_P", 0.0)
        contra_cap = _env_int("LLMPATH_STRESS_CONTRADICTORY_MAX_EXTRA", 0)

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
            # Canonicalize because we mutated claim_json
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

        # -------------------------
        # Excel-safe ID columns
        # -------------------------
        if "gene_ids" in proposed.columns:
            proposed["gene_ids_str"] = proposed["gene_ids"].map(_excel_safe_ids)
        elif "gene_ids_str" in proposed.columns:
            proposed["gene_ids_str"] = proposed["gene_ids_str"].map(_excel_safe_ids)

        if "term_ids" in proposed.columns:
            proposed["term_ids_str"] = proposed["term_ids"].map(_excel_safe_ids)
        elif "term_ids_str" in proposed.columns:
            proposed["term_ids_str"] = proposed["term_ids_str"].map(_excel_safe_ids)

        # Keep canonical display columns if they exist
        if "gene_ids_str" in proposed.columns:
            proposed["gene_ids"] = proposed["gene_ids_str"]
        if "term_ids_str" in proposed.columns:
            proposed["term_ids"] = proposed["term_ids_str"]

        # -------------------------
        # Add display-only gene symbols (optional)
        # -------------------------
        id2sym: dict[str, str] = {}
        try:
            id2sym = build_id_to_symbol_from_distilled(distilled2)
        except Exception:
            id2sym = {}

        # Optional external mapping (TSV) if provided
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

        proposed_path = outdir / "claims.proposed.tsv"
        _write_tsv(proposed, proposed_path)
        meta["artifacts"]["claims_proposed_tsv"] = str(proposed_path)

        _mark_step("audit")
        audited = _call_compat(audit_claims, proposed, distilled2, card, tau=effective_tau)

        # If audit dropped claim payload cols, restore them from proposed
        audited = _restore_claim_payload_into_audit_log(audited, proposed)

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
