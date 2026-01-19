# LLM-PathwayCurator/src/llm_pathway_curator/sample_card.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator

NA_TOKEN = "NA"
CORE_KEYS = ("disease", "tissue", "perturbation", "comparison")

MAX_EXTRA_NESTING = 8

AUDIT_KNOBS = {
    "audit_tau",
    "audit_min_gene_overlap",
    "hub_term_degree",
    "hub_frac_thr",
    "min_union_genes",
    "trust_input_survival",
}


def _norm_str(x: Any) -> str:
    if x is None:
        return NA_TOKEN
    try:
        if not isinstance(x, (list, tuple, set, dict)) and bool(pd.isna(x)):
            return NA_TOKEN
    except Exception:
        pass

    if not isinstance(x, str):
        x = str(x)

    x = x.strip().lstrip("\ufeff")
    if not x:
        return NA_TOKEN
    if x.lower() in {"na", "nan", "none"}:
        return NA_TOKEN
    return x


def _is_dict(x: Any) -> bool:
    return isinstance(x, dict)


def _flatten_extra_recursive(
    extra: dict[str, Any], *, max_depth: int = MAX_EXTRA_NESTING
) -> dict[str, Any]:
    """
    Flatten repeated {"extra": {...}} wrappers.
    Inner overrides outer (most specific wins).
    """
    if not _is_dict(extra):
        return {}

    cur: dict[str, Any] = dict(extra)
    for _ in range(max_depth):
        inner = cur.get("extra")
        if not _is_dict(inner):
            break

        outer = {k: v for k, v in cur.items() if k != "extra"}

        merged = dict(outer)
        merged.update(inner)  # inner overrides outer
        cur = merged

    return cur


def _find_in_nested_extra(
    extra: dict[str, Any], key: str, *, max_depth: int = MAX_EXTRA_NESTING
) -> Any:
    """
    Look for key in extra or nested extra.extra... up to max_depth.
    Returns the first found value (prefers outer levels).
    """
    if not _is_dict(extra):
        return None

    cur: Any = extra
    for _ in range(max_depth + 1):
        if _is_dict(cur) and key in cur:
            return cur.get(key)
        nxt = cur.get("extra") if _is_dict(cur) else None
        if _is_dict(nxt):
            cur = nxt
            continue
        break
    return None


def _canonicalize_audit_knobs(extra: dict[str, Any]) -> dict[str, Any]:
    """
    Canonicalize audit knobs into extra[KEY].
    Does NOT invent values; only hoists existing values.
    """
    if not _is_dict(extra):
        return {}

    out = _flatten_extra_recursive(extra)

    # Hoist knobs that might still be buried in original (defensive)
    for k in AUDIT_KNOBS:
        if k in out:
            continue
        v = _find_in_nested_extra(extra, k)
        if v is not None:
            out[k] = v

    return out


class SampleCard(BaseModel):
    disease: str = Field(default=NA_TOKEN)
    tissue: str = Field(default=NA_TOKEN)
    perturbation: str = Field(default=NA_TOKEN)
    comparison: str = Field(default=NA_TOKEN)

    notes: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("disease", "tissue", "perturbation", "comparison", mode="before")
    @classmethod
    def _normalize_core(cls, v: Any) -> str:
        return _norm_str(v)

    @field_validator("extra", mode="before")
    @classmethod
    def _normalize_extra(cls, v: Any) -> dict[str, Any]:
        if not _is_dict(v):
            v = {}
        return _canonicalize_audit_knobs(dict(v))

    def context_key(self) -> str:
        return "|".join([self.disease, self.tissue, self.perturbation, self.comparison])

    def context_dict(self) -> dict[str, str]:
        return {k: getattr(self, k) for k in CORE_KEYS}

    def audit_tau(self, default: float = 0.8) -> float:
        try:
            v = (self.extra or {}).get("audit_tau", None)
            return float(default) if v is None else float(v)
        except Exception:
            return float(default)

    def audit_min_gene_overlap(self, default: int = 1) -> int:
        try:
            v = (self.extra or {}).get("audit_min_gene_overlap", None)
            return int(default) if v is None else int(v)
        except Exception:
            return int(default)

    @classmethod
    def from_json(cls, path: str | Path) -> SampleCard:
        p = Path(path)
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"SampleCard JSON not found: {p}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in SampleCard: {p} ({e.msg} at line {e.lineno})") from e

        if not isinstance(obj, dict):
            raise ValueError(f"SampleCard JSON must be an object/dict: {p}")

        core_keys = ["disease", "tissue", "perturbation", "comparison", "notes"]
        known = {k: obj.get(k) for k in core_keys}

        extra: dict[str, Any] = {}
        top_extra = obj.get("extra", {})
        if isinstance(top_extra, dict):
            extra.update(top_extra)

        for k, v in obj.items():
            if k in core_keys or k == "extra":
                continue
            extra[k] = v

        known["extra"] = extra
        return cls(**known)

    def to_json(self, path: str | Path, *, indent: int = 2) -> None:
        p = Path(path)
        p.write_text(
            json.dumps(self.model_dump(), ensure_ascii=False, indent=indent) + "\n",
            encoding="utf-8",
        )

    def apply_patch(self, patch: dict[str, Any]) -> SampleCard:
        base = self.model_dump()
        for k, v in patch.items():
            if k in {"disease", "tissue", "perturbation", "comparison", "notes"}:
                base[k] = v
            elif k == "extra" and isinstance(v, dict):
                base["extra"] = {**base.get("extra", {}), **v}
            else:
                base.setdefault("extra", {})
                base["extra"][k] = v
        return SampleCard(**base)
