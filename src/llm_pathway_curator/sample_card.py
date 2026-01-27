# LLM-PathwayCurator/src/llm_pathway_curator/sample_card.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import _shared

NA_TOKEN = "NA"

# Tool-facing core context keys:
# - Use neutral "condition" (not disease/cancer) for generality.
CORE_KEYS = ("condition", "tissue", "perturbation", "comparison")

MAX_EXTRA_NESTING = 8

AUDIT_KNOBS = {
    "audit_tau",
    "audit_min_gene_overlap",
    "hub_term_degree",
    "hub_frac_thr",
    "min_union_genes",
    "trust_input_survival",
    # audit behavior knobs used by audit.py
    "pass_notes",
    "stability_gate_mode",
    "strict_evidence_check",
    # gate behavior
    "context_gate_mode",
    "stress_gate_mode",
}

# v1.1 minimal tool knobs understood by the TOOL (not paper scripts).
# Keep this list minimal and stable.
TOOL_KNOBS = {
    # audit knobs
    "audit_tau",
    "audit_min_gene_overlap",
    "hub_term_degree",
    "hub_frac_thr",
    "min_union_genes",
    "trust_input_survival",
    # audit behavior knobs used by audit.py
    "pass_notes",
    "stability_gate_mode",
    "strict_evidence_check",
    # select knobs
    "claim_mode",
    "k_claims",
    "max_per_module",
    "preselect_tau_gate",
    "enable_context_score_proxy",
    "context_review_mode",
    # gate behavior only; generation happens elsewhere
    "context_gate_mode",
    "stress_gate_mode",
}

# Backward-compat aliases that may appear in older cards.
# IMPORTANT: do NOT mix context vs stress aliases.
ALIASES = {
    # selection
    "claim_mode": {"mode", "claim_mode_env"},
    "k_claims": {"k", "n_claims"},
    "max_per_module": {"module_diversity", "per_module"},
    "preselect_tau_gate": {"tau_gate", "preselect_gate"},
    "enable_context_score_proxy": {"context_score_proxy"},
    # audit behavior
    "pass_notes": {"pass_note", "pass_note_enabled"},
    "stability_gate_mode": {"stability_gate", "stability_mode"},
    "strict_evidence_check": {"strict_evidence", "strict_evidence_genes"},
    # gates
    "context_gate_mode": {"context_gate", "context_mode"},
    "stress_gate_mode": {"stress_mode", "stress", "stress_gate"},
}

# Backward-compat aliases for CORE context keys.
# Input may use disease/cancer/tumor; tool normalizes to "condition".
CTX_ALIASES = {
    "condition": {"condition"},
    "tissue": {"tissue"},
    "perturbation": {"perturbation"},
    "comparison": {"comparison"},
    "disease": {"disease", "cancer", "tumor"},
}


def _apply_aliases(flat: dict[str, Any]) -> dict[str, Any]:
    # If canonical missing but alias exists, hoist alias value to canonical key
    for canon, aliases in ALIASES.items():
        if canon in flat and flat.get(canon) is not None:
            continue
        for a in aliases:
            if a in flat and flat.get(a) is not None:
                flat[canon] = flat.get(a)
                break
    return flat


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
    if x.lower() in _shared.NA_TOKENS_L:
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
    v1.1 contract:
      - Flatten nested extra wrappers (inner overrides outer).
      - Apply backward-compat aliases.
      - Hoist official TOOL_KNOBS if still buried (defensive).
      - Do NOT delete unknown keys (user notes / future compat).

    IMPORTANT:
      - k_claims MUST be top-level (SampleCard field). Do not keep it in extra.
        (We drop canonical + aliases here as a last line of defense.)
    """
    if not _is_dict(extra):
        return {}

    out = _flatten_extra_recursive(extra)
    out = _apply_aliases(out)

    # Hoist knobs that might still be buried in original (defensive)
    for k in TOOL_KNOBS:
        if k in out:
            continue
        v = _find_in_nested_extra(extra, k)
        if v is not None:
            out[k] = v

    # Enforce contract: k_claims is TOP-LEVEL ONLY
    out.pop("k_claims", None)
    for a in ALIASES.get("k_claims", set()):
        out.pop(a, None)

    return out


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


def _normalize_gate_mode(x: Any, default: str) -> str:
    """
    Gate mode contract (tool-facing; align with audit.py):
      - "off": ignore the gate
      - "note": do not change status, but annotate
      - "hard": gate failure/missing -> ABSTAIN/FAIL downstream (policy decided by audit)

    Backward compat:
      - "abstain" (and "on"/"enable") => "hard"
    """
    s = "" if x is None else str(x).strip().lower()
    if not s:
        s = str(default).strip().lower()

    if s in {"off", "none", "disable", "disabled"}:
        return "off"
    if s in {"note", "warn", "warning", "soft"}:
        return "note"
    if s in {"hard", "strict"}:
        return "hard"
    if s in {"abstain", "on", "enable", "enabled"}:
        return "hard"

    d = str(default).strip().lower()
    if d in {"off", "note", "hard"}:
        return d
    return "hard"


def _hoist_context_aliases(obj: dict[str, Any]) -> dict[str, Any]:
    """
    Back-compat:
      - If "condition" is missing, accept "disease"/"cancer"/"tumor" and hoist to "condition".
      - Keep the *original* keys in obj (we don't delete here); SampleCard ignores unknown keys.
    """
    if not isinstance(obj, dict):
        return obj

    if obj.get("condition") is None:
        for k in ("disease", "cancer", "tumor"):
            if obj.get(k) is not None:
                obj["condition"] = obj.get(k)
                break
    return obj


class SampleCard(BaseModel):
    """
    Tool-facing contract:
      - Core context keys are normalized strings.
      - The neutral key is "condition" (legacy disease/cancer/tumor accepted on input).
      - k_claims is TOP-LEVEL ONLY (not inside extra).
      - extra is flattened + alias-canonicalized (but keeps unknown keys).
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    condition: str = Field(default=NA_TOKEN)
    tissue: str = Field(default=NA_TOKEN)
    perturbation: str = Field(default=NA_TOKEN)
    comparison: str = Field(default=NA_TOKEN)

    notes: str | None = None

    # IMPORTANT: keep JSON key "k_claims" but avoid clashing with method name k_claims()
    k_claims_value: int = Field(default=3, alias="k_claims")

    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("condition", "tissue", "perturbation", "comparison", mode="before")
    @classmethod
    def _normalize_core(cls, v: Any) -> str:
        return _norm_str(v)

    @field_validator("k_claims_value", mode="before")
    @classmethod
    def _normalize_k_claims_value(cls, v: Any) -> int:
        # be forgiving; enforce >=1 downstream
        return _as_int(v, 3)

    @field_validator("extra", mode="before")
    @classmethod
    def _normalize_extra(cls, v: Any) -> dict[str, Any]:
        if not _is_dict(v):
            v = {}
        return _canonicalize_audit_knobs(dict(v))

    def context_key(self) -> str:
        return "|".join([self.condition, self.tissue, self.perturbation, self.comparison])

    def context_dict(self) -> dict[str, str]:
        return {k: getattr(self, k) for k in CORE_KEYS}

    # ---- audit knobs ----
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

    def hub_term_degree(self, default: int = 200) -> int:
        return max(1, _as_int((self.extra or {}).get("hub_term_degree", None), default))

    def hub_frac_thr(self, default: float = 0.5) -> float:
        v = _as_float((self.extra or {}).get("hub_frac_thr", None), default)
        return float(min(max(v, 0.0), 1.0))

    def min_union_genes(self, default: int = 3) -> int:
        return max(1, _as_int((self.extra or {}).get("min_union_genes", None), default))

    def trust_input_survival(self, default: bool = False) -> bool:
        return _as_bool((self.extra or {}).get("trust_input_survival", None), default)

    # audit behavior knobs used by audit.py
    def pass_notes(self, default: bool = True) -> bool:
        return _as_bool((self.extra or {}).get("pass_notes", None), default)

    def stability_gate_mode(self, default: str = "hard") -> str:
        v = (self.extra or {}).get("stability_gate_mode", None)
        return _normalize_gate_mode(v, default)

    def strict_evidence_check(self, default: bool = False) -> bool:
        return _as_bool((self.extra or {}).get("strict_evidence_check", None), default)

    # ---- IO ----
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

        obj = _hoist_context_aliases(obj)

        core_keys = ["condition", "tissue", "perturbation", "comparison", "notes", "k_claims"]
        known = {k: obj.get(k) for k in core_keys}

        extra: dict[str, Any] = {}
        top_extra = obj.get("extra", {})
        if isinstance(top_extra, dict):
            extra.update(top_extra)

        # Any non-core top-level keys -> extra (back-compat)
        for k, v in obj.items():
            if k in core_keys or k == "extra":
                continue
            extra[k] = v

        # Backward-compat: allow k_claims buried in extra or via aliases, but HOIST to top-level.
        # This prevents "extra.k_claims resurrection" on round-trips.
        if known.get("k_claims") is None:
            if "k_claims" in extra and extra.get("k_claims") is not None:
                known["k_claims"] = extra.get("k_claims")
            else:
                for a in ALIASES.get("k_claims", set()):
                    if a in extra and extra.get(a) is not None:
                        known["k_claims"] = extra.get(a)
                        break

        # Always remove k_claims + aliases from extra (contract)
        extra.pop("k_claims", None)
        for a in ALIASES.get("k_claims", set()):
            extra.pop(a, None)

        known["extra"] = extra
        return cls(**known)

    def to_json(self, path: str | Path, *, indent: int = 2) -> None:
        p = Path(path)
        p.write_text(
            json.dumps(self.model_dump(by_alias=True), ensure_ascii=False, indent=indent) + "\n",
            encoding="utf-8",
        )

    def apply_patch(self, patch: dict[str, Any]) -> SampleCard:
        base = self.model_dump(by_alias=True)
        for k, v in patch.items():
            if k in {"condition", "tissue", "perturbation", "comparison", "notes", "k_claims"}:
                base[k] = v
            # Back-compat: allow patch using "disease" too
            elif k in {"disease", "cancer", "tumor"} and base.get("condition") in {None, NA_TOKEN}:
                base["condition"] = v
            elif k == "extra" and isinstance(v, dict):
                base["extra"] = {**base.get("extra", {}), **v}
            else:
                base.setdefault("extra", {})
                base["extra"][k] = v

        # Enforce contract: never keep k_claims in extra
        if isinstance(base.get("extra"), dict):
            base["extra"].pop("k_claims", None)
            for a in ALIASES.get("k_claims", set()):
                base["extra"].pop(a, None)

        return SampleCard(**base)

    # ---- select / tool knobs (v1.1 minimal) ----
    def claim_mode(self, default: str = "deterministic") -> str:
        v = (self.extra or {}).get("claim_mode", None)
        s = str(v).strip().lower() if v is not None else ""
        return s if s in {"deterministic", "llm"} else str(default)

    def k_claims(self, default: int = 3) -> int:
        # TOP-LEVEL has priority; extra is fallback only (should be absent).
        v_top = getattr(self, "k_claims_value", None)
        if v_top is not None:
            return max(1, _as_int(v_top, default))
        return max(1, _as_int((self.extra or {}).get("k_claims", None), default))

    def max_per_module(self, default: int = 1) -> int:
        return max(1, _as_int((self.extra or {}).get("max_per_module", None), default))

    def preselect_tau_gate(self, default: bool = False) -> bool:
        return _as_bool((self.extra or {}).get("preselect_tau_gate", None), default)

    def enable_context_score_proxy(self, default: bool = False) -> bool:
        return _as_bool((self.extra or {}).get("enable_context_score_proxy", None), default)

    # ---- gate behavior knobs (tool contract; align with audit.py) ----
    def context_gate_mode(self, default: str = "hard") -> str:
        v = (self.extra or {}).get("context_gate_mode", None)
        return _normalize_gate_mode(v, default)

    def stress_gate_mode(self, default: str = "off") -> str:
        v = (self.extra or {}).get("stress_gate_mode", None)
        return _normalize_gate_mode(v, default)
