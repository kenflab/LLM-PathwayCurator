# LLM-PathwayCurator/src/llm_pathway_curator/sample_card.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


def _norm_str(x: Any) -> str:
    """Normalize context strings: None/empty -> 'NA', strip whitespace."""
    if x is None:
        return "NA"
    if not isinstance(x, str):
        x = str(x)
    x = x.strip()
    return x if x else "NA"


class SampleCard(BaseModel):
    disease: str = Field(default="NA")
    tissue: str = Field(default="NA")
    perturbation: str = Field(default="NA")
    comparison: str = Field(default="NA")
    notes: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    # normalize core context fields
    @field_validator("disease", "tissue", "perturbation", "comparison", mode="before")
    @classmethod
    def _normalize_core(cls, v: Any) -> str:
        return _norm_str(v)

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
        extra = {k: v for k, v in obj.items() if k not in core_keys}
        known["extra"] = extra
        return cls(**known)

    def to_json(self, path: str | Path, *, indent: int = 2) -> None:
        p = Path(path)
        payload = self.model_dump()
        p.write_text(
            json.dumps(payload, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8"
        )

    def apply_patch(self, patch: dict[str, Any]) -> SampleCard:
        """
        Create a counterfactual SampleCard by overwriting a subset of fields.
        Allowed keys: disease/tissue/perturbation/comparison/notes/extra
        Unknown keys go into extra.
        """
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
