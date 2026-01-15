# LLM-PathwayCurator/src/llm_pathway_curator/sample_card.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator


def _norm_str(x: Any) -> str:
    """
    Normalize context strings:
      - NA/None/NaN/empty/"na"/"nan"/"none" -> "NA"
      - strip whitespace + BOM
    """
    if x is None:
        return "NA"
    # pandas NA safety for scalars
    try:
        if not isinstance(x, (list, tuple, set, dict)) and bool(pd.isna(x)):
            return "NA"
    except Exception:
        pass

    if not isinstance(x, str):
        x = str(x)

    x = x.strip().lstrip("\ufeff")
    if not x:
        return "NA"
    if x.lower() in {"na", "nan", "none"}:
        return "NA"
    return x


class SampleCard(BaseModel):
    disease: str = Field(default="NA")
    tissue: str = Field(default="NA")
    perturbation: str = Field(default="NA")
    comparison: str = Field(default="NA")
    notes: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("disease", "tissue", "perturbation", "comparison", mode="before")
    @classmethod
    def _normalize_core(cls, v: Any) -> str:
        return _norm_str(v)

    def context_key(self) -> str:
        """Stable key for caching / claim IDs."""
        return "|".join([self.disease, self.tissue, self.perturbation, self.comparison])

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
        - Allowed: disease/tissue/perturbation/comparison/notes/extra
        - Unknown keys go into extra (shallow merge).
        """
        base = self.model_dump()
        for k, v in patch.items():
            if k in {"disease", "tissue", "perturbation", "comparison", "notes"}:
                base[k] = v
            elif k == "extra" and isinstance(v, dict):
                base["extra"] = {**base.get("extra", {}), **v}  # shallow merge
            else:
                base.setdefault("extra", {})
                base["extra"][k] = v
        return SampleCard(**base)

    def counterfactual(
        self,
        *,
        disease: str | None = None,
        tissue: str | None = None,
        perturbation: str | None = None,
        comparison: str | None = None,
        notes: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> SampleCard:
        patch: dict[str, Any] = {}
        if disease is not None:
            patch["disease"] = disease
        if tissue is not None:
            patch["tissue"] = tissue
        if perturbation is not None:
            patch["perturbation"] = perturbation
        if comparison is not None:
            patch["comparison"] = comparison
        if notes is not None:
            patch["notes"] = notes
        if extra is not None:
            patch["extra"] = extra
        return self.apply_patch(patch)
