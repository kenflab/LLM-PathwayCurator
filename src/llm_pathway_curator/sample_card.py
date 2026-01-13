from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SampleCard(BaseModel):
    disease: str = Field(default="NA")
    tissue: str = Field(default="NA")
    perturbation: str = Field(default="NA")
    comparison: str = Field(default="NA")
    notes: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str) -> SampleCard:
        obj = json.loads(Path(path).read_text())
        # allow unknown keys; store in extra
        known = {
            k: obj.get(k) for k in ["disease", "tissue", "perturbation", "comparison", "notes"]
        }
        extra = {k: v for k, v in obj.items() if k not in known}
        known["extra"] = extra
        return cls(**known)
