# LLM-PathwayCurator/src/llm_pathway_curator/__init__.py
from __future__ import annotations

from .pipeline import RunConfig, RunResult, run_pipeline

__all__ = ["__version__", "RunConfig", "RunResult", "run_pipeline"]

__version__ = "0.0.0"
