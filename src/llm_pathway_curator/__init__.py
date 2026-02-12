# LLM-PathwayCurator/src/llm_pathway_curator/__init__.py
from __future__ import annotations

from .pipeline import RunConfig, RunResult, run_pipeline

__all__ = ["__version__", "RunConfig", "RunResult", "run_pipeline"]

try:
    from importlib.metadata import PackageNotFoundError, version
    __version__ = version("llm-pathway-curator")
except PackageNotFoundError:
    __version__ = "0+unknown"
