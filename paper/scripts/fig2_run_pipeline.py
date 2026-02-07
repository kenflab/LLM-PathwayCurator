#!/usr/bin/env python3
# paper/scripts/run_fig2_pipeline.py

"""
Fig. 2 reproduction orchestrator (paper).

This script performs a deterministic tau-sweep over conditions × variants × gate modes
and runs the core LLM-PathwayCurator pipeline (`llm_pathway_curator.pipeline.run_pipeline`).

Inputs
------
- EvidenceTables: paper/source_data/<benchmark_id>/evidence_tables/<COND>.evidence_table.tsv
- Sample Cards : paper/source_data/<benchmark_id>/sample_cards/<COND>.<gate>.sample_card.json

Outputs (per job)
-----------------
<out_root>/<COND>/<variant>/gate_<gate>/tau_<tau>/
  - report.jsonl, audit_log.tsv, claims.proposed.tsv, modules.tsv, run_meta.json, manifest.json

Determinism
-----------
Controlled by `--seed` (default 42) and the library's fixed settings.
This script does not implement scientific logic; it orchestrates reproducible runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_pathway_curator.pipeline import RunConfig, run_pipeline
from llm_pathway_curator.schema import EvidenceTable

PAPER = Path(__file__).resolve().parents[1]  # paper/


# Runner controls pipeline stress knobs via env injection (job-local) for backward compatibility
ENV_STRESS = {
    "dropout_p": "LLMPATH_STRESS_EVIDENCE_DROPOUT_P",
    "min_keep": "LLMPATH_STRESS_EVIDENCE_DROPOUT_MIN_KEEP",
    "contra_p": "LLMPATH_STRESS_CONTRADICTORY_P",
    "contra_cap": "LLMPATH_STRESS_CONTRADICTORY_MAX_EXTRA",
}


# -------------------------
# Utils
# -------------------------
def _die(msg: str) -> None:
    """Exit the runner with a message (raises SystemExit)."""
    raise SystemExit(msg)


def _warn(msg: str) -> None:
    """Print a warning message to stderr (runner prefix included)."""
    print(f"[run_fig2] WARNING: {msg}", file=sys.stderr)


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file into a dict, or exit with a helpful error."""
    if not path.exists():
        _die(f"[run_fig2] missing file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        _die(f"[run_fig2] invalid JSON: {path}\n{e}")


def _write_json(path: Path, obj: Any) -> None:
    """Write an object as pretty JSON (sorted keys) and ensure parent dirs exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ensure_file(path: Path, label: str) -> None:
    """Fail-fast if `path` is missing, not a file, or empty."""
    if not path.exists():
        _die(f"[run_fig2] {label} not found: {path}")
    if not path.is_file():
        _die(f"[run_fig2] {label} is not a file: {path}")
    if path.stat().st_size == 0:
        _die(f"[run_fig2] {label} is empty: {path}")


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _maybe_git_rev() -> str:
    """Return current git commit hash, or an empty string if unavailable."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return ""


def _require_backend_env_for_context_review() -> None:
    """Fail-fast unless LLMPATH_BACKEND is set to a supported backend for LLM review."""
    kind = str(os.environ.get("LLMPATH_BACKEND", "")).strip().lower()
    if not kind:
        _die(
            "[run_fig2] --context-review-mode llm requires LLMPATH_BACKEND "
            "to be set (openai|ollama|gemini)."
        )
    if kind not in {"openai", "ollama", "gemini"}:
        _die(f"[run_fig2] Unknown LLMPATH_BACKEND={kind!r} (use openai|ollama|gemini)")


def _validate_card_schema(card: dict[str, Any]) -> None:
    """Fail-fast on forbidden Sample Card fields (spec enforcement)."""
    extra = card.get("extra", {})
    if isinstance(extra, dict) and "k_claims" in extra:
        _die("[run_fig2] invalid sample_card: extra.k_claims is forbidden (use top-level k_claims)")


def _validate_evidence_table(path: Path) -> None:
    """
    Validate that a TSV satisfies the EvidenceTable contract.

    Parameters
    ----------
    path
        EvidenceTable TSV path.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        If parsing fails or the contract is violated.

    Notes
    -----
    Inputs: EvidenceTable TSV.
    Outputs: none (validation only).
    Determinism: deterministic.
    Dependencies: `llm_pathway_curator.schema.EvidenceTable.read_tsv`.
    """
    try:
        EvidenceTable.read_tsv(str(path))
    except Exception as e:
        _die(f"[run_fig2] invalid EvidenceTable contract: {path}\n{e}")


def _parse_csv(arg: str) -> list[str]:
    """Parse a comma-separated CLI argument into a list of non-empty tokens."""
    return [x.strip() for x in str(arg).split(",") if x.strip()]


def _parse_taus(s: str) -> list[float]:
    """Parse a comma-separated tau list into floats (non-empty)."""
    vals: list[float] = []
    for x in _parse_csv(s):
        vals.append(float(x))
    if not vals:
        _die("[run_fig2] --taus must be non-empty")
    return vals


def _parse_gate_modes(s: str) -> list[str]:
    """Parse comma-separated gate modes (allowed: note, hard)."""
    modes = _parse_csv(s)
    allowed = {"note", "hard"}
    bad = sorted(set(modes) - allowed)
    if bad:
        _die(f"[run_fig2] unsupported gate modes: {bad} (allowed={sorted(allowed)})")
    if not modes:
        _die("[run_fig2] --gate-modes must be non-empty")
    return modes


def _parse_variants(s: str) -> list[str]:
    """Parse comma-separated variants (allowed: ours, context_swap, stress)."""
    variants = _parse_csv(s)
    allowed = {"ours", "context_swap", "stress"}
    bad = sorted(set(variants) - allowed)
    if bad:
        _die(f"[run_fig2] unsupported variants: {bad} (allowed={sorted(allowed)})")
    if not variants:
        _die("[run_fig2] --variants must be non-empty")
    return variants


def _validate_k_claims(k_claims: int) -> None:
    """Validate that k_claims is >= 1."""
    if k_claims < 1:
        _die(f"[run_fig2] --k-claims must be >= 1 (got {k_claims})")


def _validate_stress_args(
    *,
    p_drop: float,
    min_keep: int,
    p_contra: float,
    contra_cap: int,
) -> None:
    """Validate stress CLI arguments (ranges and non-negativity)."""
    if p_drop < 0.0 or p_drop > 1.0:
        _die(f"[run_fig2] --stress-evidence-dropout-p must be in [0,1] (got {p_drop})")
    if min_keep < 0:
        _die(f"[run_fig2] --stress-evidence-dropout-min-keep must be >= 0 (got {min_keep})")
    if p_contra < 0.0 or p_contra > 1.0:
        _die(f"[run_fig2] --stress-contradictory-p must be in [0,1] (got {p_contra})")
    if contra_cap < 0:
        _die(f"[run_fig2] --stress-contradictory-max-extra must be >= 0 (got {contra_cap})")


def _stress_recipe(*, p_drop: float, p_contra: float) -> str:
    """Return a human-readable stress recipe label for metadata."""
    if float(p_drop) > 0.0 and float(p_contra) > 0.0:
        return "dropout+contradiction"
    if float(p_drop) > 0.0:
        return "dropout"
    if float(p_contra) > 0.0:
        return "contradiction"
    return "off"


def _set_audit_tau(
    card: dict[str, Any],
    tau: float,
    *,
    k_claims: int | None = None,
) -> dict[str, Any]:
    """Set `extra.audit_tau` (and optionally top-level `k_claims`) on a Sample Card."""
    extra = card.get("extra", {})
    if not isinstance(extra, dict):
        extra = {}
    if "k_claims" in extra:
        extra.pop("k_claims", None)
    extra["audit_tau"] = float(tau)
    card["extra"] = extra
    if k_claims is not None:
        card["k_claims"] = int(k_claims)
    return card


def _set_stress_gate_mode(card: dict[str, Any], mode: str) -> dict[str, Any]:
    """Set `extra.stress_gate_mode` on a Sample Card (off/note/hard)."""
    mode = str(mode).strip().lower()
    allowed = {"off", "note", "hard"}
    if mode not in allowed:
        _die(f"[run_fig2] invalid stress_gate_mode={mode} (allowed={sorted(allowed)})")
    extra = card.get("extra", {})
    if not isinstance(extra, dict):
        extra = {}
    extra["stress_gate_mode"] = mode
    card["extra"] = extra
    return card


def _set_context_review_mode(card: dict[str, Any], mode: str) -> dict[str, Any]:
    """Set `extra.context_review_mode` on a Sample Card (off/llm)."""
    mode = str(mode).strip().lower()
    allowed = {"off", "llm"}
    if mode not in allowed:
        _die(f"[run_fig2] invalid context_review_mode={mode} (allowed={sorted(allowed)})")
    extra = card.get("extra", {})
    if not isinstance(extra, dict):
        extra = {}
    extra["context_review_mode"] = mode
    card["extra"] = extra
    return card


def _outdir_for(
    *, out_root: Path, condition: str, variant: str, gate_mode: str, tau: float
) -> Path:
    """Build the canonical output directory path for a job."""
    return out_root / condition / variant / f"gate_{gate_mode}" / f"tau_{tau:.2f}"


def _report_exists_ok(outdir: Path) -> bool:
    """Return True if `outdir/report.jsonl` exists and is non-empty."""
    rp = outdir / "report.jsonl"
    return rp.exists() and rp.is_file() and rp.stat().st_size > 0


def _patch_report_jsonl(
    report_path: Path,
    *,
    benchmark_id: str,
    condition: str,
    method: str,
    variant: str,
    gate_mode: str,
    tau: float,
) -> None:
    """
    Patch per-record metadata into `report.jsonl` for downstream aggregation.

    Parameters
    ----------
    report_path
        Path to `report.jsonl` produced by `run_pipeline`.
    benchmark_id, condition, method, variant, gate_mode, tau
        Metadata fields injected into every JSONL record.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        If the JSONL file is missing, empty, or contains invalid JSON.

    Notes
    -----
    Inputs: existing `report.jsonl`.
    Outputs: rewritten `report.jsonl` (same path) with additional fields.
    Determinism: deterministic transformation.
    Dependencies: JSONL schema emitted by `llm_pathway_curator.report`.
    """
    if not report_path.exists():
        _die(f"[run_fig2] report.jsonl not found for patch: {report_path}")

    out_lines: list[str] = []
    with report_path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                _die(f"[run_fig2] invalid JSON in {report_path} line {ln}\n{e}")

            rec["benchmark_id"] = benchmark_id
            rec["condition"] = condition
            rec["method"] = method
            rec["variant"] = variant
            rec["gate_mode"] = gate_mode
            rec["tau"] = float(tau)

            out_lines.append(json.dumps(rec, ensure_ascii=False))

    if not out_lines:
        _die(f"[run_fig2] empty report.jsonl after reading: {report_path}")

    report_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def _make_shuffled_mapping_all(all_conditions: list[str]) -> dict[str, str]:
    """Create a deterministic rotation map for context-swap across all conditions."""
    if len(all_conditions) < 2:
        _die("[run_fig2] context_swap requires >=2 conditions in evidence_tables/")
    rot = all_conditions[1:] + all_conditions[:1]
    return dict(zip(all_conditions, rot, strict=False))


def _with_temp_env(env_updates: dict[str, str]) -> dict[str, str]:
    """Apply env updates and return old values for later restoration."""
    old: dict[str, str] = {}
    for k, v in env_updates.items():
        if k in os.environ:
            old[k] = os.environ[k]
        else:
            old[k] = "__MISSING__"
        os.environ[k] = str(v)
    return old


def _restore_env(old_values: dict[str, str]) -> None:
    """Restore environment variables captured by `_with_temp_env`."""
    for k, old in old_values.items():
        if old == "__MISSING__":
            os.environ.pop(k, None)
        else:
            os.environ[k] = old


def _write_manifest(
    outdir: Path,
    *,
    benchmark_id: str,
    condition: str,
    method: str,
    variant: str,
    gate_mode: str,
    tau: float,
    k_claims: int,
    seed: int,
    evidence_table: Path,
    sample_card: Path,
    context_swap_from: str,
    context_swap_to: str,
    plan: dict[str, Any],
    status: str,
    stress: dict[str, Any],
    error: str = "",
) -> None:
    """
    Write a runner-level provenance manifest for one job.

    Parameters
    ----------
    outdir
        Job output directory.
    status
        "ok" or "error".
    error
        Error message when status is "error".
    plan
        High-level run plan used to reproduce the job batch.
    evidence_table, sample_card
        Input paths recorded with size/mtime/sha256.

    Returns
    -------
    None

    Notes
    -----
    Inputs: `evidence_table`, `sample_card`, and current environment metadata.
    Outputs: `outdir/manifest.json`.
    Determinism: sha256 and paths are deterministic given fixed inputs.
    Dependencies: local git (optional) and filesystem metadata.
    """
    manifest = {
        "tool": "llm-pathway-curator",
        "script": "paper/scripts/run_fig2_pipeline.py",
        "status": status,
        "error": error,
        "started_unix": plan.get("started_unix", 0),
        "finished_unix": int(time.time()),
        "git_commit": _maybe_git_rev(),
        "run": {
            "benchmark_id": benchmark_id,
            "condition": condition,
            "method": method,
            "variant": variant,
            "gate_mode": gate_mode,
            "tau": float(tau),
            "k_claims": int(k_claims),
            "seed": int(seed),
            "context_swap_from": context_swap_from,
            "context_swap_to": context_swap_to,
            "stress": stress,
        },
        "inputs": {
            "evidence_table": {
                "path": str(evidence_table),
                "size_bytes": int(evidence_table.stat().st_size) if evidence_table.exists() else 0,
                "mtime_unix": float(evidence_table.stat().st_mtime)
                if evidence_table.exists()
                else 0.0,
                "sha256": _sha256_file(evidence_table) if evidence_table.exists() else "",
            },
            "sample_card": {
                "path": str(sample_card),
                "size_bytes": int(sample_card.stat().st_size) if sample_card.exists() else 0,
                "mtime_unix": float(sample_card.stat().st_mtime) if sample_card.exists() else 0.0,
                "sha256": _sha256_file(sample_card) if sample_card.exists() else "",
            },
        },
        "env": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "cwd": str(Path.cwd()),
            "LLMPATH_K_CLAIMS": os.environ.get("LLMPATH_K_CLAIMS", ""),
            "LLMPATH_BACKEND": os.environ.get("LLMPATH_BACKEND", ""),
            ENV_STRESS["dropout_p"]: os.environ.get(ENV_STRESS["dropout_p"], ""),
            ENV_STRESS["min_keep"]: os.environ.get(ENV_STRESS["min_keep"], ""),
            ENV_STRESS["contra_p"]: os.environ.get(ENV_STRESS["contra_p"], ""),
            ENV_STRESS["contra_cap"]: os.environ.get(ENV_STRESS["contra_cap"], ""),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", ""),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", ""),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
        },
        "plan": plan,
    }
    _write_json(outdir / "manifest.json", manifest)


def _check_stress_effect(outdir: Path, *, expected_enabled: bool) -> None:
    """Warn if stress was expected but not reflected in outputs/metadata."""
    rm = outdir / "run_meta.json"
    if not rm.exists():
        _warn(f"missing run_meta.json for stress check: {rm}")
        return

    try:
        meta = json.loads(rm.read_text(encoding="utf-8"))
    except Exception:
        _warn(f"cannot parse run_meta.json for stress check: {rm}")
        return

    if not expected_enabled:
        return

    inputs = meta.get("inputs", {})
    s = inputs.get("stress", None)
    r = inputs.get("stress_runtime", None)
    if s is None and r is None:
        s = inputs.get("stress_v4", {})
        r = inputs.get("stress_v4_runtime", {})
    else:
        s = s or {}
        r = r or {}

    try:
        p_drop = float(s.get("evidence_dropout_p", 0.0) or 0.0)
        p_contra = float(s.get("contradictory_p", 0.0) or 0.0)
    except Exception:
        p_drop, p_contra = 0.0, 0.0

    if p_drop <= 0.0 and p_contra <= 0.0:
        _warn(
            "stress variant ran but pipeline recorded both evidence_dropout_p=0 and "
            "contradictory_p=0 (stress may not have been applied). "
            f"outdir={outdir}"
        )

    claims = outdir / "claims.proposed.tsv"
    if not claims.exists():
        _warn(f"missing claims.proposed.tsv for stress check: {claims}")
        return

    try:
        header = claims.open("r", encoding="utf-8").readline().strip().split("\t")
    except Exception:
        header = []

    has_stress_cols = any(h.startswith("stress_") for h in header)

    dropout_enabled = bool((r.get("dropout", {}) or {}).get("enabled", False))
    contra_enabled = bool((r.get("contradictory", {}) or {}).get("enabled", False))

    if (dropout_enabled or contra_enabled) and (not has_stress_cols):
        _warn(
            "pipeline reports stress enabled at runtime but claims.proposed.tsv has no "
            "stress_* columns (schema drift?). "
            f"outdir={outdir}"
        )


@dataclass(frozen=True)
class Job:
    """
    One tau-sweep job specification for the Fig. 2 reproduction runner.

    Parameters
    ----------
    benchmark_id
        Directory name under `paper/source_data/`.
    condition
        Condition label (e.g., HNSC, BRCA) that selects EvidenceTable and Sample Card.
    method
        Method label recorded into `report.jsonl` (e.g., ours, context_swap).
    variant
        Variant name controlling how Sample Cards are prepared (ours/context_swap/stress).
    gate_mode
        Audit gate mode (note/hard).
    tau
        Stability threshold used by the audit (`audit_tau`).
    k_claims
        Number of proposed claims before audit.
    seed
        RNG seed forwarded to `run_pipeline`.
    evidence_table
        Path to EvidenceTable TSV for this condition.
    sample_card
        Path to Sample Card JSON used for this job (may be a generated context-swap card).
    outdir
        Output directory for this job.
    context_swap_from, context_swap_to
        Human-readable labels recorded for context swap provenance.
    stress_*
        Stress knobs forwarded to the pipeline (dropout/contradiction).

    Notes
    -----
    Inputs: `evidence_table`, `sample_card`.
    Outputs: written under `outdir/` by `run_pipeline` plus `manifest.json`.
    Determinism: controlled by `seed` and fixed pipeline settings.
    Dependencies: `llm_pathway_curator.pipeline.run_pipeline`.
    """

    benchmark_id: str
    condition: str
    method: str
    variant: str
    gate_mode: str
    tau: float
    k_claims: int
    seed: int
    evidence_table: Path
    sample_card: Path
    outdir: Path
    context_swap_from: str = ""
    context_swap_to: str = ""
    stress_evidence_dropout_p: float = 0.0
    stress_evidence_dropout_min_keep: int = 1
    stress_contradictory_p: float = 0.0
    stress_contradictory_max_extra: int = 0


def _run_one(job: Job, *, force: bool, plan: dict[str, Any]) -> None:
    """
    Execute one job and write a decision-grade report under the job output directory.

    Parameters
    ----------
    job
        Fully specified job (inputs, tau, seed, stress knobs, and output directory).
    force
        If True, overwrite existing outputs. If False, skip when `report.jsonl` exists.
    plan
        Run plan metadata recorded into `manifest.json` for provenance.

    Returns
    -------
    None

    Notes
    -----
    Inputs
      - `job.evidence_table` (EvidenceTable TSV)
      - `job.sample_card` (Sample Card JSON)
    Outputs
      - `job.outdir/report.jsonl` (patched with benchmark/condition/method metadata)
      - `job.outdir/manifest.json` (runner-level provenance)
      - additional artifacts produced by `run_pipeline` (audit log, modules, etc.)
    Determinism
      - `job.seed` is forwarded to `RunConfig(seed=...)`.
    Dependencies
      - `llm_pathway_curator.pipeline.run_pipeline`
      - `llm_pathway_curator.pipeline.RunConfig`
    """
    if (not force) and _report_exists_ok(job.outdir):
        print(f"[run_fig2] SKIP (exists): {job.outdir}")
        return

    started = int(time.time())
    plan2 = dict(plan)
    plan2["started_unix"] = started

    if job.variant == "stress":
        stress = {
            "enabled": True,
            "recipe": _stress_recipe(
                p_drop=job.stress_evidence_dropout_p, p_contra=job.stress_contradictory_p
            ),
            "evidence_dropout_p": float(job.stress_evidence_dropout_p),
            "evidence_dropout_min_keep": int(job.stress_evidence_dropout_min_keep),
            "contradictory_p": float(job.stress_contradictory_p),
            "contradictory_max_extra": int(job.stress_contradictory_max_extra),
        }
    else:
        stress = {
            "enabled": False,
            "recipe": "off",
            "evidence_dropout_p": 0.0,
            "evidence_dropout_min_keep": int(job.stress_evidence_dropout_min_keep),
            "contradictory_p": 0.0,
            "contradictory_max_extra": int(job.stress_contradictory_max_extra),
        }

    env_updates = {
        ENV_STRESS["dropout_p"]: str(stress["evidence_dropout_p"]),
        ENV_STRESS["min_keep"]: str(stress["evidence_dropout_min_keep"]),
        ENV_STRESS["contra_p"]: str(stress["contradictory_p"]),
        ENV_STRESS["contra_cap"]: str(stress["contradictory_max_extra"]),
    }
    old_env = _with_temp_env(env_updates)

    try:
        run_meta_runner = {
            "benchmark_id": job.benchmark_id,
            "condition": job.condition,
            "method": job.method,
            "variant": job.variant,
            "gate_mode": job.gate_mode,
            "tau": float(job.tau),
            "k_claims": int(job.k_claims),
            "seed": int(job.seed),
            "evidence_table": str(job.evidence_table),
            "sample_card": str(job.sample_card),
            "context_swap_from": job.context_swap_from,
            "context_swap_to": job.context_swap_to,
            "stress": stress,
            "time_unix": started,
        }
        _write_json(job.outdir / "run_meta.runner.json", run_meta_runner)

        cfg = RunConfig(
            evidence_table=str(job.evidence_table),
            sample_card=str(job.sample_card),
            outdir=str(job.outdir),
            force=bool(force),
            seed=int(job.seed),
            run_meta_name="run_meta.json",
            tau=float(job.tau),
            k_claims=int(job.k_claims),
            stress_evidence_dropout_p=float(job.stress_evidence_dropout_p),
            stress_evidence_dropout_min_keep=int(job.stress_evidence_dropout_min_keep),
            stress_contradictory_p=float(job.stress_contradictory_p),
            stress_contradictory_max_extra=int(job.stress_contradictory_max_extra),
        )
        run_pipeline(cfg)

        report_path = job.outdir / "report.jsonl"
        _ensure_file(report_path, "report.jsonl")

        _patch_report_jsonl(
            report_path,
            benchmark_id=job.benchmark_id,
            condition=job.condition,
            method=job.method,
            variant=job.variant,
            gate_mode=job.gate_mode,
            tau=job.tau,
        )

        _write_manifest(
            job.outdir,
            benchmark_id=job.benchmark_id,
            condition=job.condition,
            method=job.method,
            variant=job.variant,
            gate_mode=job.gate_mode,
            tau=job.tau,
            k_claims=job.k_claims,
            seed=job.seed,
            evidence_table=job.evidence_table,
            sample_card=job.sample_card,
            context_swap_from=job.context_swap_from,
            context_swap_to=job.context_swap_to,
            plan=plan2,
            status="ok",
            stress=stress,
        )

        if job.variant == "stress":
            _check_stress_effect(job.outdir, expected_enabled=True)

    except Exception as e:
        _write_manifest(
            job.outdir,
            benchmark_id=job.benchmark_id,
            condition=job.condition,
            method=job.method,
            variant=job.variant,
            gate_mode=job.gate_mode,
            tau=job.tau,
            k_claims=job.k_claims,
            seed=job.seed,
            evidence_table=job.evidence_table,
            sample_card=job.sample_card,
            context_swap_from=job.context_swap_from,
            context_swap_to=job.context_swap_to,
            plan=plan2,
            status="error",
            stress=stress,
            error=str(e),
        )
        raise
    finally:
        _restore_env(old_env)


def _resolve_sd(benchmark_id: str) -> Path:
    """
    Resolve the benchmark root directory under `paper/source_data/`.

    Parameters
    ----------
    benchmark_id
        Directory name under `paper/source_data/` (e.g., PANCAN_TP53_v1).

    Returns
    -------
    Path
        Path to `paper/source_data/<benchmark_id>`.

    Raises
    ------
    SystemExit
        If the benchmark directory does not exist.

    Notes
    -----
    Inputs: `benchmark_id`.
    Outputs: none (path resolution only).
    Determinism: deterministic.
    Dependencies: filesystem layout under `paper/source_data/`.
    """
    p = PAPER / "source_data" / benchmark_id
    if not p.exists():
        _die(f"[run_fig2] benchmark directory not found: {p}")
    return p


def _iter_all_conditions(evid_dir: Path) -> list[str]:
    """
    List all available condition labels from EvidenceTable file names.

    Parameters
    ----------
    evid_dir
        Directory containing `*.evidence_table.tsv`.

    Returns
    -------
    list[str]
        Uppercased condition labels inferred from `<COND>.evidence_table.tsv`.

    Raises
    ------
    SystemExit
        If the directory is missing or no matching EvidenceTables are found.

    Notes
    -----
    Inputs: `evid_dir/*.evidence_table.tsv`.
    Outputs: condition list used to plan tau-sweep jobs.
    Determinism: ordering is deterministic via `sorted(glob(...))`.
    Dependencies: file naming convention `<COND>.evidence_table.tsv`.
    """
    if not evid_dir.exists():
        _die(f"[run_fig2] missing dir: {evid_dir}")
    conds: list[str] = []
    for p in sorted(evid_dir.glob("*.evidence_table.tsv")):
        conds.append(p.name.split(".")[0].upper())
    if not conds:
        _die("[run_fig2] no evidence_tables found (*.evidence_table.tsv)")
    return conds


def main() -> None:
    """
    CLI entry point for Fig. 2 reproduction runs (tau-sweep orchestrator).

    This function:
    1) Resolves benchmark directories under `paper/source_data/<benchmark_id>/`.
    2) Validates EvidenceTables and Sample Cards (fail-fast).
    3) Builds a job list across conditions × variants × gate modes × taus.
    4) Runs each job via `_run_one`, writing outputs under `<out_root>/...`.

    Returns
    -------
    None

    Notes
    -----
    Inputs
      - EvidenceTables: `<benchmark>/evidence_tables/*.evidence_table.tsv`
      - Sample Cards : `<benchmark>/sample_cards/*.sample_card.json`
    Outputs
      - Per-job outputs under:
        `<out_root>/<COND>/<variant>/gate_<gate>/tau_<tau>/`
    Determinism
      - Controlled by `--seed` (default 42) and library-level fixed settings.
    Dependencies
      - `llm_pathway_curator.pipeline.run_pipeline`
      - `llm_pathway_curator.schema.EvidenceTable` (contract validation)
    """
    ap = argparse.ArgumentParser(
        description=(
            "Fig2 (paper/supp): deterministic tau-sweep over conditions × variants × gate modes."
        )
    )
    ap.add_argument("--benchmark-id", default="PANCAN_TP53_v1")
    ap.add_argument(
        "--conditions", default="", help="Alias of --cancers (recommended for BeatAML)."
    )
    ap.add_argument("--cancers", default="ALL", help="Comma list like HNSC,LUAD or ALL")

    ap.add_argument(
        "--variants",
        default="ours,context_swap",
        help="Comma list: ours,context_swap,stress",
    )
    ap.add_argument("--taus", default="0.2,0.4,0.6,0.8,0.9", help="Comma list of tau thresholds")
    ap.add_argument(
        "--k-claims", type=int, default=100, help="Fig2: number of proposed claims before audit"
    )
    ap.add_argument(
        "--gate-modes",
        default="hard",
        help="Paper default: hard. Use 'note,hard' for debug/ablation.",
    )
    ap.add_argument("--seed", type=int, default=42, help="Paper default seed (fixed).")
    ap.add_argument(
        "--out-root",
        default="",
        help="Override output root. Default: <benchmark>/out",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs (default: resume/skip if report.jsonl exists).",
    )
    ap.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the run plan and exit (no execution).",
    )
    ap.add_argument(
        "--context-swap-to",
        default="",
        help=(
            "Optional: force context_swap target condition (single-condition runs). "
            "Example: --cancers HNSC --variants ours,context_swap --context-swap-to LUAD"
        ),
    )

    # -------------------------
    # stress knobs (paper-facing control; default OFF)
    # -------------------------
    ap.add_argument(
        "--stress-evidence-dropout-p",
        type=float,
        default=0.0,
        help="stress: probability of dropping genes from evidence lists (default: 0.0/off)",
    )
    ap.add_argument(
        "--stress-evidence-dropout-min-keep",
        type=int,
        default=1,
        help="stress: min genes kept per term evidence after dropout (default: 1)",
    )
    ap.add_argument(
        "--stress-contradictory-p",
        type=float,
        default=0.0,
        help=(
            "stress: probability of injecting contradictory direction duplicates "
            "(default: 0.0/off)",
        ),
    )
    ap.add_argument(
        "--stress-contradictory-max-extra",
        type=int,
        default=0,
        help="stress: cap the number of injected duplicates (0 = no cap) (default: 0)",
    )
    ap.add_argument(
        "--context-review-mode",
        default="off",
        choices=["off", "llm"],
        help="Enable context relevance review in proposal stage (default: off)",
    )
    args = ap.parse_args()

    if str(args.context_review_mode).strip().lower() == "llm":
        _require_backend_env_for_context_review()

    benchmark_id = str(args.benchmark_id).strip()
    if not benchmark_id:
        _die("[run_fig2] --benchmark-id must be non-empty")

    _validate_k_claims(int(args.k_claims))
    _validate_stress_args(
        p_drop=float(args.stress_evidence_dropout_p),
        min_keep=int(args.stress_evidence_dropout_min_keep),
        p_contra=float(args.stress_contradictory_p),
        contra_cap=int(args.stress_contradictory_max_extra),
    )

    # --- benchmark switch ---
    sd = _resolve_sd(benchmark_id)
    evid_dir = sd / "evidence_tables"
    card_dir = sd / "sample_cards"
    default_out_dir = sd / "out"

    out_root = (
        Path(str(args.out_root)).resolve()
        if str(args.out_root).strip()
        else default_out_dir.resolve()
    )

    # conditions list:
    # - if --conditions given: use it
    # - else fallback to --cancers (backward compatibility)
    all_conds = _iter_all_conditions(evid_dir)

    cond_arg = str(args.conditions).strip()
    cancers_arg = str(args.cancers).strip()

    # Prefer --conditions; fall back to legacy --cancers
    sel_arg = cond_arg if cond_arg else cancers_arg

    if not sel_arg:
        _die("[run_fig2] no conditions specified (use --conditions or --cancers)")

    if sel_arg.upper() == "ALL":
        conditions = all_conds
    else:
        conditions = [c.strip().upper() for c in _parse_csv(sel_arg)]

    if not conditions:
        _die("[run_fig2] no conditions selected")

    missing = sorted(set(conditions) - set(all_conds))
    if missing:
        _die(f"[run_fig2] conditions missing evidence_table.tsv: {missing}")

    variants = _parse_variants(str(args.variants))
    taus = _parse_taus(str(args.taus))
    gate_modes = _parse_gate_modes(str(args.gate_modes))

    stress_plan = {
        "evidence_dropout_p": float(args.stress_evidence_dropout_p),
        "evidence_dropout_min_keep": int(args.stress_evidence_dropout_min_keep),
        "contradictory_p": float(args.stress_contradictory_p),
        "contradictory_max_extra": int(args.stress_contradictory_max_extra),
        "recipe": _stress_recipe(
            p_drop=float(args.stress_evidence_dropout_p),
            p_contra=float(args.stress_contradictory_p),
        ),
    }

    plan = {
        "benchmark_id": benchmark_id,
        "conditions": conditions,
        "variants": variants,
        "gate_modes": gate_modes,
        "taus": [float(t) for t in taus],
        "k_claims": int(args.k_claims),
        "seed": int(args.seed),
        "out_root": str(out_root),
        "force": bool(args.force),
        "started_unix": int(time.time()),
        "stress": stress_plan,
    }

    # --- context_swap target (forced) ---
    forced_dst = str(args.context_swap_to).strip().upper()

    print("[run_fig2] plan")
    print(f"  benchmark_id: {benchmark_id}")
    print(f"  conditions: {','.join(conditions)}")
    print(f"  variants: {','.join(variants)}")
    print(f"  gate_modes: {','.join(gate_modes)}")
    print(f"  taus: {','.join([f'{t:.2f}' for t in taus])}")
    print(f"  k_claims: {int(args.k_claims)}")
    print(f"  seed: {int(args.seed)}")
    print(f"  out_root: {out_root}")
    if forced_dst:
        print(f"  context_swap_to: {forced_dst}")
    if "stress" in variants:
        print("  stress:")
        print(f"    recipe: {stress_plan['recipe']}")
        print(f"    evidence_dropout_p: {stress_plan['evidence_dropout_p']:.3f}")
        print(f"    evidence_dropout_min_keep: {stress_plan['evidence_dropout_min_keep']}")
        print(f"    contradictory_p: {stress_plan['contradictory_p']:.3f}")
        print(f"    contradictory_max_extra: {stress_plan['contradictory_max_extra']}")

    if args.plan_only:
        print("[run_fig2] plan-only: exiting.")
        return

    # Validate inputs (fail-fast).
    for condition in conditions:
        ev_path = evid_dir / f"{condition}.evidence_table.tsv"
        _ensure_file(ev_path, f"evidence_table ({condition})")
        _validate_evidence_table(ev_path)

        for gm in gate_modes:
            _ensure_file(
                card_dir / f"{condition}.{gm}.sample_card.json",
                f"sample_card {gm} ({condition})",
            )

    # context_swap policy:
    # - default: requires >=2 conditions in evidence_tables/ (for deterministic rotation)
    # - exception: if --context-swap-to is set, allow even when evidence_tables has 1 condition,
    #              but require that the target sample_card exists for every gate_mode.
    swap_map_all: dict[str, str] = {}
    if "context_swap" in variants:
        if forced_dst:
            for gm in gate_modes:
                sc = card_dir / f"{forced_dst}.{gm}.sample_card.json"
                _ensure_file(sc, f"context_swap target sample_card {gm} ({forced_dst})")
            # rotation map unused when forced
            swap_map_all = {}
        else:
            if len(all_conds) < 2:
                _warn(
                    "context_swap requested but evidence_tables has <2 conditions. "
                    "Dropping context_swap for this run (provide --context-swap-to to force)."
                )
                variants = [v for v in variants if v != "context_swap"]
                swap_map_all = {}
            else:
                swap_map_all = _make_shuffled_mapping_all(all_conds)

    # Demo-safety warning
    if "stress" in variants and stress_plan["recipe"] == "off":
        _warn(
            "variants include 'stress' but both --stress-evidence-dropout-p and "
            "--stress-contradictory-p are 0.0. Outputs may be indistinguishable from 'ours'."
        )

    jobs: list[Job] = []

    for condition in conditions:
        ev = evid_dir / f"{condition}.evidence_table.tsv"

        for tau in taus:
            for gate_mode in gate_modes:
                if "ours" in variants:
                    src_sc = card_dir / f"{condition}.{gate_mode}.sample_card.json"
                    outdir = _outdir_for(
                        out_root=out_root,
                        condition=condition,
                        variant="ours",
                        gate_mode=gate_mode,
                        tau=tau,
                    )
                    outdir.mkdir(parents=True, exist_ok=True)
                    jobs.append(
                        Job(
                            benchmark_id=benchmark_id,
                            condition=condition,
                            method="ours",
                            variant="ours",
                            gate_mode=gate_mode,
                            tau=float(tau),
                            k_claims=int(args.k_claims),
                            seed=int(args.seed),
                            evidence_table=ev,
                            sample_card=src_sc,
                            outdir=outdir,
                            stress_evidence_dropout_p=0.0,
                            stress_evidence_dropout_min_keep=int(
                                args.stress_evidence_dropout_min_keep
                            ),
                            stress_contradictory_p=0.0,
                            stress_contradictory_max_extra=int(args.stress_contradictory_max_extra),
                        )
                    )

                if "stress" in variants:
                    src_sc = card_dir / f"{condition}.{gate_mode}.sample_card.json"
                    outdir = _outdir_for(
                        out_root=out_root,
                        condition=condition,
                        variant="stress",
                        gate_mode=gate_mode,
                        tau=tau,
                    )
                    outdir.mkdir(parents=True, exist_ok=True)
                    jobs.append(
                        Job(
                            benchmark_id=benchmark_id,
                            condition=condition,
                            method="ours",
                            variant="stress",
                            gate_mode=gate_mode,
                            tau=float(tau),
                            k_claims=int(args.k_claims),
                            seed=int(args.seed),
                            evidence_table=ev,
                            sample_card=src_sc,
                            outdir=outdir,
                            stress_evidence_dropout_p=float(args.stress_evidence_dropout_p),
                            stress_evidence_dropout_min_keep=int(
                                args.stress_evidence_dropout_min_keep
                            ),
                            stress_contradictory_p=float(args.stress_contradictory_p),
                            stress_contradictory_max_extra=int(args.stress_contradictory_max_extra),
                        )
                    )

                if "context_swap" in variants:
                    dst_cond = forced_dst or swap_map_all.get(condition, "")
                    if not dst_cond:
                        _die("[run_fig2] internal error: context swap map missing condition")

                    src_sc_path = card_dir / f"{condition}.{gate_mode}.sample_card.json"
                    src = _read_json(src_sc_path)

                    dst_sc_path = card_dir / f"{dst_cond}.{gate_mode}.sample_card.json"
                    if not dst_sc_path.exists():
                        _die(f"[run_fig2] context_swap target card missing: {dst_sc_path}")
                    dst = _read_json(dst_sc_path)

                    extra = src.get("extra", {})
                    if not isinstance(extra, dict):
                        extra = {}

                    extra["context_swap_from"] = str(
                        src.get("condition") or src.get("disease") or ""
                    ).strip()
                    extra["context_swap_to"] = str(
                        dst.get("condition") or dst.get("disease") or ""
                    ).strip()
                    extra["context_swap_to_condition"] = dst_cond
                    src["extra"] = extra

                    for k in ("condition", "disease", "tissue", "perturbation", "comparison"):
                        v = dst.get(k, "")
                        if isinstance(v, str) and v.strip():
                            src[k] = v
                        elif v not in (None, "", []):
                            src[k] = v

                    src = _set_audit_tau(src, float(tau), k_claims=int(args.k_claims))
                    src = _set_context_review_mode(src, str(args.context_review_mode))

                    outdir2 = _outdir_for(
                        out_root=out_root,
                        condition=condition,
                        variant="context_swap",
                        gate_mode=gate_mode,
                        tau=tau,
                    )
                    outdir2.mkdir(parents=True, exist_ok=True)

                    swap_card_path = (
                        outdir2
                        / f"sample_card.context_swap.{dst_cond}.{gate_mode}.tau_{tau:.2f}.json"
                    )
                    _write_json(swap_card_path, src)

                    jobs.append(
                        Job(
                            benchmark_id=benchmark_id,
                            condition=condition,
                            method="context_swap",
                            variant="context_swap",
                            gate_mode=gate_mode,
                            tau=float(tau),
                            k_claims=int(args.k_claims),
                            seed=int(args.seed),
                            evidence_table=ev,
                            sample_card=swap_card_path,
                            outdir=outdir2,
                            context_swap_from=str(extra.get("context_swap_from", "")),
                            context_swap_to=str(extra.get("context_swap_to", "")),
                            stress_evidence_dropout_p=0.0,
                            stress_evidence_dropout_min_keep=int(
                                args.stress_evidence_dropout_min_keep
                            ),
                            stress_contradictory_p=0.0,
                            stress_contradictory_max_extra=int(args.stress_contradictory_max_extra),
                        )
                    )

    n_total = len(jobs)
    print(f"[run_fig2] jobs: {n_total}")
    for i, job in enumerate(jobs, start=1):
        print(
            f"[run_fig2] ({i}/{n_total}) {job.condition} {job.variant} "
            f"gate={job.gate_mode} tau={job.tau:.2f}"
        )
        _run_one(job, force=bool(args.force), plan=plan)

    print("[run_fig2] OK")
    print(f"  out_root: {out_root}")


if __name__ == "__main__":
    main()
