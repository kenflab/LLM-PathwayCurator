#!/usr/bin/env python3
# paper/scripts/run_fig2_pipeline.py
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

from llm_pathway_curator.backends import GeminiBackend, OllamaBackend, OpenAIBackend
from llm_pathway_curator.pipeline import RunConfig, run_pipeline
from llm_pathway_curator.schema import EvidenceTable

PAPER = Path(__file__).resolve().parents[1]  # paper/
SD = PAPER / "source_data" / "PANCAN_TP53_v1"

EVID_DIR = SD / "evidence_tables"
CARD_DIR = SD / "sample_cards"
OUT_DIR = SD / "out"

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
    raise SystemExit(msg)


def _warn(msg: str) -> None:
    print(f"[run_fig2] WARNING: {msg}", file=sys.stderr)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        _die(f"[run_fig2] missing file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        _die(f"[run_fig2] invalid JSON: {path}\n{e}")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        _die(f"[run_fig2] {label} not found: {path}")
    if not path.is_file():
        _die(f"[run_fig2] {label} is not a file: {path}")
    if path.stat().st_size == 0:
        _die(f"[run_fig2] {label} is empty: {path}")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _maybe_git_rev() -> str:
    # Best-effort: works in git checkout; safe in tarball (returns "").
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return ""


def _build_backend_from_env():
    """
    Fig2 runner: construct LLM backend if requested by context review.

    Minimal env contract:
      LLMPATH_BACKEND=openai|ollama|gemini
    Backend-specific secrets/config are handled inside each Backend class.
    """
    kind = str(os.environ.get("LLMPATH_BACKEND", "")).strip().lower()
    if not kind:
        return None
    if kind == "ollama":
        return OllamaBackend()
    if kind == "gemini":
        return GeminiBackend()
    if kind == "openai":
        return OpenAIBackend()
    raise ValueError(f"Unknown LLMPATH_BACKEND={kind!r} (use openai|ollama|gemini)")


def _validate_card_schema(card: dict[str, Any]) -> None:
    extra = card.get("extra", {})
    if isinstance(extra, dict) and "k_claims" in extra:
        _die("[run_fig2] invalid sample_card: extra.k_claims is forbidden (use top-level k_claims)")


def _validate_evidence_table(path: Path) -> None:
    try:
        EvidenceTable.read_tsv(str(path))
    except Exception as e:
        _die(f"[run_fig2] invalid EvidenceTable contract: {path}\n{e}")


def _iter_all_cancers() -> list[str]:
    if not EVID_DIR.exists():
        _die(f"[run_fig2] missing dir: {EVID_DIR}")
    cancers: list[str] = []
    for p in sorted(EVID_DIR.glob("*.evidence_table.tsv")):
        cancers.append(p.name.split(".")[0].upper())
    if not cancers:
        _die("[run_fig2] no evidence_tables found (*.evidence_table.tsv)")
    return cancers


def _parse_csv(arg: str) -> list[str]:
    return [x.strip() for x in str(arg).split(",") if x.strip()]


def _parse_taus(s: str) -> list[float]:
    vals: list[float] = []
    for x in _parse_csv(s):
        vals.append(float(x))
    if not vals:
        _die("[run_fig2] --taus must be non-empty")
    return vals


def _parse_gate_modes(s: str) -> list[str]:
    modes = _parse_csv(s)
    allowed = {"note", "hard"}
    bad = sorted(set(modes) - allowed)
    if bad:
        _die(f"[run_fig2] unsupported gate modes: {bad} (allowed={sorted(allowed)})")
    if not modes:
        _die("[run_fig2] --gate-modes must be non-empty")
    return modes


def _parse_variants(s: str) -> list[str]:
    variants = _parse_csv(s)
    # Meaningful names only; no internal versioning.
    # - ours: baseline
    # - context_swap: swap disease context in sample card (paper ablation)
    # - stress: enable stress knobs (dropout / contradiction) for robustness demo
    allowed = {"ours", "context_swap", "stress"}
    bad = sorted(set(variants) - allowed)
    if bad:
        _die(f"[run_fig2] unsupported variants: {bad} (allowed={sorted(allowed)})")
    if not variants:
        _die("[run_fig2] --variants must be non-empty")
    return variants


def _validate_k_claims(k_claims: int) -> None:
    if k_claims < 1:
        _die(f"[run_fig2] --k-claims must be >= 1 (got {k_claims})")


def _validate_stress_args(
    *,
    p_drop: float,
    min_keep: int,
    p_contra: float,
    contra_cap: int,
) -> None:
    if p_drop < 0.0 or p_drop > 1.0:
        _die(f"[run_fig2] --stress-evidence-dropout-p must be in [0,1] (got {p_drop})")
    if min_keep < 0:
        _die(f"[run_fig2] --stress-evidence-dropout-min-keep must be >= 0 (got {min_keep})")
    if p_contra < 0.0 or p_contra > 1.0:
        _die(f"[run_fig2] --stress-contradictory-p must be in [0,1] (got {p_contra})")
    if contra_cap < 0:
        _die(f"[run_fig2] --stress-contradictory-max-extra must be >= 0 (got {contra_cap})")


def _stress_recipe(*, p_drop: float, p_contra: float) -> str:
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
    extra = card.get("extra", {})
    if not isinstance(extra, dict):
        extra = {}

    # ---- schema normalization (paper contract) ----
    # k_claims is top-level only (never inside extra)
    if "k_claims" in extra:
        extra.pop("k_claims", None)

    extra["audit_tau"] = float(tau)
    card["extra"] = extra

    if k_claims is not None:
        card["k_claims"] = int(k_claims)

    return card


def _set_stress_gate_mode(card: dict[str, Any], mode: str) -> dict[str, Any]:
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


def _write_card_with_tau(
    src_card_path: Path,
    dst_card_path: Path,
    *,
    tau: float,
    k_claims: int,
    stress_gate_mode: str | None = None,
    context_review_mode: str | None = None,
) -> None:
    if src_card_path.resolve() == dst_card_path.resolve():
        _die(f"[run_fig2] _write_card_with_tau src==dst: {src_card_path}")
    card = _read_json(src_card_path)
    card = _set_audit_tau(card, tau, k_claims=int(k_claims))
    if stress_gate_mode is not None:
        card = _set_stress_gate_mode(card, stress_gate_mode)
    if context_review_mode is not None:
        card = _set_context_review_mode(card, context_review_mode)
    _validate_card_schema(card)
    _write_json(dst_card_path, card)


def _outdir_for(*, out_root: Path, cancer: str, variant: str, gate_mode: str, tau: float) -> Path:
    # Paper-friendly layout:
    # out/HNSC/ours/gate_hard/tau_0.20/
    return out_root / cancer / variant / f"gate_{gate_mode}" / f"tau_{tau:.2f}"


def _report_exists_ok(outdir: Path) -> bool:
    rp = outdir / "report.jsonl"
    return rp.exists() and rp.is_file() and rp.stat().st_size > 0


def _patch_report_jsonl(
    report_path: Path,
    *,
    benchmark_id: str,
    cancer: str,
    method: str,
    variant: str,
    gate_mode: str,
    tau: float,
) -> None:
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

            # Stable Fig2 metadata for aggregation (downstream scripts depend on these).
            rec["benchmark_id"] = benchmark_id
            rec["cancer"] = cancer
            rec["method"] = method  # e.g. "ours"
            rec["variant"] = variant  # ours / context_swap / stress
            rec["gate_mode"] = gate_mode  # hard / note
            rec["tau"] = float(tau)

            out_lines.append(json.dumps(rec, ensure_ascii=False))

    if not out_lines:
        _die(f"[run_fig2] empty report.jsonl after reading: {report_path}")

    report_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def _make_shuffled_mapping_all(all_cancers: list[str]) -> dict[str, str]:
    # Deterministic mapping independent of the selected subset.
    if len(all_cancers) < 2:
        _die("[run_fig2] context_swap requires >=2 cancers in evidence_tables/")
    rot = all_cancers[1:] + all_cancers[:1]
    return dict(zip(all_cancers, rot, strict=False))


def _with_temp_env(env_updates: dict[str, str]) -> dict[str, str]:
    """
    Apply env_updates to os.environ and return previous values (or missing marker).
    Caller must restore via _restore_env.
    """
    old: dict[str, str] = {}
    for k, v in env_updates.items():
        if k in os.environ:
            old[k] = os.environ[k]
        else:
            old[k] = "__MISSING__"
        os.environ[k] = str(v)
    return old


def _restore_env(old_values: dict[str, str]) -> None:
    for k, old in old_values.items():
        if old == "__MISSING__":
            os.environ.pop(k, None)
        else:
            os.environ[k] = old


def _write_manifest(
    outdir: Path,
    *,
    benchmark_id: str,
    cancer: str,
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
            "cancer": cancer,
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
            # stress env snapshot (explicit for reproducibility)
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
    """
    Best-effort validation that stress was actually applied.
    - reads run_meta.json written by pipeline (not runner)
    - warns if pipeline saw zeros despite runner intending non-zeros
    - warns if claims.proposed.tsv lacks stress_* columns when expected
    """
    rm = outdir / "run_meta.json"
    if not rm.exists():
        _warn(f"missing run_meta.json for stress check: {rm}")
        return

    try:
        meta = json.loads(rm.read_text(encoding="utf-8"))
    except Exception:
        _warn(f"cannot parse run_meta.json for stress check: {rm}")
        return

    inputs = meta.get("inputs", {})
    s = inputs.get("stress_v4", {})  # pipeline currently writes stress_v4 (legacy key)
    r = inputs.get("stress_v4_runtime", {})

    if not expected_enabled:
        return

    try:
        p_drop = float(s.get("evidence_dropout_p", 0.0) or 0.0)
        p_contra = float(s.get("contradictory_p", 0.0) or 0.0)
    except Exception:
        p_drop, p_contra = 0.0, 0.0

    if p_drop <= 0.0 and p_contra <= 0.0:
        _warn(
            "stress variant ran but pipeline recorded both evidence_dropout_p=0 and "
            "contradictory_p=0 "
            f"(stress may not have been applied). outdir={outdir}"
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
    dropout_enabled = bool(r.get("dropout", {}).get("enabled", False))
    contra_enabled = bool(r.get("contradictory", {}).get("enabled", False))
    if (dropout_enabled or contra_enabled) and (not has_stress_cols):
        _warn(
            "pipeline reports stress enabled at runtime but claims.proposed.tsv has no "
            "stress_* columns "
            f"(schema drift?). outdir={outdir}"
        )


@dataclass(frozen=True)
class Job:
    benchmark_id: str
    cancer: str
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
    # stress (job-local)
    stress_evidence_dropout_p: float = 0.0
    stress_evidence_dropout_min_keep: int = 1
    stress_contradictory_p: float = 0.0
    stress_contradictory_max_extra: int = 0


def _run_one(job: Job, *, force: bool, plan: dict[str, Any], backend: Any | None) -> None:
    if (not force) and _report_exists_ok(job.outdir):
        print(f"[run_fig2] SKIP (exists): {job.outdir}")
        return

    started = int(time.time())
    plan2 = dict(plan)
    plan2["started_unix"] = started

    # Decide stress knobs for this job:
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

    # Inject job-local env (pipeline currently reads these knobs from env)
    env_updates = {
        ENV_STRESS["dropout_p"]: str(stress["evidence_dropout_p"]),
        ENV_STRESS["min_keep"]: str(stress["evidence_dropout_min_keep"]),
        ENV_STRESS["contra_p"]: str(stress["contradictory_p"]),
        ENV_STRESS["contra_cap"]: str(stress["contradictory_max_extra"]),
    }
    old_env = _with_temp_env(env_updates)

    try:
        # Runner-level run meta (do not collide with pipeline's run_meta.json)
        run_meta_runner = {
            "benchmark_id": job.benchmark_id,
            "cancer": job.cancer,
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
        )
        run_pipeline(cfg)

        report_path = job.outdir / "report.jsonl"
        _ensure_file(report_path, "report.jsonl")

        _patch_report_jsonl(
            report_path,
            benchmark_id=job.benchmark_id,
            cancer=job.cancer,
            method=job.method,
            variant=job.variant,
            gate_mode=job.gate_mode,
            tau=job.tau,
        )

        _write_manifest(
            job.outdir,
            benchmark_id=job.benchmark_id,
            cancer=job.cancer,
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
            cancer=job.cancer,
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fig2 (paper): deterministic tau-sweep over cancers × variants × gate modes."
    )
    ap.add_argument("--benchmark-id", default="PANCAN_TP53_v1")
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
        default=str(OUT_DIR),
        help="Output root directory (default: paper/source_data/.../out)",
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
            "Optional: force context_swap target cancer (single-cancer runs). "
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

    # Build backend once per runner invocation (jobs are sequential).
    backend = None
    if str(args.context_review_mode).strip().lower() == "llm":
        try:
            backend = _build_backend_from_env()
        except Exception as e:
            _die(f"[run_fig2] failed to build backend (LLMPATH_BACKEND): {e}")
        if backend is None:
            _die(
                "[run_fig2] --context-review-mode llm requires LLMPATH_BACKEND "
                "to be set (openai|ollama|gemini)."
            )
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

    out_root = Path(str(args.out_root)).resolve()

    all_cancers = _iter_all_cancers()
    if str(args.cancers).strip().upper() == "ALL":
        cancers = all_cancers
    else:
        cancers = [c.strip().upper() for c in _parse_csv(str(args.cancers))]
        if not cancers:
            _die("[run_fig2] --cancers must not be empty")
        missing = sorted(set(cancers) - set(all_cancers))
        if missing:
            _die(f"[run_fig2] cancers missing evidence_table.tsv: {missing}")

    variants = _parse_variants(str(args.variants))
    taus = _parse_taus(str(args.taus))
    gate_modes = _parse_gate_modes(str(args.gate_modes))

    # Demo-safety: if stress is requested, warn on "off"
    if "stress" in variants:
        if (
            _stress_recipe(
                p_drop=float(args.stress_evidence_dropout_p),
                p_contra=float(args.stress_contradictory_p),
            )
            == "off"
        ):
            _warn(
                "variants include 'stress' but both --stress-evidence-dropout-p and "
                "--stress-contradictory-p are 0.0. This will likely produce outputs "
                "indistinguishable from 'ours'."
            )

    # Validate inputs (fail-fast).
    for cancer in cancers:
        ev_path = EVID_DIR / f"{cancer}.evidence_table.tsv"
        _ensure_file(ev_path, f"evidence_table ({cancer})")
        _validate_evidence_table(ev_path)

        for gm in gate_modes:
            _ensure_file(
                CARD_DIR / f"{cancer}.{gm}.sample_card.json", f"sample_card {gm} ({cancer})"
            )

    # Deterministic mapping independent of selected subset for context_swap
    swap_map_all = _make_shuffled_mapping_all(all_cancers) if "context_swap" in variants else {}

    forced_dst = str(args.context_swap_to).strip().upper()
    if forced_dst:
        if forced_dst not in set(all_cancers):
            _die(
                "[run_fig2] --context-swap-to must be one of cancers with evidence_tables "
                f"(got {forced_dst})"
            )

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
        "cancers": cancers,
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

    print("[run_fig2] plan")
    print(f"  benchmark_id: {benchmark_id}")
    print(f"  cancers: {','.join(cancers)}")
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

    jobs: list[Job] = []

    for cancer in cancers:
        ev = EVID_DIR / f"{cancer}.evidence_table.tsv"

        for tau in taus:
            for gate_mode in gate_modes:
                # -------------------------
                # Variant: ours
                # -------------------------
                if "ours" in variants:
                    src_sc = CARD_DIR / f"{cancer}.{gate_mode}.sample_card.json"
                    outdir = _outdir_for(
                        out_root=out_root,
                        cancer=cancer,
                        variant="ours",
                        gate_mode=gate_mode,
                        tau=tau,
                    )
                    outdir.mkdir(parents=True, exist_ok=True)

                    tau_card = outdir / "sample_card.tau.json"
                    _write_card_with_tau(
                        src_sc,
                        tau_card,
                        tau=tau,
                        k_claims=int(args.k_claims),
                        context_review_mode=str(args.context_review_mode),
                    )

                    jobs.append(
                        Job(
                            benchmark_id=benchmark_id,
                            cancer=cancer,
                            method="ours",
                            variant="ours",
                            gate_mode=gate_mode,
                            tau=float(tau),
                            k_claims=int(args.k_claims),
                            seed=int(args.seed),
                            evidence_table=ev,
                            sample_card=tau_card,
                            outdir=outdir,
                            # ensure stress off for non-stress variants
                            stress_evidence_dropout_p=0.0,
                            stress_evidence_dropout_min_keep=int(
                                args.stress_evidence_dropout_min_keep
                            ),
                            stress_contradictory_p=0.0,
                            stress_contradictory_max_extra=int(args.stress_contradictory_max_extra),
                        )
                    )

                # -------------------------
                # Variant: stress (robustness demo)
                # -------------------------
                if "stress" in variants:
                    src_sc = CARD_DIR / f"{cancer}.{gate_mode}.sample_card.json"
                    outdir = _outdir_for(
                        out_root=out_root,
                        cancer=cancer,
                        variant="stress",
                        gate_mode=gate_mode,
                        tau=tau,
                    )
                    outdir.mkdir(parents=True, exist_ok=True)

                    tau_card = outdir / "sample_card.tau.json"
                    _write_card_with_tau(
                        src_sc,
                        tau_card,
                        tau=tau,
                        k_claims=int(args.k_claims),
                        stress_gate_mode="hard",
                        context_review_mode=str(args.context_review_mode),
                    )

                    jobs.append(
                        Job(
                            benchmark_id=benchmark_id,
                            cancer=cancer,
                            method="ours",
                            variant="stress",
                            gate_mode=gate_mode,
                            tau=float(tau),
                            k_claims=int(args.k_claims),
                            seed=int(args.seed),
                            evidence_table=ev,
                            sample_card=tau_card,
                            outdir=outdir,
                            stress_evidence_dropout_p=float(args.stress_evidence_dropout_p),
                            stress_evidence_dropout_min_keep=int(
                                args.stress_evidence_dropout_min_keep
                            ),
                            stress_contradictory_p=float(args.stress_contradictory_p),
                            stress_contradictory_max_extra=int(args.stress_contradictory_max_extra),
                        )
                    )

                # -------------------------
                # Variant: context_swap (paper ablation; formerly shuffled_context)
                # -------------------------
                if "context_swap" in variants:
                    dst_cancer = forced_dst or swap_map_all.get(cancer, "")
                    if not dst_cancer:
                        _die("[run_fig2] internal error: context swap map missing cancer")

                    src_sc_path = CARD_DIR / f"{cancer}.{gate_mode}.sample_card.json"
                    src = _read_json(src_sc_path)

                    extra = src.get("extra", {})
                    if not isinstance(extra, dict):
                        extra = {}

                    extra["context_swap_from"] = str(src.get("disease", "")).strip()
                    extra["context_swap_to"] = dst_cancer
                    src["extra"] = extra

                    # Counterfactual: swap disease field (template-only).
                    src["disease"] = dst_cancer

                    # Ensure tau + k_claims are set in the card copy.
                    src = _set_audit_tau(src, float(tau), k_claims=int(args.k_claims))
                    src = _set_context_review_mode(src, str(args.context_review_mode))

                    outdir2 = _outdir_for(
                        out_root=out_root,
                        cancer=cancer,
                        variant="context_swap",
                        gate_mode=gate_mode,
                        tau=tau,
                    )
                    outdir2.mkdir(parents=True, exist_ok=True)

                    tmp_card = outdir2 / "sample_card.tau.json"
                    _validate_card_schema(src)
                    _write_json(tmp_card, src)

                    jobs.append(
                        Job(
                            benchmark_id=benchmark_id,
                            cancer=cancer,
                            method="ours",
                            variant="context_swap",
                            gate_mode=gate_mode,
                            tau=float(tau),
                            k_claims=int(args.k_claims),
                            seed=int(args.seed),
                            evidence_table=ev,
                            sample_card=tmp_card,
                            outdir=outdir2,
                            context_swap_from=str(extra.get("context_swap_from", "")),
                            context_swap_to=str(extra.get("context_swap_to", "")),
                            # ensure stress off
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
            f"[run_fig2] ({i}/{n_total}) {job.cancer} {job.variant} "
            f"gate={job.gate_mode} tau={job.tau:.2f}"
        )
        _run_one(job, force=bool(args.force), plan=plan, backend=backend)

    print("[run_fig2] OK")
    print(f"  out_root: {out_root}")


if __name__ == "__main__":
    main()
