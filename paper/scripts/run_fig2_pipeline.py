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

from llm_pathway_curator.pipeline import RunConfig, run_pipeline
from llm_pathway_curator.schema import EvidenceTable

PAPER = Path(__file__).resolve().parents[1]  # paper/
SD = PAPER / "source_data" / "PANCAN_TP53_v1"

EVID_DIR = SD / "evidence_tables"
CARD_DIR = SD / "sample_cards"
OUT_DIR = SD / "out"


# -------------------------
# Utils
# -------------------------
def _die(msg: str) -> None:
    raise SystemExit(msg)


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
    allowed = {"ours", "shuffled_context"}
    bad = sorted(set(variants) - allowed)
    if bad:
        _die(f"[run_fig2] unsupported variants in v1: {bad} (allowed={sorted(allowed)})")
    if not variants:
        _die("[run_fig2] --variants must be non-empty")
    return variants


def _validate_k_claims(k_claims: int) -> None:
    if k_claims < 1:
        _die(f"[run_fig2] --k-claims must be >= 1 (got {k_claims})")


def _set_audit_tau(
    card: dict[str, Any],
    tau: float,
    *,
    k_claims: int | None = None,
) -> dict[str, Any]:
    extra = card.get("extra", {})
    if not isinstance(extra, dict):
        extra = {}
    extra["audit_tau"] = float(tau)
    card["extra"] = extra
    if k_claims is not None:
        # SampleCard reads top-level k_claims (not extra)
        card["k_claims"] = int(k_claims)
    return card


def _write_card_with_tau(
    src_card_path: Path, dst_card_path: Path, *, tau: float, k_claims: int
) -> None:
    if src_card_path.resolve() == dst_card_path.resolve():
        _die(f"[run_fig2] _write_card_with_tau src==dst: {src_card_path}")
    card = _read_json(src_card_path)
    card = _set_audit_tau(card, tau, k_claims=int(k_claims))
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
            rec["variant"] = variant  # e.g. "ours" or "shuffled_context"
            rec["gate_mode"] = gate_mode  # "hard" or "note"
            rec["tau"] = float(tau)

            out_lines.append(json.dumps(rec, ensure_ascii=False))

    if not out_lines:
        _die(f"[run_fig2] empty report.jsonl after reading: {report_path}")

    report_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def _make_shuffled_mapping_all(all_cancers: list[str]) -> dict[str, str]:
    # Deterministic mapping independent of the selected subset.
    if len(all_cancers) < 2:
        _die("[run_fig2] shuffled_context requires >=2 cancers in evidence_tables/")
    rot = all_cancers[1:] + all_cancers[:1]
    return dict(zip(all_cancers, rot, strict=False))


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
    context_shuffle_from: str,
    context_shuffle_to: str,
    plan: dict[str, Any],
    status: str,
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
            "context_shuffle_from": context_shuffle_from,
            "context_shuffle_to": context_shuffle_to,
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
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", ""),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", ""),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
        },
        "plan": plan,
    }
    _write_json(outdir / "manifest.json", manifest)


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
    context_shuffle_from: str = ""
    context_shuffle_to: str = ""


def _run_one(job: Job, *, force: bool, plan: dict[str, Any]) -> None:
    job.outdir.mkdir(parents=True, exist_ok=True)

    if (not force) and _report_exists_ok(job.outdir):
        # Resume-friendly default: do not overwrite paper outputs.
        print(f"[run_fig2] SKIP (exists): {job.outdir}")
        return

    started = int(time.time())
    plan2 = dict(plan)
    plan2["started_unix"] = started

    try:
        # Write a minimal run_meta.json for human inspection.
        run_meta = {
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
            "context_shuffle_from": job.context_shuffle_from,
            "context_shuffle_to": job.context_shuffle_to,
            "time_unix": started,
        }
        _write_json(job.outdir / "run_meta.json", run_meta)

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
            context_shuffle_from=job.context_shuffle_from,
            context_shuffle_to=job.context_shuffle_to,
            plan=plan2,
            status="ok",
        )

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
            context_shuffle_from=job.context_shuffle_from,
            context_shuffle_to=job.context_shuffle_to,
            plan=plan2,
            status="error",
            error=str(e),
        )
        raise


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fig2 (paper): deterministic tau-sweep over cancers × variants × gate modes."
    )
    ap.add_argument("--benchmark-id", default="PANCAN_TP53_v1")
    ap.add_argument("--cancers", default="ALL", help="Comma list like HNSC,LUAD or ALL")
    ap.add_argument(
        "--variants", default="ours,shuffled_context", help="Comma list: ours,shuffled_context"
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
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Paper default seed (fixed for reproducibility).",
    )
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
        "--shuffled-context-to",
        default="",
        help=(
            "Optional: force shuffled_context target cancer (useful for single-cancer runs). "
            "Example: --cancers HNSC --variants ours,shuffled_context --shuffled-context-to LUAD"
        ),
    )
    args = ap.parse_args()

    benchmark_id = str(args.benchmark_id).strip()
    if not benchmark_id:
        _die("[run_fig2] --benchmark-id must be non-empty")

    _validate_k_claims(int(args.k_claims))

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

    # Validate inputs (fail-fast).
    for cancer in cancers:
        ev_path = EVID_DIR / f"{cancer}.evidence_table.tsv"
        _ensure_file(ev_path, f"evidence_table ({cancer})")
        _validate_evidence_table(ev_path)

        for gm in gate_modes:
            _ensure_file(
                CARD_DIR / f"{cancer}.{gm}.sample_card.json", f"sample_card {gm} ({cancer})"
            )

    # Deterministic shuffle mapping independent of selected subset.
    shuffle_map_all = (
        _make_shuffled_mapping_all(all_cancers) if "shuffled_context" in variants else {}
    )

    forced_dst = str(args.shuffled_context_to).strip().upper()
    if forced_dst:
        if forced_dst not in set(all_cancers):
            _die(
                
                    "[run_fig2] --shuffled-context-to must be one of cancers with "
                    f"evidence_tables (got {forced_dst})"
                
            )

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
        print(f"  shuffled_context_to: {forced_dst}")
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
                    _write_card_with_tau(src_sc, tau_card, tau=tau, k_claims=int(args.k_claims))

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
                        )
                    )

                # -------------------------
                # Variant: shuffled_context (stress test)
                # -------------------------
                if "shuffled_context" in variants:
                    dst_cancer = forced_dst or shuffle_map_all.get(cancer, "")
                    if not dst_cancer:
                        _die("[run_fig2] internal error: shuffle_map missing cancer")

                    src_sc_path = CARD_DIR / f"{cancer}.{gate_mode}.sample_card.json"
                    src = _read_json(src_sc_path)

                    extra = src.get("extra", {})
                    if not isinstance(extra, dict):
                        extra = {}

                    extra["context_shuffle_from"] = str(src.get("disease", "")).strip()
                    extra["context_shuffle_to"] = dst_cancer
                    src["extra"] = extra

                    # Counterfactual: swap disease field (template-only).
                    src["disease"] = dst_cancer

                    # Ensure tau + k_claims are set in the card copy.
                    src = _set_audit_tau(src, float(tau), k_claims=int(args.k_claims))

                    outdir2 = _outdir_for(
                        out_root=out_root,
                        cancer=cancer,
                        variant="shuffled_context",
                        gate_mode=gate_mode,
                        tau=tau,
                    )
                    outdir2.mkdir(parents=True, exist_ok=True)

                    tmp_card = outdir2 / "sample_card.tau.json"
                    _write_json(tmp_card, src)

                    jobs.append(
                        Job(
                            benchmark_id=benchmark_id,
                            cancer=cancer,
                            method="ours",
                            variant="shuffled_context",
                            gate_mode=gate_mode,
                            tau=float(tau),
                            k_claims=int(args.k_claims),
                            seed=int(args.seed),
                            evidence_table=ev,
                            sample_card=tmp_card,
                            outdir=outdir2,
                            context_shuffle_from=str(extra.get("context_shuffle_from", "")),
                            context_shuffle_to=str(extra.get("context_shuffle_to", "")),
                        )
                    )

    # Execute jobs
    n_total = len(jobs)
    print(f"[run_fig2] jobs: {n_total}")
    for i, job in enumerate(jobs, start=1):
        print(
            
                f"[run_fig2] ({i}/{n_total}) {job.cancer} {job.variant} "
                f"gate={job.gate_mode} tau={job.tau:.2f}"
            
        )
        _run_one(job, force=bool(args.force), plan=plan)

    print("[run_fig2] OK")
    print(f"  out_root: {out_root}")


if __name__ == "__main__":
    main()
