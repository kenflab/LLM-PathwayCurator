#!/usr/bin/env python3
# paper/scripts/run_fig2_pipeline.py
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from llm_pathway_curator.pipeline import RunConfig, run_pipeline

PAPER = Path(__file__).resolve().parents[1]  # paper/
SD = PAPER / "source_data" / "PANCAN_TP53_v1"

EVID_DIR = SD / "evidence_tables"
CARD_DIR = SD / "sample_cards"
OUT_DIR = SD / "out"


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


def _write_card_with_tau(src_card_path: Path, dst_card_path: Path, *, tau: float) -> None:
    if src_card_path.resolve() == dst_card_path.resolve():
        _die(f"[run_fig2] _write_card_with_tau src==dst: {src_card_path}")
    card = _read_json(src_card_path)
    extra = card.get("extra", {})
    if not isinstance(extra, dict):
        extra = {}
    extra["audit_tau"] = float(tau)
    card["extra"] = extra
    _write_json(dst_card_path, card)


def _ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        _die(f"[run_fig2] {label} not found: {path}")
    if not path.is_file():
        _die(f"[run_fig2] {label} is not a file: {path}")
    if path.stat().st_size == 0:
        _die(f"[run_fig2] {label} is empty: {path}")


def _iter_cancers() -> list[str]:
    if not EVID_DIR.exists():
        _die(f"[run_fig2] missing dir: {EVID_DIR}")
    cancers: list[str] = []
    for p in sorted(EVID_DIR.glob("*.evidence_table.tsv")):
        cancers.append(p.name.split(".")[0].upper())
    if not cancers:
        _die("[run_fig2] no evidence_tables found (*.evidence_table.tsv)")
    return cancers


def _make_shuffled_mapping(cancers: list[str]) -> dict[str, str]:
    if len(cancers) < 2:
        _die("[run_fig2] shuffled_context requires >=2 cancers")
    rot = cancers[1:] + cancers[:1]
    return dict(zip(cancers, rot, strict=False))


def _patch_report_jsonl(
    report_path: Path, *, benchmark_id: str, cancer: str, method: str, tau: float
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

            rec["benchmark_id"] = benchmark_id
            rec["cancer"] = cancer
            rec["method"] = method
            rec["tau"] = float(tau)

            out_lines.append(json.dumps(rec, ensure_ascii=False))

    if not out_lines:
        _die(f"[run_fig2] empty report.jsonl after reading: {report_path}")

    report_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def _run_one(
    *,
    benchmark_id: str,
    cancer: str,
    variant: str,
    tau: float,
    evidence_table: Path,
    sample_card: Path,
    outdir: Path,
    seed: int | None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "benchmark_id": benchmark_id,
        "cancer": cancer,
        "variant": variant,
        "tau": tau,
        "evidence_table": str(evidence_table),
        "sample_card": str(sample_card),
        "time_unix": int(time.time()),
        "python": sys.version.split()[0],
        "cwd": str(Path.cwd()),
        "env": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", ""),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", ""),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
        },
    }
    _write_json(outdir / "run_meta.json", run_meta)

    method = variant  # Fig2 "method" field
    cfg_kwargs = dict(
        evidence_table=str(evidence_table),
        sample_card=str(sample_card),
        outdir=str(outdir),
        force=True,
        seed=seed,
        run_meta_name="run_meta.json",
    )
    cfg = RunConfig(**cfg_kwargs)
    run_pipeline(cfg)

    report_path = outdir / "report.jsonl"
    _ensure_file(report_path, "report.jsonl")
    _patch_report_jsonl(
        report_path,
        benchmark_id=benchmark_id,
        cancer=cancer,
        method=method,
        tau=tau,
    )


def _parse_taus(s: str) -> list[float]:
    vals: list[float] = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    if not vals:
        _die("[run_fig2] --taus must be non-empty")
    return vals


def main() -> None:
    ap = argparse.ArgumentParser(description="Fig2 v1: run pipeline for each cancer/variant/tau.")
    ap.add_argument("--benchmark-id", default="PANCAN_TP53_v1")
    ap.add_argument("--cancers", default="ALL", help="Comma list like HNSC,LUAD or ALL")
    ap.add_argument(
        "--variants", default="ours,shuffled_context", help="Comma list: ours,shuffled_context"
    )
    ap.add_argument("--taus", default="0.2,0.4,0.6,0.8,0.9", help="Comma list of tau thresholds")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    benchmark_id = str(args.benchmark_id).strip()
    if not benchmark_id:
        _die("[run_fig2] --benchmark-id must be non-empty")

    all_cancers = _iter_cancers()
    if str(args.cancers).strip().upper() == "ALL":
        cancers = all_cancers
    else:
        cancers = [c.strip().upper() for c in str(args.cancers).split(",") if c.strip()]
        if not cancers:
            _die("[run_fig2] --cancers must not be empty")
        missing = sorted(set(cancers) - set(all_cancers))
        if missing:
            _die(f"[run_fig2] cancers missing evidence_table.tsv: {missing}")

    variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    allowed = {"ours", "shuffled_context"}
    bad = sorted(set(variants) - allowed)
    if bad:
        _die(f"[run_fig2] unsupported variants in v1: {bad} (allowed={sorted(allowed)})")

    taus = _parse_taus(str(args.taus))

    shuffle_map = _make_shuffled_mapping(cancers) if "shuffled_context" in variants else {}

    # fail-fast inputs
    for cancer in cancers:
        _ensure_file(EVID_DIR / f"{cancer}.evidence_table.tsv", f"evidence_table ({cancer})")
        _ensure_file(CARD_DIR / f"{cancer}.sample_card.json", f"sample_card ({cancer})")

    for cancer in cancers:
        for tau in taus:
            for variant in variants:
                ev = EVID_DIR / f"{cancer}.evidence_table.tsv"

                if variant == "ours":
                    src_sc = CARD_DIR / f"{cancer}.sample_card.json"
                    outdir = OUT_DIR / cancer / "ours" / f"tau_{tau:.2f}"
                    outdir.mkdir(parents=True, exist_ok=True)

                    tau_card = outdir / "sample_card.tau.json"
                    _write_card_with_tau(src_sc, tau_card, tau=tau)

                    _run_one(
                        benchmark_id=benchmark_id,
                        cancer=cancer,
                        variant="ours",
                        tau=tau,
                        evidence_table=ev,
                        sample_card=tau_card,
                        outdir=outdir,
                        seed=args.seed,
                    )

                elif variant == "shuffled_context":
                    src = _read_json(CARD_DIR / f"{cancer}.sample_card.json")
                    dst_cancer = shuffle_map.get(cancer, "")
                    if not dst_cancer:
                        _die("[run_fig2] internal error: shuffle_map missing cancer")

                    extra = src.get("extra", {})
                    if not isinstance(extra, dict):
                        extra = {}
                    extra["context_shuffle_from"] = src.get("disease", "")
                    extra["context_shuffle_to"] = dst_cancer
                    extra["audit_tau"] = float(tau)
                    src["extra"] = extra

                    # stress test: swap disease field
                    src["disease"] = dst_cancer

                    outdir = OUT_DIR / cancer / "shuffled_context" / f"tau_{tau:.2f}"
                    outdir.mkdir(parents=True, exist_ok=True)

                    tmp_card = outdir / "sample_card.json"
                    _write_json(tmp_card, src)

                    _run_one(
                        benchmark_id=benchmark_id,
                        cancer=cancer,
                        variant="shuffled_context",
                        tau=tau,
                        evidence_table=ev,
                        sample_card=tmp_card,
                        outdir=outdir,
                        seed=args.seed,
                    )

    print("[run_fig2] OK")
    print(f"  out: {OUT_DIR}")


if __name__ == "__main__":
    main()
