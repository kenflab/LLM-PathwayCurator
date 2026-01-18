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

# =========================
# Fixed paths (v1)
# =========================
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
    # rotate list: HNSC->LUAD->...->HNSC
    if len(cancers) < 2:
        _die("[run_fig2] shuffled_context requires >=2 cancers")
    rot = cancers[1:] + cancers[:1]
    return dict(zip(cancers, rot, strict=False))


def _patch_report_jsonl(
    report_path: Path,
    *,
    benchmark_id: str,
    cancer: str,
    method: str,
) -> None:
    """
    Enforce Fig2 input contract: benchmark_id/cancer/method must exist.
    Also keep any existing fields; only fill/overwrite these meta keys.
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
            rec["cancer"] = cancer
            rec["method"] = method

            out_lines.append(json.dumps(rec, ensure_ascii=False))

    if not out_lines:
        _die(f"[run_fig2] empty report.jsonl after reading: {report_path}")

    report_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def _run_one(
    *,
    benchmark_id: str,
    cancer: str,
    variant: str,
    evidence_table: Path,
    sample_card: Path,
    outdir: Path,
    seed: int | None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # run_meta: minimal provenance
    run_meta = {
        "benchmark_id": benchmark_id,
        "cancer": cancer,
        "variant": variant,
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

    # Try passing method into RunConfig (if supported).
    # If not supported, we still run, then patch report.jsonl to satisfy Fig2 contract.
    method = variant  # Fig2 method field
    cfg_kwargs: dict[str, Any] = dict(
        evidence_table=str(evidence_table),
        sample_card=str(sample_card),
        outdir=str(outdir),
        force=True,  # paper scripts are allowed to overwrite
        seed=seed,
        run_meta_name="run_meta.json",
    )

    cfg = None
    try:
        cfg = RunConfig(**cfg_kwargs, method=method)  # type: ignore[arg-type]
    except TypeError:
        cfg = RunConfig(**cfg_kwargs)

    run_pipeline(cfg)

    report_path = outdir / "report.jsonl"
    _ensure_file(report_path, "report.jsonl")
    _patch_report_jsonl(report_path, benchmark_id=benchmark_id, cancer=cancer, method=method)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fig2 v1: run pipeline to produce report.jsonl for each cancer and variant."
    )
    ap.add_argument(
        "--benchmark-id",
        default="PANCAN_TP53_v1",
        help="benchmark_id to stamp into report.jsonl",
    )
    ap.add_argument(
        "--cancers",
        default="ALL",
        help="Comma list like HNSC,LUAD or ALL (default: ALL from evidence_tables/)",
    )
    ap.add_argument(
        "--variants",
        default="ours,shuffled_context",
        help="Comma list (v1 supports: ours,shuffled_context)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed plumbing",
    )
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

    shuffle_map = _make_shuffled_mapping(cancers) if "shuffled_context" in variants else {}

    # Fail-fast on required inputs
    for cancer in cancers:
        ev = EVID_DIR / f"{cancer}.evidence_table.tsv"
        sc = CARD_DIR / f"{cancer}.sample_card.json"
        _ensure_file(ev, f"evidence_table ({cancer})")
        _ensure_file(sc, f"sample_card ({cancer})")

    # Run
    for cancer in cancers:
        for variant in variants:
            ev = EVID_DIR / f"{cancer}.evidence_table.tsv"

            if variant == "ours":
                sc = CARD_DIR / f"{cancer}.sample_card.json"
                outdir = OUT_DIR / cancer / "ours"
                _run_one(
                    benchmark_id=benchmark_id,
                    cancer=cancer,
                    variant="ours",
                    evidence_table=ev,
                    sample_card=sc,
                    outdir=outdir,
                    seed=args.seed,
                )

            elif variant == "shuffled_context":
                # Temporary shuffled card: swap disease only (minimal stress test)
                src_card = _read_json(CARD_DIR / f"{cancer}.sample_card.json")
                dst_cancer = shuffle_map.get(cancer, "")
                if not dst_cancer:
                    _die("[run_fig2] internal error: shuffle_map missing cancer")

                src_card["disease"] = dst_cancer  # key stress
                tmp_dir = OUT_DIR / cancer / "shuffled_context"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp_card = tmp_dir / "sample_card.shuffled.json"
                _write_json(tmp_card, src_card)

                outdir = OUT_DIR / cancer / "shuffled_context"
                _run_one(
                    benchmark_id=benchmark_id,
                    cancer=cancer,
                    variant="shuffled_context",
                    evidence_table=ev,
                    sample_card=tmp_card,
                    outdir=outdir,
                    seed=args.seed,
                )

            else:
                _die(f"[run_fig2] unreachable variant: {variant}")

    print("[run_fig2] OK")
    print(f"  out: {OUT_DIR}")


if __name__ == "__main__":
    main()
