# LLM-PathwayCurator/src/llm_pathway_curator/cli.py
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from .adapters.fgsea import convert_fgsea_table_to_evidence_tsv
from .adapters.metascape import MetascapeAdapterConfig, convert_metascape_table_to_evidence_tsv
from .audit import audit_claims
from .distill import distill_evidence
from .modules import attach_module_ids, factorize_modules_connected_components
from .report import write_report, write_report_jsonl
from .sample_card import SampleCard
from .schema import EvidenceTable
from .select import select_claims


@dataclass(frozen=True)
class RunConfig:
    evidence_table: str
    sample_card: str
    outdir: str
    force: bool = False
    seed: int | None = None
    run_meta_name: str = "run_meta.json"


def _file_fingerprint(path: str) -> dict[str, Any]:
    p = Path(path)
    st = p.stat()
    return {"path": str(p), "size_bytes": int(st.st_size), "mtime_epoch": float(st.st_mtime)}


def _safe_mkdir_outdir(outdir: str, force: bool) -> None:
    p = Path(outdir)
    if p.exists():
        if not p.is_dir():
            raise NotADirectoryError(f"outdir exists but is not a directory: {outdir}")
        if not force and any(p.iterdir()):
            raise FileExistsError(f"outdir is not empty: {outdir} (use --force)")
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def _sha256_text(s: str) -> str:
    h = hashlib.sha256()
    h.update(s.encode("utf-8"))
    return h.hexdigest()


def _env_fingerprint() -> dict[str, Any]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "pandas": getattr(pd, "__version__", "unknown"),
    }


def cmd_run(args: argparse.Namespace) -> None:
    cfg = RunConfig(
        evidence_table=args.evidence_table,
        sample_card=args.sample_card,
        outdir=args.outdir,
        force=args.force,
        seed=args.seed,
        run_meta_name=args.run_meta,
    )

    _safe_mkdir_outdir(cfg.outdir, cfg.force)

    outdir = Path(cfg.outdir)
    meta_path = outdir / cfg.run_meta_name

    # run_id: stable enough to disambiguate runs into same outdir
    run_id = f"{int(time.time())}_{_sha256_text(json.dumps(asdict(cfg), sort_keys=True))[:10]}"

    meta: dict[str, Any] = {
        "tool": "llm-pathway-curator",
        "cmd": "run",
        "run_id": run_id,
        "status": "running",
        "started_epoch": time.time(),
        "config": asdict(cfg),
        "env": _env_fingerprint(),
        "inputs": {
            "evidence_table": _file_fingerprint(cfg.evidence_table),
            "sample_card": _file_fingerprint(cfg.sample_card),
        },
        "artifacts": {},
    }
    _write_json(meta_path, meta)

    def _mark_step(step: str) -> None:
        meta["step"] = step
        _write_json(meta_path, meta)

    try:
        # 0) normalize inputs under internal contract
        _mark_step("normalize_inputs")
        ev_tbl = EvidenceTable.read_tsv(cfg.evidence_table)
        ev_tbl.write_tsv(str(outdir / "evidence.normalized.tsv"))
        meta["artifacts"]["evidence_normalized_tsv"] = str(outdir / "evidence.normalized.tsv")

        # fingerprint normalized evidence for audit reproducibility
        norm_text = (outdir / "evidence.normalized.tsv").read_text(encoding="utf-8")
        meta["inputs"]["evidence_normalized_sha256"] = _sha256_text(norm_text)

        ev = ev_tbl.df.copy()
        card = SampleCard.from_json(cfg.sample_card)

        _write_json(outdir / "sample_card.resolved.json", card.model_dump())
        meta["artifacts"]["sample_card_resolved_json"] = str(outdir / "sample_card.resolved.json")

        # 1) distill (evidence hygiene)
        _mark_step("distill")
        distilled = distill_evidence(ev, card, seed=cfg.seed)
        _write_tsv(distilled, outdir / "distilled.tsv")
        meta["artifacts"]["distilled_tsv"] = str(outdir / "distilled.tsv")

        # 2) modules
        _mark_step("modules")
        mod_out = factorize_modules_connected_components(distilled)
        _write_tsv(mod_out.modules_df, outdir / "modules.tsv")
        _write_tsv(mod_out.term_modules_df, outdir / "term_modules.tsv")
        _write_tsv(mod_out.edges_df, outdir / "term_gene_edges.tsv")
        meta["artifacts"].update(
            {
                "modules_tsv": str(outdir / "modules.tsv"),
                "term_modules_tsv": str(outdir / "term_modules.tsv"),
                "term_gene_edges_tsv": str(outdir / "term_gene_edges.tsv"),
            }
        )

        distilled2 = attach_module_ids(distilled, mod_out.term_modules_df)
        _write_tsv(distilled2, outdir / "distilled.with_modules.tsv")
        meta["artifacts"]["distilled_with_modules_tsv"] = str(outdir / "distilled.with_modules.tsv")

        # 3) propose claims (LLM-min, schema-locked)
        _mark_step("select_claims")
        proposed = select_claims(distilled2, card)
        _write_tsv(proposed, outdir / "claims.proposed.tsv")
        meta["artifacts"]["claims_proposed_tsv"] = str(outdir / "claims.proposed.tsv")

        # 4) mechanical audit (decider)
        _mark_step("audit")
        audited = audit_claims(proposed, distilled2, card)
        _write_tsv(audited, outdir / "audit_log.tsv")
        meta["artifacts"]["audit_log_tsv"] = str(outdir / "audit_log.tsv")

        # 5) report (human-facing, backed by audit log)
        _mark_step("report")
        write_report(audited, distilled2, card, str(outdir))
        meta["artifacts"]["report_dir"] = str(outdir)

        # 6) report.jsonl (Fig2-ready; explicit contract)
        _mark_step("report_jsonl")
        jsonl_path = write_report_jsonl(
            audit_log=audited,
            card=card,
            outdir=str(outdir),
            run_id=run_id,
        )
        meta["artifacts"]["report_jsonl"] = str(jsonl_path)

        meta["status"] = "ok"
        meta["finished_epoch"] = time.time()
        _write_json(meta_path, meta)

    except KeyboardInterrupt:
        meta["status"] = "aborted"
        meta["finished_epoch"] = time.time()
        _write_json(meta_path, meta)
        raise
    except Exception as e:
        meta["status"] = "error"
        meta["finished_epoch"] = time.time()
        meta["error"] = {"type": type(e).__name__, "message": str(e), "step": meta.get("step")}
        _write_json(meta_path, meta)
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        raise


AdaptFormat = Literal["metascape", "fgsea"]


def cmd_adapt(args: argparse.Namespace) -> None:
    fmt: AdaptFormat = args.format
    in_path = args.input
    out_path = args.output

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    if fmt == "metascape":
        cfg = MetascapeAdapterConfig(
            include_summary=bool(args.include_summary),
            source_name=str(args.source_name or "metascape"),
        )
        convert_metascape_table_to_evidence_tsv(in_path, out_path, config=cfg)
    elif fmt == "fgsea":
        convert_fgsea_table_to_evidence_tsv(in_path, out_path)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    # Validate under internal contract (raises on failure)
    EvidenceTable.read_tsv(out_path)
    print(f"[OK] wrote EvidenceTable TSV: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llm-pathway-curator")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run distill → modules → claims → audit → report")
    p_run.add_argument("--evidence-table", required=True, help="TSV EvidenceTable (term×gene)")
    p_run.add_argument("--sample-card", required=True, help="sample_card.json")
    p_run.add_argument("--outdir", required=True, help="output directory")
    p_run.add_argument(
        "--force", action="store_true", help="Allow writing into an existing non-empty outdir"
    )
    p_run.add_argument(
        "--seed", type=int, default=None, help="Optional seed (plumbing; v0 deterministic)"
    )
    p_run.add_argument(
        "--run-meta", default="run_meta.json", help="Filename for run metadata JSON inside outdir"
    )
    p_run.add_argument(
        "--benchmark-id",
        default="",
        help="(optional) benchmark identifier for report.jsonl (e.g., PANCAN_TP53_v1)",
    )
    p_run.add_argument(
        "--cancer",
        default="",
        help="(optional) cancer label for report.jsonl (default: SampleCard.disease)",
    )
    p_run.add_argument(
        "--method",
        default="ours",
        help="(optional) method label for report.jsonl (e.g., ours, baseline_qonly)",
    )
    p_run.set_defaults(func=cmd_run)

    p_adapt = sub.add_parser(
        "adapt", help="Convert external enrichment outputs to EvidenceTable TSV"
    )
    p_adapt.add_argument(
        "--format", required=True, choices=["metascape", "fgsea"], help="Input format"
    )
    p_adapt.add_argument("--input", required=True, help="Input table path (TSV/CSV ok)")
    p_adapt.add_argument("--output", required=True, help="Output EvidenceTable TSV path")
    p_adapt.add_argument(
        "--include-summary",
        action="store_true",
        help="(metascape) include *_Summary rows (default: exclude)",
    )
    p_adapt.add_argument(
        "--source-name", default=None, help="(metascape) override source field (default: metascape)"
    )
    p_adapt.set_defaults(func=cmd_adapt)

    return p


def main(argv: list[str] | None = None) -> None:
    p = build_parser()
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
