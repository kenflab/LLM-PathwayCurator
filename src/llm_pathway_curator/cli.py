# LLM-PathwayCurator/src/llm_pathway_curator/cli.py
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

# adapters
from .adapters.fgsea import convert_fgsea_table_to_evidence_tsv
from .adapters.metascape import MetascapeAdapterConfig, convert_metascape_table_to_evidence_tsv
from .audit import audit_claims
from .distill import distill_evidence
from .modules import attach_module_ids, factorize_modules_connected_components
from .report import write_report
from .sample_card import SampleCard
from .schema import EvidenceTable
from .select import select_claims


@dataclass(frozen=True)
class RunConfig:
    evidence_table: str
    sample_card: str
    outdir: str
    force: bool = False
    dump_modules: bool = False
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
            raise FileExistsError(
                f"outdir is not empty: {outdir} (use --force to overwrite/mix artifacts)"
            )
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def cmd_run(args: argparse.Namespace) -> None:
    cfg = RunConfig(
        evidence_table=args.evidence_table,
        sample_card=args.sample_card,
        outdir=args.outdir,
        force=args.force,
        dump_modules=args.dump_modules,
        seed=args.seed,
        run_meta_name=args.run_meta,
    )

    _safe_mkdir_outdir(cfg.outdir, cfg.force)

    outdir = Path(cfg.outdir)
    meta_path = outdir / cfg.run_meta_name

    meta: dict[str, Any] = {
        "tool": "llm-pathway-curator",
        "cmd": "run",
        "status": "running",
        "started_epoch": time.time(),
        "config": asdict(cfg),
        "inputs": {
            "evidence_table": _file_fingerprint(cfg.evidence_table),
            "sample_card": _file_fingerprint(cfg.sample_card),
        },
    }
    _write_json(meta_path, meta)

    try:
        ev_tbl = EvidenceTable.read_tsv(cfg.evidence_table)
        ev = ev_tbl.df
        card = SampleCard.from_json(cfg.sample_card)

        distilled = distill_evidence(ev, card, seed=cfg.seed)

        mod_out = factorize_modules_connected_components(distilled)
        distilled = attach_module_ids(distilled, mod_out.term_modules_df)

        proposed = select_claims(distilled, card)
        audited = audit_claims(proposed, distilled, card)

        write_report(audited, distilled, card, cfg.outdir)
        _write_json(outdir / "sample_card.resolved.json", card.model_dump())

        if cfg.dump_modules:
            mod_out.modules_df.to_csv(outdir / "modules.tsv", sep="\t", index=False)
            mod_out.term_modules_df.to_csv(outdir / "term_modules.tsv", sep="\t", index=False)
            mod_out.edges_df.to_csv(outdir / "term_gene_edges.tsv", sep="\t", index=False)

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
        meta["error"] = {"type": type(e).__name__, "message": str(e)}
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
        # fgsea adapter already hard-fails if leadingEdge is empty (by design)
        convert_fgsea_table_to_evidence_tsv(in_path, out_path)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    # sanity: ensure the output is readable by EvidenceTable (your internal contract)
    EvidenceTable.read_tsv(out_path)
    print(f"[OK] wrote EvidenceTable TSV: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llm-pathway-curator")
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- run ---
    p_run = sub.add_parser("run", help="Run distill → modules → claims → audit → report")
    p_run.add_argument("--evidence-table", required=True, help="TSV EvidenceTable (term×gene)")
    p_run.add_argument("--sample-card", required=True, help="sample_card.json")
    p_run.add_argument("--outdir", required=True, help="output directory")
    p_run.add_argument(
        "--force", action="store_true", help="Allow writing into an existing non-empty outdir"
    )
    p_run.add_argument(
        "--dump-modules",
        action="store_true",
        help="Also write modules.tsv / term_modules.tsv / term_gene_edges.tsv",
    )
    p_run.add_argument(
        "--seed", type=int, default=None, help="Optional seed (plumbing; v0 deterministic)"
    )
    p_run.add_argument(
        "--run-meta", default="run_meta.json", help="Filename for run metadata JSON inside outdir"
    )
    p_run.set_defaults(func=cmd_run)

    # --- adapt ---
    p_adapt = sub.add_parser(
        "adapt", help="Convert external enrichment outputs to EvidenceTable TSV"
    )
    p_adapt.add_argument(
        "--format", required=True, choices=["metascape", "fgsea"], help="Input format"
    )
    p_adapt.add_argument("--input", required=True, help="Input table path (TSV/CSV ok)")
    p_adapt.add_argument("--output", required=True, help="Output EvidenceTable TSV path")
    # metascape options
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
