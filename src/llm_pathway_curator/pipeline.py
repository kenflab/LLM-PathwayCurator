# src/llm_pathway_curator/pipeline.py
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

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


@dataclass(frozen=True)
class RunResult:
    run_id: str
    outdir: str
    artifacts: dict[str, str]
    meta_path: str


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


def _make_run_id(cfg: RunConfig) -> str:
    payload = json.dumps(asdict(cfg), sort_keys=True, ensure_ascii=False)
    return f"{int(time.time())}_{_sha256_text(payload)[:10]}"


def run_pipeline(cfg: RunConfig, *, run_id: str | None = None) -> RunResult:
    """
    Run the tool pipeline: distill → modules → claims → audit → report (+report.jsonl).

    This is a user-facing tool API and intentionally contains no paper/benchmark logic.
    For reproducibility, it writes run metadata (run_meta.json) and stable artifacts
    into cfg.outdir.
    """
    _safe_mkdir_outdir(cfg.outdir, cfg.force)

    outdir = Path(cfg.outdir)
    meta_path = outdir / cfg.run_meta_name

    rid = str(run_id or _make_run_id(cfg))

    meta: dict[str, Any] = {
        "tool": "llm-pathway-curator",
        "cmd": "run",
        "run_id": rid,
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
        ev_norm_path = outdir / "evidence.normalized.tsv"
        ev_tbl.write_tsv(str(ev_norm_path))
        meta["artifacts"]["evidence_normalized_tsv"] = str(ev_norm_path)

        norm_text = ev_norm_path.read_text(encoding="utf-8")
        meta["inputs"]["evidence_normalized_sha256"] = _sha256_text(norm_text)

        ev = ev_tbl.df.copy()
        card = SampleCard.from_json(cfg.sample_card)

        sc_path = outdir / "sample_card.resolved.json"
        _write_json(sc_path, card.model_dump())
        meta["artifacts"]["sample_card_resolved_json"] = str(sc_path)

        # 1) distill (evidence hygiene)
        _mark_step("distill")
        distilled = distill_evidence(ev, card, seed=cfg.seed)
        dist_path = outdir / "distilled.tsv"
        _write_tsv(distilled, dist_path)
        meta["artifacts"]["distilled_tsv"] = str(dist_path)

        # 2) modules
        _mark_step("modules")
        mod_out = factorize_modules_connected_components(distilled)

        modules_path = outdir / "modules.tsv"
        term_modules_path = outdir / "term_modules.tsv"
        edges_path = outdir / "term_gene_edges.tsv"
        _write_tsv(mod_out.modules_df, modules_path)
        _write_tsv(mod_out.term_modules_df, term_modules_path)
        _write_tsv(mod_out.edges_df, edges_path)

        meta["artifacts"].update(
            {
                "modules_tsv": str(modules_path),
                "term_modules_tsv": str(term_modules_path),
                "term_gene_edges_tsv": str(edges_path),
            }
        )

        distilled2 = attach_module_ids(distilled, mod_out.term_modules_df)
        dist2_path = outdir / "distilled.with_modules.tsv"
        _write_tsv(distilled2, dist2_path)
        meta["artifacts"]["distilled_with_modules_tsv"] = str(dist2_path)

        # 3) propose claims (schema-locked)
        _mark_step("select_claims")
        proposed = select_claims(distilled2, card)
        proposed_path = outdir / "claims.proposed.tsv"
        _write_tsv(proposed, proposed_path)
        meta["artifacts"]["claims_proposed_tsv"] = str(proposed_path)

        # 4) mechanical audit (decider)
        _mark_step("audit")
        audited = audit_claims(proposed, distilled2, card)
        audit_path = outdir / "audit_log.tsv"
        _write_tsv(audited, audit_path)
        meta["artifacts"]["audit_log_tsv"] = str(audit_path)

        # 5) report (human-facing)
        _mark_step("report")
        write_report(audited, distilled2, card, str(outdir))
        meta["artifacts"]["report_dir"] = str(outdir)

        # 6) report.jsonl (stable contract)
        _mark_step("report_jsonl")
        jsonl_path = write_report_jsonl(
            audit_log=audited,
            card=card,
            outdir=str(outdir),
            run_id=rid,
        )
        meta["artifacts"]["report_jsonl"] = str(jsonl_path)

        meta["status"] = "ok"
        meta["finished_epoch"] = time.time()
        _write_json(meta_path, meta)

        return RunResult(
            run_id=rid,
            outdir=str(outdir),
            artifacts={k: str(v) for k, v in meta["artifacts"].items()},
            meta_path=str(meta_path),
        )

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
        raise
