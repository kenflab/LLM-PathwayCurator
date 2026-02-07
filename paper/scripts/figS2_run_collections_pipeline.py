#!/usr/bin/env python3
# paper/scripts/run_figS2_collections_pipeline.py

"""
Supp Fig S2 reproduction orchestrator (collections sensitivity).

This script runs the core LLM-PathwayCurator pipeline at a fixed tau (default 0.80)
across gene set collections (Hallmark/GO BP/Reactome/KEGG) and variants
(ours/context_swap/stress), over multiple cancers and audit gate modes.

Inputs
------
- EvidenceTables:
  paper/source_data/<benchmark_id>/evidence_tables/<CANCER>.<suffix>.evidence_table.tsv
- Sample Cards:
  paper/source_data/<benchmark_id>/sample_cards/<CANCER>.<gate>.sample_card.json

Outputs (per job)
-----------------
<out_root>/<collection_slug>/<CANCER>/<variant>/gate_<gate>/tau_<tau>/
  - report.jsonl, audit_log.tsv, claims.proposed.tsv, modules.tsv, run_meta.json
  - run_meta.runner.json (runner-added metadata)

Determinism
-----------
Controlled by `--seed` (default 42) and library-level fixed settings.

Dependencies
------------
`llm_pathway_curator.pipeline.run_pipeline` and the EvidenceTable TSV contract.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_pathway_curator.pipeline import RunConfig, run_pipeline
from llm_pathway_curator.schema import EvidenceTable

PAPER = Path(__file__).resolve().parents[1]  # paper/


# -------------------------
# Utils
# -------------------------
def _die(msg: str) -> None:
    """Exit the runner with a message (raises SystemExit)."""
    raise SystemExit(msg)


def _warn(msg: str) -> None:
    """Print a warning message to stderr (runner prefix included)."""
    print(f"[run_figS2] WARNING: {msg}", file=sys.stderr)


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file into a dict, or exit with a helpful error."""
    if not path.exists():
        _die(f"[run_figS2] missing file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        _die(f"[run_figS2] invalid JSON: {path}\n{e}")


def _write_json(path: Path, obj: Any) -> None:
    """Write an object as pretty JSON (sorted keys) and ensure parent dirs exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ensure_file(path: Path, label: str) -> None:
    """Fail-fast if `path` is missing, not a file, or empty."""
    if not path.exists():
        _die(f"[run_figS2] {label} not found: {path}")
    if not path.is_file():
        _die(f"[run_figS2] {label} is not a file: {path}")
    if path.stat().st_size == 0:
        _die(f"[run_figS2] {label} is empty: {path}")


def _parse_csv(arg: str) -> list[str]:
    """Parse a comma-separated CLI argument into a list of non-empty tokens."""
    return [x.strip() for x in str(arg).split(",") if x.strip()]


def _validate_evidence_table(path: Path) -> None:
    """Validate EvidenceTable TSV contract via `EvidenceTable.read_tsv`."""
    try:
        EvidenceTable.read_tsv(str(path))
    except Exception as e:
        _die(f"[run_figS2] invalid EvidenceTable contract: {path}\n{e}")


def _report_exists_ok(outdir: Path) -> bool:
    """Return True if `outdir/report.jsonl` exists and is non-empty."""
    rp = outdir / "report.jsonl"
    return rp.exists() and rp.is_file() and rp.stat().st_size > 0


def _outdir_for(
    *, out_root: Path, condition: str, variant: str, gate_mode: str, tau: float
) -> Path:
    """Build a canonical output directory path for one job."""
    return out_root / condition / variant / f"gate_{gate_mode}" / f"tau_{tau:.2f}"


def _make_rotation_map(conds: list[str]) -> dict[str, str]:
    """Create a deterministic rotation map used for context_swap within the run."""
    if len(conds) < 2:
        _die("[run_figS2] context_swap requires >=2 conditions (cancers) in this run")
    rot = conds[1:] + conds[:1]
    return dict(zip(conds, rot, strict=False))


def _set_card_extra(card: dict[str, Any], key: str, val: Any) -> dict[str, Any]:
    """Set `card['extra'][key] = val` (creating a dict if needed)."""
    extra = card.get("extra", {})
    if not isinstance(extra, dict):
        extra = {}
    extra[key] = val
    card["extra"] = extra
    return card


def _set_k_claims(card: dict[str, Any], k_claims: int) -> dict[str, Any]:
    """Set top-level `k_claims` and remove forbidden `extra.k_claims` if present."""
    extra = card.get("extra", {})
    if isinstance(extra, dict) and "k_claims" in extra:
        extra.pop("k_claims", None)
        card["extra"] = extra
    card["k_claims"] = int(k_claims)
    return card


def _set_audit_tau(card: dict[str, Any], tau: float) -> dict[str, Any]:
    """Set `extra.audit_tau` for audit configuration metadata."""
    return _set_card_extra(card, "audit_tau", float(tau))


def _set_context_review_mode(card: dict[str, Any], mode: str) -> dict[str, Any]:
    """Set `extra.context_review_mode` (off/llm) with validation."""
    mode = str(mode).strip().lower()
    if mode not in {"off", "llm"}:
        _die(f"[run_figS2] invalid context_review_mode={mode} (allowed: off,llm)")
    return _set_card_extra(card, "context_review_mode", mode)


def _require_backend_env_for_context_review() -> None:
    """Fail-fast unless `LLMPATH_BACKEND` is set to a supported backend."""
    kind = str(os.environ.get("LLMPATH_BACKEND", "")).strip().lower()
    if not kind:
        _die(
            "[run_figS2] --context-review-mode llm requires LLMPATH_BACKEND "
            "to be set (openai|ollama|gemini)."
        )
    if kind not in {"openai", "ollama", "gemini"}:
        _die(f"[run_figS2] Unknown LLMPATH_BACKEND={kind!r} (use openai|ollama|gemini)")


# -------------------------
# Collection spec
# -------------------------
@dataclass(frozen=True)
class CollectionSpec:
    """
    Gene set collection specification for the collections sensitivity pipeline.

    Parameters
    ----------
    name
        Paper-facing label (e.g., Hallmark, GO_BP, Reactome, KEGG_MEDICUS).
    suffix
        EvidenceTable filename suffix produced by the paper-side fgsea scripts.
        Expected file pattern: `<CANCER>.<suffix>.evidence_table.tsv`.
    out_slug
        Directory slug under `out_root/` used to group outputs by collection.

    Notes
    -----
    Inputs: used only for planning and file resolution.
    Outputs: affects output directory layout under `out_root/`.
    Determinism: deterministic.
    Dependencies: the file naming convention in `evidence_tables/`.
    """

    name: str  # paper-facing label (Hallmark/GO_BP/Reactome/KEGG_MEDICUS)
    suffix: str  # evidence_tables filename suffix (matches your fgsea script output)
    out_slug: str  # directory slug under out_root


def _default_collections() -> list[CollectionSpec]:
    """
    Return the default collection set used for the paper.

    Returns
    -------
    list[CollectionSpec]
        Default collections in paper order.

    Notes
    -----
    Inputs: none.
    Outputs: collection specs used to resolve EvidenceTables and output paths.
    Determinism: deterministic.
    Dependencies: must match suffixes produced by the paper-side fgsea scripts.
    """
    return [
        CollectionSpec(name="Hallmark", suffix="H", out_slug="H"),
        CollectionSpec(name="GO_BP", suffix="C5_GO_BP", out_slug="C5_GO_BP"),
        CollectionSpec(name="Reactome", suffix="C2_CP_REACTOME", out_slug="C2_CP_REACTOME"),
        CollectionSpec(
            name="KEGG_MEDICUS", suffix="C2_CP_KEGG_MEDICUS", out_slug="C2_CP_KEGG_MEDICUS"
        ),
    ]


def _find_evidence_table(evid_dir: Path, cancer: str, suffix: str) -> Path:
    """
    Resolve and validate an EvidenceTable TSV for a given cancer and collection suffix.

    Parameters
    ----------
    evid_dir
        Directory containing EvidenceTables.
    cancer
        Cancer label (uppercased), used as `<CANCER>` in the expected file name.
    suffix
        Collection suffix used as `<suffix>` in the expected file name.

    Returns
    -------
    Path
        Path to `<CANCER>.<suffix>.evidence_table.tsv`.

    Notes
    -----
    Inputs: filesystem path `<evid_dir>/<CANCER>.<suffix>.evidence_table.tsv`.
    Outputs: none (path resolution + validation only).
    Determinism: deterministic.
    Dependencies: `EvidenceTable.read_tsv` contract validation.
    """
    # expects: <CANCER>.<suffix>.evidence_table.tsv
    p = evid_dir / f"{cancer}.{suffix}.evidence_table.tsv"
    _ensure_file(p, f"evidence_table ({cancer}, {suffix})")
    _validate_evidence_table(p)
    return p


def _sample_card_path(sd: Path, cancer: str, gate_mode: str) -> Path:
    """
    Resolve a Sample Card JSON path for a given cancer and gate mode.

    Parameters
    ----------
    sd
        Benchmark root directory under `paper/source_data/<benchmark_id>/`.
    cancer
        Cancer label (uppercased).
    gate_mode
        Gate mode string used in file naming (e.g., hard, note).

    Returns
    -------
    Path
        Path to `<sd>/sample_cards/<CANCER>.<gate_mode>.sample_card.json`.

    Notes
    -----
    Inputs: filesystem path under `<sd>/sample_cards/`.
    Outputs: none (path resolution only).
    Determinism: deterministic.
    Dependencies: file naming convention in `sample_cards/`.
    """
    p = sd / "sample_cards" / f"{cancer}.{gate_mode}.sample_card.json"
    _ensure_file(p, f"sample_card ({cancer}, gate={gate_mode})")
    return p


# -------------------------
# Runner
# -------------------------
@dataclass(frozen=True)
class Job:
    """
    One job configuration for the collections pipeline.

    Parameters
    ----------
    benchmark_id
        Directory name under `paper/source_data/`.
    collection
        Paper-facing collection label (e.g., Hallmark).
    collection_suffix
        EvidenceTable filename suffix for this collection.
    condition
        Cancer/condition label (e.g., BRCA, HNSC).
    variant
        Variant name: ours, context_swap, or stress.
    gate_mode
        Audit gate mode: hard or note.
    tau
        Stability threshold used by the audit.
    k_claims
        Number of proposed claims before audit.
    seed
        RNG seed forwarded to `run_pipeline`.
    evidence_table
        EvidenceTable TSV for this (condition, collection).
    sample_card
        Sample Card JSON used for proposal/audit (may be generated for context_swap).
    outdir
        Output directory for this job.
    stress_*
        Stress knobs forwarded to the pipeline when `variant == "stress"`.

    Notes
    -----
    Inputs: `evidence_table`, `sample_card`.
    Outputs: written under `outdir/` by `run_pipeline` plus `run_meta.runner.json`.
    Determinism: controlled by `seed` and fixed pipeline settings.
    Dependencies: `llm_pathway_curator.pipeline.run_pipeline`.
    """

    benchmark_id: str
    collection: str
    collection_suffix: str
    condition: str
    variant: str
    gate_mode: str
    tau: float
    k_claims: int
    seed: int
    evidence_table: Path
    sample_card: Path
    outdir: Path
    # stress knobs (only used when variant==stress)
    stress_evidence_dropout_p: float = 0.0
    stress_evidence_dropout_min_keep: int = 1
    stress_contradictory_p: float = 0.0
    stress_contradictory_max_extra: int = 0


def _run_one(job: Job, *, force: bool) -> None:
    """
    Run the core pipeline for one (collection, condition, variant) job.

    Parameters
    ----------
    job
        Fully specified job including inputs, tau/seed, stress knobs, and output dir.
    force
        If True, overwrite existing outputs. If False, skip when `report.jsonl` exists.

    Returns
    -------
    None

    Notes
    -----
    Inputs
      - `job.evidence_table` (EvidenceTable TSV)
      - `job.sample_card` (Sample Card JSON)
    Outputs
      - `job.outdir/report.jsonl` and pipeline artifacts (audit log, modules, etc.)
      - `job.outdir/run_meta.runner.json` (runner-added metadata)
    Determinism
      - `job.seed` is forwarded to `RunConfig(seed=...)`.
    Dependencies
      - `llm_pathway_curator.pipeline.RunConfig`
      - `llm_pathway_curator.pipeline.run_pipeline`
    """
    if (not force) and _report_exists_ok(job.outdir):
        print(f"[run_figS2] SKIP (exists): {job.outdir}")
        return

    job.outdir.mkdir(parents=True, exist_ok=True)

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

    # annotate minimal runner metadata (non-invasive)
    meta = {
        "runner": "paper/scripts/run_figS2_collections_pipeline.py",
        "benchmark_id": job.benchmark_id,
        "collection": job.collection,
        "collection_suffix": job.collection_suffix,
        "condition": job.condition,
        "variant": job.variant,
        "gate_mode": job.gate_mode,
        "tau": float(job.tau),
        "k_claims": int(job.k_claims),
        "seed": int(job.seed),
        "time_unix": int(time.time()),
    }
    _write_json(job.outdir / "run_meta.runner.json", meta)


def main() -> None:
    """
    CLI entry point for the collections sensitivity pipeline (Supp Fig S2).

    This function:
    1) Resolves `paper/source_data/<benchmark_id>/` and validates required inputs.
    2) Selects collections and builds jobs across collections × cancers × variants × gates.
    3) Optionally generates context-swap Sample Cards per job.
    4) Runs each job via `_run_one`, writing outputs under `<out_root>/`.

    Returns
    -------
    None

    Notes
    -----
    Inputs
      - EvidenceTables:
        `<benchmark>/evidence_tables/<CANCER>.<suffix>.evidence_table.tsv`
      - Sample Cards:
        `<benchmark>/sample_cards/<CANCER>.<gate>.sample_card.json`
    Outputs
      - Per-job outputs under:
        `<out_root>/<collection_slug>/<CANCER>/<variant>/gate_<gate>/tau_<tau>/`
    Determinism
      - Controlled by `--seed` (default 42) and library-level fixed settings.
    Dependencies
      - `llm_pathway_curator.pipeline.run_pipeline`
      - `llm_pathway_curator.schema.EvidenceTable` (contract validation)
    """
    ap = argparse.ArgumentParser(
        description="Supp Fig Sx: run tau=0.8 pipelines across gene set collections."
    )
    ap.add_argument("--benchmark-id", default="PANCAN_TP53_v1")
    ap.add_argument("--cancers", default="HNSC,LUAD,LUSC,BRCA,OV,UCEC,SKCM")
    ap.add_argument("--variants", default="ours,context_swap,stress")
    ap.add_argument("--gate-modes", default="hard")
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--k-claims", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--collections",
        default="Hallmark,GO_BP,Reactome,KEGG_MEDICUS",
        help="Comma list: Hallmark,GO_BP,Reactome,KEGG_MEDICUS (default all)",
    )
    ap.add_argument(
        "--out-root",
        default="",
        help="Default: <benchmark>/out_supp_collections_tau0.80/",
    )
    ap.add_argument("--force", action="store_true")

    # stress knobs (same defaults as fig2 run you showed)
    ap.add_argument("--stress-evidence-dropout-p", type=float, default=0.05)
    ap.add_argument("--stress-evidence-dropout-min-keep", type=int, default=1)
    ap.add_argument("--stress-contradictory-p", type=float, default=0.0)
    ap.add_argument("--stress-contradictory-max-extra", type=int, default=0)

    # optional: context review in proposal stage (llm/off)
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
        _die("[run_figS2] --benchmark-id must be non-empty")

    tau = float(args.tau)
    if tau < 0.0 or tau > 1.0:
        _die(f"[run_figS2] --tau must be in [0,1] (got {tau})")

    if int(args.k_claims) < 1:
        _die(f"[run_figS2] --k-claims must be >=1 (got {args.k_claims})")

    cancers = [c.strip().upper() for c in _parse_csv(args.cancers)]
    if not cancers:
        _die("[run_figS2] --cancers must be non-empty")

    variants = [v.strip() for v in _parse_csv(args.variants)]
    allowed_variants = {"ours", "context_swap", "stress"}
    bad = sorted(set(variants) - allowed_variants)
    if bad:
        _die(f"[run_figS2] unsupported variants: {bad} (allowed={sorted(allowed_variants)})")
    if not variants:
        _die("[run_figS2] --variants must be non-empty")

    gate_modes = [g.strip().lower() for g in _parse_csv(args.gate_modes)]
    allowed_gates = {"hard", "note"}
    badg = sorted(set(gate_modes) - allowed_gates)
    if badg:
        _die(f"[run_figS2] unsupported gate modes: {badg} (allowed={sorted(allowed_gates)})")
    if not gate_modes:
        _die("[run_figS2] --gate-modes must be non-empty")

    # resolve benchmark dir
    sd = PAPER / "source_data" / benchmark_id
    if not sd.exists():
        _die(f"[run_figS2] benchmark directory not found: {sd}")
    evid_dir = sd / "evidence_tables"
    if not evid_dir.exists():
        _die(f"[run_figS2] missing evidence_tables dir: {evid_dir}")

    # select collections
    all_specs = {c.name: c for c in _default_collections()}
    want = [x.strip() for x in _parse_csv(args.collections)]
    missing = sorted(set(want) - set(all_specs))
    if missing:
        _die(f"[run_figS2] unknown collections: {missing} (known={sorted(all_specs)})")
    col_specs = [all_specs[n] for n in want]

    # out root
    out_root = (
        Path(str(args.out_root)).resolve()
        if str(args.out_root).strip()
        else (sd / f"out_supp_collections_tau_{tau:.2f}").resolve()
    )

    # context_swap map (within the cancers list)
    swap_map = _make_rotation_map(cancers) if "context_swap" in variants else {}

    # Validate inputs first (fail-fast)
    for c in cancers:
        for spec in col_specs:
            _find_evidence_table(evid_dir, c, spec.suffix)
        for gm in gate_modes:
            _sample_card_path(sd, c, gm)

    # Plan + run
    jobs: list[Job] = []
    for spec in col_specs:
        out_root_c = out_root / spec.out_slug
        for cancer in cancers:
            ev = _find_evidence_table(evid_dir, cancer, spec.suffix)
            for gm in gate_modes:
                base_card_path = _sample_card_path(sd, cancer, gm)

                # ours
                if "ours" in variants:
                    outdir = _outdir_for(
                        out_root=out_root_c,
                        condition=cancer,
                        variant="ours",
                        gate_mode=gm,
                        tau=tau,
                    )
                    jobs.append(
                        Job(
                            benchmark_id=benchmark_id,
                            collection=spec.name,
                            collection_suffix=spec.suffix,
                            condition=cancer,
                            variant="ours",
                            gate_mode=gm,
                            tau=tau,
                            k_claims=int(args.k_claims),
                            seed=int(args.seed),
                            evidence_table=ev,
                            sample_card=base_card_path,
                            outdir=outdir,
                        )
                    )

                # stress (dropout)
                if "stress" in variants:
                    outdir = _outdir_for(
                        out_root=out_root_c,
                        condition=cancer,
                        variant="stress",
                        gate_mode=gm,
                        tau=tau,
                    )
                    jobs.append(
                        Job(
                            benchmark_id=benchmark_id,
                            collection=spec.name,
                            collection_suffix=spec.suffix,
                            condition=cancer,
                            variant="stress",
                            gate_mode=gm,
                            tau=tau,
                            k_claims=int(args.k_claims),
                            seed=int(args.seed),
                            evidence_table=ev,
                            sample_card=base_card_path,
                            outdir=outdir,
                            stress_evidence_dropout_p=float(args.stress_evidence_dropout_p),
                            stress_evidence_dropout_min_keep=int(
                                args.stress_evidence_dropout_min_keep
                            ),
                            stress_contradictory_p=float(args.stress_contradictory_p),
                            stress_contradictory_max_extra=int(args.stress_contradictory_max_extra),
                        )
                    )

                # context_swap (swap sample_card within same collection run)
                if "context_swap" in variants:
                    dst = swap_map.get(cancer, "")
                    if not dst:
                        _die("[run_figS2] internal error: missing swap map entry")

                    src = _read_json(base_card_path)
                    dst_card_path = _sample_card_path(sd, dst, gm)
                    dst_card = _read_json(dst_card_path)

                    # patch card fields (same as fig2 runner intent)
                    src = _set_k_claims(src, int(args.k_claims))
                    src = _set_audit_tau(src, float(tau))
                    src = _set_context_review_mode(src, str(args.context_review_mode))

                    src = _set_card_extra(src, "context_swap_from", cancer)
                    src = _set_card_extra(src, "context_swap_to_condition", dst)

                    # copy key context fields from dst card into src card
                    for k in ("condition", "disease", "tissue", "perturbation", "comparison"):
                        v = dst_card.get(k, None)
                        if isinstance(v, str) and v.strip():
                            src[k] = v
                        elif v not in (None, "", []):
                            src[k] = v

                    outdir = _outdir_for(
                        out_root=out_root_c,
                        condition=cancer,
                        variant="context_swap",
                        gate_mode=gm,
                        tau=tau,
                    )
                    outdir.mkdir(parents=True, exist_ok=True)
                    swap_card = outdir / f"sample_card.context_swap.{dst}.{gm}.tau_{tau:.2f}.json"
                    _write_json(swap_card, src)

                    jobs.append(
                        Job(
                            benchmark_id=benchmark_id,
                            collection=spec.name,
                            collection_suffix=spec.suffix,
                            condition=cancer,
                            variant="context_swap",
                            gate_mode=gm,
                            tau=tau,
                            k_claims=int(args.k_claims),
                            seed=int(args.seed),
                            evidence_table=ev,
                            sample_card=swap_card,
                            outdir=outdir,
                        )
                    )

    print("[run_figS2] plan")
    print(f"  benchmark_id: {benchmark_id}")
    print(f"  tau: {tau:.2f}")
    print(f"  cancers: {','.join(cancers)}")
    print(f"  collections: {','.join([c.name for c in col_specs])}")
    print(f"  variants: {','.join(variants)}")
    print(f"  gate_modes: {','.join(gate_modes)}")
    print(f"  k_claims: {int(args.k_claims)}")
    print(f"  seed: {int(args.seed)}")
    print(f"  out_root: {out_root}")
    if "stress" in variants:
        print("  stress:")
        print(f"    evidence_dropout_p: {float(args.stress_evidence_dropout_p):.3f}")
        print(f"    evidence_dropout_min_keep: {int(args.stress_evidence_dropout_min_keep)}")
        print(f"    contradictory_p: {float(args.stress_contradictory_p):.3f}")
        print(f"    contradictory_max_extra: {int(args.stress_contradictory_max_extra)}")
    if "context_swap" in variants:
        print("  context_swap: rotation within cancers list")

    n = len(jobs)
    print(f"[run_figS2] jobs: {n}")
    for i, job in enumerate(jobs, start=1):
        print(
            f"[run_figS2] ({i}/{n}) {job.collection} {job.condition} {job.variant} "
            f"gate={job.gate_mode} tau={job.tau:.2f}"
        )
        _run_one(job, force=bool(args.force))

    print("[run_figS2] OK")
    print(f"  out_root: {out_root}")


if __name__ == "__main__":
    main()
