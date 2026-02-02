# LLM-PathwayCurator/src/llm_pathway_curator/cli.py

"""
Command-line interface for LLM-PathwayCurator.

This module provides two commands:
- run: execute distill → modules → claims → audit → report
- adapt: convert external enrichment outputs into EvidenceTable TSV

Design goals
------------
- Fail fast with actionable error messages (SystemExit).
- Validate EvidenceTable contract early for clear CLI feedback.
- Keep outputs reproducible via a minimal run config echo.

Notes
-----
This CLI is intentionally small and delegates all core logic to pipeline,
schema, and adapter layers.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Literal

from .adapters.fgsea import convert_fgsea_table_to_evidence_tsv
from .adapters.metascape import MetascapeAdapterConfig, convert_metascape_table_to_evidence_tsv
from .pipeline import RunConfig, run_pipeline
from .schema import EvidenceTable


def _p(path: str | Path) -> Path:
    """
    Normalize a filesystem path for CLI usage.

    Parameters
    ----------
    path : str or pathlib.Path
        Input path.

    Returns
    -------
    pathlib.Path
        Expanded and resolved absolute path.

    Notes
    -----
    This performs:
    - `expanduser()`
    - `resolve()`
    """
    return Path(path).expanduser().resolve()


def _require_file(path: Path, label: str) -> None:
    """
    Require that a path exists and is a regular file.

    Parameters
    ----------
    path : pathlib.Path
        Path to validate.
    label : str
        Human-readable label used in error messages.

    Raises
    ------
    SystemExit
        If the file does not exist or is not a regular file.
    """
    if not path.exists():
        raise SystemExit(f"[ERROR] {label} not found: {path}")
    if not path.is_file():
        raise SystemExit(f"[ERROR] {label} is not a file: {path}")


def _ensure_outdir(outdir: Path, force: bool) -> None:
    """
    Create an output directory and enforce overwrite safety.

    Parameters
    ----------
    outdir : pathlib.Path
        Output directory path.
    force : bool
        If True, allow writing into an existing non-empty directory.

    Raises
    ------
    SystemExit
        If `outdir` exists but is not a directory, or if it is non-empty
        and `force` is False.

    Notes
    -----
    The non-empty check is performed by attempting to read one entry from
    `outdir.iterdir()`.
    """
    if outdir.exists() and not outdir.is_dir():
        raise SystemExit(f"[ERROR] outdir exists but is not a directory: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    # refuse writing into non-empty outdir unless --force
    if not force:
        try:
            next(outdir.iterdir())
            raise SystemExit(
                f"[ERROR] outdir is not empty: {outdir} (use --force to allow overwrite)"
            )
        except StopIteration:
            pass  # empty OK


def _env_int(name: str) -> int | None:
    """
    Parse an integer environment variable.

    Parameters
    ----------
    name : str
        Environment variable name.

    Returns
    -------
    int or None
        Parsed integer value, or None if unset/empty.

    Raises
    ------
    SystemExit
        If the variable is set but cannot be parsed as an integer.
    """
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "":
        return None
    try:
        return int(raw)
    except ValueError as e:
        raise SystemExit(f"[ERROR] invalid env {name}={raw!r} (expected int)") from e


def _resolve_k_claims(cli_value: int | None) -> tuple[int | None, str]:
    """
    Resolve k_claims with explicit precedence and provenance.

    Parameters
    ----------
    cli_value : int or None
        Value provided via CLI `--k-claims`.

    Returns
    -------
    (int or None, str)
        Tuple of:
        - resolved k_claims (or None)
        - source string in {"cli", "env", "default"}

    Notes
    -----
    Precedence:
    1) CLI `--k-claims`
    2) env `LLMPATH_K_CLAIMS`
    3) None (downstream defaults)
    """
    if cli_value is not None:
        return cli_value, "cli"

    env_value = _env_int("LLMPATH_K_CLAIMS")
    if env_value is not None:
        return env_value, "env"

    return None, "default"


def _validate_positive_int(name: str, value: int | None) -> None:
    """
    Validate that an optional integer is positive.

    Parameters
    ----------
    name : str
        Label used in error messages.
    value : int or None
        Value to validate.

    Raises
    ------
    SystemExit
        If `value` is not None and is less than 1.
    """
    if value is None:
        return
    if value < 1:
        raise SystemExit(f"[ERROR] {name} must be >= 1 (got {value})")


def cmd_run(args: argparse.Namespace) -> None:
    """
    CLI entrypoint for `run`.

    This command executes the end-to-end pipeline:
    distill → modules → claims → audit → report.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. Expected fields include:
        evidence_table, sample_card, outdir, force, seed, tau, k_claims,
        run_meta.

    Raises
    ------
    SystemExit
        If inputs are missing, the outdir is unsafe to write, or the
        EvidenceTable TSV violates the contract.

    Notes
    -----
    The EvidenceTable contract is validated before running the pipeline
    to provide clear CLI errors.
    """
    evidence_table = _p(args.evidence_table)
    sample_card = _p(args.sample_card)
    outdir = _p(args.outdir)

    _require_file(evidence_table, "--evidence-table")
    _require_file(sample_card, "--sample-card")
    _ensure_outdir(outdir, force=bool(args.force))

    # contract gate: validate evidence table early (clear CLI error)
    try:
        EvidenceTable.read_tsv(str(evidence_table))
    except Exception as e:
        raise SystemExit(f"[ERROR] invalid --evidence-table (EvidenceTable contract): {e}") from e

    k_claims, k_src = _resolve_k_claims(args.k_claims)
    _validate_positive_int("--k-claims / LLMPATH_K_CLAIMS", k_claims)

    # Single-line run config echo for reproducibility/debugging (Fig2-friendly)
    tau_str = "default" if args.tau is None else str(args.tau)
    seed_str = "default" if args.seed is None else str(args.seed)
    k_str = "default" if k_claims is None else f"{k_claims} ({k_src})"
    run_meta_rel = str(args.run_meta)
    print(f"[INFO] tau={tau_str} seed={seed_str} k_claims={k_str} run_meta={run_meta_rel}")

    cfg = RunConfig(
        evidence_table=str(evidence_table),
        sample_card=str(sample_card),
        outdir=str(outdir),
        force=bool(args.force),
        seed=args.seed,
        run_meta_name=run_meta_rel,
        tau=args.tau,
        k_claims=k_claims,
    )

    res = run_pipeline(cfg)
    print(f"[OK] wrote artifacts to: {res.outdir}")


AdaptFormat = Literal["metascape", "fgsea"]


def cmd_adapt(args: argparse.Namespace) -> None:
    """
    CLI entrypoint for `adapt`.

    Convert external enrichment outputs into an EvidenceTable TSV and
    validate the written TSV against the EvidenceTable contract.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. Expected fields include:
        format, input, output, include_summary, source_name.

    Raises
    ------
    SystemExit
        If inputs are missing, output is invalid, or the generated TSV
        violates the EvidenceTable contract.
    """
    fmt: AdaptFormat = args.format
    in_path = _p(args.input)
    out_path = _p(args.output)

    _require_file(in_path, "--input")
    if out_path.exists() and out_path.is_dir():
        raise SystemExit(f"[ERROR] --output is a directory: {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "metascape":
        cfg = MetascapeAdapterConfig(
            include_summary=bool(args.include_summary),
            source_name=str(args.source_name or "metascape"),
        )
        convert_metascape_table_to_evidence_tsv(str(in_path), str(out_path), config=cfg)
    elif fmt == "fgsea":
        convert_fgsea_table_to_evidence_tsv(str(in_path), str(out_path))
    else:
        raise SystemExit(f"[ERROR] Unknown format: {fmt}")

    # validate EvidenceTable TSV (contract gate)
    try:
        EvidenceTable.read_tsv(str(out_path))
    except Exception as e:
        raise SystemExit(f"[ERROR] Invalid EvidenceTable TSV written: {out_path}\n{e}") from e

    print(f"[OK] wrote EvidenceTable TSV: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    """
    Build the top-level argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with subcommands:
        - run
        - adapt

    Notes
    -----
    The subparser is created with `required=True` to ensure a subcommand
    is always provided.
    """
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
        "--tau",
        type=float,
        default=None,
        help="Audit stability threshold tau (overrides sample_card.audit_tau() if set)",
    )
    p_run.add_argument(
        "--k-claims",
        type=int,
        default=None,
        help=(
            "Number of claims to propose before audit "
            "(CLI overrides env LLMPATH_K_CLAIMS; downstream may fall back to "
            "sample_card.k_claims())"
        ),
    )
    p_run.add_argument(
        "--run-meta",
        default="run_meta.json",
        help="Relative path for run metadata JSON inside outdir (default: run_meta.json)",
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
        "--source-name",
        default=None,
        help="(metascape) override source field (default: metascape)",
    )
    p_adapt.set_defaults(func=cmd_adapt)

    return p


def main(argv: list[str] | None = None) -> None:
    """
    Main entrypoint for the `llm-pathway-curator` CLI.

    Parameters
    ----------
    argv : list of str or None, optional
        Argument vector without the program name. If None, argparse uses
        `sys.argv` implicitly.

    Notes
    -----
    This function delegates to the selected subcommand via `args.func`.
    """
    p = build_parser()
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
