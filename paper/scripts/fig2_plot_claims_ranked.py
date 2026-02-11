#!/usr/bin/env python3
# paper/scripts/fig2_plot_claims_ranked.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if src.exists():
        sys.path.insert(0, str(src))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plot ranked claims (packed circles or bars). "
            "Thin wrapper delegating to src/llm_pathway_curator/viz_ranked.py"
        )
    )

    p.add_argument("--mode", type=str, default="packed", choices=["packed", "bars"])
    p.add_argument(
        "--in-tsv", type=str, default="", help="claims_ranked.tsv (recommended) or audit_log.tsv"
    )
    p.add_argument("--run-dir", type=str, default="", help="Auto-detect TSV under this directory")
    p.add_argument("--out-png", type=str, required=True)

    p.add_argument(
        "--decision",
        type=str,
        default="PASS",
        help="Filter decision/status (e.g., PASS or PASS,ABSTAIN)",
    )
    p.add_argument("--drop-hallmark-prefix", action="store_true")
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--annotate", action="store_true")

    # packed
    p.add_argument("--top-modules", type=int, default=12)
    p.add_argument("--top-terms-per-module", type=int, default=8)
    p.add_argument("--size-gamma", type=float, default=3.5)
    p.add_argument("--size-min-norm", type=float, default=0.03)
    p.add_argument("--min-term-label-r", type=float, default=0.09)
    p.add_argument("--dark", action="store_true")
    p.add_argument("--module-cmap", type=str, default="tab20")
    p.add_argument(
        "--term-color-mode", type=str, default="module", choices=["score", "module", "direction"]
    )
    p.add_argument("--term-cmap", type=str, default="coolwarm")
    p.add_argument("--term-vmin", type=float, default=float("nan"))
    p.add_argument("--term-vmax", type=float, default=float("nan"))
    p.add_argument("--term-font-scale", type=float, default=60.0)
    p.add_argument("--term-font-min", type=float, default=7.0)
    p.add_argument("--term-font-max", type=float, default=18.0)
    p.add_argument("--module-font-scale", type=float, default=34.0)
    p.add_argument("--module-font-min", type=float, default=10.0)
    p.add_argument("--module-font-max", type=float, default=20.0)

    # bars
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--group-by-module", action="store_true")
    p.add_argument("--left-strip", action="store_true")
    p.add_argument("--strip-labels", action="store_true")
    p.add_argument("--xlabel", type=str, default="")
    p.add_argument(
        "--bar-color-mode", type=str, default="score", choices=["score", "module", "direction"]
    )
    p.add_argument("--bar-cmap", type=str, default="coolwarm")
    p.add_argument("--bar-vmin", type=float, default=float("nan"))
    p.add_argument("--bar-vmax", type=float, default=float("nan"))

    return p


def main(argv: list[str] | None = None) -> int:
    _ensure_src_on_path()

    from llm_pathway_curator.viz_ranked import PlotRankedConfig, plot_ranked  # noqa: PLC0415

    a = build_parser().parse_args(argv)

    cfg = PlotRankedConfig(
        mode=str(a.mode),
        in_tsv=str(a.in_tsv),
        run_dir=str(a.run_dir),
        out_png=str(a.out_png),
        decision=str(a.decision),
        drop_hallmark_prefix=bool(a.drop_hallmark_prefix),
        dpi=int(a.dpi),
        annotate=bool(a.annotate),
        top_modules=int(a.top_modules),
        top_terms_per_module=int(a.top_terms_per_module),
        size_gamma=float(a.size_gamma),
        size_min_norm=float(a.size_min_norm),
        min_term_label_r=float(a.min_term_label_r),
        dark=bool(a.dark),
        module_cmap=str(a.module_cmap),
        term_color_mode=str(a.term_color_mode),
        term_cmap=str(a.term_cmap),
        term_vmin=float(a.term_vmin),
        term_vmax=float(a.term_vmax),
        term_font_scale=float(a.term_font_scale),
        term_font_min=float(a.term_font_min),
        term_font_max=float(a.term_font_max),
        module_font_scale=float(a.module_font_scale),
        module_font_min=float(a.module_font_min),
        module_font_max=float(a.module_font_max),
        top_n=int(a.top_n),
        group_by_module=bool(a.group_by_module),
        left_strip=bool(a.left_strip),
        strip_labels=bool(a.strip_labels),
        xlabel=str(a.xlabel),
        bar_color_mode=str(a.bar_color_mode),
        bar_cmap=str(a.bar_cmap),
        bar_vmin=float(a.bar_vmin),
        bar_vmax=float(a.bar_vmax),
    )

    out = plot_ranked(cfg)
    print(f"[OK] wrote plot: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
