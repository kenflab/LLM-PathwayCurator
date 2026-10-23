"""Plot saved source reports without reranking pathways or accepting prose."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D

CHECK_STATES = ("VIOLATION_DETECTED", "REVIEW_FLAGGED", "NO_LIMITED_FLAG")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source_plot(run_dir: Path) -> tuple[list[dict], dict]:
    """Verify the report receipt and retain source values and draft counts separately."""
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    if meta.get("workflow") != "source" or meta.get("status") != "COMPLETE":
        raise ValueError("plot-source requires a completed source-workflow report")
    report = run_dir / "report.jsonl"
    expected = meta.get("artifacts", {}).get("report_jsonl", {}).get("sha256")
    if not expected or _sha(report) != expected:
        raise ValueError("report.jsonl does not match its run_meta.json receipt")
    rows, seen = [], set()
    for line in report.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("schema_version") != "pathway-source-report/1":
            raise ValueError("Unsupported source-report schema")
        ev = record["evidence"]
        uid = ev["term_uid"]
        if not isinstance(uid, str) or not uid.strip() or uid in seen:
            raise ValueError("Source identities must be nonblank and unique")
        seen.add(uid)
        for key in ("stat", "qval"):
            value = ev[key]
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Invalid {key} for {uid}")
        q = ev["qval"]
        if q is not None and not 0 <= q <= 1:
            raise ValueError(f"Adjusted value outside [0, 1] for {uid}")
        if not isinstance(record["selected"], bool):
            raise ValueError("selected must be a Boolean")
        counts = Counter(r["limited_checks_status"] for r in record["submitted_reviews"])
        if set(counts) - set(CHECK_STATES):
            raise ValueError("Unknown draft-check status")
        rows.append(
            {
                "term_uid": uid,
                "term_name": ev["term_name"],
                "stat_kind": ev["stat_kind"],
                "stat": ev["stat"],
                "q_kind": ev.get("q_kind", "adjusted value"),
                "qval": q,
                "selected": record["selected"],
                "reason_code": record["reason_code"],
                "evidence_sha256": ev["evidence_sha256"],
                "draft_count": sum(counts.values()),
                **{state: counts[state] for state in CHECK_STATES},
            }
        )
    if not rows or len(rows) != meta["candidate_count"]:
        raise ValueError("Source candidate count does not match run metadata")
    if sum(r["selected"] for r in rows) != meta["selected_count"]:
        raise ValueError("Source selection count does not match run metadata")
    cutoff = meta["config"]["q_threshold"]
    if not isinstance(cutoff, (int, float)) or not math.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("Invalid recorded adjusted-value cutoff")
    return sorted(rows, key=lambda r: (r["qval"] is None, r["qval"] or 0, r["term_uid"])), meta


def plot_source(run_dir: str | Path, outdir: str | Path, *, top_n: int = 50) -> Path:
    """Render PNG/PDF/SVG, exact plotted rows and a receipt into a new directory.

    Rows are ordered by adjusted value, with missing values last. ``top_n=0``
    shows all candidates. No selection, text check or model is rerun. The draft
    count column refers only to ``submitted_reviews`` in report.jsonl; structured
    proposals and support modules remain in the linked HTML report.
    """
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 0:
        raise ValueError("top_n must be a nonnegative integer (0 means all)")
    run_dir, outdir = Path(run_dir), Path(outdir)
    if outdir.exists():
        raise ValueError("Use a new plot output directory")
    rows, meta = load_source_plot(run_dir)
    shown = rows[:top_n] if top_n else rows
    stat_kind, q_kind = shown[0]["stat_kind"], shown[0]["q_kind"]
    mixed_stats = len({r["stat_kind"] for r in rows}) > 1
    if len({r["q_kind"] for r in rows}) > 1:
        q_kind = "adjusted value"
    stats = np.array([r["stat"] if r["stat"] is not None else np.nan for r in shown])
    finite = stats[np.isfinite(stats)]
    signed = str(stat_kind).upper() in {"NES", "ES"}
    if signed:
        bound = max(float(np.max(np.abs(finite))) if len(finite) else 0, 1e-9)
        norm, cmap = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), plt.get_cmap("RdBu_r")
    else:
        lo, hi = (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0)
        norm, cmap = Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1), plt.get_cmap("viridis")
    with plt.rc_context({"font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, (ax, notes) = plt.subplots(
            1,
            2,
            figsize=(13, max(4.6, 0.36 * len(shown) + 2.1)),
            gridspec_kw={"width_ratios": [1.4, 1]},
            layout="constrained",
        )
        for i, r in enumerate(shown):
            q, stat = r["qval"], r["stat"]
            color = "#aaaaaa" if stat is None or mixed_stats else cmap(norm(stat))
            if q is None:
                ax.plot(0, i, marker="x", color="#444444", clip_on=False)
            else:
                ax.barh(
                    i,
                    -math.log10(max(q, 1e-300)),
                    color=color,
                    hatch=None if r["selected"] else "///",
                    edgecolor="#555555",
                    linewidth=0.4,
                )
                if q == 1:
                    ax.plot(0, i, marker="|", color=color, clip_on=False)
            value = "NA" if stat is None else f"{stat:.3g}"
            if mixed_stats:
                value = f"{r['stat_kind']} {value}"
            qtext = "NA" if q is None else f"{q:.3g}"
            draft = (
                "no draft" if not r["draft_count"] else "/".join(str(r[s]) for s in CHECK_STATES)
            )
            notes.text(0, i, value, va="center")
            notes.text(0.32, i, qtext, va="center")
            notes.text(0.55, i, "yes" if r["selected"] else "no", va="center")
            notes.text(0.78, i, draft, va="center")
        ax.set_yticks(
            range(len(shown)),
            [
                textwrap.fill(str(r["term_name"]).removeprefix("HALLMARK_").replace("_", " "), 32)
                for r in shown
            ],
        )
        ax.set_ylim(len(shown) - 0.5, -0.5)
        notes.set_ylim(ax.get_ylim())
        ax.set_xlim(left=0)
        cutoff = meta["config"]["q_threshold"]
        if cutoff > 0:
            ax.axvline(-math.log10(cutoff), color="#555555", linestyle=":", linewidth=1)
        ax.set_xlabel(f"−log10(adjusted P value: {q_kind})")
        ax.spines[["top", "right"]].set_visible(False)
        notes.set_xlim(0, 1)
        notes.axis("off")
        for x, label in [
            (0, "Statistic" if mixed_stats else stat_kind),
            (0.32, "Adj. value"),
            (0.55, "Selected"),
            (0.78, "Drafts V/R/N"),
        ]:
            notes.text(x, 1.02, label, transform=notes.transAxes, fontweight="bold")
        fig.suptitle(
            f"Source evidence: {len(shown)} of {len(rows)} pathways shown", fontweight="bold"
        )
        if len(finite) and not mixed_stats:
            fig.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax,
                fraction=0.025,
                pad=0.02,
                label=stat_kind,
            )
        handles = [Line2D([], [], color="#555555", linestyle=":", label=f"Cutoff {cutoff:g}")]
        ax.legend(handles=handles, loc="upper right", frameon=False)
        fig.supxlabel(
            "Hatching: not selected by the saved source rule; ×: adjusted value missing.\n"
            "Draft counts V/R/N: violation / review flag only / no limited flag. "
            "No flag is not approval.\n"
            "Comparison, evidence links, modules and structured proposals: "
            "see the source HTML report."
            + (
                "\nZero adjusted values are drawn at 300; exact values remain in the table."
                if any(r["qval"] == 0 for r in shown)
                else ""
            ),
            fontsize=8,
        )
        outdir.mkdir(parents=True)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(outdir / f"source_plot.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)
    with (outdir / "source_plot.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(shown[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(shown)
    receipt = {
        "schema": "source-plot/1",
        "run_id": meta["run_id"],
        "ruleset": meta["ruleset"],
        "candidate_count": len(rows),
        "shown_count": len(shown),
        "top_n": top_n,
        "selected_count": meta["selected_count"],
        "q_threshold": cutoff,
        "inputs_sha256": {name: _sha(run_dir / name) for name in ("report.jsonl", "run_meta.json")},
        "source_implementation_sha256": meta["implementation_sha256"],
        "plot_implementation_sha256": _sha(Path(__file__)),
        "outputs_sha256": {p.name: _sha(p) for p in sorted(outdir.iterdir())},
        "model_calls": 0,
        "selection_recomputed": False,
        "prose_approved": False,
        "scope": "Display of saved source facts and per-term flat-draft counts, not accuracy.",
    }
    (outdir / "plot_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return outdir / "source_plot.png"
