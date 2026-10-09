#!/usr/bin/env python3
"""Inventory archived labels without treating claim-ID reuse as task equivalence.

This is a provenance diagnostic, not a new biological performance evaluation.
It does not overwrite frozen study files or impute missing labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def unique(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    result = {}
    for row in rows:
        value = row[key]
        if not value or value in result:
            raise ValueError(f"Missing or duplicate {key}: {value!r}")
        result[value] = row
    return result


def inspect_labels(labels: list[dict[str, str]], audit: list[dict[str, str]]) -> dict:
    """Do not silently collapse raters, conflicting labels or unmatched records."""
    lab = unique(labels, "claim_id")
    target = unique(audit, "claim_id")
    invalid = {r["human_label"] for r in labels} - {"ACCEPT", "SHOULD_ABSTAIN", "REJECT"}
    if invalid:
        raise ValueError(f"Unknown human labels: {sorted(invalid)}")
    matches = []
    for cid in sorted(lab.keys() & target.keys()):
        a, b = lab[cid], target[cid]
        fields = ["entity", "direction", "gene_symbols_str"]
        matches.append(
            {
                "claim_id": cid,
                "source_record_fields_match": all(a.get(f) == b.get(f) for f in fields),
                "label_source": a.get("_src", ""),
                "rater_id": a.get("rater_id", ""),
                "human_label": a["human_label"],
                "target_status": b["status"],
                "task_equivalence": "NOT_ESTABLISHED_BY_CLAIM_ID",
            }
        )
    return {
        "label_rows": len(labels),
        "target_rows": len(audit),
        "raters_in_file": sorted({r.get("rater_id", "") for r in labels}),
        "label_counts": dict(Counter(r["human_label"] for r in labels)),
        "label_source_counts": dict(sorted(Counter(r.get("_src", "") for r in labels).items())),
        "matched_ids": len(matches),
        "matched_field_mismatches": sum(not r["source_record_fields_match"] for r in matches),
        "target_ids_without_labels": sorted(target.keys() - lab.keys()),
        "label_ids_outside_target": sorted(lab.keys() - target.keys()),
        "matches": matches,
        "inter_rater_agreement": "NOT_ESTIMATED: no paired independent ratings supplied",
        "performance_comparison": (
            "NOT_ESTIMATED: context and presented-text equivalence unverified"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[1] / "source_data"
    )
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)
    root = args.source_root
    inputs = []

    def load(relative):
        path = root / relative
        inputs.append({"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        return read_tsv(path)

    cases = {
        "HNSC": (
            "PANCAN_TP53_v1/labels/labels_hnsc_R1_v1.20260203.tsv",
            "PANCAN_TP53_v1/out_fig2/HNSC/ours/gate_hard/tau_0.80/audit_log.tsv",
        ),
        "BeatAML": (
            "BEATAML_TP53_v1/labels/labels_aml_R1_v1.20260203.tsv",
            "BEATAML_TP53_v1/out_figS4/BEATAML/ours/gate_hard/tau_0.80/audit_log.tsv",
        ),
    }
    result = {
        name: inspect_labels(load(label), load(audit)) for name, (label, audit) in cases.items()
    }
    prefix = "PANCAN_TP53_v1/out_figS3/HNSC/ours/gate_hard/tau_0.80/"
    wnt = "HALLMARK_WNT_BETA_CATENIN_SIGNALING"
    evidence = unique(load(prefix + "evidence.normalized.tsv"), "term_id")[wnt]
    record = unique(load(prefix + "audit_log.tsv"), "entity")[wnt]
    vignette = {
        "term_id": wnt,
        "claim_id": record["claim_id"],
        "stat": float(evidence["stat"]),
        "stat_kind": "NES (fgsea source)",
        "qval": float(evidence["qval"]),
        "source_direction": evidence["direction"],
        "historical_status": record["status"],
        "historical_context_method": record["context_method"],
        "historical_context_reason": record["context_reason"],
        "source_q_eligible_at_0_05": float(evidence["qval"]) <= 0.05,
        "interpretation": (
            "The saved model rationale is not biological ground truth. "
            "The source q-value does not support an adjusted-significance claim at 0.05."
        ),
    }
    output = {
        "schema": "saved-label-provenance/1",
        "scope": "Archived-file inventory and source verification; no new ratings or model calls",
        "cohorts": result,
        "worked_example": vignette,
        "inputs": inputs,
    }
    (args.outdir / "provenance_audit.json").write_text(json.dumps(output, indent=2) + "\n")
    for name, details in result.items():
        print(
            f"{name}: {details['label_rows']} labels; {details['matched_ids']} matching IDs; "
            f"raters={details['raters_in_file']}"
        )
    print(f"WNT: NES={vignette['stat']:.6f}; q={vignette['qval']:.6f}")


if __name__ == "__main__":
    main()
