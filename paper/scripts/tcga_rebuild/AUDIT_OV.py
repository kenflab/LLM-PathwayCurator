#!/usr/bin/env python3
"""Separate MC3 call status, analysis eligibility and current GDC discordance."""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--inputs", required=True, type=Path)
    args = ap.parse_args()
    root = args.run_dir
    bundle = args.inputs
    ledger = pd.read_csv(root / "sample_ledger.tsv", sep="\t").fillna("")
    ov = ledger[ledger.cancer.eq("OV")].copy()
    previous_path = bundle / "superseded_reference/previous_sample_ledger.tsv"
    previous = pd.read_csv(previous_path, sep="\t").fillna("") if previous_path.is_file() else None
    if previous is not None:
        other = ledger[ledger.cancer.ne("OV")].set_index("sample")
        old_other = previous[previous.cancer.ne("OV")].set_index("sample")
        if not other.group.equals(old_other.group):
            raise RuntimeError("Unexpected group change outside OV")
    stages = []
    states = ov.TP53_call_status_before_RNA_QC
    stages.append(
        {
            "stage": "MC3 genotype eligibility before RNA QC and GDC guard",
            "MUT": int(states.eq("MUT").sum()),
            "MC3_call_negative": int(states.eq("MC3_call_negative").sum()),
            "excluded_unknown": int(states.eq("unknown").sum()),
        }
    )
    for label, eligible in [
        ("After RNA QC, before GDC guard", ov.eligible_before_GDC_guard),
        ("Final: known GDC-positive/MC3-negative cases excluded", ov.group.ne("TP53_unknown")),
    ]:
        stages.append(
            {
                "stage": label,
                "MUT": int((eligible & ov.TP53_accepted_MC3_call).sum()),
                "MC3_call_negative": int((eligible & ~ov.TP53_accepted_MC3_call).sum()),
                "excluded_unknown": int((~eligible).sum()),
            }
        )
    pd.DataFrame(stages).to_csv(root / "OV_denominators.tsv", sep="\t", index=False)
    if previous is not None:
        changed = ov.merge(
            previous[previous.cancer.eq("OV")][["sample", "group", "exclusion_reasons"]],
            on="sample",
            suffixes=("_corrected", "_previous"),
            validate="one_to_one",
        )
        changed.to_csv(root / "OV_sample_changes.tsv", sep="\t", index=False)
    hits = json.loads((bundle / "public_inputs/gdc_OV_TP53_occurrences.json").read_text())["data"][
        "hits"
    ]
    discordant = set(ov.loc[ov.MC3_GDC_discordant, "sample"].str[:12])
    evidence = []
    for hit in hits:
        case = hit["case"]["submitter_id"]
        if case not in discordant:
            continue
        terms = sorted(
            {
                t["transcript"]["consequence_type"]
                for t in hit["ssm"].get("consequence", [])
                if t.get("transcript", {}).get("gene", {}).get("gene_id") == "ENSG00000141510"
            }
        )
        evidence.append(
            {
                "case": case,
                "ssm_occurrence_id": hit["id"],
                "GRCh38_genomic_change": hit["ssm"].get("genomic_dna_change", ""),
                "TP53_consequences": ";".join(terms),
                "excluded_after_RNA_QC": bool(
                    ov.loc[ov["sample"].str[:12].eq(case), "eligible_before_GDC_guard"].any()
                ),
                "match_scope": "case only; no genotype relabeling",
            }
        )
    pd.DataFrame(evidence).to_csv(root / "OV_GDC_discordance_evidence.tsv", sep="\t", index=False)
    result = {
        "other_six_cohort_groups_unchanged": True if previous is not None else None,
        "OV_stages": stages,
        "OV_known_GDC_positive_MC3_negative_excluded_after_QC": int(
            (ov.MC3_GDC_discordant & ov.eligible_before_GDC_guard).sum()
        ),
        "legacy_OV_3_MUT_1_comparator": (
            "superseded over-exclusion result, not TCGA-OV mutation prevalence"
        ),
        "WT_confirmed": False,
        "GDC_snapshot_used_for_frequency_denominator": False,
    }
    (root / "OV_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
