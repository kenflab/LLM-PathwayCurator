"""Compile a report and import clearly labeled synthetic proposal examples.

No model requests or biological-performance estimates. Run from any directory
after installing the candidate package, or set PYTHONPATH to its src directory.
"""

import argparse
import json
from pathlib import Path

from llm_pathway_curator import ReviewConfig, review_enrichment


def main():
    example = Path(__file__).resolve().parents[1] / "source_report"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-table", default=str(example / "evidence_table.tsv"))
    parser.add_argument("--sample-card", default=str(example / "sample_card.json"))
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    out = Path(args.outdir).resolve()
    if out.exists():
        parser.error("Use a new demo output directory")
    first = review_enrichment(
        ReviewConfig(args.evidence_table, args.sample_card, str(out / "packet"), modules=True)
    )
    packet = json.loads(Path(first.artifacts["proposal_packet"]).read_text())
    records = packet["source_records"]
    source = records[0]["evidence"]
    refs = [{"term_uid": source["term_uid"], "evidence_sha256": source["evidence_sha256"]}]
    base = {
        "comparison": packet["sample_card"]["comparison"],
        "generator": {"name": "synthetic software demonstration", "model_called": "false"},
        "supporting_genes": [],
    }
    module = max(packet["support_modules"]["modules"], key=lambda m: len(m["member_term_uids"]))
    proposals = [
        {
            **base,
            "claim_id": "DEMO_OBSERVATION",
            "claim_type": "statistical_observation",
            "text": records[0]["source_statement"],
            "evidence_refs": refs,
        },
        {
            **base,
            "claim_id": "DEMO_SUPPORT_SUMMARY",
            "claim_type": "support_summary",
            "text": module["source_summary"],
            "evidence_refs": module["evidence_refs"],
        },
        {
            **base,
            "claim_id": "DEMO_HYPOTHESIS",
            "claim_type": "hypothesis",
            "text": "A hypothesis to investigate is whether shared supporting genes contribute "
            "to repeated enrichment patterns. This would require separate analysis and "
            "does not establish a shared mechanism.",
            "evidence_refs": module["evidence_refs"],
        },
    ]
    path = out / "synthetic_proposals.jsonl"
    path.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in proposals))
    result = review_enrichment(
        ReviewConfig(
            args.evidence_table,
            args.sample_card,
            str(out / "reviewed"),
            modules=True,
            proposals_file=str(path),
        )
    )
    print(result.artifacts["report_html"])


if __name__ == "__main__":
    main()
