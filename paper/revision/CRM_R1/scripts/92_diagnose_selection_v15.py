#!/usr/bin/env python3
"""Join recorded audit reasons to the existing frozen join; no new method selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from v15_diagnostic_common import (
    KEY,
    METHODS,
    PAIR,
    boolean,
    finish,
    new_output,
    numeric,
    read_tsv,
    record,
    require,
    sha256,
    write_json,
    write_tsv,
)


def load_claims(path, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    require(
        manifest["schema_version"] == "CRM_R1_P2B_FIGURE2_SOURCE_v14_1_3",
        "Expected frozen V14.1.3 figure source manifest",
    )
    require(
        sha256(path) == manifest["outputs"]["claim_source_private"]["sha256"],
        "Frozen joined claim table SHA mismatch",
    )
    data = read_tsv(path)
    require(
        set(KEY + ["method_id", "selected", "replicated"]).issubset(data),
        "Missing joined-source columns",
    )
    require(not data.duplicated(KEY + ["method_id"]).any(), "Duplicate source key")
    require(set(data.method_id) == set(METHODS), "Unexpected methods")
    for col in ["selected", "replicated", "full_audit_pass", "direction_concordant"]:
        data[col] = boolean(data[col])
    for col in ["discovery_q_value", "validation_q_value", "discovery_stability_score"]:
        data[col] = numeric(data[col], bounded=True)
    for col in ["discovery_direction", "validation_direction"]:
        require(set(data[col]) <= {"UP", "DOWN"}, f"Invalid {col}")
    common = [c for c in data if c not in ["method_id", "selected"] + KEY]
    require(
        data.groupby(KEY)[common].nunique(dropna=False).eq(1).all().all(),
        "Method copies disagree on a claim outcome or input",
    )
    require(data.groupby(KEY).size().eq(4).all(), "Incomplete method copies")
    expected = data.discovery_direction.eq(data.validation_direction)
    require(expected.eq(data.direction_concordant).all(), "Concordance mismatch")
    threshold = manifest["endpoint"]["validation_q_value_threshold"]
    require(
        (expected & data.validation_q_value.le(threshold)).eq(data.replicated).all(),
        "Endpoint mismatch; no rewriting allowed",
    )
    counts = data.groupby(PAIR + ["method_id"]).size()
    require(counts.eq(50).all(), "Expected 50 candidates per method/partition")
    require(data.cohort_id.nunique() == manifest["cohorts"], "Cohort count mismatch")
    require(
        data[PAIR].drop_duplicates().groupby("cohort_id").size().eq(manifest["repeats"]).all(),
        "Split count mismatch",
    )
    selections = data.pivot(index=KEY, columns="method_id", values="selected")
    require(selections.raw_pool.all(), "Raw pool must include all candidates")
    sizes = selections.groupby(PAIR).sum()
    require(sizes.full_audit.gt(0).all(), "Zero K in frozen source")
    for method in ["q_value_matched", "stability_matched"]:
        require(sizes[method].eq(sizes.full_audit).all(), "Matched K mismatch")
    base = data.loc[data.method_id.eq("full_audit")].drop(columns=["method_id", "selected"])
    base = base.merge(selections.reset_index(), on=KEY, validate="one_to_one")
    require(base.full_audit.eq(base.full_audit_pass).all(), "PASS/selection mismatch")
    return data, base, manifest


def load_audits(work, base):
    blocks, inputs = [], []
    for (cohort, split), block in base.groupby(PAIR, sort=True):
        path = Path(work) / "jobs" / cohort / split / "audit" / "audit_log.tsv"
        audit = read_tsv(path)
        required = [
            "entity",
            "status",
            "context_method",
            "context_review_mode",
            "context_evaluated",
            "context_reason",
            "context_confidence",
            "context_status",
            "term_survival_agg",
            "direction",
        ]
        require(set(required).issubset(audit), f"Missing audit fields: {path}")
        require(
            len(audit) == 50 and not audit.entity.duplicated().any(),
            f"Invalid audit census: {cohort}/{split}",
        )
        require(set(audit.entity) == set(block.claim_uid), "Audit/source term mismatch")
        for col in ["context_method", "context_review_mode"]:
            require(audit[col].str.lower().eq("llm").all(), "Non-LLM audit row")
        require(boolean(audit.context_evaluated).all(), "Unevaluated audit row")
        require(audit.context_reason.str.strip().ne("").all(), "Missing LLM context reason")
        audit["context_confidence"] = numeric(audit.context_confidence, bounded=True)
        audit["term_survival_agg"] = numeric(audit.term_survival_agg, bounded=True)
        require(set(audit.status) <= {"PASS", "FAIL", "ABSTAIN"}, "Invalid audit status")
        require(set(audit.context_status) <= {"PASS", "FAIL", "WARN"}, "Invalid context status")
        # Error caches can coexist with otherwise valid-looking output: inspect if supplied.
        caches = sorted(path.parent.glob("context_review_cache*.json"))
        for cache_path in caches:
            cache = json.loads(cache_path.read_text())
            require(isinstance(cache, dict), "Malformed context cache")
            require(
                not any(isinstance(v, dict) and v.get("error") for v in cache.values()),
                f"Technical error in cache: {cache_path}",
            )
            inputs.append(record(cache_path))
        fields = required + [
            c
            for c in ["fail_reason", "abstain_reason", "context_notes", "gene_ids", "claim_id"]
            if c in audit
        ]
        audit = audit[fields].rename(columns={"entity": "claim_uid"})
        audit["cohort_id"], audit["split_id"] = cohort, split
        audit["raw_cache_inspected"] = bool(caches)
        blocks.append(audit)
        inputs.append(record(path))
    result = base.merge(pd.concat(blocks), on=KEY, validate="one_to_one")
    require(result.status.eq("PASS").eq(result.full_audit).all(), "Audit PASS mismatch")
    require(
        result.direction.str.upper().eq(result.discovery_direction).all(),
        "Audit direction mismatch",
    )
    require(
        np.allclose(
            result.term_survival_agg, result.discovery_stability_score, rtol=1e-12, atol=1e-15
        ),
        "Audit stability mismatch",
    )
    result["gate"] = result.status
    for status, reason in [("FAIL", "fail_reason"), ("ABSTAIN", "abstain_reason")]:
        require(reason in result, f"Missing {reason}")
        mask = result.status.eq(status)
        require(result.loc[mask, reason].str.strip().ne("").all(), f"Blank {reason}")
        result.loc[mask, "gate"] = status + ":" + result.loc[mask, reason]
    result["reason_at_160_char_limit"] = result.context_reason.str.len().ge(160)
    return result, inputs


def diagnostic_tables(joined, policy):
    frames, effects = [], []
    for method in ["q_value_matched", "stability_matched"]:
        current = joined.copy()
        current["comparator"] = method
        current["selection_group"] = np.select(
            [current.full_audit & current[method], current.full_audit, current[method]],
            ["both", "full_only", "comparator_only"],
            default="neither",
        )
        current["K"] = current.groupby(PAIR).full_audit.transform("sum")
        current["signed_replication_contribution"] = (
            (current.full_audit.astype(int) - current[method].astype(int))
            * current.replicated.astype(int)
            / current.K
        )
        frames.append(current)
        by_split = current.groupby(PAIR).signed_replication_contribution.sum()
        for cohort, value in by_split.groupby("cohort_id").mean().items():
            effects.append(
                {"cohort_id": cohort, "comparator": method, "full_minus_comparator": value}
            )
    rows = pd.concat(frames, ignore_index=True)
    patterns = policy["lexical_flags"]
    for label, regex in patterns.items():
        rows[label] = rows.context_reason.str.contains(regex, case=False, regex=True)
    strata = ["comparator", "selection_group", "gate"]
    pooled = rows.groupby(strata, as_index=False).agg(
        appearances=("replicated", "size"),
        replicated_appearances=("replicated", "sum"),
        reasons_at_limit=("reason_at_160_char_limit", "sum"),
    )
    pooled["appearance_weighted_replication"] = pooled.replicated_appearances / pooled.appearances
    per_cohort = rows.groupby(["cohort_id"] + strata, as_index=False).agg(
        appearances=("replicated", "size"), replicated_appearances=("replicated", "sum")
    )
    # Zero-pad absent gates/selection groups within every cohort/split before equal averaging.
    split_effect = rows.groupby(PAIR + strata).signed_replication_contribution.sum()
    pair_frame = rows[PAIR].drop_duplicates()
    grid = pair_frame.merge(rows[strata].drop_duplicates(), how="cross")
    split_effect = grid.merge(split_effect.reset_index(), on=PAIR + strata, how="left")
    split_effect["signed_replication_contribution"] = (
        split_effect.signed_replication_contribution.fillna(0)
    )
    cohort_effect = split_effect.groupby(["cohort_id"] + strata, as_index=False)[
        "signed_replication_contribution"
    ].mean()
    summary_effect = cohort_effect.groupby(strata, as_index=False)[
        "signed_replication_contribution"
    ].mean()
    contrasts = pd.DataFrame(effects)
    check = (
        cohort_effect.groupby(["cohort_id", "comparator"])["signed_replication_contribution"]
        .sum()
        .sort_index()
    )
    expected = contrasts.set_index(["cohort_id", "comparator"])[
        "full_minus_comparator"
    ].sort_index()
    require(np.allclose(check, expected, atol=1e-14, rtol=0), "Decomposition does not sum")
    # These flags describe words in stored text; they are not correctness labels.
    lexical = []
    for label in patterns:
        subset = rows.loc[rows[label]]
        block = subset.groupby(strata, as_index=False).agg(
            appearances=("replicated", "size"), replicated_appearances=("replicated", "sum")
        )
        block["lexical_flag"] = label
        lexical.append(block)
    examples = rows.loc[
        rows.comparator.eq("q_value_matched")
        & rows.selection_group.eq("comparator_only")
        & rows.gate.eq("FAIL:context_fail")
    ].copy()
    examples["example_order"] = examples.apply(
        lambda r: hashlib.sha256("|".join(str(r[k]) for k in KEY).encode()).hexdigest(), axis=1
    )
    examples = (
        examples.sort_values("example_order")
        .groupby(["cohort_id", "replicated"], sort=True)
        .head(policy["examples_per_cohort_outcome"])
    )
    examples["review_status"] = "UNREVIEWED_DIAGNOSTIC_EXAMPLE"
    pathway = rows.groupby(["cohort_id", "claim_uid"] + strata, as_index=False).agg(
        appearances=("replicated", "size"), replicated_appearances=("replicated", "sum")
    )
    repetition = joined.groupby(["cohort_id", "claim_uid"], as_index=False).agg(
        splits=("split_id", "nunique"),
        distinct_context_verdicts=("context_status", "nunique"),
        distinct_recorded_reasons=("context_reason", "nunique"),
        context_fail_splits=("context_status", lambda s: int(s.eq("FAIL").sum())),
        replicated_splits=("replicated", "sum"),
        reasons_at_limit=("reason_at_160_char_limit", "sum"),
    )
    return {
        "joined_reasons.private.tsv": rows,
        "selection_gate_appearance_summary.tsv": pooled,
        "selection_gate_by_cohort.tsv": per_cohort,
        "gate_contribution_by_cohort.tsv": cohort_effect,
        "gate_contribution_summary.tsv": summary_effect,
        "cohort_contrasts_reproduced.tsv": contrasts,
        "lexical_flag_summary.tsv": pd.concat(lexical, ignore_index=True),
        "rationale_examples.private.tsv": examples,
        "pathway_selection_by_cohort.tsv": pathway,
        "context_repeat_by_cohort_pathway.tsv": repetition,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--claim-source", type=Path, required=True)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--audit-work", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    policy = json.loads(args.policy.read_text())
    require(policy["analysis_role"] == "post_hoc_diagnostic", "Policy role mismatch")
    data, base, manifest = load_claims(args.claim_source, args.source_manifest)
    joined, inputs = load_audits(args.audit_work, base)
    tables = diagnostic_tables(joined, policy)
    method_split = data.loc[data.selected].groupby(PAIR + ["method_id"]).replicated.mean()
    method_cohort = method_split.groupby(["cohort_id", "method_id"]).mean()
    method_summary = method_cohort.groupby("method_id").mean().to_dict()
    contrasts = (
        tables["cohort_contrasts_reproduced.tsv"]
        .groupby("comparator")["full_minus_comparator"]
        .mean()
        .to_dict()
    )
    summary = {
        "cohorts": int(joined.cohort_id.nunique()),
        "claims": len(joined),
        "partitions": len(joined[PAIR].drop_duplicates()),
        "method_means": method_summary,
        "contrasts": contrasts,
        "reason_at_160_char_limit": int(joined.reason_at_160_char_limit.sum()),
        "raw_cache_partitions_inspected": int(
            joined.loc[joined.raw_cache_inspected, PAIR].drop_duplicates().shape[0]
        ),
        "causal_gate_effect_estimated": False,
        "lexical_flags_are_truth_labels": False,
        "uncertainty": "Use the original frozen cohort-bootstrap CIs; no new tests.",
        "schema": manifest["schema_version"],
    }
    inputs += [
        record(x)
        for x in [
            args.claim_source,
            args.source_manifest,
            args.policy,
            Path(__file__),
            Path(__file__).with_name("v15_diagnostic_common.py"),
        ]
    ]
    with new_output(args.output) as output:
        for name, table in tables.items():
            write_tsv(output / name, table)
        write_json(output / "diagnostic_summary.json", summary)
        finish(output, inputs, summary)
    print("[PASS] Read-only selection diagnosis written:", args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
