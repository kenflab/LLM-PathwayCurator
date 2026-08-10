#!/usr/bin/env python3
"""Evaluate frozen P5 audit outputs against GO and Reactome hierarchies once."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BENCHMARK_ID = "PANCAN_TP53_v1_HNSC_R1_P5"
COLLECTIONS = {
    "C5_GO_BP": "GO Biological Process",
    "C2_CP_REACTOME": "Reactome",
}
PAIR_COLUMNS = [
    "collection",
    "relation_scope",
    "parent_claim_id",
    "child_claim_id",
    "parent_id",
    "child_id",
    "parent_name",
    "child_name",
    "parent_depth",
    "child_depth",
    "depth_gap",
    "parent_status",
    "child_status",
    "status_pattern",
    "parent_direction",
    "child_direction",
    "directional_contradiction",
    "parent_gene_n",
    "child_gene_n",
    "intersection_n",
    "leading_edge_jaccard",
    "child_covered_by_parent",
    "parent_covered_by_child",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[4]


def normalize_label(value: str, prefix: str) -> str:
    text = str(value).strip().upper()
    if text.startswith(prefix):
        text = text[len(prefix) :]
    return re.sub(r"[^A-Z0-9]+", "_", text).strip("_")


def parse_gene_ids(value: Any) -> frozenset[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return frozenset()
    return frozenset(re.findall(r"\d+", str(value)))


def parse_go_obo(path: Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    names: dict[str, str] = {}
    parents: dict[str, set[str]] = defaultdict(set)
    current: dict[str, Any] | None = None

    def commit(term: dict[str, Any] | None) -> None:
        if not term or term.get("obsolete") or term.get("namespace") != "biological_process":
            return
        term_id = term.get("id")
        name = term.get("name")
        if not term_id or not name:
            return
        names[term_id] = name
        parents[term_id].update(term.get("parents", set()))

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line == "[Term]":
                commit(current)
                current = {"parents": set(), "obsolete": False}
                continue
            if line.startswith("["):
                commit(current)
                current = None
                continue
            if current is None:
                continue
            if line.startswith("id: "):
                current["id"] = line[4:].strip()
            elif line.startswith("name: "):
                current["name"] = line[6:].strip()
            elif line.startswith("namespace: "):
                current["namespace"] = line[11:].strip()
            elif line.startswith("is_obsolete: true"):
                current["obsolete"] = True
            elif line.startswith("is_a: "):
                current["parents"].add(line.split()[1])
            elif line.startswith("relationship: part_of "):
                current["parents"].add(line.split()[2])
            # All regulates/has_part relationships are intentionally ignored.
    commit(current)
    for child in list(parents):
        parents[child] = {parent for parent in parents[child] if parent in names}
    return names, dict(parents)


def parse_reactome(
    pathways_path: Path, relations_path: Path
) -> tuple[dict[str, str], dict[str, set[str]]]:
    names: dict[str, str] = {}
    with pathways_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 3 and fields[2] == "Homo sapiens":
                names[fields[0]] = fields[1]
    parents: dict[str, set[str]] = defaultdict(set)
    with relations_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 2 and fields[0] in names and fields[1] in names:
                parents[fields[1]].add(fields[0])
    return names, dict(parents)


def minimum_depths(names: dict[str, str], parents: dict[str, set[str]]) -> dict[str, int]:
    children: dict[str, set[str]] = defaultdict(set)
    indegree = {node: 0 for node in names}
    for child, node_parents in parents.items():
        for parent in node_parents:
            if parent in names and child in names:
                children[parent].add(child)
                indegree[child] += 1
    roots = sorted(node for node, degree in indegree.items() if degree == 0)
    depth = {root: 0 for root in roots}
    queue = list(roots)
    cursor = 0
    while cursor < len(queue):
        parent = queue[cursor]
        cursor += 1
        for child in sorted(children.get(parent, set())):
            proposed = depth[parent] + 1
            if child not in depth or proposed < depth[child]:
                depth[child] = proposed
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    require(
        len(depth) == len(names),
        "Ontology hierarchy is cyclic or disconnected from roots",
    )
    return depth


def ancestor_sets(names: dict[str, str], parents: dict[str, set[str]]) -> dict[str, set[str]]:
    memo: dict[str, set[str]] = {}
    visiting: set[str] = set()

    def visit(node: str) -> set[str]:
        if node in memo:
            return memo[node]
        require(node not in visiting, f"Cycle detected at {node}")
        visiting.add(node)
        result: set[str] = set()
        for parent in parents.get(node, set()):
            if parent in names:
                result.add(parent)
                result.update(visit(parent))
        visiting.remove(node)
        memo[node] = result
        return result

    for node in names:
        visit(node)
    return memo


def unique_label_index(names: dict[str, str]) -> dict[str, str]:
    candidates: dict[str, list[str]] = defaultdict(list)
    for term_id, name in names.items():
        candidates[normalize_label(name, "")].append(term_id)
    return {label: values[0] for label, values in candidates.items() if len(values) == 1}


def map_audit_terms(
    audit: pd.DataFrame,
    *,
    collection: str,
    names: dict[str, str],
    depths: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefix = "GOBP_" if collection == "C5_GO_BP" else "REACTOME_"
    index = unique_label_index(names)
    rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    for record in audit.to_dict("records"):
        normalized = normalize_label(record["entity"], prefix)
        ontology_id = index.get(normalized)
        qc_rows.append(
            {
                "collection": collection,
                "claim_id": record["claim_id"],
                "entity": record["entity"],
                "normalized_label": normalized,
                "ontology_id": ontology_id,
                "mapping_status": "MAPPED_UNIQUE" if ontology_id else "UNMAPPED_OR_AMBIGUOUS",
            }
        )
        if ontology_id is None:
            continue
        genes = parse_gene_ids(record.get("gene_ids"))
        rows.append(
            {
                "collection": collection,
                "claim_id": record["claim_id"],
                "entity": record["entity"],
                "ontology_id": ontology_id,
                "ontology_name": names[ontology_id],
                "depth": depths[ontology_id],
                "status": str(record["status"]).upper(),
                "direction": str(record["direction"]).lower(),
                "gene_ids": genes,
                "gene_n": len(genes),
            }
        )
    mapped = pd.DataFrame(rows)
    require(not mapped.empty, f"No {collection} audit terms mapped to ontology")
    require(
        mapped["ontology_id"].is_unique,
        f"{collection}: multiple claims map to one term",
    )
    return mapped, pd.DataFrame(qc_rows)


def pair_metrics(parent: pd.Series, child: pd.Series, scope: str) -> dict[str, Any]:
    parent_genes = parent["gene_ids"]
    child_genes = child["gene_ids"]
    intersection = parent_genes & child_genes
    union = parent_genes | child_genes
    return {
        "collection": parent["collection"],
        "relation_scope": scope,
        "parent_claim_id": parent["claim_id"],
        "child_claim_id": child["claim_id"],
        "parent_id": parent["ontology_id"],
        "child_id": child["ontology_id"],
        "parent_name": parent["ontology_name"],
        "child_name": child["ontology_name"],
        "parent_depth": int(parent["depth"]),
        "child_depth": int(child["depth"]),
        "depth_gap": int(child["depth"] - parent["depth"]),
        "parent_status": parent["status"],
        "child_status": child["status"],
        "status_pattern": f"{child['status']}->{parent['status']}",
        "parent_direction": parent["direction"],
        "child_direction": child["direction"],
        "directional_contradiction": int(parent["direction"] != child["direction"]),
        "parent_gene_n": int(parent["gene_n"]),
        "child_gene_n": int(child["gene_n"]),
        "intersection_n": len(intersection),
        "leading_edge_jaccard": len(intersection) / len(union) if union else np.nan,
        "child_covered_by_parent": (
            len(intersection) / len(child_genes) if child_genes else np.nan
        ),
        "parent_covered_by_child": (
            len(intersection) / len(parent_genes) if parent_genes else np.nan
        ),
    }


def build_pairs(
    mapped: pd.DataFrame,
    parents: dict[str, set[str]],
    ancestors: dict[str, set[str]],
) -> pd.DataFrame:
    by_id = {row["ontology_id"]: row for _, row in mapped.iterrows()}
    rows: list[dict[str, Any]] = []
    for child_id, child in by_id.items():
        for parent_id in sorted(parents.get(child_id, set())):
            if parent_id in by_id:
                rows.append(pair_metrics(by_id[parent_id], child, "direct_parent_child"))
        for parent_id in sorted(ancestors.get(child_id, set())):
            if parent_id in by_id:
                rows.append(pair_metrics(by_id[parent_id], child, "safe_ancestor_descendant"))
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def matched_nonedge_reference(
    mapped: pd.DataFrame,
    observed: pd.DataFrame,
    ancestors: dict[str, set[str]],
    *,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if observed.empty:
        return pd.DataFrame(), {
            "null_draws": draws,
            "contradiction_null_mean": np.nan,
            "child_coverage_null_mean": np.nan,
            "contradiction_p_one_sided_lower": np.nan,
            "child_coverage_p_one_sided_greater": np.nan,
        }
    terms = [row for _, row in mapped.iterrows()]
    candidates: list[dict[str, Any]] = []
    for parent in terms:
        for child in terms:
            parent_id = parent["ontology_id"]
            child_id = child["ontology_id"]
            if parent_id == child_id or parent["depth"] >= child["depth"]:
                continue
            if parent_id in ancestors.get(child_id, set()):
                continue
            if child_id in ancestors.get(parent_id, set()):
                continue
            parent_genes = parent["gene_ids"]
            child_genes = child["gene_ids"]
            intersection = parent_genes & child_genes
            candidates.append(
                {
                    "parent_depth": int(parent["depth"]),
                    "child_depth": int(child["depth"]),
                    "parent_log_gene_bin": round(math.log2(parent["gene_n"] + 1)),
                    "child_log_gene_bin": round(math.log2(child["gene_n"] + 1)),
                    "directional_contradiction": int(parent["direction"] != child["direction"]),
                    "child_covered_by_parent": (
                        len(intersection) / len(child_genes) if child_genes else np.nan
                    ),
                }
            )
    require(candidates, "No eligible nonedge pairs for matched reference")

    candidate_table = pd.DataFrame(candidates)
    matched: list[pd.DataFrame] = []
    for _, pair in observed.iterrows():
        distance = (
            (candidate_table["parent_depth"] - pair["parent_depth"]).abs()
            + (candidate_table["child_depth"] - pair["child_depth"]).abs()
            + (
                candidate_table["parent_log_gene_bin"] - round(math.log2(pair["parent_gene_n"] + 1))
            ).abs()
            + (
                candidate_table["child_log_gene_bin"] - round(math.log2(pair["child_gene_n"] + 1))
            ).abs()
        )
        matched.append(candidate_table.loc[distance.eq(distance.min())].reset_index(drop=True))

    rng = np.random.default_rng(seed)
    contradiction_samples = np.empty((draws, len(matched)), dtype=float)
    coverage_samples = np.empty((draws, len(matched)), dtype=float)
    for column, table in enumerate(matched):
        indices = rng.integers(0, len(table), size=draws)
        contradiction_samples[:, column] = table["directional_contradiction"].to_numpy()[indices]
        coverage_samples[:, column] = table["child_covered_by_parent"].to_numpy()[indices]
    null = pd.DataFrame(
        {
            "draw": np.arange(1, draws + 1),
            "contradiction_fraction": np.mean(contradiction_samples, axis=1),
            "median_child_coverage": np.nanmedian(coverage_samples, axis=1),
        }
    )
    observed_contradiction = observed["directional_contradiction"].mean()
    observed_coverage = observed["child_covered_by_parent"].median()
    summary = {
        "null_draws": draws,
        "contradiction_null_mean": float(null["contradiction_fraction"].mean()),
        "child_coverage_null_mean": float(null["median_child_coverage"].mean()),
        "contradiction_p_one_sided_lower": float(
            (1 + null["contradiction_fraction"].le(observed_contradiction).sum()) / (draws + 1)
        ),
        "child_coverage_p_one_sided_greater": float(
            (1 + null["median_child_coverage"].ge(observed_coverage).sum()) / (draws + 1)
        ),
    }
    return null, summary


def summarize_pairs(
    pairs: pd.DataFrame,
    mapped_by_collection: dict[str, pd.DataFrame],
    ancestors_by_collection: dict[str, dict[str, set[str]]],
    *,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    null_rows: list[pd.DataFrame] = []
    scopes = ("direct_parent_child", "safe_ancestor_descendant")
    for collection in COLLECTIONS:
        for scope_index, scope in enumerate(scopes):
            observed = pairs.loc[
                pairs["collection"].eq(collection) & pairs["relation_scope"].eq(scope)
            ].copy()
            null, null_summary = matched_nonedge_reference(
                mapped_by_collection[collection],
                observed,
                ancestors_by_collection[collection],
                draws=draws,
                seed=seed + 100 * list(COLLECTIONS).index(collection) + scope_index,
            )
            if not null.empty:
                null.insert(0, "relation_scope", scope)
                null.insert(0, "collection", collection)
                null_rows.append(null)
            n = len(observed)
            contradictions = int(observed["directional_contradiction"].sum()) if n else 0
            ci_low, ci_high = wilson_interval(contradictions, n)
            if n == 0:
                estimate_status = "NOT_ESTIMABLE_NO_PAIRS"
            elif scope == "direct_parent_child" and n < 10:
                estimate_status = "NOT_ESTIMABLE_LT10_PRIMARY_PAIRS"
            else:
                estimate_status = "ESTIMABLE"
            rows.append(
                {
                    "collection": collection,
                    "relation_scope": scope,
                    "n_pairs": n,
                    "estimate_status": estimate_status,
                    "n_directional_contradictions": contradictions,
                    "directional_contradiction_fraction": contradictions / n if n else np.nan,
                    "directional_contradiction_ci_low": ci_low,
                    "directional_contradiction_ci_high": ci_high,
                    "median_leading_edge_jaccard": (
                        observed["leading_edge_jaccard"].median() if n else np.nan
                    ),
                    "median_child_covered_by_parent": (
                        observed["child_covered_by_parent"].median() if n else np.nan
                    ),
                    "median_parent_covered_by_child": (
                        observed["parent_covered_by_child"].median() if n else np.nan
                    ),
                    **null_summary,
                }
            )
    null_table = pd.concat(null_rows, ignore_index=True) if null_rows else pd.DataFrame()
    return pd.DataFrame(rows), null_table


def locate_external(manifest: dict[str, Any], data_root: Path, filename: str) -> Path:
    matches = [
        data_root / record["path"]
        for record in manifest["external_files"]
        if Path(record["path"]).name == filename
    ]
    require(len(matches) == 1, f"Expected one frozen {filename}; found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    repo_root = repo_root_from_script()
    checker = repo_root / "paper/revision/CRM_R1/scripts/51_check_priority5_freeze.py"
    subprocess.run(
        [sys.executable, str(checker), "--data-root", str(data_root)],
        cwd=repo_root,
        check=True,
    )

    p5_root = data_root / "output/priority5" / BENCHMARK_ID
    final_dir = p5_root / "ontology"
    require(not final_dir.exists(), f"P5 ontology output already exists: {final_dir}")
    manifest_path = p5_root / "freeze/priority5_input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol_path = repo_root / "paper/revision/CRM_R1/config/priority5_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    draws = int(protocol["matched_nonedge_reference"]["draws"])
    seed = int(protocol["matched_nonedge_reference"]["seed"])

    go_names, go_parents = parse_go_obo(locate_external(manifest, data_root, "go-basic.obo"))
    reactome_names, reactome_parents = parse_reactome(
        locate_external(manifest, data_root, "ReactomePathways.txt"),
        locate_external(manifest, data_root, "ReactomePathwaysRelation.txt"),
    )
    ontologies = {
        "C5_GO_BP": (go_names, go_parents),
        "C2_CP_REACTOME": (reactome_names, reactome_parents),
    }

    mapped_by_collection: dict[str, pd.DataFrame] = {}
    ancestors_by_collection: dict[str, dict[str, set[str]]] = {}
    mapping_tables: list[pd.DataFrame] = []
    pair_tables: list[pd.DataFrame] = []
    term_tables: list[pd.DataFrame] = []
    for collection, (names, parents) in ontologies.items():
        depths = minimum_depths(names, parents)
        ancestors = ancestor_sets(names, parents)
        ancestors_by_collection[collection] = ancestors
        audit_candidates = [
            data_root / record["path"]
            for record in manifest["external_files"]
            if record["path"].endswith(f"/{collection}/HNSC/ours/gate_hard/tau_0.90/audit_log.tsv")
        ]
        require(len(audit_candidates) == 1, f"Frozen audit log not unique: {collection}")
        audit = pd.read_csv(audit_candidates[0], sep="\t", low_memory=False)
        mapped, mapping = map_audit_terms(audit, collection=collection, names=names, depths=depths)
        mapped_by_collection[collection] = mapped
        mapping_tables.append(mapping)
        pair_tables.append(build_pairs(mapped, parents, ancestors))
        term_table = mapped.drop(columns=["gene_ids"]).copy()
        term_tables.append(term_table)

    mapping_qc = pd.concat(mapping_tables, ignore_index=True)
    pairs = (
        pd.concat([table for table in pair_tables if not table.empty], ignore_index=True)
        if any(not table.empty for table in pair_tables)
        else pd.DataFrame(columns=PAIR_COLUMNS)
    )
    terms = pd.concat(term_tables, ignore_index=True)
    metrics, null_draws = summarize_pairs(
        pairs,
        mapped_by_collection,
        ancestors_by_collection,
        draws=draws,
        seed=seed,
    )

    temporary = Path(tempfile.mkdtemp(prefix="priority5-ontology-", dir=p5_root))
    try:
        mapping_qc.to_csv(temporary / "mapping_qc.tsv", sep="\t", index=False)
        pairs.to_csv(temporary / "hierarchy_pairs.tsv", sep="\t", index=False)
        metrics.to_csv(temporary / "hierarchy_metrics.tsv", sep="\t", index=False)
        terms.to_csv(temporary / "ontology_term_metrics.tsv", sep="\t", index=False)
        null_draws.to_csv(temporary / "matched_nonedge_null_draws.tsv", sep="\t", index=False)
        outputs = [
            temporary / name
            for name in (
                "mapping_qc.tsv",
                "hierarchy_pairs.tsv",
                "hierarchy_metrics.tsv",
                "ontology_term_metrics.tsv",
                "matched_nonedge_null_draws.tsv",
            )
        ]
        run_meta = {
            "schema_version": "CRM_R1_PRIORITY5_ONTOLOGY_v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "benchmark_id": BENCHMARK_ID,
            "freeze_manifest_sha256": sha256_file(manifest_path),
            "protocol_sha256": sha256_file(protocol_path),
            "matched_nonedge_draws": draws,
            "matched_nonedge_seed": seed,
            "mapping": {
                collection: {
                    "mapped": int(
                        mapping_qc.loc[mapping_qc["collection"].eq(collection), "mapping_status"]
                        .eq("MAPPED_UNIQUE")
                        .sum()
                    ),
                    "total": int(mapping_qc.loc[mapping_qc["collection"].eq(collection)].shape[0]),
                }
                for collection in COLLECTIONS
            },
            "output_sha256": {path.name: sha256_file(path) for path in outputs},
            "priority3_grades_read": False,
            "priority4_ratings_read": False,
            "hierarchy_used_by_audit": False,
        }
        (temporary / "ontology_evaluation.run_meta.json").write_text(
            json.dumps(run_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, final_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print("[PASS] Priority 5 ontology hierarchy evaluation complete")
    for row in metrics.to_dict("records"):
        print(
            "[RESULT] "
            f"{row['collection']} {row['relation_scope']}: "
            f"n={row['n_pairs']}; contradiction={row['directional_contradiction_fraction']}"
        )
    print("[INFO] P3 grades and P4 ratings were not read")
    print(f"[INFO] Wrote: {final_dir / 'hierarchy_metrics.tsv'}")


if __name__ == "__main__":
    main()
