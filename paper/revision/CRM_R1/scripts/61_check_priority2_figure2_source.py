#!/usr/bin/env python3
"""Verify frozen V11 Figure 2 source tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from v11_lock_common import read_json, require, sha256_file

BENCHMARK_ID = "PANCAN_TP53_v1_HNSC_R1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()
    root = (
        args.data_root.expanduser().resolve() / "output" / "priority2" / BENCHMARK_ID / "final_v11"
    )
    manifest_path = root / "figure2_source_manifest.json"
    digest_path = root / "figure2_source_manifest.sha256"
    require(
        manifest_path.is_file() and digest_path.is_file(),
        "Missing Figure 2 source lock",
    )
    require(
        sha256_file(manifest_path) == digest_path.read_text().split()[0],
        "Figure 2 manifest digest mismatch",
    )
    manifest = read_json(manifest_path)
    require(
        manifest.get("status") == "FIGURE2_SOURCE_FROZEN_AFTER_P3_P4_LOCK",
        "Figure 2 source status drift",
    )
    paths = {}
    for label, record in {**manifest["inputs"], **manifest["outputs"]}.items():
        path = Path(record["path"])
        require(path.is_file(), f"Missing Figure 2 file: {label}")
        require(sha256_file(path) == record["sha256"], f"Figure 2 hash mismatch: {label}")
        paths[label] = path
    metrics = pd.read_csv(paths["panel_BC"], sep="\t")
    require(
        set(metrics["method_id"])
        == {"raw_pool", "q_value_matched", "stability_matched", "full_audit"},
        "Figure 2 method inventory drift",
    )
    require(
        set(metrics["outcome"]) >= {"primary_independent_support", "major_overstatement_majority"},
        "Figure 2 outcomes incomplete",
    )
    print("[PASS] Figure 2 V11 source bundle is complete and hash-consistent")
    print(f"[INFO] benchmark={BENCHMARK_ID}; claims=50; matched K={manifest['matched_k']}")
    print("[GO] Figure 2 may be rendered without recomputing outcomes")


if __name__ == "__main__":
    main()
