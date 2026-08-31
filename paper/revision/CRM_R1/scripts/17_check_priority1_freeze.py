#!/usr/bin/env python3
"""Verify the immutable Priority 1 freeze bundle before held-out 72 h analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

CRM_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CRM_DIR.parents[2]
DEFAULT_CONFIG = CRM_DIR / "config" / "priority1_protocol.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def scoped_path(entry: dict[str, Any], *, data_root: Path) -> Path:
    scope = entry.get("scope")
    relative = Path(str(entry.get("path", "")))
    require(not relative.is_absolute(), f"Manifest path must be relative: {relative}")
    require(".." not in relative.parts, f"Manifest path escapes its scope: {relative}")
    if scope == "data_root":
        return data_root / relative
    if scope == "repository":
        return REPO_ROOT / relative
    raise ValueError(f"Unknown manifest scope: {scope}")


def parse_sidecar(path: Path, *, expected_name: str) -> str:
    fields = path.read_text(encoding="utf-8").strip().split()
    require(len(fields) == 2, f"Malformed SHA-256 sidecar: {path}")
    require(fields[1] == expected_name, f"Sidecar filename mismatch: {path}")
    require(len(fields[0]) == 64, f"Malformed SHA-256 digest: {path}")
    return fields[0]


def selected_ids(table: pd.DataFrame, column: str) -> set[str]:
    require(column in table.columns, f"Missing membership column: {column}")
    values = table[column]
    if pd.api.types.is_bool_dtype(values.dtype):
        mask = values.fillna(False).astype(bool)
    else:
        normalized = values.fillna("").astype(str).str.strip().str.lower()
        require(set(normalized) <= {"true", "false"}, f"Invalid Boolean column: {column}")
        mask = normalized.eq("true")
    return set(table.loc[mask, "claim_id"].astype(str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--allow-post-validation",
        action="store_true",
        help="Allow known 72 h outputs when rechecking provenance after validation.",
    )
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    config = read_json(config_path)
    require(config["status"] == "FROZEN", "Protocol is not FROZEN")
    require(config["protocol_version"] == "CRM_R1_PRIORITY1_v5", "Protocol version drift")
    primary_tau = float(config["empirical_stability"]["primary_tau"])
    require(math.isclose(primary_tau, 0.8, abs_tol=1e-12), "Primary tau is not 0.80")

    benchmark_dir = data_root / "output" / "priority1" / str(config["benchmark_id"])
    metrics_dir = benchmark_dir / "metrics"
    manifest_path = metrics_dir / "priority1_freeze_manifest.json"
    sidecar_path = metrics_dir / "priority1_freeze_manifest.sha256"
    require(manifest_path.is_file(), f"Missing freeze manifest: {manifest_path}")
    require(sidecar_path.is_file(), f"Missing freeze sidecar: {sidecar_path}")
    expected_manifest_hash = parse_sidecar(sidecar_path, expected_name=manifest_path.name)
    require(sha256_file(manifest_path) == expected_manifest_hash, "Freeze manifest hash mismatch")
    manifest = read_json(manifest_path)

    require(
        manifest.get("manifest_schema_version") == "CRM_R1_PRIORITY1_FREEZE_MANIFEST_v1",
        "Freeze manifest schema drift",
    )
    require(manifest.get("benchmark_id") == config["benchmark_id"], "Benchmark drift")
    require(manifest.get("held_out_expression_outcomes_calculated") is False, "Held-out flag drift")
    require(manifest.get("validation_output_absent_at_freeze") is True, "Freeze timing not proven")
    require(math.isclose(float(manifest["primary_tau"]), primary_tau), "Manifest tau drift")
    require(int(manifest["selected_k"]) == 23, "Manifest K drift")
    require(manifest["protocol"]["status"] == "FROZEN", "Manifest protocol status drift")
    require(
        manifest["protocol"]["version"] == config["protocol_version"],
        "Manifest protocol version drift",
    )
    require(sha256_file(config_path) == manifest["protocol"]["sha256"], "Protocol hash drift")

    inventory = manifest.get("input_inventory")
    require(isinstance(inventory, list) and inventory, "Freeze inventory is empty")
    seen: set[tuple[str, str]] = set()
    for entry in inventory:
        require(isinstance(entry, dict), "Malformed inventory entry")
        key = (str(entry.get("scope")), str(entry.get("path")))
        require(key not in seen, f"Duplicate inventory entry: {key}")
        seen.add(key)
        path = scoped_path(entry, data_root=data_root)
        require(path.is_file(), f"Frozen input is missing: {path}")
        require(path.stat().st_size == int(entry["size_bytes"]), f"Frozen input size drift: {path}")
        require(sha256_file(path) == entry["sha256"], f"Frozen input hash drift: {path}")

    outputs = manifest["outputs"]
    membership_path = data_root / outputs["primary_membership"]["path"]
    tau_grid_path = data_root / outputs["tau_grid_membership"]["path"]
    for name, path in (
        ("primary_membership", membership_path),
        ("tau_grid_membership", tau_grid_path),
    ):
        require(path.is_file(), f"Frozen output is missing: {path}")
        require(sha256_file(path) == outputs[name]["sha256"], f"Frozen output hash drift: {path}")

    membership = pd.read_csv(membership_path, sep="\t")
    require(len(membership) == 50, "Frozen candidate-pool row count drift")
    require(membership["claim_id"].nunique() == 50, "Frozen claim_id uniqueness drift")
    require(set(membership["selection_status"].astype(str)) == {"FROZEN"}, "Status drift")
    require(
        set(pd.to_numeric(membership["primary_tau"], errors="raise")) == {primary_tau},
        "Membership tau drift",
    )
    manifest_memberships = manifest["memberships"]
    method_columns = {
        "empirical_stability_audit": "empirical_selected",
        "q_value_matched": "q_value_matched_selected",
        "q_value_and_leading_edge_size_matched": "q_value_size_matched_selected",
    }
    observed: dict[str, set[str]] = {}
    for method, column in method_columns.items():
        observed[method] = selected_ids(membership, column)
        require(len(observed[method]) == 23, f"Frozen K drift for {method}")
        require(
            observed[method] == set(manifest_memberships[method]),
            f"Manifest/table membership drift for {method}",
        )
    require(
        len(observed["empirical_stability_audit"] & observed["q_value_matched"]) == 16,
        "Empirical/q-value overlap drift",
    )
    require(
        len(
            observed["empirical_stability_audit"]
            & observed["q_value_and_leading_edge_size_matched"]
        )
        == 17,
        "Empirical/size-matched overlap drift",
    )

    tau_grid = pd.read_csv(tau_grid_path, sep="\t")
    require(len(tau_grid) == 200, "Frozen tau-grid row count drift")
    require(set(tau_grid["selection_status"].astype(str)) == {"FROZEN"}, "Grid status drift")
    expected_counts = {0.8: 23, 0.9: 15, 0.95: 10, 0.98: 5}
    for tau, expected in expected_counts.items():
        rows = tau_grid.loc[np_isclose(tau_grid["tau"], tau)]
        require(len(rows) == 50, f"Tau-grid candidate count drift at {tau}")
        require(
            int(rows["status"].astype(str).str.upper().eq("PASS").sum()) == expected,
            f"PASS drift at {tau}",
        )

    validation_dir = benchmark_dir / "validation"
    known_72h_outputs = sorted(validation_dir.glob("*72h*")) if validation_dir.exists() else []
    if not args.allow_post_validation:
        require(
            not known_72h_outputs,
            "Known 72 h output exists; use --allow-post-validation only for provenance rechecks",
        )

    print("[PASS] Priority 1 freeze bundle is complete and hash-consistent")
    selected_k = manifest["selected_k"]
    freeze_label = manifest["freeze_label"]
    print(f"[INFO] tau={primary_tau:.2f}; K={selected_k}; label={freeze_label}")
    print(f"[INFO] Freeze commit: {manifest['code']['commit_at_freeze']}")
    if args.allow_post_validation:
        print("[INFO] Post-validation provenance check; existing 72 h outputs were allowed.")
    else:
        print("[GO] Held-out 72 h analysis may now be implemented and run exactly once.")


def np_isclose(values: pd.Series, target: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    return (numeric - target).abs() <= 1e-12


if __name__ == "__main__":
    main()
