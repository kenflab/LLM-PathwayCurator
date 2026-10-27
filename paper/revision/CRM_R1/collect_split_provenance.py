#!/usr/bin/env python3
"""Collect original split-study manifests and code for author provenance review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

LOCK_DIRS = (
    "design_lock_v14_1",
    "numerical_lock_v14_1_3",
    "discovery_stats_lock_v14_1_3",
    "membership_lock_v14_1_3",
    "validation_design_lock_v14_1_3",
    "validation_lock_v14_1_3",
)
EXPECTED = {
    "design_lock_v14_1/design_manifest.json": (
        "f88017ae625e45ba135f519227a6159df77b302626bc58898892fe987bfe6e78"
    ),
    "membership_lock_v14_1_3/membership_manifest.json": (
        "839d0921cd0e54abe6915736cb4d0b48bf42eda9aa3ae24d7c0ce3aeaf2dbe6a"
    ),
    "validation_lock_v14_1_3/validation_manifest.json": (
        "cdbca7e228f787cf52c1dcb30e474f9cd367b979c75af3ba0cd9c93768c602d8"
    ),
}
SUFFIXES = {".json", ".py", ".r", ".sha256", ".yml", ".yaml"}
SKIP_DIRS = {
    "out",
    "output",
    "input",
    "raw",
    "source_data",
    "node_modules",
    "venv",
    "__pycache__",
    "site",
    "dist",
    "build",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_projects(root, candidates):
    """Read Git identities and candidate split-study code; never prune worktrees."""
    records = []
    if not root.is_dir():
        return [{"path": str(root), "status": "MISSING"}]
    for project in sorted(root.iterdir()):
        if not project.name.startswith(("LLM-PathwayCurator", "LLM_PathwayCurator")):
            continue
        if not project.is_dir() or project.is_symlink():
            records.append({"name": project.name, "status": "NOT_SCANNED_FILE_OR_SYMLINK"})
            continue
        record = {"name": project.name, "status": "SCANNED", "git": {}, "code_files": 0}
        for key, arguments in (
            ("head", ["rev-parse", "HEAD"]),
            ("branch", ["symbolic-ref", "--short", "-q", "HEAD"]),
            ("status", ["status", "--porcelain=v1", "--untracked-files=normal"]),
        ):
            try:
                result = subprocess.run(
                    ["git", "--no-optional-locks", "-C", str(project), *arguments],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                record["git"][key] = result.stdout.strip() if result.returncode == 0 else None
            except (OSError, subprocess.TimeoutExpired):
                record["git"][key] = None
        seen = 0
        for folder, dirs, names in os.walk(project, followlinks=False):
            dirs[:] = sorted(
                d
                for d in dirs
                if not d.startswith(".")
                and d not in SKIP_DIRS
                and not (Path(folder) / d).is_symlink()
            )
            seen += len(names)
            if seen > 50000:
                record["status"] = "FILE_SCAN_LIMIT_REACHED"
                break
            for name in sorted(names):
                path = Path(folder) / name
                if (
                    path.suffix.lower() not in {".py", ".r"}
                    or path.is_symlink()
                    or ".private." in name
                ):
                    continue
                if "priority2b" not in name.lower() and not name.startswith("v14_1"):
                    continue
                if not path.resolve().is_relative_to(project.resolve()):
                    continue
                relative = Path(project.name) / path.relative_to(project)
                candidates["project_code/" + str(relative)] = path
                record["code_files"] += 1
        records.append(record)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--projects-root",
        type=Path,
        help="Optional parent folder containing old checkouts and backups",
    )
    args = parser.parse_args(argv)
    base = (args.data_root / "output/priority2b").resolve()
    out = args.outdir.resolve()
    archive = out.with_name(out.name + ".zip")
    if out.exists() or archive.exists():
        raise ValueError("Use a new output directory and ZIP name")
    candidates, missing = {}, []
    for name in LOCK_DIRS:
        folder = base / name
        if not folder.is_dir():
            missing.append(name)
            continue
        for path in sorted(folder.rglob("*")):
            if (
                path.is_file()
                and not path.is_symlink()
                and path.suffix.lower() in SUFFIXES
                and ".private." not in path.name
                and path.resolve().is_relative_to(base)
            ):
                candidates["locks/" + str(path.relative_to(base))] = path
    scripts = (args.repo / "paper/revision/CRM_R1/scripts").resolve()
    for pattern in ("*priority2b*.py", "*priority2b*.R", "v14_1*objective*.py"):
        for path in sorted(scripts.glob(pattern)):
            if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(scripts):
                candidates["checkout_scripts/" + path.name] = path
    projects = (
        inspect_projects(args.projects_root.resolve(), candidates) if args.projects_root else []
    )
    if any(p.stat().st_size > 8 * 1024**2 for p in candidates.values()):
        raise ValueError("A manifest/code file exceeds the 8 MiB collection limit")
    if sum(p.stat().st_size for p in candidates.values()) > 64 * 1024**2:
        raise ValueError("Collection exceeds the 64 MiB limit")
    records = []
    for name, expected in EXPECTED.items():
        path = candidates.get("locks/" + name)
        actual = sha(path) if path else None
        records.append(
            {
                "path": name,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "status": "MISSING"
                if actual is None
                else "MATCH"
                if actual == expected
                else "DIFFERENT_BYTES",
            }
        )
    out.mkdir(parents=True)
    files = {}
    for name, path in sorted(candidates.items()):
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if sha(path) != sha(target):
            raise ValueError("Source changed during collection: " + name)
        files[name] = {"sha256": sha(target), "bytes": target.stat().st_size}
    receipt = {
        "schema": "split-provenance-collection/1",
        "files": files,
        "known_lock_checks": records,
        "missing_directories": missing,
        "project_inventory": projects,
        "model_calls": 0,
        "study_relocked": False,
        "inputs_changed": False,
        "scope": "Author review packet; local paths may occur in original manifests. "
        "Collection does not certify the executed fit code.",
    }
    (out / "COLLECTION.json").write_text(json.dumps(receipt, indent=2) + "\n")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                z.write(path, str(Path(out.name) / path.relative_to(out)))
    print(
        json.dumps(
            {
                "outdir": str(out),
                "return_zip": str(archive),
                "files": len(files),
                "known_lock_checks": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
