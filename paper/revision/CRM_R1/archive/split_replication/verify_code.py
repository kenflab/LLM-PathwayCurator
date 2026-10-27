#!/usr/bin/env python3
"""Verify archived split-analysis code without importing or executing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

RUNNER_SHA256 = "6bb63f6932872c0caa7d8a39eabeab2dfacf26e79b28ee416d07db31228c5794"
PACKAGE_SHA256 = "44ba5bba30e69c6a60e40e7a4183787fa33046119f9003457e9d37874197a22c"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def checked_file(root, name, expected):
    path = root / name
    require(path.resolve().is_relative_to(root.resolve()), "File outside archive")
    require(path.is_file() and not path.is_symlink(), f"Missing archive file: {name}")
    raw = path.read_bytes()
    require(digest(raw) == expected, f"Checksum differs: {name}")
    return raw


def verify(root: Path) -> dict:
    provenance = json.loads((root / "PROVENANCE.json").read_text())
    require(provenance["schema"] == "historical-split-code-provenance/2", "Unknown schema")
    for entry in provenance["files"]:
        signature = json.loads(checked_file(root, entry["signature"], entry["signature_sha256"]))
        require(
            signature["script_sha256"] == entry["script_sha256"],
            f"Signature differs: {entry['stage']}",
        )
        checked_file(root, entry["script"], entry["script_sha256"])

    record_entry = provenance["membership_execution_record"]
    record = json.loads(checked_file(root, record_entry["filename"], record_entry["sha256"]))
    signature = record["audit_run_signature"]
    runner = provenance["membership_runner"]
    require(
        runner["sha256"] == signature["script_sha256"] == RUNNER_SHA256,
        "Membership runner identity differs",
    )
    checked_file(root, runner["filename"], RUNNER_SHA256)

    package = provenance["membership_package"]
    require(
        package["package_source_sha256"] == signature["package_source_sha256"] == PACKAGE_SHA256,
        "Package identity differs",
    )
    require(package["git_commit"] == signature["git_head"], "Package commit differs")
    checked_file(root, package["archive"], package["archive_sha256"])
    expected = package["files"]
    require(len(expected) == 24, "Package file count differs")
    aggregate = hashlib.sha256()
    with zipfile.ZipFile(root / package["archive"]) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate package archive entries")
        require(
            set(names) == {"llm_pathway_curator/" + name for name in expected},
            "Package archive entries differ",
        )
        for name in sorted(expected):
            path = PurePosixPath(name)
            require(
                not path.is_absolute() and ".." not in path.parts and path.suffix == ".py",
                "Invalid package path",
            )
            raw = archive.read("llm_pathway_curator/" + name)
            require(len(raw) == expected[name]["bytes"], f"Package size differs: {name}")
            require(digest(raw) == expected[name]["sha256"], f"Package hash differs: {name}")
            aggregate.update(name.encode("utf-8"))
            aggregate.update(b"\0")
            aggregate.update(raw)
            aggregate.update(b"\0")
    require(aggregate.hexdigest() == PACKAGE_SHA256, "Aggregate package digest differs")

    helper = provenance["distribution_helper"]
    checked_file(root, helper["filename"], helper["sha256"])
    require(helper["execution_hash_verified"] is False, "Helper scope differs")
    return {
        "status": "ARCHIVED_CODE_VERIFIED",
        "R_runners": len(provenance["files"]),
        "membership_runner_sha256": RUNNER_SHA256,
        "package_python_files": len(expected),
        "package_source_sha256": PACKAGE_SHA256,
        "distribution_helper_execution_hash_verified": False,
        "model_calls": 0,
        "study_rerun": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.archive_dir), indent=2))


if __name__ == "__main__":
    main()
