#!/usr/bin/env python3
"""Read remaining split-study metadata and recover matching code from local Git.

Uses only the Python standard library and read-only Git commands. It never
imports the project, calls a model, changes a lock, fetches, checks out, commits,
pushes, deletes a source file, or collects expression/rating tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

MEMBERSHIP_SHA256 = "839d0921cd0e54abe6915736cb4d0b48bf42eda9aa3ae24d7c0ce3aeaf2dbe6a"
EXPORT_SHA256 = "8abe03b1df8fb74097e99a4a07d1944782233918d543af5b989c860bf5a31209"
PACKAGE_PREFIX = "src/llm_pathway_curator/"
MAX_FILE = 8 * 1024**2
MAX_TOTAL = 64 * 1024**2


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def relative_record_path(record, data_root, repo):
    """Rebase only complete CRM_R1 or LLM-PathwayCurator roots, never basenames."""
    raw = record.get("path", "")
    require(isinstance(raw, str) and raw, "Missing recorded path")
    path = Path(raw).expanduser()
    for root in (data_root, repo):
        resolved = path.resolve()
        if resolved.is_relative_to(root):
            return resolved
    for token, root in (("{ANALYSIS_ROOT}/", data_root), ("{REPO_ROOT}/", repo)):
        if raw.startswith(token):
            candidate = (root / raw[len(token) :]).resolve()
            require(candidate.is_relative_to(root), "Recorded path escapes root")
            return candidate
    for marker, root in (("/CRM_R1/", data_root), ("/LLM-PathwayCurator/", repo)):
        if marker in raw:
            candidate = (root / raw.split(marker, 1)[1]).resolve()
            require(candidate.is_relative_to(root), "Recorded path escapes root")
            return candidate
    raise ValueError("Recorded path is outside the declared data and repository roots")


def git_bytes(repo, *args):
    result = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(repo), *args],
        capture_output=True,
        timeout=30,
        check=False,
    )
    require(result.returncode == 0, "Local Git object or path is unavailable")
    require(len(result.stdout) <= MAX_FILE, "Git output exceeds the size limit")
    return result.stdout


def package_digest(files):
    result = hashlib.sha256()
    for name in sorted(files):
        result.update(name.encode("utf-8"))
        result.update(b"\0")
        result.update(files[name])
        result.update(b"\0")
    return result.hexdigest()


def recover_git(repo, signature, put):
    revision = signature.get("git_head", "")
    expected_package = signature.get("package_source_sha256", "")
    expected_script = signature.get("script_sha256", "")
    result = {
        "recorded_git_head": revision,
        "recorded_package_sha256": expected_package,
        "recorded_runner_sha256": expected_script,
        "package_status": "NOT_RECOVERED",
        "runner_status": "NOT_RECOVERED",
    }
    require(
        re.fullmatch(r"[0-9a-fA-F]{40,64}", revision) is not None,
        "The recorded Git commit is missing or malformed",
    )
    entries = git_bytes(
        repo,
        "ls-tree",
        "-r",
        "--name-only",
        revision,
        "--",
        "src/llm_pathway_curator",
        "paper/revision/CRM_R1/scripts",
    )
    paths = entries.decode("utf-8").splitlines()
    require(len(paths) <= 5000, "Git tree exceeds the collection limit")
    package = {}
    for name in paths:
        if name.startswith(PACKAGE_PREFIX) and name.endswith(".py"):
            relative = name[len(PACKAGE_PREFIX) :]
            require(".." not in PurePosixPath(relative).parts, "Invalid Git path")
            package[relative] = git_bytes(repo, "show", f"{revision}:{name}")
    if package:
        actual = package_digest(package)
        result.update(
            actual_package_sha256=actual,
            package_python_files=len(package),
            package_status="MATCH" if actual == expected_package else "DIFFERENT_BYTES",
        )
        if actual == expected_package:
            for name, raw in package.items():
                put("matched_git_code/src/llm_pathway_curator/" + name, raw)
    candidate_hashes = []
    for name in paths:
        if not name.endswith(".py") or "priority2b" not in PurePosixPath(name).name:
            continue
        raw = git_bytes(repo, "show", f"{revision}:{name}")
        actual = digest(raw)
        candidate_hashes.append({"path": name, "sha256": actual})
        if actual == expected_script:
            put("matched_git_code/" + name, raw)
            result.update(runner_status="MATCH", runner_path=name)
    result["candidate_runner_hashes"] = candidate_hashes
    # This helper is context at the recorded commit; it is not independently pinned
    # by the package digest, which covers only src/llm_pathway_curator/*.py.
    helper = "paper/revision/CRM_R1/scripts/v14_1_objective_common.py"
    if helper in paths:
        raw = git_bytes(repo, "show", f"{revision}:{helper}")
        put("git_context/" + helper, raw)
        result["helper_context"] = {
            "path": helper,
            "sha256": digest(raw),
            "scope": "Recorded Git commit; no separate execution hash",
        }
    return result


def collect(data_root, repo, out):
    data_root, repo, out = data_root.resolve(), repo.resolve(), out.resolve()
    require(data_root.is_dir() and repo.is_dir(), "Data root or repository is missing")
    archive = out.with_name(out.name + ".zip")
    require(
        not out.exists() and not archive.exists(),
        "Use a new output directory and ZIP name",
    )
    out.mkdir(parents=True)
    receipt = {
        "schema": "remaining-split-provenance/1",
        "files": {},
        "checks": [],
        "model_calls": 0,
        "git_changes": False,
        "study_relocked": False,
        "research_outputs_changed": False,
        "scope": "Private provenance review packet; source paths may be present. "
        "Matching bytes do not rerun the study or certify its full environment.",
    }

    def put(name, raw):
        target = out / name
        require(target.resolve().is_relative_to(out), "Invalid output path")
        require(len(raw) <= MAX_FILE, "File exceeds the 8 MiB collection limit")
        require(
            sum(x["bytes"] for x in receipt["files"].values()) + len(raw) <= MAX_TOTAL,
            "Packet exceeds the 64 MiB collection limit",
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        receipt["files"][name] = {"bytes": len(raw), "sha256": digest(raw)}

    def verified_json(label, path, expected):
        require(path.is_file() and not path.is_symlink(), "Missing metadata file: " + label)
        require(path.stat().st_size <= MAX_FILE, "Metadata file exceeds the size limit")
        raw = path.read_bytes()
        actual = digest(raw)
        receipt["checks"].append(
            {
                "label": label,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "status": "MATCH" if actual == expected else "DIFFERENT_BYTES",
            }
        )
        require(actual == expected, "Recorded metadata hash differs: " + label)
        put("metadata/" + label + ".json", raw)
        value = json.loads(raw)
        require(isinstance(value, dict), "Expected a metadata object: " + label)
        return value

    try:
        membership = verified_json(
            "membership_manifest",
            data_root / "output/priority2b/membership_lock_v14_1_3/membership_manifest.json",
            MEMBERSHIP_SHA256,
        )
        exported = membership["inputs"]["membership_export_manifest"]
        require(exported["sha256"] == EXPORT_SHA256, "Membership export identity differs")
        manifest = verified_json(
            "membership_export_manifest",
            relative_record_path(exported, data_root, repo),
            EXPORT_SHA256,
        )
        record = manifest["inputs"]["audit_run_signature"]
        signature = verified_json(
            "audit_run_signature",
            relative_record_path(record, data_root, repo),
            record["sha256"],
        )
        receipt["signature_fields"] = {
            key: signature.get(key)
            for key in (
                "git_head",
                "script_sha256",
                "package_source_sha256",
                "context_backend",
                "context_model",
                "context_temperature",
                "python",
                "ollama",
            )
        }
        try:
            receipt["git_recovery"] = recover_git(repo, signature, put)
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            receipt["git_recovery"] = {"status": "NOT_RECOVERED", "reason": str(error)}
        receipt["status"] = "METADATA_VERIFIED"
    except (ValueError, OSError, KeyError) as error:
        receipt.update(status="INCOMPLETE", reason=str(error))
    (out / "REMAINING_COLLECTION.json").write_text(json.dumps(receipt, indent=2) + "\n")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                z.write(path, str(Path(out.name) / path.relative_to(out)))
    return {
        "status": receipt["status"],
        "outdir": str(out),
        "return_zip": str(archive),
        "files": len(receipt["files"]),
        "git_recovery": receipt.get("git_recovery"),
        "reason": receipt.get("reason"),
        "model_calls": 0,
        "git_changes": False,
        "study_relocked": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--outdir", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out = args.outdir or (args.data_root / "output/revision_v17" / ("split_remaining_" + stamp))
    result = collect(args.data_root, args.repo, out)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "METADATA_VERIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
