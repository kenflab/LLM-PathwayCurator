"""Apply an exact revision bundle to an existing checkout, preserving local work.

Default is a read-only preflight. --apply writes only listed code files after
all comparisons succeed. --run uses this Python to execute the bundle's milestone. --publish
requires the existing main checkout and performs a normal, restricted Git commit
and fast-forward push, using the user's existing Git authentication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(repo, *args, check=True):
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode:
        raise ValueError(f"Git {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def file_path(root, relative):
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, "Unsafe bundle path")
    require(path.parts and path.parts[0] != ".git", "Bundle cannot edit .git")
    result = root / path
    require(result.resolve().is_relative_to(root), "Bundle path escapes its root")
    require(
        not any(p.is_symlink() for p in [result, *result.parents] if p != root.parent),
        f"Symlink in bundle target: {relative}",
    )
    return result


def prepare(bundle, repo, data_root):
    repo, data_root = Path(repo).expanduser().resolve(), Path(data_root).expanduser().resolve()
    top = git(repo, "rev-parse", "--show-toplevel").stdout.strip()
    require(Path(top).resolve() == repo, "Use the existing repository root")
    require(
        (data_root / "input").is_dir() and (data_root / "output").is_dir(),
        "Data root must contain input/ and output/",
    )
    require(not data_root.is_relative_to(repo), "CRM_R1 data root must be outside Git")
    manifest = json.loads((bundle / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))
    require(
        manifest.get("schema")
        in {
            "CRM_R1_CODE_BUNDLE_v17_R01",
            "CRM_R1_CODE_BUNDLE_v17_R02",
            "CRM_R1_CODE_BUNDLE_v17_R03",
            "CRM_R1_CODE_BUNDLE_v17_R04",
            "CRM_R1_CODE_BUNDLE_v17_R05",
            "CRM_R1_CODE_BUNDLE_v17_R06",
        },
        "Unknown bundle schema",
    )
    require(manifest.get("repository") == "kenflab/LLM-PathwayCurator", "Wrong bundle repository")
    if manifest["schema"] == "CRM_R1_CODE_BUNDLE_v17_R02":
        data_files = manifest.get("data_files", {})
        require(
            "metadata_snapshot/METADATA_MANIFEST.json" in data_files,
            "R02 metadata snapshot missing from bundle",
        )
        for relative, expected in data_files.items():
            require(Path(relative).parts[0] == "metadata_snapshot", "Invalid R02 data path")
            raw = file_path(bundle, relative).read_bytes()
            require(digest(raw) == expected, f"R02 metadata bundle hash mismatch: {relative}")
    if manifest["schema"] == "CRM_R1_CODE_BUNDLE_v17_R03":
        data_files = manifest.get("data_files", {})
        require(
            "baseline_snapshot/v16_1_acceptance_results.zip" in data_files,
            "R03 historical baseline missing from bundle",
        )
        for relative, expected in data_files.items():
            require(Path(relative).parts[0] == "baseline_snapshot", "Invalid R03 data path")
            raw = file_path(bundle, relative).read_bytes()
            require(digest(raw) == expected, f"R03 baseline bundle hash mismatch: {relative}")
    if manifest["schema"] == "CRM_R1_CODE_BUNDLE_v17_R04":
        prerequisites = manifest.get("prerequisite_files", {})
        require(prerequisites, "R04 replay prerequisites missing from bundle")
        for relative, expected in prerequisites.items():
            require(
                digest(file_path(repo, relative).read_bytes()) == expected,
                f"R04 replay source conflict preserved: {relative}",
            )
        data_files = manifest.get("data_files", {})
        require(
            "returned_r03/r03_20261004T231625491225Z.zip" in data_files,
            "R04 returned R03 snapshot missing from bundle",
        )
        for relative, expected in data_files.items():
            require(Path(relative).parts[0] == "returned_r03", "Invalid R04 data path")
            raw = file_path(bundle, relative).read_bytes()
            require(digest(raw) == expected, f"R04 returned snapshot hash mismatch: {relative}")
    if manifest["schema"] == "CRM_R1_CODE_BUNDLE_v17_R05":
        prerequisites = manifest.get("prerequisite_files", {})
        require(prerequisites, "R05 replay prerequisites missing from bundle")
        for relative, expected in prerequisites.items():
            require(
                digest(file_path(repo, relative).read_bytes()) == expected,
                f"R05 replay source conflict preserved: {relative}",
            )
        data_files = manifest.get("data_files", {})
        require(
            "returned_r04/r04_20261005T100701108792Z.zip" in data_files,
            "R05 returned R04 snapshot missing from bundle",
        )
        for relative, expected in data_files.items():
            require(Path(relative).parts[0] == "returned_r04", "Invalid R05 data path")
            require(
                digest(file_path(bundle, relative).read_bytes()) == expected,
                f"R05 returned snapshot hash mismatch: {relative}",
            )
    rows, payloads, originals, conflicts = [], {}, {}, []
    require(manifest.get("files"), "Empty bundle")
    for relative, item in manifest["files"].items():
        source, target = file_path(bundle / "payload", relative), file_path(repo, relative)
        raw = source.read_bytes()
        require(digest(raw) == item["sha256"], f"Bundle hash mismatch: {relative}")
        require(not target.exists() or target.is_file(), f"Target is not a file: {relative}")
        original = target.read_bytes() if target.exists() else None
        current = digest(original) if original is not None else None
        if current == item["sha256"]:
            action = "ALREADY_IDENTICAL"
        elif current in (
            item.get("base_sha256s", [item["base_sha256"]])
            if manifest["schema"] == "CRM_R1_CODE_BUNDLE_v17_R06"
            else [item["base_sha256"]]
        ):
            action = "CREATE" if current is None else "UPDATE"
        else:
            action = "CONFLICT_PRESERVED"
            conflicts.append(relative)
        rows.append(
            {
                "path": relative,
                "action": action,
                "old_sha256": current,
                "new_sha256": item["sha256"],
            }
        )
        originals[relative], payloads[relative] = original, raw
    require(not conflicts, "Local differences preserved; no files changed: " + ", ".join(conflicts))
    return repo, data_root, manifest, rows, payloads, originals


def publish_preflight(repo, *, fetch):
    require(
        git(repo, "branch", "--show-current").stdout.strip() == "main",
        "Publishing requires the existing main checkout; no branches are switched automatically",
    )
    origin = git(repo, "remote", "get-url", "origin").stdout.strip().removesuffix(".git")
    require(
        origin
        in {
            "https://github.com/kenflab/LLM-PathwayCurator",
            "git@github.com:kenflab/LLM-PathwayCurator",
            "ssh://git@github.com/kenflab/LLM-PathwayCurator",
        },
        "origin must be kenflab/LLM-PathwayCurator",
    )
    if fetch:
        git(repo, "fetch", "origin", "main")
    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    remote = git(repo, "rev-parse", "refs/remotes/origin/main").stdout.strip()
    require(
        head == remote,
        "Local HEAD differs from origin/main. "
        "Preserve local work and inspect git status before updating",
    )
    return head


def write_json(path, data):
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def apply_bundle(bundle, repo, data_root, *, apply=False, run=False, publish=False):
    require(apply or not (run or publish), "--run and --publish require --apply")
    repo, data_root, manifest, rows, payloads, originals = prepare(bundle, repo, data_root)
    if run and manifest["schema"] == "CRM_R1_CODE_BUNDLE_v17_R02":
        require(
            (bundle / "metadata_snapshot").resolve().is_relative_to(data_root / "input"),
            "Place the R02 bundle under CRM_R1/input/ before --run",
        )
    if run and manifest["schema"] in {
        "CRM_R1_CODE_BUNDLE_v17_R03",
        "CRM_R1_CODE_BUNDLE_v17_R04",
        "CRM_R1_CODE_BUNDLE_v17_R05",
    }:
        require(
            bundle.resolve().is_relative_to(data_root / "input"),
            "Place the R03/R04/R05 bundle under CRM_R1/input/ before --run",
        )
    for row in rows:
        print(f"{row['action']}: {row['path']}", flush=True)
    if not apply:
        print("PREFLIGHT_OK; no files changed", flush=True)
        return rows
    if publish:
        publish_preflight(repo, fetch=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup = file_path(data_root, f"output/revision_v17/code_install_{stamp}")
    backup.mkdir(parents=True)
    log = {
        "schema": manifest["schema"],
        "repository": str(repo),
        "rows": rows,
        "head_before": git(repo, "rev-parse", "HEAD").stdout.strip(),
        "run_requested": run,
        "publish_requested": publish,
    }
    write_json(backup / "STARTED.private.json", log)
    try:
        for row in rows:
            relative = row["path"]
            target = file_path(repo, relative)
            current = target.read_bytes() if target.exists() else None
            require(current == originals[relative], f"File changed during preflight: {relative}")
            if row["action"] == "ALREADY_IDENTICAL":
                continue
            mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else 0o644
            if originals[relative] is not None:
                saved = backup / "originals" / relative
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, saved)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(payloads[relative])
            try:
                temporary.chmod(mode)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        if run:
            r02 = manifest["schema"] == "CRM_R1_CODE_BUNDLE_v17_R02"
            r03 = manifest["schema"] == "CRM_R1_CODE_BUNDLE_v17_R03"
            r04 = manifest["schema"] == "CRM_R1_CODE_BUNDLE_v17_R04"
            r05 = manifest["schema"] == "CRM_R1_CODE_BUNDLE_v17_R05"
            r06 = manifest["schema"] == "CRM_R1_CODE_BUNDLE_v17_R06"
            entry = (
                "65_revision_r06.py"
                if r06
                else "64_revision_r05.py"
                if r05
                else "63_revision_r04.py"
                if r04
                else "62_revision_r03.py"
                if r03
                else "61_revision_r02.py"
                if r02
                else "60_revision_r01.py"
            )
            command = [
                sys.executable,
                str(repo / "paper/revision/CRM_R1/scripts" / entry),
                "--data-root",
                str(data_root),
            ]
            if r02:
                command.extend(["--metadata-snapshot", str(bundle / "metadata_snapshot")])
            if r03 or r04 or r05:
                command.extend(["--source-bundle", str(bundle)])
            subprocess.run(command, check=True)
        if publish:
            publish_preflight(repo, fetch=True)
            paths = list(manifest["files"])
            for relative in paths:
                require(
                    digest(file_path(repo, relative).read_bytes())
                    == manifest["files"][relative]["sha256"],
                    f"Code changed after installation: {relative}",
                )
            git(repo, "add", "--", *paths)
            differences = git(repo, "diff", "--cached", "--quiet", "--", *paths, check=False)
            require(differences.returncode in {0, 1}, "Cannot inspect staged code changes")
            if differences.returncode:
                git(
                    repo,
                    "commit",
                    "--only",
                    "-m",
                    "Add CRM R1 R06 source-linked existing-rater baseline reanalysis"
                    if manifest["schema"] == "CRM_R1_CODE_BUNDLE_v17_R06"
                    else (
                        "Add CRM R1 R05 fixed-task backend capacity comparison "
                        "and revision decision"
                    )
                    if manifest["schema"] == "CRM_R1_CODE_BUNDLE_v17_R05"
                    else "Add CRM R1 R04 saved-response diagnosis and source-only reading probe"
                    if manifest["schema"] == "CRM_R1_CODE_BUNDLE_v17_R04"
                    else "Add CRM R1 R03 bounded atomic semantic development pilot"
                    if manifest["schema"] == "CRM_R1_CODE_BUNDLE_v17_R03"
                    else "Add CRM R1 R02 figure provenance and external metadata preflight"
                    if manifest["schema"] == "CRM_R1_CODE_BUNDLE_v17_R02"
                    else "Integrate CRM R1 contract API and source-checked R01 workflow",
                    "--",
                    *paths,
                )
            git(repo, "push", "origin", "HEAD:main")
        log.update(status="COMPLETE", head_after=git(repo, "rev-parse", "HEAD").stdout.strip())
        write_json(backup / "COMPLETE.private.json", log)
        print(f"COMPLETE; installation record and originals: {backup}", flush=True)
        return rows
    except Exception as error:
        write_json(backup / "FAILED.private.json", log | {"status": "FAILED", "error": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    try:
        apply_bundle(
            args.bundle.resolve(),
            args.repo,
            args.data_root,
            apply=args.apply,
            run=args.run,
            publish=args.publish,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"[STOP] {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
