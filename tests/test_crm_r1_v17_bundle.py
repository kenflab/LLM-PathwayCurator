"""Synthetic Git repositories; no remote writes, credentials, or real data."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "paper/revision/CRM_R1/scripts/install_revision_bundle.py"
)
SPEC = importlib.util.spec_from_file_location("crm_revision_bundle", SCRIPT)
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


def r02_layout(layout):
    repo, data, bundle = layout
    target = data / "input/revision_code/test_r02"
    target.parent.mkdir(parents=True)
    shutil.move(bundle, target)
    bundle = target
    (bundle / "metadata_snapshot").mkdir()
    snapshot = bundle / "metadata_snapshot/METADATA_MANIFEST.json"
    snapshot.write_text('{"synthetic":true}')
    path = bundle / "BUNDLE_MANIFEST.json"
    manifest = json.loads(path.read_text())
    manifest["schema"] = "CRM_R1_CODE_BUNDLE_v17_R02"
    manifest["data_files"] = {
        "metadata_snapshot/METADATA_MANIFEST.json": installer.digest(snapshot.read_bytes())
    }
    path.write_text(json.dumps(manifest))
    return repo, data, bundle


def test_r02_snapshot_conflict_stops_before_any_code_write(layout):
    repo, data, bundle = r02_layout(layout)
    (bundle / "metadata_snapshot/METADATA_MANIFEST.json").write_text("changed")
    with pytest.raises(ValueError, match="metadata bundle hash mismatch"):
        installer.apply_bundle(bundle, repo, data, apply=True, run=True)
    assert (repo / "README.md").read_bytes() == b"old\n"
    assert not list((data / "output").iterdir())


def test_r02_run_uses_r02_entrypoint_and_metadata_without_rerunning_r01(layout, monkeypatch):
    repo, data, bundle = r02_layout(layout)
    real_run, executions = installer.subprocess.run, []

    def run(command, *args, **kwargs):
        if command[0] == sys.executable:
            executions.append(command)
            return subprocess.CompletedProcess(command, 0)
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(installer.subprocess, "run", run)
    installer.apply_bundle(bundle, repo, data, apply=True, run=True)
    assert len(executions) == 1
    assert executions[0][1].endswith("61_revision_r02.py")
    assert executions[0][-2:] == ["--metadata-snapshot", str(bundle / "metadata_snapshot")]


def test_r02_run_from_downloads_stops_before_code_installation(layout):
    repo, data, bundle = r02_layout(layout)
    misplaced = bundle.parent.parent.parent.parent / "Downloads/r02"
    misplaced.parent.mkdir(parents=True)
    shutil.move(bundle, misplaced)
    with pytest.raises(ValueError, match="under CRM_R1/input"):
        installer.apply_bundle(misplaced, repo, data, apply=True, run=True)
    assert (repo / "README.md").read_bytes() == b"old\n"
    assert not list((data / "output").iterdir())


@pytest.fixture
def layout(tmp_path):
    repo, data, bundle = tmp_path / "repo", tmp_path / "CRM_R1", tmp_path / "bundle"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    installer.git(repo, "config", "user.name", "Synthetic test")
    installer.git(repo, "config", "user.email", "test@example.invalid")
    (repo / "README.md").write_text("old\n")
    installer.git(repo, "add", "README.md")
    installer.git(repo, "commit", "-qm", "Synthetic base")
    head = installer.git(repo, "rev-parse", "HEAD").stdout.strip()
    installer.git(
        repo, "remote", "add", "origin", "https://github.com/kenflab/LLM-PathwayCurator.git"
    )
    installer.git(repo, "update-ref", "refs/remotes/origin/main", head)
    (data / "input").mkdir(parents=True)
    (data / "output").mkdir()
    files = {}
    for name, raw, old in (
        ("README.md", b"new\n", b"old\n"),
        ("src/demo.py", b"# synthetic\n", None),
    ):
        path = bundle / "payload" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        files[name] = {
            "sha256": installer.digest(raw),
            "base_sha256": installer.digest(old) if old else None,
        }
    (bundle / "BUNDLE_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema": "CRM_R1_CODE_BUNDLE_v17_R01",
                "repository": "kenflab/LLM-PathwayCurator",
                "files": files,
            }
        )
    )
    return repo, data, bundle


def test_preflight_changes_nothing(layout):
    repo, data, bundle = layout
    installer.apply_bundle(bundle, repo, data)
    assert (repo / "README.md").read_bytes() == b"old\n"
    assert not (repo / "src/demo.py").exists()
    assert not list((data / "output").iterdir())


def test_apply_backs_up_originals_and_identical_files_are_reusable(layout):
    repo, data, bundle = layout
    installer.apply_bundle(bundle, repo, data, apply=True)
    assert (repo / "README.md").read_bytes() == b"new\n"
    assert (repo / "src/demo.py").read_bytes() == b"# synthetic\n"
    backups = list((data / "output/revision_v17").glob("code_install_*/originals/README.md"))
    assert len(backups) == 1 and backups[0].read_bytes() == b"old\n"
    rows = installer.apply_bundle(bundle, repo, data)
    assert all(row["action"] == "ALREADY_IDENTICAL" for row in rows)


@pytest.mark.parametrize("change", ["local_edit", "deleted_tracked_file", "bundle_tamper"])
def test_conflicts_stop_before_any_code_or_result_write(layout, change):
    repo, data, bundle = layout
    if change == "local_edit":
        (repo / "README.md").write_text("local work\n")
    elif change == "deleted_tracked_file":
        (repo / "README.md").unlink()
    else:
        (bundle / "payload/README.md").write_text("tampered\n")
    before = (repo / "README.md").read_bytes() if (repo / "README.md").exists() else None
    with pytest.raises(ValueError):
        installer.apply_bundle(bundle, repo, data, apply=True)
    after = (repo / "README.md").read_bytes() if (repo / "README.md").exists() else None
    assert before == after
    assert not (repo / "src/demo.py").exists()
    assert not list((data / "output").iterdir())


def test_symlink_and_git_metadata_paths_are_rejected(layout):
    repo, _, _ = layout
    (repo / "alias").symlink_to(repo, target_is_directory=True)
    for path in ("alias/README.md", ".git/config", "../README.md"):
        with pytest.raises(ValueError):
            installer.file_path(repo, path)


def test_publish_from_other_branch_stops_before_code_write(layout):
    repo, data, bundle = layout
    installer.git(repo, "checkout", "-qb", "existing_local_branch")
    with pytest.raises(ValueError, match="main checkout"):
        installer.apply_bundle(bundle, repo, data, apply=True, publish=True)
    assert (repo / "README.md").read_bytes() == b"old\n"
    assert not list((data / "output").iterdir())


def test_publish_rejects_ahead_or_diverged_history(layout):
    repo, _, _ = layout
    (repo / "unrelated.txt").write_text("local commit\n")
    installer.git(repo, "add", "unrelated.txt")
    installer.git(repo, "commit", "-qm", "Unrelated local history")
    with pytest.raises(ValueError, match="differs from origin/main"):
        installer.publish_preflight(repo, fetch=False)


def test_restricted_commit_preserves_unrelated_staged_files(layout, monkeypatch):
    repo, data, bundle = layout
    (repo / "unrelated.txt").write_text("do not include\n")
    installer.git(repo, "add", "unrelated.txt")
    real_git, calls = installer.git, []

    def local_git(repo, *args, check=True):
        if args[0] in {"fetch", "push"}:
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        return real_git(repo, *args, check=check)

    monkeypatch.setattr(installer, "git", local_git)
    installer.apply_bundle(bundle, repo, data, apply=True, publish=True)
    committed = real_git(repo, "show", "--format=", "--name-only", "HEAD").stdout.splitlines()
    assert set(committed) == {"README.md", "src/demo.py"}
    assert real_git(repo, "diff", "--cached", "--name-only").stdout.strip() == "unrelated.txt"
    assert calls[-1] == ("push", "origin", "HEAD:main")
    assert all("--force" not in call for call in calls)
