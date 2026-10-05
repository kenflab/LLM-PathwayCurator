"""R07 installer dispatch and restricted publication, with mocked remotes."""

import json
import subprocess
import sys

import pytest
import test_crm_r1_v17_bundle as base


@pytest.fixture
def layout(tmp_path):
    repo, data, bundle = base.layout.__wrapped__(tmp_path)
    path = bundle / "BUNDLE_MANIFEST.json"
    manifest = json.loads(path.read_text())
    manifest["schema"] = "CRM_R1_CODE_BUNDLE_v17_R07"
    reference = repo / "source_reference.json"
    reference.write_text('{"synthetic_r01_reference":true}')
    manifest["prerequisite_files"] = {
        reference.name: base.installer.digest(reference.read_bytes())
    }
    for item in manifest["files"].values():
        item["base_sha256s"] = [item["base_sha256"]]
    path.write_text(json.dumps(manifest))
    return repo, data, bundle


def test_run_dispatch_is_offline_r07_without_paid_backend(layout, monkeypatch):
    repo, data, bundle = layout
    real_run, executions = base.installer.subprocess.run, []

    def run(command, *args, **kwargs):
        if command[0] == sys.executable:
            executions.append(command)
            return subprocess.CompletedProcess(command, 0)
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(base.installer.subprocess, "run", run)
    base.installer.apply_bundle(bundle, repo, data, apply=True, run=True)
    assert len(executions) == 1
    assert executions[0][1].endswith("66_revision_r07.py")
    assert "--live" not in executions[0] and "--source-bundle" not in executions[0]


def test_r07_conflict_preserves_every_local_file(layout):
    repo, data, bundle = layout
    (repo / "README.md").write_text("unrelated local revision\n")
    with pytest.raises(ValueError, match="Local differences preserved"):
        base.installer.apply_bundle(bundle, repo, data, apply=True)
    assert (repo / "README.md").read_text() == "unrelated local revision\n"
    assert not (repo / "src/demo.py").exists()


def test_r07_changed_adapter_blocks_before_code_installation(layout):
    repo, data, bundle = layout
    (repo / "source_reference.json").write_text("changed adapter")
    with pytest.raises(ValueError, match="R07 source-adapter conflict preserved"):
        base.installer.apply_bundle(bundle, repo, data, apply=True)
    assert (repo / "README.md").read_text() == "old\n"


def test_r07_publish_excludes_unrelated_staged_work(layout, monkeypatch):
    repo, data, bundle = layout
    (repo / "unrelated.txt").write_text("retain staged\n")
    base.installer.git(repo, "add", "unrelated.txt")
    actual, remote_calls = base.installer.git, []

    def git(root, *args, **kwargs):
        if args[0] in {"fetch", "push"}:
            remote_calls.append(args)
            return subprocess.CompletedProcess(args, 0, "", "")
        return actual(root, *args, **kwargs)

    monkeypatch.setattr(base.installer, "git", git)
    base.installer.apply_bundle(bundle, repo, data, apply=True, publish=True)
    assert actual(repo, "diff", "--cached", "--name-only").stdout.strip() == "unrelated.txt"
    assert actual(repo, "show", "--format=", "--name-only", "HEAD").stdout.strip().splitlines() == [
        "README.md",
        "src/demo.py",
    ]
    assert ("push", "origin", "HEAD:main") in remote_calls
