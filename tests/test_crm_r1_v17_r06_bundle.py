"""R06 installer tests in synthetic repositories; all fetch/push calls are mocked."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "paper/revision/CRM_R1/scripts"
SPEC = importlib.util.spec_from_file_location(
    "r06_bundle_test", SCRIPT / "install_revision_bundle.py"
)
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


@pytest.fixture
def layout(tmp_path):
    repo, root, bundle = tmp_path / "repo", tmp_path / "CRM_R1", tmp_path / "bundle"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    installer.git(repo, "config", "user.name", "Synthetic test")
    installer.git(repo, "config", "user.email", "test@example.invalid")
    (repo / "README.md").write_bytes(b"r04\n")
    installer.git(repo, "add", "README.md")
    installer.git(repo, "commit", "-qm", "Synthetic R04 base")
    head = installer.git(repo, "rev-parse", "HEAD").stdout.strip()
    installer.git(
        repo, "remote", "add", "origin", "https://github.com/kenflab/LLM-PathwayCurator.git"
    )
    installer.git(repo, "update-ref", "refs/remotes/origin/main", head)
    (root / "input").mkdir(parents=True)
    (root / "output").mkdir()
    files = {}
    for relative, raw, bases in (
        ("README.md", b"r06\n", [installer.digest(b"r04\n"), installer.digest(b"r05\n")]),
        ("paper/revision/CRM_R1/scripts/65_revision_r06.py", b"# synthetic\n", [None]),
    ):
        path = bundle / "payload" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        files[relative] = {
            "sha256": installer.digest(raw),
            "base_sha256": bases[0],
            "base_sha256s": bases,
        }
    (bundle / "BUNDLE_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema": "CRM_R1_CODE_BUNDLE_v17_R06",
                "repository": "kenflab/LLM-PathwayCurator",
                "files": files,
            }
        )
    )
    return repo, root, bundle


@pytest.mark.parametrize("base", [b"r04\n", b"r05\n"])
def test_r06_accepts_exact_r04_or_prepared_r05_versions_without_r05_dependency(layout, base):
    repo, root, bundle = layout
    (repo / "README.md").write_bytes(base)
    installer.apply_bundle(bundle, repo, root, apply=True)
    assert (repo / "README.md").read_bytes() == b"r06\n"


def test_r06_unknown_local_difference_is_preserved_before_writes(layout):
    repo, root, bundle = layout
    (repo / "README.md").write_bytes(b"unrelated local work\n")
    with pytest.raises(ValueError, match="Local differences preserved"):
        installer.apply_bundle(bundle, repo, root, apply=True)
    assert (repo / "README.md").read_bytes() == b"unrelated local work\n"
    assert not (repo / "paper").exists()
    assert not list((root / "output").iterdir())


def test_r06_run_executes_only_existing_rating_reanalysis_without_live_or_p3(layout, monkeypatch):
    repo, root, bundle = layout
    real_run, commands = installer.subprocess.run, []

    def local(command, *args, **kwargs):
        if command[0] == sys.executable:
            commands.append(command)
            return subprocess.CompletedProcess(command, 0)
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(installer.subprocess, "run", local)
    installer.apply_bundle(bundle, repo, root, apply=True, run=True)
    assert len(commands) == 1
    assert commands[0][1].endswith("65_revision_r06.py")
    assert commands[0][2:] == ["--data-root", str(root)]


def test_r06_restricted_publish_excludes_prior_backend_and_private_or_unrelated_files(
    layout, monkeypatch
):
    repo, root, bundle = layout
    (repo / "unrelated.txt").write_text("Keep staged but do not commit\n")
    (repo / "previous_backend.py").write_text("# Untracked old preparation\n")
    installer.git(repo, "add", "unrelated.txt")
    real_git, remote_calls = installer.git, []

    def local_git(repo, *args, check=True):
        if args[0] in {"fetch", "push"}:
            remote_calls.append(args)
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        return real_git(repo, *args, check=check)

    monkeypatch.setattr(installer, "git", local_git)
    installer.apply_bundle(bundle, repo, root, apply=True, publish=True)
    paths = real_git(repo, "show", "--format=", "--name-only", "HEAD").stdout.splitlines()
    assert set(paths) == {"README.md", "paper/revision/CRM_R1/scripts/65_revision_r06.py"}
    assert real_git(repo, "diff", "--cached", "--name-only").stdout.strip() == "unrelated.txt"
    assert "R06" in real_git(repo, "log", "-1", "--format=%s").stdout
    assert remote_calls[-1] == ("push", "origin", "HEAD:main")
    assert all("--force" not in command for command in remote_calls)
