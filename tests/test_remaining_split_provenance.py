"""Read-only provenance recovery must preserve live code and reject hash drift."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "paper/revision/CRM_R1/collect_remaining_split_provenance.py"


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("remaining_collector", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return (
            subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )

    git("init", "-q", "-b", "work")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@localhost")
    package = {
        "__init__.py": b'raise RuntimeError("Do not import archived code")\n',
        "pipeline.py": b"# Original package\n",
    }
    for name, raw in package.items():
        path = repo / module.PACKAGE_PREFIX / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    runner = repo / "paper/revision/CRM_R1/scripts/28c_run_priority2b_discovery_audit_v14_1_3.py"
    runner.parent.mkdir(parents=True)
    runner.write_text('raise RuntimeError("Do not run the archived model pipeline")\n')
    git("add", ".")
    git("commit", "-qm", "Original execution code")
    head = git("rev-parse", "HEAD")
    data = tmp_path / "data"
    signature_path = (
        data
        / "output/priority2b/discovery_work_v14_1_3/audit_final_v14_1_3_1/audit_run_signature.json"
    )
    signature_path.parent.mkdir(parents=True)
    signature = {
        "git_head": head,
        "package_source_sha256": module.package_digest(package),
        "script_sha256": module.digest(runner.read_bytes()),
    }
    signature_path.write_text(json.dumps(signature))
    export = data / "input/priority2b_v14_1_3/discovery_membership_export_manifest.json"
    export.parent.mkdir(parents=True)

    def records():
        export.write_text(
            json.dumps(
                {
                    "inputs": {
                        "audit_run_signature": {
                            "path": str(signature_path),
                            "sha256": module.digest(signature_path.read_bytes()),
                        }
                    }
                }
            )
        )
        membership = data / "output/priority2b/membership_lock_v14_1_3/membership_manifest.json"
        membership.parent.mkdir(parents=True, exist_ok=True)
        membership.write_text(
            json.dumps(
                {
                    "inputs": {
                        "membership_export_manifest": {
                            "path": str(export),
                            "sha256": module.digest(export.read_bytes()),
                        }
                    }
                }
            )
        )
        monkeypatch.setattr(module, "MEMBERSHIP_SHA256", module.digest(membership.read_bytes()))
        monkeypatch.setattr(module, "EXPORT_SHA256", module.digest(export.read_bytes()))

    records()
    (repo / module.PACKAGE_PREFIX / "pipeline.py").write_text("# Current dirty work\n")
    return module, repo, data, signature_path, export, records, git


def test_recover_recorded_git_bytes_without_touching_live_code(fixture, tmp_path):
    module, repo, data, _, _, _, git = fixture
    live = repo / module.PACKAGE_PREFIX / "pipeline.py"
    before, status = live.read_bytes(), git("status", "--porcelain")
    result = module.collect(data, repo, tmp_path / "review")
    assert result["status"] == "METADATA_VERIFIED"
    assert result["git_recovery"]["package_status"] == "MATCH"
    assert result["git_recovery"]["runner_status"] == "MATCH"
    assert not result["git_changes"] and result["model_calls"] == 0
    assert live.read_bytes() == before and git("status", "--porcelain") == status
    recovered = tmp_path / "review/matched_git_code/src/llm_pathway_curator/pipeline.py"
    assert recovered.read_text() == "# Original package\n"
    assert Path(result["return_zip"]).is_file()


def test_manifest_drift_is_reported_without_following_it(fixture, tmp_path):
    module, repo, data, _, export, _, _ = fixture
    export.write_text('{"tampered": true}\n')
    result = module.collect(data, repo, tmp_path / "review")
    assert result["status"] == "INCOMPLETE"
    assert "hash differs" in result["reason"]
    assert not (tmp_path / "review/metadata/audit_run_signature.json").exists()


def test_different_package_hash_is_never_marked_recovered(fixture, tmp_path):
    module, repo, data, signature_path, _, records, _ = fixture
    value = json.loads(signature_path.read_text())
    value["package_source_sha256"] = "0" * 64
    signature_path.write_text(json.dumps(value))
    records()
    result = module.collect(data, repo, tmp_path / "review")
    assert result["git_recovery"]["package_status"] == "DIFFERENT_BYTES"
    assert not (tmp_path / "review/matched_git_code/src").exists()
    assert result["git_recovery"]["runner_status"] == "MATCH"


def test_missing_git_object_keeps_verified_metadata(fixture, tmp_path):
    module, repo, data, signature_path, _, records, _ = fixture
    value = json.loads(signature_path.read_text())
    value["git_head"] = "f" * 40
    signature_path.write_text(json.dumps(value))
    records()
    result = module.collect(data, repo, tmp_path / "review")
    assert result["status"] == "METADATA_VERIFIED"
    assert result["git_recovery"]["status"] == "NOT_RECOVERED"
    assert (tmp_path / "review/metadata/audit_run_signature.json").is_file()


def test_existing_output_and_outside_paths_are_rejected(fixture, tmp_path):
    module, repo, data, _, _, _, _ = fixture
    out = tmp_path / "existing"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_text("existing result")
    with pytest.raises(ValueError, match="new output"):
        module.collect(data, repo, out)
    assert sentinel.read_text() == "existing result"
    with pytest.raises(ValueError, match="outside"):
        module.relative_record_path({"path": "/unrelated/secrets.json"}, data, repo)
    with pytest.raises(ValueError, match="escapes"):
        module.relative_record_path({"path": "{ANALYSIS_ROOT}/../../escape.json"}, data, repo)
