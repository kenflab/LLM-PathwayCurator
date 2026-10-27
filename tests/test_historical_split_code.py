"""Protect execution identities while keeping historical code out of imports."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

ARCHIVE = Path(__file__).resolve().parents[1] / "paper/revision/CRM_R1/archive/split_replication"


def verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_historical_code", ARCHIVE / "verify_code.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recorded_code_identity_without_loading_old_package():
    before = {
        name: getattr(module, "__file__", None)
        for name, module in sys.modules.items()
        if name.startswith("llm_pathway_curator")
    }
    result = verifier().verify(ARCHIVE)
    assert result["status"] == "ARCHIVED_CODE_VERIFIED"
    assert result["R_runners"] == 2 and result["package_python_files"] == 24
    assert not result["distribution_helper_execution_hash_verified"]
    assert result["model_calls"] == 0 and not result["study_rerun"]
    after = {
        name: getattr(module, "__file__", None)
        for name, module in sys.modules.items()
        if name.startswith("llm_pathway_curator")
    }
    assert before == after


@pytest.mark.parametrize(
    "name",
    ["28c_run_priority2b_discovery_audit_v14_1_3.py", "llm_pathway_curator_e4698b8.zip"],
)
def test_altered_historical_code_is_rejected(tmp_path, name):
    archive = tmp_path / "archive"
    shutil.copytree(ARCHIVE, archive)
    path = archive / name
    path.write_bytes(path.read_bytes() + b"\n# altered\n")
    with pytest.raises(ValueError, match="Checksum differs"):
        verifier().verify(archive)
