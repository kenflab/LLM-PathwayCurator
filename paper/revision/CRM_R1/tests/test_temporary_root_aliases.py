"""Run real provenance/boundary tests through macOS-style temporary aliases."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest

HERE = Path(__file__).resolve().parent


@pytest.mark.parametrize(
    ("filename", "classname"),
    [
        ("test_crm_r1_external_dex.py", "FrozenInputs"),
        ("test_crm_r1_submission_assembly.py", "SourceAndManuscriptBoundaries"),
    ],
)
def test_provenance_and_boundaries_with_temporary_root_alias(filename, classname):
    # /var and /private/var can refer to the same macOS temporary directory.
    # Model that relationship without depending on a macOS runner.
    original_temporary_directory = tempfile.TemporaryDirectory

    class AliasDirectory:
        def __init__(self, *args, **kwargs):
            self.original = original_temporary_directory(*args, **kwargs)
            self.alias = Path(self.original.name + ".alias")
            self.alias.symlink_to(Path(self.original.name).resolve(), target_is_directory=True)
            self.name = str(self.alias)

        def cleanup(self):
            self.alias.unlink(missing_ok=True)
            self.original.cleanup()

        def __enter__(self):
            return self.name

        def __exit__(self, *_):
            self.cleanup()

    spec = importlib.util.spec_from_file_location("temp_alias_" + classname, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(getattr(module, classname))
    result = unittest.TestResult()
    with mock.patch.object(tempfile, "TemporaryDirectory", AliasDirectory):
        suite.run(result)
    diagnostics = "\n".join(trace for _, trace in result.errors + result.failures)
    assert result.testsRun > 0
    assert result.wasSuccessful(), diagnostics
