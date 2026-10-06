"""Protect saved-result integrity and interpretation of matched comparisons."""
import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[4] / "paper/revision/CRM_R1/experiments/external_dex/report.py"
SPEC = importlib.util.spec_from_file_location("dex_report", SOURCE)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def table(q="0.8", nes="1"):
    return {f"TERM_{i:02}": {"term_id": f"TERM_{i:02}", "status": "ESTIMABLE", "q": q, "pval": "0.01", "NES": nes} for i in range(50)}


class InterpretationChecks(unittest.TestCase):
    def test_equal_rates_can_hide_different_selected_ids(self):
        discovery = table()
        discovery["TERM_00"]["q"] = "0.001"
        discovery["TERM_01"]["q"] = "0.01"
        folds = [copy.deepcopy(discovery) for _ in range(4)]
        folds[0]["TERM_00"]["q"] = "0.8"
        folds[1]["TERM_00"]["q"] = "0.8"
        rows, selected, _ = REPORT.comparisons(discovery, table("0.001"), folds)
        self.assertEqual(rows[2]["replication_rate"], rows[3]["replication_rate"])
        self.assertNotEqual(set(selected[REPORT.METHODS[2]]), set(selected[REPORT.METHODS[3]]))
        self.assertEqual(selected[REPORT.METHODS[2]], ["TERM_01"])
        self.assertEqual(selected[REPORT.METHODS[3]], ["TERM_00"])

    def test_non_estimable_validation_remains_in_denominator(self):
        discovery, validation = table("0.001"), table("0.001")
        validation["TERM_00"].update(status="NUMERICAL_FAILURE", NES="NA", q="NA")
        rows, _, _ = REPORT.comparisons(discovery, validation, [table("0.001") for _ in range(4)])
        self.assertEqual(rows[2]["selected_count"], 50)
        self.assertEqual(rows[2]["validation_estimable_count"], 49)
        self.assertEqual(rows[2]["replication_rate"], 49 / 50)

    def test_empty_selection_has_undefined_rate(self):
        rows, selected, _ = REPORT.comparisons(table(), table("0.001"), [table() for _ in range(4)])
        self.assertEqual(selected[REPORT.METHODS[2]], [])
        self.assertIsNone(rows[2]["replication_rate"])

    def test_validation_changes_cannot_select_different_terms(self):
        discovery = table("0.001")
        first = REPORT.comparisons(discovery, table("0.001"), [table("0.001") for _ in range(4)])
        second = REPORT.comparisons(discovery, table("0.9", "-1"), [table("0.001") for _ in range(4)])
        self.assertEqual(first[1], second[1])
        self.assertEqual(first[0][2]["replicated_count"], 50)
        self.assertEqual(second[0][2]["replicated_count"], 0)

    def test_bh_retains_family_50_for_two_estimable_terms(self):
        rows = [{"term_id": "A", "status": "ESTIMABLE", "pval": "0.001"},
                {"term_id": "B", "status": "ESTIMABLE", "pval": "0.01"},
                {"term_id": "C", "status": "NUMERICAL_FAILURE", "pval": "NA"}]
        self.assertEqual(REPORT.bh50(rows), {"B": 0.25, "A": 0.05})

    def test_zero_nes_is_not_same_direction(self):
        a = table("0.001", "0")["TERM_00"]
        b = table("0.001", "0")["TERM_00"]
        self.assertFalse(REPORT.endpoint(a, b))


class ArchiveChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "source.zip"

    def test_changed_payload_is_rejected_without_extraction(self):
        with zipfile.ZipFile(self.path, "w") as archive:
            archive.writestr("data.tsv", "changed data\n")
            archive.writestr("EXPORT_SHA256.json", json.dumps({"files": {"data.tsv": hashlib.sha256(b"original data\n").hexdigest()}}))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            REPORT.read_archive(self.path)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_unrecorded_extra_file_is_rejected(self):
        with zipfile.ZipFile(self.path, "w") as archive:
            archive.writestr("data.tsv", "original\n")
            archive.writestr("extra.tsv", "unrecorded\n")
            archive.writestr("EXPORT_SHA256.json", json.dumps({"files": {"data.tsv": hashlib.sha256(b"original\n").hexdigest()}}))
        with self.assertRaisesRegex(ValueError, "Unrecorded"):
            REPORT.read_archive(self.path)

    def test_path_traversal_is_rejected_without_extraction(self):
        with zipfile.ZipFile(self.path, "w") as archive:
            archive.writestr("../escaped.txt", "unsafe\n")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            REPORT.read_archive(self.path)
        self.assertFalse((self.path.parent.parent / "escaped.txt").exists())


if __name__ == "__main__":
    unittest.main()
