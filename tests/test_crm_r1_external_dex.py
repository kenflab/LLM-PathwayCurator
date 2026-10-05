"""Scientific denominator, provenance and input-safety regressions for the dex case."""
import copy
import gzip
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "paper/revision/CRM_R1/experiments/external_dex/run.py"
SPEC = importlib.util.spec_from_file_location("external_dex", SOURCE)
DEX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEX)


def table(q="0.4", nes="1.5"):
    return {t: {"term_id": t, "pval": "0.001", "q": q, "NES": nes, "status": "ESTIMABLE"} for t in DEX.TERMS}


class ScientificComparisons(unittest.TestCase):
    def test_zero_selection_has_no_rate_or_threshold_repair(self):
        discovery = table()
        records, sets, _, _, delta = DEX.compare(discovery, table("0.001"), [table() for _ in range(4)])
        self.assertEqual(len(sets["ALL_50_CANDIDATES"]), 50)
        self.assertEqual(sets["MATCHED_Q_K"], [])
        self.assertIsNone(delta)
        self.assertIsNone(records[2]["replication_rate"])

    def test_non_estimable_validation_stays_in_selected_denominator(self):
        discovery, validation = table("0.001"), table("0.001")
        validation[DEX.TERMS[0]].update(status="NUMERICAL_FAILURE", NES="NA", q="NA")
        records, _, _, _, _ = DEX.compare(discovery, validation, [table("0.001") for _ in range(4)])
        self.assertEqual(records[2]["selected_count"], 50)
        self.assertEqual(records[2]["validation_estimable_count"], 49)
        self.assertEqual(records[2]["replication_rate"], 49 / 50)

    def test_selections_do_not_depend_on_validation_sign_or_p_values(self):
        discovery = table()
        first, second = DEX.TERMS[:2]
        discovery[first]["q"], discovery[second]["q"] = "0.01", "0.001"
        folds = [copy.deepcopy(discovery) for _ in range(4)]
        folds[2][second].update(status="INCOMPLETE_FIXED_UNIVERSE", NES="NA", q="NA")
        folds[3][second].update(status="INCOMPLETE_FIXED_UNIVERSE", NES="NA", q="NA")
        a = DEX.compare(discovery, table("0.001", "1"), folds)
        b = DEX.compare(discovery, table("0.9", "-1"), folds)
        self.assertEqual(a[1], b[1])
        self.assertEqual(a[1]["Q_PLUS_DONOR_LOO_3_OF_4"], [first])
        self.assertEqual(a[1]["MATCHED_Q_K"], [second])
        self.assertEqual(a[2][second], 2)
        self.assertNotEqual(a[3], b[3])

    def test_direction_agreement_without_significance_is_not_replication(self):
        records, _, _, outcomes, _ = DEX.compare(table("0.001"), table("0.8"), [table("0.001") for _ in range(4)])
        self.assertEqual(records[0]["same_direction_count"], 50)
        self.assertFalse(any(outcomes.values()))

    def test_zero_direction_is_not_a_match(self):
        records, _, _, _, _ = DEX.compare(table("0.001", "0"), table("0.001", "0"), [table("0.001", "0") for _ in range(4)])
        self.assertEqual(records[0]["same_direction_count"], 0)
        self.assertEqual(records[0]["replicated_count"], 0)


class FrozenInputs(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "input").mkdir()
        (self.root / "output").mkdir()
        self.hallmark = self.root / "input/hallmark.tsv"
        self.hallmark.write_text("term_id\tgene_id\n" + "".join(f"{t}\tGENE{i}\n" for i, t in enumerate(DEX.TERMS)))

    def test_freeze_is_immutable_and_a_changed_original_is_rejected(self):
        directory, first = DEX.freeze(self.root, self.hallmark)
        self.assertFalse(first["expression_outcomes_loaded_by_freeze"])
        self.assertEqual(DEX.freeze(self.root)[1], first)
        original = (directory / "DESIGN_LOCK.json").read_bytes()
        self.hallmark.write_text(self.hallmark.read_text().replace("GENE0", "OTHER0"))
        with self.assertRaisesRegex(ValueError, "Original Hallmark"):
            DEX.freeze(self.root)
        self.assertEqual((directory / "DESIGN_LOCK.json").read_bytes(), original)

    def test_duplicate_and_non_symbol_memberships_are_rejected(self):
        original = self.hallmark.read_text()
        self.hallmark.write_text(original + original.splitlines()[1] + "\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            DEX.load_memberships(self.hallmark)
        self.hallmark.write_text(original.replace("GENE0", "1234"))
        with self.assertRaisesRegex(ValueError, "non-symbol"):
            DEX.load_memberships(self.hallmark)

    def test_missing_term_cannot_reduce_census(self):
        lines = self.hallmark.read_text().splitlines()
        self.hallmark.write_text("\n".join(lines[:-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "50 canonical"):
            DEX.load_memberships(self.hallmark)

    def test_symlink_cannot_redirect_output_to_another_directory(self):
        outside = self.root / "outside"
        outside.mkdir()
        (self.root / "output/revision_v17").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            DEX.freeze(self.root, self.hallmark)

    def test_download_reuse_checks_sha_without_network(self):
        path = self.root / "input/local.bin"
        path.write_bytes(b"first public input")
        receipt = {"url": "https://example.test/file", "design_sha256": "test", "sha256": DEX.sha(path)}
        DEX.write_json(path.with_name(path.name + ".receipt.json"), receipt)
        self.assertEqual(DEX.download(path, receipt["url"], "test"), receipt)
        path.write_bytes(b"changed input")
        with self.assertRaisesRegex(ValueError, "changed"):
            DEX.download(path, receipt["url"], "test")

    def test_sample_suffixes_never_create_pairing(self):
        protocol = DEX.read_json(DEX.HERE / "protocol.json")
        samples = protocol["validation"]["samples"]
        text = "!Sample_geo_accession\t" + "\t".join(s["geo_accession"] for s in samples) + "\n"
        text += "!Sample_title\t" + "\t".join(s["title"] for s in samples) + "\n"
        text += "!Sample_platform_id\t" + "\t".join(["GPL6480"] * 10) + "\n!series_matrix_table_begin\nexpression intentionally not parsed\n"
        matrix = self.root / "input/matrix.gz"
        with gzip.open(matrix, "wt") as h:
            h.write(text)
        DEX.check_sample_metadata(matrix, protocol)
        self.assertIn("UNVERIFIED", protocol["validation"]["pairing"])
        with gzip.open(matrix, "wt") as h:
            h.write(text.replace("dex24hr_3", "dex24hr_4"))
        with self.assertRaisesRegex(ValueError, "titles"):
            DEX.check_sample_metadata(matrix, protocol)

    def test_archive_path_traversal_is_rejected_without_extraction(self):
        archive = self.root / "input/raw.tar"
        with tarfile.open(archive, "w") as h:
            item = tarfile.TarInfo("../GSM847200_bad.txt")
            payload = b"gMedianSignal gBGMedianSignal gIsWellAboveBG"
            item.size = len(payload)
            h.addfile(item, io.BytesIO(payload))
        target = self.root / "arrays"
        target.mkdir()
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            DEX.extract_arrays(archive, target, DEX.read_json(DEX.HERE / "protocol.json"))
        self.assertEqual(list(target.iterdir()), [])

    def test_successful_annotation_parse_and_html_rejection(self):
        path = self.root / "input/gpl.txt"
        path.write_text("^PLATFORM = GPL6480\n!platform_table_begin\nID\tGENE_SYMBOL\tCONTROL_TYPE\nA_23_TEST\tTP53\t0\n!platform_table_end\n")
        self.assertEqual(DEX.platform_mapping(path)[0]["gene_symbol"], "TP53")
        path.write_text("<html>checking your browser</html>")
        with self.assertRaisesRegex(ValueError, "SOFT annotation"):
            DEX.platform_mapping(path)

    def test_coordinator_seals_complete_synthetic_output_and_reuses_without_rerun(self):
        # Synthetic wiring fixture only: R statistics are NOT executed by this test.
        lock_dir, lock = DEX.freeze(self.root, self.hallmark)
        directory = self.root / "input/external_dex_r08_v1"
        directory.mkdir()
        protocol = DEX.read_json(DEX.HERE / "protocol.json")
        samples = protocol["validation"]["samples"]
        text = "!Sample_geo_accession\t" + "\t".join(s["geo_accession"] for s in samples) + "\n"
        text += "!Sample_title\t" + "\t".join(s["title"] for s in samples) + "\n"
        text += "!Sample_platform_id\t" + "\t".join(["GPL6480"] * 10) + "\n!series_matrix_table_begin\n"
        with gzip.open(directory / "GSE34313_series_matrix.txt.gz", "wt") as h:
            h.write(text)
        (directory / "GPL6480.soft.txt").write_text("^PLATFORM = GPL6480\n!platform_table_begin\nID\tGENE_SYMBOL\tCONTROL_TYPE\nA_TEST\tTP53\t0\n!platform_table_end\n")
        with tarfile.open(directory / "GSE34313_RAW.tar", "w") as h:
            for sample in samples:
                item = tarfile.TarInfo(sample["geo_accession"] + "_fixture.txt")
                payload = b"gMedianSignal gBGMedianSignal gIsWellAboveBG"
                item.size = len(payload)
                h.addfile(item, io.BytesIO(payload))
        inputs = {"schema": "CRM_R1_EXTERNAL_DEX_INPUTS_v1", "design_sha256": DEX.object_sha(lock), "files": {}}
        for name, url in DEX.DOWNLOADS.items():
            receipt = {"url": url, "sha256": DEX.sha(directory / name), "design_sha256": DEX.object_sha(lock)}
            inputs["files"][name] = receipt
            DEX.write_json(directory / (name + ".receipt.json"), receipt)
        DEX.write_json(directory / "INPUT_MANIFEST.json", inputs)
        identity = {"ready": True, "missing": [], "R": "SYNTHETIC_TEST_ONLY", "packages": {}}
        def fake_run(command, **kwargs):
            job = DEX.read_json(command[-1])
            out = Path(job["outdir"])
            names = ["discovery", "validation_24h", "validation_4h_secondary"] + ["donor_loo_" + d for d in sorted({s["donor"] for s in protocol["discovery"]["samples"]})]
            for name in names:
                rows = list(table("0.001").values())
                DEX.write_tsv(out / (name + ".tsv"), rows, list(rows[0]))
            DEX.write_tsv(out / "common_gene_universe.tsv", [{"gene_symbol": "GENE" + str(i)} for i in range(1000)], ["gene_symbol"])
            DEX.write_json(out / "R_STATISTICS_COMPLETE.json", {"schema": "CRM_R1_EXTERNAL_DEX_R_STATS_R08", "donor_folds": 4, "model_calls": 0, "common_genes": 1000, "expression_outcomes_loaded": True})
            return type("Result", (), {"returncode": 0})()
        with mock.patch.object(DEX, "runtime_identity", return_value=identity), mock.patch.object(DEX.shutil, "which", return_value="SYNTHETIC_RSCRIPT"), mock.patch.object(DEX.subprocess, "run", side_effect=fake_run) as called:
            DEX.analyze(self.root, lock_dir, lock, directory, inputs)
            self.assertEqual(called.call_count, 1)
            DEX.analyze(self.root, lock_dir, lock, directory, inputs)
            self.assertEqual(called.call_count, 1)
        completed = DEX.read_json(lock_dir / "COMPLETED_ANALYSIS.json")
        out = self.root / completed["outdir_relative_path"]
        summary = DEX.read_json(out / "SUMMARY.json")
        self.assertEqual(summary["candidate_retained_count"], 50)
        self.assertFalse(summary["full_audit_performance_estimated"])
        self.assertTrue(Path(str(out) + ".zip").is_file())
        (out / "SUMMARY.json").write_text("modified")
        with mock.patch.object(DEX, "runtime_identity", return_value=identity):
            with self.assertRaisesRegex(ValueError, "export changed"):
                DEX.analyze(self.root, lock_dir, lock, directory, inputs)


if __name__ == "__main__":
    unittest.main()
