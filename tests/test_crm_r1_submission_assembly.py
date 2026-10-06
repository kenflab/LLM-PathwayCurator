"""Protect manuscript and evidence boundaries of the read-only collector."""
import importlib.util
import io
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / 'paper/revision/CRM_R1/experiments/submission/assemble.py'
SPEC = importlib.util.spec_from_file_location('crm_r1_submission_assembly', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SourceAndManuscriptBoundaries(unittest.TestCase):
    def test_deleted_word_text_and_comments_are_separate_from_final_prose(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as z:
            z.writestr('word/document.xml', '''<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Original </w:t></w:r><w:del><w:r><w:delText>obsolete claim</w:delText></w:r></w:del><w:ins><w:r><w:t>revised claim</w:t></w:r></w:ins></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>cell A</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>cell B</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>''')
            z.writestr('word/comments.xml', '''<w:comments xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:comment w:id="0"><w:p><w:r><w:t>review note</w:t></w:r></w:p></w:comment></w:comments>''')
        raw = stream.getvalue()
        result = MODULE.docx_text(raw)
        self.assertEqual(result['paragraphs'], ['Original revised claim', 'cell A', 'cell B'])
        self.assertEqual(result['deleted_text_fragments'], ['obsolete claim'])
        self.assertEqual(result['comments'][0]['text'], 'review note')
        self.assertEqual((result['tracked_insertions'], result['tracked_deletions']), (1, 1))
        self.assertEqual(stream.getvalue(), raw)

    def test_historical_census_phrase_is_a_review_flag_not_an_error_label(self):
        flags = MODULE.manuscript_flags(['The original study had 100 claims and two raters.'])
        self.assertTrue(any(r['review_topic'] == 'HISTORICAL_RATER_CENSUS' for r in flags))
        self.assertTrue(all(r['status'] == 'AUTHOR_REVIEW_REQUIRED_NOT_AUTOMATIC_ERROR_LABEL' for r in flags))

    def test_original_full_path_rebases_by_complete_path_and_traversal_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'CRM_R1'
            root.mkdir()
            self.assertEqual(MODULE.local_path(root, '/Users/old/OneDrive/CRM_R1/output/p1/source.tsv'), root / 'output/p1/source.tsv')
            with self.assertRaises(ValueError):
                MODULE.local_path(root, '../secret.tsv')
            with self.assertRaises(ValueError):
                MODULE.local_path(root, '/Users/old/source.tsv')

    def test_blank_grading_remains_unknown_without_support_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / 'CRM_R1', Path(tmp) / 'report'
            root.mkdir(); out.mkdir()
            name = f'output/priority3/{MODULE.BENCHMARK}/grading_working/record_screening_P3C1.private.tsv'
            path = root / name
            path.parent.mkdir(parents=True)
            path.write_text('\t'.join(MODULE.P3_FIELDS) + '\n' + '\t' * (len(MODULE.P3_FIELDS) - 1) + '\n')
            original = path.read_bytes()
            result = MODULE.collect_p3(MODULE.Collection(root, out))
            self.assertEqual(result['rows_with_nonblank_eligible_curator_and_note'], 0)
            self.assertFalse(result['complete_locked_grading_authenticated'])
            self.assertFalse(result['grading_performed_by_this_run'])
            self.assertEqual(path.read_bytes(), original)
            self.assertNotIn('independent_support_rate', result)

    def test_missing_sources_cannot_become_submission_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, repo = Path(tmp) / 'CRM_R1', Path(tmp) / 'repo'
            (root / 'input').mkdir(parents=True)
            (root / 'output').mkdir()
            for name in MODULE.GUIDES:
                target = repo / 'paper/revision/CRM_R1' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('Guide fixture only\n')
            out = root / 'output/report'
            result = MODULE.build(root, out, repo)
            self.assertFalse(result['submission_ready'])
            self.assertFalse(result['current_full_audit_advantage_established'])
            self.assertEqual(result['source_status']['current_manuscript']['status'], 'MISSING_OR_NOT_VERIFIED')
            self.assertNotIn('Results — existing-rater comparison', (out / 'RESULTS_AND_DISCUSSION_DRAFT_EN.txt').read_text())
            with self.assertRaises(ValueError):
                MODULE.build(root, out, repo)

    def test_oversized_output_is_stream_verified_without_blocking_small_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / 'CRM_R1', Path(tmp) / 'report'
            folder = root / 'output/frozen'
            folder.mkdir(parents=True); out.mkdir()
            (folder / 'large.tsv').write_bytes(b'x' * (8 * 1024 * 1024 + 1))
            (folder / 'summary.tsv').write_text('method\tn\nraw\t50\n')
            manifest = {"outputs": {n: {"path": str(folder / n),
                        "sha256": MODULE.file_sha(folder / n)}
                        for n in ('large.tsv', 'summary.tsv')}}
            raw = json.dumps(manifest).encode()
            (folder / 'manifest.json').write_bytes(raw)
            (folder / 'manifest.sha256').write_text(hashlib.sha256(raw).hexdigest() + '\n')
            collection = MODULE.Collection(root, out)
            result = collection.verify_output_manifest('output/frozen/manifest.json')
            self.assertEqual(result['recorded_output_files_checked'], 2)
            self.assertEqual(result['copied_output_file_count'], 1)
            self.assertFalse((out / 'frozen_source_data/output/frozen/large.tsv').exists())
            self.assertTrue((out / 'frozen_source_data/output/frozen/summary.tsv').exists())
            self.assertEqual(result['verified_but_not_copied'][0]['sha256'], MODULE.file_sha(folder / 'large.tsv'))
            self.assertEqual(collection.inputs['output/frozen/large.tsv'], MODULE.file_sha(folder / 'large.tsv'))
            self.assertTrue((out / 'frozen_source_data/output/frozen/manifest.sha256').exists())

    def test_oversized_output_hash_mismatch_is_not_excused_by_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / 'CRM_R1', Path(tmp) / 'report'
            folder = root / 'output/frozen'
            folder.mkdir(parents=True); out.mkdir()
            path = folder / 'large.tsv'
            path.write_bytes(b'x' * (8 * 1024 * 1024 + 1))
            (folder / 'manifest.json').write_text(json.dumps({"outputs": {
                "large": {"path": str(path), "sha256": '0' * 64}}}))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                MODULE.Collection(root, out).verify_output_manifest('output/frozen/manifest.json')


if __name__ == '__main__':
    unittest.main()
