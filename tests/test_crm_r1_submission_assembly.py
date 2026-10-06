"""Protect manuscript and evidence boundaries of the read-only collector."""
import importlib.util
import io
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


if __name__ == '__main__':
    unittest.main()
