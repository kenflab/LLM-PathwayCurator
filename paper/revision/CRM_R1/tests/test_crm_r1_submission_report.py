"""Boundaries for saved-result reporting; fixtures are not scientific results."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

SCRIPT=Path(__file__).resolve().parents[4]/'paper/revision/CRM_R1/experiments/submission/report.py'
spec=importlib.util.spec_from_file_location('submission_report',SCRIPT)
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


class ArchiveBoundaries(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def archive(self,changes=None,extras=None,missing=None):
        files={'SUMMARY.json':json.dumps({'schema':'CRM_R1_SUBMISSION_ASSEMBLY_R10','missing_or_unverified_components':[]}).encode(),
               'fixture.tsv':b'NON_SCIENTIFIC_FIXTURE\n'}
        manifest={'files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}}
        files['EXPORT_SHA256.json']=json.dumps(manifest).encode()
        files.update(changes or {});files.update(extras or {})
        for n in missing or []:files.pop(n)
        path=self.root/'input.zip'
        with zipfile.ZipFile(path,'w') as z:
            for n,raw in files.items():z.writestr(n,raw)
        return path

    def test_changed_export_stops_before_output_creation(self):
        path=self.archive(changes={'fixture.tsv':b'changed'})
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            report.build_report(path,self.root/'output')
        self.assertFalse((self.root/'output').exists())

    def test_unmanifested_or_missing_exports_are_rejected(self):
        for change in ({'extras':{'extra.txt':b'unknown'}},{'missing':['fixture.tsv']}):
            path=self.archive(**change)
            with self.assertRaisesRegex(ValueError,'Unmanifested or missing'):
                report.verify_archive(path)

    def test_archive_traversal_is_rejected(self):
        path=self.archive(extras={'../escape.txt':b'unsafe'})
        with self.assertRaisesRegex(ValueError,'Unsafe ZIP'):
            report.verify_archive(path)
        self.assertFalse((self.root.parent/'escape.txt').exists())

    def test_duplicate_paths_are_rejected(self):
        path=self.archive()
        with zipfile.ZipFile(path,'a') as z:z.writestr('fixture.tsv',b'duplicate')
        with self.assertRaisesRegex(ValueError,'Duplicate ZIP'):
            report.verify_archive(path)


if __name__=='__main__':unittest.main()
