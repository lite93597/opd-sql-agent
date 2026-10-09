from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile


DATA_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "data"
spec = importlib.util.spec_from_file_location("download_bird", DATA_SCRIPTS / "download_bird.py")
downloader = importlib.util.module_from_spec(spec)
sys.modules["download_bird"] = downloader
spec.loader.exec_module(downloader)
spec = importlib.util.spec_from_file_location("prepare_bird_test_module", DATA_SCRIPTS / "prepare_bird.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class BIRDArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "raw").mkdir()
        (self.root / "work").mkdir()
        db_path = self.root / "tiny.sqlite"
        with sqlite3.connect(db_path) as connection:
            connection.execute("CREATE TABLE t(id INTEGER PRIMARY KEY)")
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, "w") as archive:
            archive.writestr("train_databases/tiny/tiny.sqlite", db_path.read_bytes())
            archive.writestr("train_databases/tiny/database_description/t.csv", "column_name,column_description\nid,identifier\n")
            archive.writestr("train_databases/tiny/database_description/._bad.csv", b"\x00\x05\x16\x07BAD_METADATA")
            archive.writestr("__MACOSX/train_databases/tiny/tiny.sqlite", b"NOT_SQLITE")
        self.archive_path = self.root / "raw/train.zip"
        with zipfile.ZipFile(self.archive_path, "w") as archive:
            archive.writestr("train/train_databases.zip", inner.getvalue())
            archive.writestr("__MACOSX/train/._train_databases.zip", b"\x00\x05\x16\x07NOT_A_ZIP")
            archive.writestr("train/train.json", json.dumps([{"db_id": "tiny", "question": "Count rows", "SQL": "SELECT COUNT(*) FROM t"}]))
        sha, _ = downloader.digest_file(self.archive_path)
        self.sha = sha
        downloader.write_json(self.root / "work/train-download.json", {"status": "complete", "zip_crc_verified": True, "sha256": sha})

    def test_mac_resource_forks_are_ignored_throughout_preparation(self):
        records, report = prepare.prepare_split(self.root, "train", None)
        self.assertEqual(report["count"], 1)
        self.assertEqual(records[0]["source_split"], "train")
        self.assertEqual(report["databases"]["tiny"]["description_files"], 1)
        self.assertEqual(report["databases"]["tiny"]["quick_check"], "ok")
        self.assertIn("CREATE TABLE t", records[0]["schema"])
        self.assertFalse(list((self.root / "extracted").rglob("._*")))
        self.assertFalse(list((self.root / "extracted").rglob("__MACOSX")))

    def test_retry_reuses_outer_extraction_after_nested_failure(self):
        destination = self.root / "extracted/train"
        with zipfile.ZipFile(self.archive_path) as archive:
            # Reproduce the previous version: outer extraction finished, then
            # the first fake nested resource-fork ZIP made preparation fail.
            archive.extractall(destination)
        original_extract = prepare.extract
        with patch.object(prepare, "extract", wraps=original_extract) as extractor:
            records, _ = prepare.prepare_split(self.root, "train", None)
        self.assertEqual(len(records), 1)
        self.assertEqual(extractor.call_count, 1)
        self.assertEqual(extractor.call_args.args[0].name, "train_databases.zip")
        self.assertEqual((destination / ".outer-archive-sha256").read_text().strip(), self.sha)

    def test_corrupt_outer_extraction_is_not_reused(self):
        destination = self.root / "extracted/train"
        with zipfile.ZipFile(self.archive_path) as archive:
            archive.extractall(destination)
        nested = destination / "train/train_databases.zip"
        raw = nested.read_bytes()
        nested.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
        self.assertFalse(prepare.extracted_matches_archive(self.archive_path, destination))
        records, _ = prepare.prepare_split(self.root, "train", None)
        self.assertEqual(len(records), 1)


if __name__ == "__main__":
    unittest.main()
