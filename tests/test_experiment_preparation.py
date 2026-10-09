from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


DATA_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "data"
for name in ("download_bird", "prepare_bird", "prepare_experiment"):
    spec = importlib.util.spec_from_file_location(name, DATA_SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
experiment = sys.modules["prepare_experiment"]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class ExperimentPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.processed = self.root / "processed"
        self.processed.mkdir()
        self.train = [self.record("train", i, f"train_db_{i // 100}") for i in range(1000)]
        self.dev = [self.record("dev", i, f"dev_db_{i // 150}") for i in range(450)]
        self.baseline = self.dev[:120]
        self.write_sources()

    def record(self, split, index, db_id):
        return {"id": f"{split}:{index}", "db_id": db_id, "question": f"Question {index}",
                "evidence": "", "gold_sql": "SELECT 1", "schema": "CREATE TABLE t(id INTEGER);",
                "db_path": f"/fixture/{db_id}.sqlite", "source_split": split,
                "difficulty": "unknown" if split == "train" else ("simple", "moderate", "challenging")[index % 3]}

    def write_sources(self):
        hashes, splits = {}, {}
        for split, rows in (("train", self.train), ("dev", self.dev)):
            path = self.processed / f"{split}.jsonl"
            if path.exists():
                path.unlink()
            hashes[path.name] = experiment.write_jsonl(path, rows)
            db_ids = sorted({r["db_id"] for r in rows})
            splits[split] = {"status": "complete", "count": len(rows), "db_ids": db_ids,
                             "archive": {"sha256": "a" * 64},
                             "databases": {db: {"quick_check": "ok", "sqlite_sha256": "b" * 64} for db in db_ids}}
        path = self.processed / "baseline-dev-120.jsonl"
        if path.exists():
            path.unlink()
        hashes[path.name] = experiment.write_jsonl(path, self.baseline)
        self.manifest = {"split_disjointness_verified": True, "splits": splits, "processed_sha256": hashes,
                         "baseline_sample": {"ids": [r["id"] for r in self.baseline]}, "license": "CC BY-SA 4.0"}
        (self.processed / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_complete_partition_and_no_database_or_baseline_leakage(self):
        before = {p.name: experiment.file_sha(p) for p in self.processed.glob("*.json*")}
        manifest = experiment.prepare(self.root)
        out = self.processed / "experiment-v1"
        train = read_jsonl(out / "internal_train.jsonl")
        validation = read_jsonl(out / "internal_validation.jsonl")
        sample = read_jsonl(out / "internal-validation-120.jsonl")
        test = read_jsonl(out / "dev-heldout-test-300.jsonl")
        self.assertEqual((len(train), len(validation), len(sample), len(test)), (800, 200, 120, 300))
        self.assertEqual({r["id"] for r in train + validation}, {r["id"] for r in self.train})
        self.assertFalse({r["db_id"] for r in train} & {r["db_id"] for r in validation})
        self.assertFalse({r["id"] for r in test} & {r["id"] for r in self.baseline})
        self.assertTrue({r["id"] for r in sample} <= {r["id"] for r in validation})
        self.assertTrue(all(r["source_split"] == "train" and r["split"] == "internal_train" for r in train))
        self.assertTrue(all(r["source_split"] == "train" and r["split"] == "internal_validation" for r in sample))
        self.assertTrue(all(r["source_split"] == "dev" and r["split"] == "dev_heldout_test" for r in test))
        self.assertTrue(manifest["selection"]["train_difficulty_missing"])
        for filename, report in manifest["files"].items():
            self.assertEqual(experiment.file_sha(out / filename), report["sha256"])
        self.assertEqual(before, {p.name: experiment.file_sha(p) for p in self.processed.glob("*.json*")})
        audit = read_jsonl(out / "record-audit.jsonl")
        self.assertEqual(len(audit), 1420)
        row = next(r for r in audit if r["file"] == "internal_train.jsonl")
        source = next(r for r in self.train if r["id"] == row["id"])
        self.assertEqual(row["source_record_sha256"], experiment.object_sha(source))
        self.assertEqual(row["record_sha256"], experiment.object_sha(next(r for r in train if r["id"] == row["id"])))

    def test_repeat_is_identical_and_gold_changes_cannot_change_selection(self):
        first = experiment.prepare(self.root, self.processed / "first")
        second = experiment.prepare(self.root, self.processed / "second")
        self.assertEqual(first["files"], second["files"])
        self.assertEqual(first["database_split"], second["database_split"])
        for row in self.train + self.dev:
            row["gold_sql"] = "INVALID SQL THAT CANNOT EXECUTE"
        self.write_sources()
        changed = experiment.prepare(self.root, self.processed / "changed-gold")
        self.assertEqual(first["database_split"], changed["database_split"])
        for filename in first["files"]:
            self.assertEqual(first["files"][filename]["ids"], changed["files"][filename]["ids"])

    def test_tampered_source_fails_before_creating_experiment(self):
        with (self.processed / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "changed since preparation"):
            experiment.prepare(self.root)
        self.assertFalse((self.processed / "experiment-v1").exists())

    def test_sample_shortfall_fails_and_frozen_output_cannot_be_overwritten(self):
        with self.assertRaisesRegex(ValueError, "sample exceeds"):
            experiment.prepare(self.root, validation_size=201)
        self.assertFalse((self.processed / "experiment-v1").exists())
        experiment.prepare(self.root)
        with self.assertRaises(FileExistsError):
            experiment.prepare(self.root)

    def test_database_overlap_rejected_even_if_source_manifest_claims_disjoint(self):
        self.dev[0]["db_id"] = "train_db_0"
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "database overlap"):
            experiment.prepare(self.root)


if __name__ == "__main__":
    unittest.main()
