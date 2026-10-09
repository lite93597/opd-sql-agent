from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from opd_sql.evaluation import compare_reports, evaluate_predictions
from opd_sql.bird_evaluation import main as evaluate_cli, resolve_db_paths
from opd_sql.sqlite_tools import execute_readonly, get_schema


class SQLiteEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / "space and # mark.sqlite"
        with sqlite3.connect(self.db) as connection:
            connection.execute("CREATE TABLE sales (region TEXT, amount INTEGER)")
            connection.executemany("INSERT INTO sales VALUES (?, ?)",
                                   [("east", 10), ("east", 10), ("west", 30)])

    def record(self, record_id="q1", gold_sql="SELECT SUM(amount) FROM sales"):
        return {"id": record_id, "db_id": "sales", "question": "total?",
                "evidence": "", "gold_sql": gold_sql, "db_path": str(self.db),
                "schema": get_schema(self.db)}

    def test_correct_and_executable_but_incorrect(self):
        report = evaluate_predictions(
            [self.record("right"), self.record("wrong")],
            [{"id": "wrong", "sql": "SELECT MAX(amount) FROM sales"},
             {"id": "right", "sql": "SELECT 50 AS answer"}],
        )
        self.assertEqual(report["summary"]["accuracy"], 0.5)
        self.assertEqual(report["summary"]["execution_success_rate"], 1.0)
        self.assertEqual(report["results"][1]["status"], "incorrect")

    def test_row_order_ignored_but_duplicates_preserved(self):
        record = self.record(gold_sql="SELECT amount FROM sales ORDER BY amount")
        reversed_rows = evaluate_predictions([record], [
            {"id": "q1", "sql": "SELECT amount FROM sales ORDER BY amount DESC"}])
        distinct_rows = evaluate_predictions([record], [
            {"id": "q1", "sql": "SELECT DISTINCT amount FROM sales"}])
        self.assertTrue(reversed_rows["results"][0]["correct"])
        self.assertFalse(distinct_rows["results"][0]["correct"])
        self.assertTrue(distinct_rows["results"][0]["bird_correct"])
        self.assertEqual(distinct_rows["summary"]["bird_execution_accuracy"], 1.0)

    def test_writes_attach_pragma_and_extensions_rejected(self):
        for sql in (
            "DELETE FROM sales", "CREATE TABLE attack (x)",
            "ATTACH DATABASE ':memory:' AS extra", "PRAGMA query_only = OFF",
            "SELECT load_extension('attack')", "VACUUM", "BEGIN",
        ):
            with self.subTest(sql=sql):
                self.assertEqual(execute_readonly(self.db, sql)["status"], "error")
        result = execute_readonly(self.db, "SELECT COUNT(*) FROM sales")
        self.assertEqual(result["rows"], [[3]])

    def test_limit_does_not_return_truncated_rows(self):
        result = execute_readonly(self.db, "SELECT amount FROM sales", max_rows=2)
        self.assertEqual(result["status"], "row_limit")
        self.assertIsNone(result["rows"])
        report = evaluate_predictions(
            [self.record(gold_sql="SELECT amount FROM sales")],
            [{"id": "q1", "sql": "SELECT amount FROM sales"}], max_rows=2)
        self.assertEqual(report["summary"]["correct"], 0)
        self.assertEqual(report["summary"]["invalid_gold"], 1)

    def test_vm_loop_timeout(self):
        sql = ("WITH RECURSIVE numbers(n) AS (SELECT 1 UNION ALL "
               "SELECT n+1 FROM numbers WHERE n < 1000000000) "
               "SELECT SUM(n) FROM numbers")
        result = execute_readonly(self.db, sql, timeout_seconds=0.03)
        self.assertEqual(result["status"], "timeout")
        self.assertIsNone(result["rows"])

    def test_hard_process_deadline_and_bounded_result(self):
        result = execute_readonly(self.db, "SELECT 1", timeout_seconds=0.000001)
        self.assertEqual(result["status"], "timeout")
        self.assertTrue(result["hard_timeout"])
        result = execute_readonly(self.db, "SELECT zeroblob(1024)", max_result_bytes=128)
        self.assertEqual(result["status"], "result_limit")
        self.assertIsNone(result["rows"])
        self.assertLess(len(json.dumps(result)), 1000)

    def test_missing_duplicate_and_unknown_ids_do_not_inflate_accuracy(self):
        report = evaluate_predictions(
            [self.record("missing"), self.record("duplicate"), self.record("correct")],
            [{"id": "duplicate", "sql": "SELECT 50"},
             {"id": "duplicate", "sql": "SELECT 50"},
             {"id": "correct", "sql": "SELECT 50"},
             {"id": "unknown", "sql": "SELECT 50"}, {"sql": "SELECT 50"}],
        )
        self.assertEqual(report["summary"]["total"], 3)
        self.assertEqual(report["summary"]["accuracy"], 1 / 3)
        self.assertEqual(report["summary"]["missing_predictions"], 1)
        self.assertEqual(report["summary"]["duplicate_predictions"], 1)
        self.assertEqual(report["diagnostics"]["unknown_prediction_ids"], ["unknown"])
        self.assertEqual(report["diagnostics"]["malformed_prediction_indices"], [4])

    def test_invalid_gold_never_counts_as_correct(self):
        report = evaluate_predictions(
            [self.record(gold_sql="SELECT nonexistent FROM sales")],
            [{"id": "q1", "sql": "SELECT nonexistent FROM sales"}])
        self.assertEqual(report["summary"]["accuracy"], 0)
        self.assertEqual(report["summary"]["invalid_gold"], 1)
        self.assertIsNone(report["summary"]["accuracy_on_valid_gold"])

    def test_duplicate_reference_ids_raise(self):
        with self.assertRaisesRegex(ValueError, "Duplicate reference"):
            evaluate_predictions([self.record(), self.record()], [])

    def test_json_serializable_blob_and_schema(self):
        result = execute_readonly(self.db, "SELECT X'00FF', NULL, 1, 1.5")
        self.assertEqual(result["status"], "ok")
        json.dumps(result, allow_nan=False)
        self.assertIn("CREATE TABLE sales", get_schema(self.db))

    def test_multiple_statements_rejected_and_missing_db_not_created(self):
        result = execute_readonly(self.db, "SELECT 1; DELETE FROM sales")
        self.assertEqual(result["status"], "error")
        missing = self.db.parent / "missing.sqlite"
        self.assertEqual(execute_readonly(missing, "SELECT 1")["status"], "error")
        self.assertFalse(missing.exists())

    def test_empty_results_and_column_order(self):
        empty = self.record(gold_sql="SELECT amount FROM sales WHERE 0")
        report = evaluate_predictions([empty], [
            {"id": "q1", "sql": "SELECT amount FROM sales WHERE amount < 0"}])
        self.assertEqual(report["summary"]["accuracy"], 1.0)
        report = evaluate_predictions([empty], [
            {"id": "q1", "sql": "SELECT region, amount FROM sales WHERE 0"}])
        self.assertEqual(report["summary"]["accuracy"], 0.0)
        self.assertEqual(report["summary"]["bird_execution_accuracy"], 1.0)
        columns = self.record(gold_sql="SELECT 1, 2")
        report = evaluate_predictions([columns], [{"id": "q1", "sql": "SELECT 2, 1"}])
        self.assertEqual(report["summary"]["accuracy"], 0.0)
        self.assertEqual(report["summary"]["bird_execution_accuracy"], 0.0)

    def test_failed_gold_missing_and_duplicate_still_count_inputs(self):
        records = [self.record("missing", "SELECT nonexistent"),
                   self.record("duplicate", "SELECT nonexistent")]
        report = evaluate_predictions(records, [
            {"id": "duplicate", "sql": "SELECT 1"},
            {"id": "duplicate", "sql": "SELECT 1"},
        ])
        self.assertEqual(report["summary"]["missing_predictions"], 1)
        self.assertEqual(report["summary"]["duplicate_predictions"], 1)
        self.assertEqual(report["summary"]["bird_correct"], 0)

    def test_gold_timeout_and_missing_database_keep_fixed_denominator(self):
        # Even an apparently identical prediction cannot convert failed gold to a pass.
        loop = ("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n) "
                "SELECT SUM(x) FROM n")
        record = self.record(gold_sql=loop)
        report = evaluate_predictions([record], [{"id": "q1", "sql": loop}],
                                      timeout_seconds=0.2)
        self.assertEqual(report["summary"]["total"], 1)
        self.assertEqual(report["summary"]["invalid_gold"], 1)
        self.assertEqual(report["summary"]["bird_execution_accuracy"], 0)
        self.assertEqual(report["summary"]["gold_execution_status_counts"], {"timeout": 1})
        moved = resolve_db_paths([self.record()], self.db.parent / "no_databases")
        report = evaluate_predictions(moved, [{"id": "q1", "sql": "SELECT 50"}])
        self.assertEqual(report["summary"]["invalid_gold"], 1)
        self.assertEqual(report["summary"]["bird_correct"], 0)
        self.assertFalse(Path(moved[0]["db_path"]).exists())

    def test_generation_error_markers_reject_retained_sql_but_length_is_scored(self):
        predictions = [
            {"id": "top_error", "sql": "SELECT 50", "error": "failed"},
            {"id": "finish_error", "sql": "SELECT 50", "finish_reason": "error"},
            {"id": "attempt_error", "sql": "SELECT 50",
             "attempts": [{"status": "generation_error", "execution_error": "failed"}]},
            {"id": "length", "sql": "SELECT 50", "finish_reason": "length"},
        ]
        report = evaluate_predictions([self.record(item["id"]) for item in predictions], predictions)
        self.assertEqual(report["summary"]["bird_correct"], 1)
        self.assertEqual(report["summary"]["generation_failures"], 3)
        self.assertEqual(report["summary"]["generation_length_limits"], 1)
        self.assertTrue(report["results"][-1]["bird_correct"])
        self.assertTrue(all(item["prediction_execution"] is None for item in report["results"][:3]))

    def test_paired_breakdowns_generation_failure_and_cli_relocation(self):
        records = [dict(self.record("simple"), difficulty="simple"),
                   dict(self.record("moderate"), difficulty="moderate")]
        student_predictions = [{"id": "simple", "sql": "SELECT 50"},
                               {"id": "moderate", "sql": "SELECT 50", "error": "generation failed"}]
        teacher_predictions = [{"id": "simple", "sql": "SELECT 40"},
                               {"id": "moderate", "sql": "SELECT 50"}]
        student = evaluate_predictions(records, student_predictions)
        teacher = evaluate_predictions(records, teacher_predictions)
        paired = compare_reports(student, teacher)
        self.assertEqual(paired["summary"]["student_only"], 1)
        self.assertEqual(paired["summary"]["teacher_only"], 1)
        self.assertEqual(student["summary"]["bird_execution_accuracy"], 0.5)
        self.assertEqual(student["breakdowns"]["difficulty"]["moderate"]["bird_correct"], 0)
        with self.assertRaisesRegex(ValueError, "same reference ids"):
            compare_reports(student, evaluate_predictions(records[:1], teacher_predictions))
        root = self.db.parent / "relocated"
        relocated_db = root / "sales" / "sales.sqlite"
        relocated_db.parent.mkdir(parents=True)
        relocated_db.write_bytes(self.db.read_bytes())
        moved = resolve_db_paths(records, root)
        self.assertEqual(moved[0]["db_path"], str(relocated_db.resolve()))
        with self.assertRaisesRegex(ValueError, "Invalid db_id"):
            resolve_db_paths([{**records[0], "db_id": "../sales"}], root)
        files = {}
        for key, values in (("records", records), ("student", student_predictions),
                            ("teacher", teacher_predictions)):
            path = self.db.parent / (key + ".jsonl")
            path.write_text("".join(json.dumps(item) + "\n" for item in values), encoding="utf-8")
            files[key] = path
        output = self.db.parent / "evaluation.json"
        report = evaluate_cli(["--records", str(files["records"]),
                               "--predictions", str(files["student"]),
                               "--comparison-predictions", str(files["teacher"]),
                               "--db-root", str(root), "--output", str(output)])
        self.assertEqual(report["comparison"]["summary"]["total"], 2)
        self.assertEqual(len(report["input_files"]["records"]["sha256"]), 64)
        self.assertEqual(json.loads(output.read_text())["summary"]["bird_correct"], 1)


if __name__ == "__main__":
    unittest.main()
