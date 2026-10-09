"""Run a synthetic SQLite evaluation demo without models, GPUs or downloads.

Install the package first: python -m pip install -e .
The SQL predictions below are handwritten fixtures, not model generations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from opd_sql.evaluation import evaluate_predictions
from opd_sql.sqlite_tools import execute_readonly


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "work" / "demo",
        help="New directory for the toy database and reports; existing directories are refused",
    )
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite {output}; choose a new --output-dir")
    output.mkdir(parents=True, exist_ok=False)

    database = output / "sales.sqlite"
    schema = "CREATE TABLE sales (region TEXT, amount INTEGER);"
    with sqlite3.connect(database) as connection:
        connection.execute(schema)
        connection.executemany(
            "INSERT INTO sales VALUES (?, ?)",
            [("east", 10), ("east", 10), ("west", 30)],
        )

    questions = [
        ("count", "How many sales rows are there?", "SELECT COUNT(*) FROM sales"),
        ("total", "What is the total sales amount?", "SELECT SUM(amount) FROM sales"),
        ("amounts", "Return the amount from each sales row.", "SELECT amount FROM sales"),
    ]
    records = [
        {"id": f"synthetic:{name}", "db_id": "sales", "question": question,
         "evidence": "", "gold_sql": sql, "schema": schema,
         "db_path": str(database), "difficulty": "synthetic", "source_split": "synthetic"}
        for name, question, sql in questions
    ]
    reference_predictions = [
        {"id": "synthetic:count", "sql": "SELECT COUNT(*) FROM sales"},
        {"id": "synthetic:total", "sql": "SELECT MAX(amount) FROM sales"},
        {"id": "synthetic:amounts", "sql": "SELECT DISTINCT amount FROM sales"},
    ]
    candidate_predictions = [{"id": row["id"], "sql": row["gold_sql"]} for row in records]
    write_jsonl(output / "records.jsonl", records)
    write_jsonl(output / "reference-predictions.jsonl", reference_predictions)
    write_jsonl(output / "candidate-predictions.jsonl", candidate_predictions)

    reports = {}
    print("SYNTHETIC DEMO: handwritten SQL fixtures; no SFT/OPD model was trained or run.")
    print("Rows: ('east', 10), ('east', 10), ('west', 30)")
    for label, predictions in (("reference", reference_predictions), ("candidate", candidate_predictions)):
        report = evaluate_predictions(records, predictions)
        report["demo_metadata"] = {
            "synthetic": True,
            "prediction_origin": "handwritten fixtures",
            "model_inference_performed": False,
            "training_performed": False,
        }
        reports[label] = report
        with (output / f"{label}-report.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        summary = report["summary"]
        print(f"\n{label}: set equality {summary['bird_correct']}/{summary['total']}; "
              f"row multiset {summary['correct']}/{summary['total']}")
        for prediction in predictions:
            result = execute_readonly(database, prediction["sql"])
            print(f"  {prediction['id']}: {prediction['sql']} -> {result['rows']}")

    reference_rows = {row["id"]: row for row in reports["reference"]["results"]}
    gained = [row["id"] for row in reports["candidate"]["results"]
              if row["bird_correct"] and not reference_rows[row["id"]]["bird_correct"]]
    lost = [row["id"] for row in reports["candidate"]["results"]
            if not row["bird_correct"] and reference_rows[row["id"]]["bird_correct"]]
    print(f"\nToy paired comparison: gained={len(gained)}, lost={len(lost)}, net={len(gained) - len(lost)}")
    print("Set equality ignores duplicate rows; the row-multiset metric preserves them.")
    print(f"Artifacts: {output}")
    print("Evaluate these predictions again with the package CLI (use a new report path):")
    print(f'python -m opd_sql.bird_evaluation --records "{output / "records.jsonl"}" '
          f'--predictions "{output / "candidate-predictions.jsonl"}" '
          f'--output "{output / "cli-report.json"}"')


if __name__ == "__main__":
    main()
