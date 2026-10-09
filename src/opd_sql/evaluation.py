"""Safe local execution evaluation with two explicitly named comparisons.

Rows are compared as multisets: row order and column names are ignored, column
order and duplicate row counts are preserved. Numeric 1 and 1.0 compare equal;
no string coercion, floating tolerance, or database perturbation is performed.
Consequently this metric can miss ordering errors and queries that accidentally
produce the right answer on one database instance.

The additional BIRD comparison faithfully uses set(predicted_rows) ==
set(gold_rows), including its duplicate / empty-result semantics. Execution is
still our bounded read-only runner, not the unmodified official harness.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .sqlite_tools import execute_readonly


BIRD_EVALUATOR_COMMIT = "188835d4f9948563a6b9c8ac50cd0f3ae4021ed6"
BIRD_EVALUATOR_URL = (
    "https://github.com/AlibabaResearch/DAMO-ConvAI/blob/"
    + BIRD_EVALUATOR_COMMIT + "/bird/llm/src/evaluation.py"
)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return ("sqlite_tag", tuple(sorted((key, _freeze(val)) for key, val in value.items())))
    if isinstance(value, list):
        return tuple(_freeze(cell) for cell in value)
    return value


def _same_rows(left: list[list[Any]], right: list[list[Any]]) -> bool:
    return Counter(_freeze(row) for row in left) == Counter(_freeze(row) for row in right)


def _same_row_sets(left: list[list[Any]], right: list[list[Any]]) -> bool:
    return set(_freeze(row) for row in left) == set(_freeze(row) for row in right)


def _execution_metadata(execution: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in execution.items() if key != "rows"}


def _generation_failure(prediction: dict[str, Any]) -> str | None:
    """Reject failed generation even if a caller accidentally retained old SQL."""
    if prediction.get("error"):
        return "error_field"
    if prediction.get("finish_reason") == "error":
        return "finish_reason_error"
    attempts = prediction.get("attempts")
    if isinstance(attempts, list) and attempts and isinstance(attempts[-1], dict):
        final = attempts[-1]
        if final.get("status") == "generation_error" or final.get("finish_reason") == "error":
            return "final_attempt_generation_error"
    return None


def evaluate_predictions(records: list[dict[str, Any]], predictions: list[dict[str, Any]],
                         *, timeout_seconds: float = 5,
                         max_rows: int = 10000,
                         max_result_bytes: int = 8 * 1024 * 1024) -> dict[str, Any]:
    """Evaluate by id with an explicit, fixed denominator and input diagnostics.

    All reference records count in ``summary.accuracy``. Missing/duplicate or
    invalid predictions and invalid gold queries cannot score as correct.
    ``accuracy_on_valid_gold`` excludes invalid reference queries but retains
    missing predictions. Unknown prediction ids are reported, never matched by
    position. Duplicate or malformed *reference* ids raise ValueError because
    the benchmark membership would otherwise be ambiguous.
    """
    ids: set[str] = set()
    for index, record in enumerate(records):
        record_id = record.get("id") if isinstance(record, dict) else None
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"Reference record {index} requires a non-empty string id")
        if record_id in ids:
            raise ValueError(f"Duplicate reference id: {record_id}")
        ids.add(record_id)

    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    malformed: list[int] = []
    for index, prediction in enumerate(predictions):
        prediction_id = prediction.get("id") if isinstance(prediction, dict) else None
        if not isinstance(prediction_id, str) or not prediction_id:
            malformed.append(index)
        else:
            by_id[prediction_id].append(prediction)

    diagnostics = {
        "unknown_prediction_ids": sorted(set(by_id) - ids),
        "duplicate_prediction_ids": sorted(key for key, vals in by_id.items() if len(vals) > 1),
        "malformed_prediction_indices": malformed,
    }
    results: list[dict[str, Any]] = []
    for record in records:
        record_id = record["id"]
        gold = execute_readonly(record.get("db_path"), record.get("gold_sql"),
                                timeout_seconds, max_rows, max_result_bytes)
        gold_ok = gold["status"] == "ok"
        item: dict[str, Any] = {
            "id": record_id, "db_id": record.get("db_id"),
            "difficulty": record.get("difficulty") or "unknown",
            "correct": False, "bird_correct": False, "status": "missing_prediction",
            "prediction_input_status": "missing",
            "gold_execution": _execution_metadata(gold),
            "prediction_execution": None, "attempt_count": 0,
            "latency_seconds": None, "generation_failure_reason": None,
        }
        matches = by_id.get(record_id, [])
        if len(matches) > 1:
            item["status"] = "duplicate_prediction"
            item["prediction_input_status"] = "duplicate"
        elif matches:
            item["prediction_input_status"] = "present"
            prediction = matches[0]
            attempts = prediction.get("attempts")
            item["attempt_count"] = len(attempts) if isinstance(attempts, list) else 0
            item["latency_seconds"] = prediction.get("latency_seconds")
            for key in ("model", "input_tokens", "output_tokens", "generation_seconds",
                        "finish_reason", "error"):
                item[key] = prediction.get(key)
            failure = _generation_failure(prediction)
            if failure:
                item["generation_failure_reason"] = failure
                item["status"] = "prediction_generation_error"
            else:
                execution = execute_readonly(record.get("db_path"), prediction.get("sql"),
                                             timeout_seconds, max_rows, max_result_bytes)
                item["prediction_execution"] = _execution_metadata(execution)
                if execution["status"] == "ok":
                    item["correct"] = (
                        gold_ok
                        and len(gold["columns"]) == len(execution["columns"])
                        and _same_rows(gold["rows"], execution["rows"])
                    )
                    item["bird_correct"] = gold_ok and _same_row_sets(
                        gold["rows"], execution["rows"])
                    item["status"] = "correct" if item["correct"] else "incorrect"
                else:
                    item["status"] = "prediction_" + execution["status"]
        if not gold_ok:
            item["status"] = "invalid_gold"
        results.append(item)

    summary = _summarize(results)
    summary.update(unknown_prediction_ids=len(diagnostics["unknown_prediction_ids"]),
                   malformed_predictions=len(malformed))
    breakdowns: dict[str, Any] = {}
    for dimension in ("difficulty", "db_id"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in results:
            groups[str(item[dimension] or "unknown")].append(item)
        breakdowns[dimension] = {key: _summarize(group) for key, group in sorted(groups.items())}
    return {
        "metric": "local_sqlite_execution_row_multiset",
        "metric_notes": (
            "Not official BIRD evaluation. Ignores row order and column aliases; "
            "preserves column order and duplicate rows. No answer tolerance. "
            "Main accuracy denominator includes every reference record."
        ),
        "additional_metric": {
            "name": "bird_execution_set_equality",
            "official_comparison": "set(predicted_res) == set(ground_truth_res)",
            "official_source": BIRD_EVALUATOR_URL,
            "official_harness": False,
            "notes": "Ignores duplicate rows, row order and column aliases. Preserves "
                     "column order for non-empty rows. Empty results compare equal even "
                     "with different column counts. Bounded read-only execution differs "
                     "from the official harness; see docs/evaluation-protocol.md.",
        },
        "execution_limits": {"timeout_seconds": timeout_seconds, "max_rows": max_rows,
                             "max_result_bytes": max_result_bytes,
                             "timeout_scope": "per query subprocess including startup and IPC"},
        "summary": summary,
        "breakdowns": breakdowns,
        "diagnostics": diagnostics,
        "results": results,
    }


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    correct = sum(item["correct"] for item in results)
    bird_correct = sum(item["bird_correct"] for item in results)
    valid_gold = sum(item["gold_execution"]["status"] == "ok" for item in results)
    executable = sum(bool(item["prediction_execution"]
                          and item["prediction_execution"]["status"] == "ok") for item in results)
    return {
        "total": total, "correct": correct,
        "accuracy": correct / total if total else None,
        "bird_correct": bird_correct,
        "bird_execution_accuracy": bird_correct / total if total else None,
        "valid_gold": valid_gold, "invalid_gold": total - valid_gold,
        "accuracy_on_valid_gold": correct / valid_gold if valid_gold else None,
        "bird_accuracy_on_valid_gold": bird_correct / valid_gold if valid_gold else None,
        "executable_predictions": executable,
        "execution_success_rate": executable / total if total else None,
        "missing_predictions": sum(item["prediction_input_status"] == "missing" for item in results),
        "duplicate_predictions": sum(item["prediction_input_status"] == "duplicate" for item in results),
        "generation_failures": sum(item["generation_failure_reason"] is not None for item in results),
        "generation_length_limits": sum(item.get("finish_reason") == "length" for item in results),
        "status_counts": dict(Counter(item["status"] for item in results)),
        "gold_execution_status_counts": dict(Counter(item["gold_execution"]["status"] for item in results)),
        "prediction_execution_status_counts": dict(Counter(
            item["prediction_execution"]["status"] if item["prediction_execution"] else "not_executed"
            for item in results)),
    }


def compare_reports(student: dict[str, Any], teacher: dict[str, Any]) -> dict[str, Any]:
    """Pair every reference id, retaining the same fixed denominator."""
    left = {item["id"]: item for item in student["results"]}
    right = {item["id"]: item for item in teacher["results"]}
    if set(left) != set(right):
        raise ValueError("Paired reports must evaluate exactly the same reference ids")
    paired = []
    for record_id, first in left.items():
        second = right[record_id]
        a, b = first["bird_correct"], second["bird_correct"]
        paired.append({"id": record_id, "db_id": first["db_id"],
                       "difficulty": first["difficulty"], "student_correct": a,
                       "teacher_correct": b,
                       "outcome": ("both_correct" if a and b else "student_only" if a
                                   else "teacher_only" if b else "both_incorrect")})
    def counts(items):
        categories = Counter(item["outcome"] for item in items)
        total = len(items)
        gain = categories["teacher_only"] - categories["student_only"]
        return {"total": total, **{key: categories[key] for key in (
            "both_correct", "student_only", "teacher_only", "both_incorrect")},
            "teacher_minus_student_accuracy": gain / total if total else None}
    breakdowns = {}
    for dimension in ("difficulty", "db_id"):
        groups = defaultdict(list)
        for item in paired:
            groups[str(item[dimension] or "unknown")].append(item)
        breakdowns[dimension] = {key: counts(group) for key, group in sorted(groups.items())}
    return {"metric": "bird_execution_set_equality", "summary": counts(paired),
            "breakdowns": breakdowns, "results": paired,
            "notes": "Fixed reference denominator; all failures count as incorrect. "
                     "Descriptive paired comparison, not a significance test."}
