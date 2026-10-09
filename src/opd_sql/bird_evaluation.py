"""Evaluate portable BIRD JSONL predictions and optional teacher/student pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .evaluation import compare_reports, evaluate_predictions


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_db_paths(records: list[dict], db_root: str | Path | None) -> list[dict]:
    """A root override maps db_id to <root>/<db_id>/<db_id>.sqlite."""
    if db_root is None:
        return records
    root = Path(db_root).expanduser().resolve()
    relocated = []
    for record in records:
        db_id = record.get("db_id")
        if (not isinstance(db_id, str) or not db_id or db_id in (".", "..")
                or any(char in db_id for char in "/\\:")):
            raise ValueError(f"Invalid db_id for root relocation: {db_id!r}")
        relocated.append({**record, "db_path": str(root / db_id / (db_id + ".sqlite"))})
    return relocated


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--predictions", required=True, help="Student predictions JSONL")
    parser.add_argument("--comparison-predictions", help="Teacher predictions on identical records")
    parser.add_argument("--output", required=True)
    parser.add_argument("--db-root")
    parser.add_argument("--timeout-seconds", type=float, default=30)
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--max-result-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; use a new path or --overwrite")
    records = resolve_db_paths(read_jsonl(args.records), args.db_root)
    if not records:
        raise ValueError("Reference records must not be empty")
    kwargs = {"timeout_seconds": args.timeout_seconds, "max_rows": args.max_rows,
              "max_result_bytes": args.max_result_bytes}
    report = evaluate_predictions(records, read_jsonl(args.predictions), **kwargs)
    sources = {"records": args.records, "predictions": args.predictions}
    if args.comparison_predictions:
        teacher = evaluate_predictions(records, read_jsonl(args.comparison_predictions), **kwargs)
        report["comparison"] = compare_reports(report, teacher)
        report["teacher_report"] = teacher
        sources["comparison_predictions"] = args.comparison_predictions
    report["input_files"] = {
        key: {"path": str(Path(path).resolve()),
              "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        for key, path in sources.items()
    }
    report["reference_id_sha256"] = hashlib.sha256(
        json.dumps([record["id"] for record in records], separators=(",", ":")).encode()).hexdigest()
    report["db_root_override"] = str(Path(args.db_root).resolve()) if args.db_root else None
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                      encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "summary": report["summary"]},
                     ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
