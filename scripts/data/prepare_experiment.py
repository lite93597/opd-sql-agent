"""Freeze database-disjoint train/validation and an untouched official-dev test set."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from collections import Counter

from prepare_bird import stratified_sample

SEED = 20261004
FIELDS = {"id", "db_id", "question", "evidence", "gold_sql", "schema", "db_path", "difficulty", "source_split"}


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def read_records(path: Path, source_split: str) -> list[dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    ids = set()
    for record in records:
        if FIELDS - record.keys():
            raise ValueError(f"Incomplete record in {path}: {record.get('id')}")
        if record["source_split"] != source_split or record.get("split", source_split) != source_split:
            raise ValueError(f"Expected unmodified official {source_split} record: {record['id']}")
        if not isinstance(record["id"], str) or record["id"] in ids:
            raise ValueError(f"Non-string or duplicate record ID: {record['id']}")
        prefix, numeric_id = record["id"].split(":", 1)
        if prefix != source_split or not numeric_id.isdigit():
            raise ValueError(f"Official record ID must be {source_split}:<integer>: {record['id']}")
        ids.add(record["id"])
    if not records:
        raise ValueError(f"No records in {path}")
    return sorted(records, key=lambda record: int(record["id"].split(":", 1)[1]))


def write_jsonl(path: Path, records: list[dict]) -> str:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return file_sha(path)


def label(records: list[dict], split: str, purpose: str) -> list[dict]:
    return [{**record, "split": split, "purpose": purpose} for record in records]


def describe(records: list[dict], filename: str, sha: str, purpose: str) -> dict:
    ids = [record["id"] for record in records]
    db_ids = sorted({record["db_id"] for record in records})
    return {"filename": filename, "sha256": sha, "count": len(records), "ids": ids,
            "record_ids_sha256": object_sha(ids), "db_ids": db_ids, "db_ids_sha256": object_sha(db_ids),
            "db_count": len(db_ids), "difficulty_counts": dict(Counter(r["difficulty"] for r in records)),
            "db_counts": dict(Counter(r["db_id"] for r in records)), "purpose": purpose}


def prepare(root: Path, output: Path | None = None, baseline_records: Path | None = None,
            validation_size: int = 120, heldout_size: int = 300) -> dict:
    root = root.resolve()
    processed = root / "processed"
    output = (output or processed / "experiment-v1").resolve()
    # Never overwrite historical files, and never replace a frozen experiment.
    if output.exists():
        raise FileExistsError(f"Frozen output already exists: {output}")
    baseline_records = baseline_records or processed / "baseline-dev-120.jsonl"
    source_manifest_path = processed / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if not source_manifest.get("split_disjointness_verified"):
        raise ValueError("Official train/dev preparation must first pass database disjointness")
    official, input_files = {}, {}
    for split in ("train", "dev"):
        report = source_manifest["splits"][split]
        path = processed / f"{split}.jsonl"
        sha = file_sha(path)
        if report["status"] != "complete" or sha != source_manifest["processed_sha256"][path.name]:
            raise ValueError(f"Official {split} source is incomplete or changed since preparation")
        rows = read_records(path, split)
        db_ids = sorted({row["db_id"] for row in rows})
        if len(rows) != report["count"] or db_ids != sorted(report["db_ids"]):
            raise ValueError(f"Official {split} records disagree with preparation manifest")
        for db_id in db_ids:
            if report["databases"][db_id].get("quick_check") != "ok":
                raise ValueError(f"Database is unverified: {split}/{db_id}")
        official[split] = rows
        input_files[split] = {"path": str(path), "sha256": sha, "archive_sha256": report["archive"]["sha256"]}
    train_dbs = sorted({row["db_id"] for row in official["train"]})
    dev_dbs = {row["db_id"] for row in official["dev"]}
    if set(train_dbs) & dev_dbs:
        raise ValueError("Official train/dev database overlap")
    if len(train_dbs) < 2:
        raise ValueError("Need at least two train databases for an internal holdout")
    shuffled_dbs = train_dbs.copy()
    random.Random(SEED).shuffle(shuffled_dbs)
    validation_db_count = max(1, min(len(train_dbs) - 1, math.floor(len(train_dbs) * 0.2 + 0.5)))
    validation_dbs = set(shuffled_dbs[:validation_db_count])
    train = [r for r in official["train"] if r["db_id"] not in validation_dbs]
    validation = [r for r in official["train"] if r["db_id"] in validation_dbs]
    sampled_validation = stratified_sample(validation, validation_size, SEED)
    baseline = read_records(baseline_records, "dev")
    baseline_ids = {r["id"] for r in baseline}
    expected_baseline_ids = set(source_manifest["baseline_sample"]["ids"])
    if len(baseline) != 120 or baseline_ids != expected_baseline_ids:
        raise ValueError("Historical baseline must contain the original fixed 120 IDs")
    all_dev_ids = {r["id"] for r in official["dev"]}
    if not baseline_ids <= all_dev_ids:
        raise ValueError("Historical baseline IDs are missing from official dev")
    heldout_pool = [r for r in official["dev"] if r["id"] not in baseline_ids]
    heldout = stratified_sample(heldout_pool, heldout_size, SEED)
    output_sets = {
        "internal_train.jsonl": label(train, "internal_train", "optimizer_updates_only"),
        "internal_validation.jsonl": label(validation, "internal_validation", "checkpoint_selection_only"),
        f"internal-validation-{validation_size}.jsonl": label(sampled_validation, "internal_validation", "checkpoint_selection_only"),
        f"dev-heldout-test-{heldout_size}.jsonl": label(heldout, "dev_heldout_test", "final_test_only_never_checkpoint_selection"),
    }
    sources_by_id = {r["id"]: r for rows in official.values() for r in rows}
    audit = []
    for filename, rows in output_sets.items():
        for row in rows:
            original = sources_by_id[row["id"]]
            db_report = source_manifest["splits"][row["source_split"]]["databases"][row["db_id"]]
            audit.append({"file": filename, "id": row["id"], "db_id": row["db_id"],
                          "source_split": row["source_split"], "split": row["split"],
                          "source_record_sha256": object_sha(original), "record_sha256": object_sha(row),
                          "source_sqlite_sha256": db_report["sqlite_sha256"],
                          "schema_sha256": hashlib.sha256(row["schema"].encode("utf-8")).hexdigest()})
    manifest = {"experiment": "experiment-v1", "seed": SEED, "status": "complete",
                "source_manifest": {"path": str(source_manifest_path), "sha256": file_sha(source_manifest_path)},
                "source_files": input_files,
                "excluded_baseline": {"path": str(baseline_records.resolve()), "sha256": file_sha(baseline_records),
                                      "ids": sorted(baseline_ids), "record_ids_sha256": object_sha(sorted(baseline_ids))},
                "database_split": {"method": "Sorted official train db_ids; Python Random(seed).shuffle; first nearest 20% are validation",
                                   "internal_train_db_ids": sorted(set(train_dbs) - validation_dbs),
                                   "internal_validation_db_ids": sorted(validation_dbs),
                                   "db_assignment_sha256": object_sha({db: ("internal_validation" if db in validation_dbs else "internal_train")
                                                                        for db in train_dbs})},
                "selection": {"method": "Hamilton allocation over (db_id,difficulty), seeded per-stratum shuffle",
                              "uses_gold_sql_or_model_feedback": False, "schema_truncated": False,
                              "train_difficulty_missing": all(r["difficulty"] == "unknown" for r in official["train"])},
                "isolation": {"all_official_train_records_assigned_exactly_once": len(train) + len(validation) == len(official["train"]),
                              "train_validation_database_overlap": [], "train_dev_database_overlap": [],
                              "dev_test_baseline_id_overlap": []},
                "test_policy": "No test SQL, execution feedback or model score may be used for training, hyperparameters or checkpoint selection; only final evaluation after checkpoint selection is frozen.",
                "files": {}, "databases": {split: report["databases"] for split, report in source_manifest["splits"].items()},
                "code_sha256": file_sha(Path(__file__)),
                "source_license": source_manifest.get("license"), "source_website": source_manifest.get("official_website")}
    # All integrity checks and sample-size checks happen before creating output.
    staging = output.with_name(output.name + ".incomplete")
    staging.mkdir(parents=True, exist_ok=False)
    for filename, rows in output_sets.items():
        sha = write_jsonl(staging / filename, rows)
        manifest["files"][filename] = describe(rows, filename, sha, rows[0]["purpose"])
    manifest["record_audit"] = {"filename": "record-audit.jsonl", "sha256": write_jsonl(staging / "record-audit.jsonl", audit),
                                "count": len(audit), "note": "Validation sample intentionally duplicates IDs from full internal-validation membership"}
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    staging.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-records", type=Path, help="Historical baseline snapshot; its IDs must match the original fixed 120")
    args = parser.parse_args()
    manifest = prepare(args.root, args.output, args.baseline_records)
    print(json.dumps({"experiment": manifest["experiment"], "files": {name: {k: report[k] for k in ("count", "db_count", "sha256")}
                       for name, report in manifest["files"].items()}, "database_split": manifest["database_split"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
