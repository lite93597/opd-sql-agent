"""Prepare official BIRD train/dev without moving dev examples into training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import shutil
import sqlite3
import zipfile
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from download_bird import digest_file, write_json

OFFICIAL_REPO_COMMIT = "188835d4f9948563a6b9c8ac50cd0f3ae4021ed6"


def is_metadata(path: Path) -> bool:
    return path.name.startswith("._") or "__MACOSX" in path.parts


def extracted_matches_archive(archive_path: Path, destination: Path) -> bool:
    """Reuse a fully extracted outer ZIP after a nested-stage failure, verifying CRCs."""
    with zipfile.ZipFile(archive_path) as archive:
        base = destination.resolve()
        for info in archive.infolist():
            if info.is_dir() or is_metadata(Path(info.filename)):
                continue
            target = (destination / info.filename).resolve()
            if base not in target.parents or not target.is_file() or target.stat().st_size != info.file_size:
                return False
            crc = 0
            with target.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    crc = zlib.crc32(chunk, crc)
            if crc & 0xFFFFFFFF != info.CRC:
                return False
    return True


def extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        # Outer and nested archives are both CRC checked before trusting extraction.
        bad = archive.testzip()
        if bad:
            raise ValueError(f"ZIP CRC failed: {archive_path}: {bad}")
        base = destination.resolve()
        members = [info for info in archive.infolist() if not is_metadata(Path(info.filename))]
        for info in members:
            target = (destination / info.filename).resolve()
            if target != base and base not in target.parents:
                raise ValueError(f"Unsafe archive path: {info.filename}")
        unpacked = sum(info.file_size for info in members)
        free = shutil.disk_usage(destination).free
        # Conservatively reserve every member's full size, even on a partial rerun.
        if free < unpacked + 1024 ** 3:
            raise ValueError(f"Insufficient extraction space: {archive_path}: need {unpacked} + 1 GiB, free {free}")
        archive.extractall(destination, members=members)


def database_schema(db_path: Path) -> tuple[str, dict]:
    """Use every user table/view DDL, PRAGMA metadata and original column descriptions."""
    connection = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        check = connection.execute("PRAGMA quick_check").fetchall()
        if check != [("ok",)]:
            raise ValueError(f"SQLite quick_check failed: {db_path}: {check}")
        objects = connection.execute(
            "SELECT type,name,sql FROM sqlite_master WHERE type IN ('table','view') "
            "AND name NOT LIKE 'sqlite_%' ORDER BY type,name").fetchall()
        sections, tables = [], {}
        for object_type, name, ddl in objects:
            if not ddl:
                raise ValueError(f"Missing DDL for {db_path.name}/{name}")
            quoted = '"' + name.replace('"', '""') + '"'
            columns = connection.execute(f"PRAGMA table_xinfo({quoted})").fetchall()
            keys = connection.execute(f"PRAGMA foreign_key_list({quoted})").fetchall()
            tables[name] = {"type": object_type, "columns": columns, "foreign_keys": keys}
            sections.append(ddl.rstrip(";") + ";")
        description_dir = db_path.parent / "database_description"
        descriptions = {}
        if description_dir.exists():
            for path in sorted(description_dir.glob("*.csv")):
                if is_metadata(path):
                    continue
                # Official files include UTF-8 and legacy byte sequences.
                raw = path.read_bytes()
                try:
                    text = raw.decode("utf-8-sig")
                except UnicodeDecodeError:
                    text = raw.decode("cp1252")
                rows = list(csv.DictReader(io.StringIO(text)))
                descriptions[path.stem] = rows
        # Structured comments cannot execute; the original descriptions help resolve column meanings.
        if descriptions:
            sections.append("\n-- Column descriptions (official BIRD database_description):")
            for table, rows in descriptions.items():
                sections.append("-- " + json.dumps({"table": table, "columns": rows}, ensure_ascii=False).replace("\n", "\\n"))
        return "\n\n".join(sections), {"tables": tables, "descriptions": descriptions}
    finally:
        connection.close()


def prepare_split(root: Path, split: str, db_path_root: Path | None) -> tuple[list[dict], dict] | None:
    archive_path = root / "raw" / f"{split}.zip"
    status_path = root / "work" / f"{split}-download.json"
    if not archive_path.exists():
        return None
    if not status_path.exists():
        raise ValueError(f"Run download_bird.py to verify {split}.zip first")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "complete" or not status.get("zip_crc_verified"):
        raise ValueError(f"{split} archive has not completed verified downloading")
    sha, _ = digest_file(archive_path)
    if sha != status["sha256"]:
        raise ValueError(f"{split}.zip changed since download verification")
    extracted = root / "extracted" / split
    marker = extracted / ".archive-sha256"
    if not marker.exists() or marker.read_text().strip() != sha:
        outer_marker = extracted / ".outer-archive-sha256"
        outer_done = outer_marker.exists() and outer_marker.read_text().strip() == sha
        if not outer_done:
            if not extracted_matches_archive(archive_path, extracted):
                extract(archive_path, extracted)
            outer_marker.write_text(sha + "\n", encoding="ascii")
        for nested in sorted(extracted.rglob("*.zip")):
            # macOS resource forks are metadata, not ZIP archives even when their
            # names end in .zip (e.g. __MACOSX/train/._train_databases.zip).
            if is_metadata(nested.relative_to(extracted)):
                continue
            extract(nested, nested.parent)
        marker.write_text(sha + "\n", encoding="ascii")
    data_paths = sorted(path for path in extracted.rglob(f"{split}.json") if not is_metadata(path.relative_to(extracted)))
    if len(data_paths) != 1:
        raise ValueError(f"Expected exactly one {split}.json, found {data_paths}")
    data_path = data_paths[0]
    rows = json.loads(data_path.read_text(encoding="utf-8"))
    db_files = {path.stem: path for path in extracted.rglob("*.sqlite") if not is_metadata(path.relative_to(extracted))}
    required_dbs = sorted({row["db_id"] for row in rows})
    missing = set(required_dbs) - db_files.keys()
    if missing:
        raise ValueError(f"Missing {split} databases: {sorted(missing)}")
    schemas, schema_report = {}, {}
    for db_id in required_dbs:
        schema, metadata = database_schema(db_files[db_id])
        schemas[db_id] = schema
        db_sha, _ = digest_file(db_files[db_id])
        schema_report[db_id] = {"sqlite_sha256": db_sha, "bytes": db_files[db_id].stat().st_size,
                                "quick_check": "ok", "table_count": len(metadata["tables"]),
                                "description_files": len(metadata["descriptions"]),
                                "schema_characters": len(schema),
                                "schema_utf8_bytes": len(schema.encode("utf-8")),
                                "ddl_utf8_bytes": len(schema.split("-- Column descriptions", 1)[0].encode("utf-8")),
                                "relative_path": db_files[db_id].relative_to(root).as_posix()}
    records = []
    for index, row in enumerate(rows):
        relative_db = db_files[row["db_id"]].relative_to(root)
        path = ((db_path_root / relative_db).as_posix() if db_path_root else str(db_files[row["db_id"]].resolve()))
        records.append({"id": f"{split}:{row.get('question_id', index)}", "db_id": row["db_id"],
                        "question": row["question"], "evidence": row.get("evidence", ""),
                        "gold_sql": row["SQL"], "schema": schemas[row["db_id"]], "db_path": path,
                        "difficulty": row.get("difficulty", "unknown"), "source_split": split})
    if len({record["id"] for record in records}) != len(records):
        raise ValueError(f"Duplicate IDs in {split}")
    report = {"status": "complete", "count": len(records), "db_count": len(required_dbs),
              "db_ids": required_dbs, "difficulty_counts": dict(Counter(r["difficulty"] for r in records)),
              "archive": status, "version_directory": data_path.parent.name,
              "schema_utf8_bytes_mean": sum(len(r["schema"].encode("utf-8")) for r in records) / len(records),
              "schema_utf8_bytes_max": max(len(r["schema"].encode("utf-8")) for r in records),
              "databases": schema_report}
    return records, report


def stratified_sample(records: list[dict], count: int, seed: int) -> list[dict]:
    """Hamilton allocation over (db_id,difficulty); never inspect SQL or execution outcomes."""
    if count > len(records):
        raise ValueError("Requested sample exceeds available dev records")
    groups = defaultdict(list)
    for record in records:
        groups[(record["db_id"], record["difficulty"])].append(record)
    quotas = {key: len(group) * count // len(records) for key, group in groups.items()}
    remainder = count - sum(quotas.values())
    priority = sorted(groups, key=lambda key: (-(len(groups[key]) * count % len(records)), key))
    for key in priority[:remainder]:
        quotas[key] += 1
    rng, selected = random.Random(seed), []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda item: item["id"])
        rng.shuffle(group)
        selected.extend(group[:quotas[key]])
    return sorted(selected, key=lambda item: int(item["id"].split(":", 1)[1]))


def write_jsonl(path: Path, records: list[dict]) -> str:
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--db-path-root", type=Path, help="Override only recorded db paths, e.g. the server root")
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "processed"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"prepared_at": datetime.now(timezone.utc).isoformat(),
                "official_website": "https://bird-bench.github.io/",
                "license": "CC BY-SA 4.0",
                "license_source": f"https://github.com/AlibabaResearch/DAMO-ConvAI/blob/{OFFICIAL_REPO_COMMIT}/bird/README.md",
                "official_eval_url": f"https://github.com/AlibabaResearch/DAMO-ConvAI/blob/{OFFICIAL_REPO_COMMIT}/bird/llm/src/evaluation.py",
                "split_policy": "Official train only for training; official dev only for validation and baseline",
                "prompt_policy": "Question, evidence and complete schema only; gold_sql never enters model prompt",
                "splits": {}, "processed_sha256": {}}
    records_by_split = {}
    for split in ("train", "dev"):
        result = prepare_split(root, split, args.db_path_root)
        if result is None:
            manifest["splits"][split] = {"status": "pending", "count": 0, "db_ids": [],
                                         "reason": "Official archive not downloaded and verified; no substitute split"}
            continue
        records, report = result
        records_by_split[split] = records
        manifest["splits"][split] = report
        manifest["processed_sha256"][f"{split}.jsonl"] = write_jsonl(output / f"{split}.jsonl", records)
    if "dev" in records_by_split:
        selected = stratified_sample(records_by_split["dev"], args.sample_size, args.seed)
        filename = f"baseline-dev-{args.sample_size}.jsonl"
        manifest["processed_sha256"][filename] = write_jsonl(output / filename, selected)
        manifest["baseline_sample"] = {"source_split": "dev", "count": len(selected), "seed": args.seed,
                                       "method": "Hamilton proportional allocation over (db_id,difficulty), seeded shuffle",
                                       "selection_uses_gold_sql": False,
                                       "ids": [r["id"] for r in selected],
                                       "ids_sha256": hashlib.sha256("\n".join(r["id"] for r in selected).encode("utf-8")).hexdigest(),
                                       "difficulty_counts": dict(Counter(r["difficulty"] for r in selected)),
                                       "db_counts": dict(Counter(r["db_id"] for r in selected))}
    intersection = sorted(set(manifest["splits"]["train"]["db_ids"]) & set(manifest["splits"]["dev"]["db_ids"]))
    manifest["train_dev_db_intersection"] = intersection
    manifest["split_disjointness_verified"] = all(s["status"] == "complete" for s in manifest["splits"].values()) and not intersection
    if intersection:
        raise ValueError(f"Train/dev database overlap: {intersection}")
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"splits": {key: {field: value[field] for field in ("status", "count", "db_count") if field in value}
                                      for key, value in manifest["splits"].items()},
                      "baseline_sample": manifest.get("baseline_sample"), "manifest": str(output / "manifest.json")},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
