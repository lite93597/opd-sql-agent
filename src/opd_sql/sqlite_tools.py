"""Bounded, read-only SQLite execution for local Text-to-SQL experiments.

Each query runs in a disposable subprocess with a wall-clock deadline and a
bounded JSON result. This is a guard for SELECT queries against trusted local
database files, not an OS sandbox or a native memory limit.
"""

from __future__ import annotations

import math
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from typing import Any


_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}
_BLOCKED_FUNCTIONS = {"load_extension", "readfile", "writefile"}


def _authorize(action: int, arg1: str | None, arg2: str | None,
               database: str | None, trigger: str | None) -> int:
    if action not in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        function = (arg2 or arg1 or "").lower()
        if function in _BLOCKED_FUNCTIONS:
            return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _json_cell(value: Any) -> Any:
    """Preserve blob/non-finite values without emitting invalid JSON."""
    if isinstance(value, bytes):
        return {"__sqlite_blob_hex__": value.hex()}
    if isinstance(value, float) and not math.isfinite(value):
        return {"__sqlite_nonfinite__": str(value)}
    return value


def _execute_inprocess(db_path: str | Path, sql: str, timeout_seconds: float = 5,
                       max_rows: int = 10000,
                       max_result_bytes: int = 8 * 1024 * 1024) -> dict[str, Any]:
    """Worker implementation; callers use ``execute_readonly`` for isolation.

    A successful result has status ``ok`` and complete ``rows``. All other
    statuses have ``rows=None``; in particular a row cap never returns a
    truncated answer that the evaluator could mistakenly mark correct.

    ``timeout_seconds`` controls SQLite VM interruption and caps lock waiting.
    Read-only URI mode, query_only, and an authorizer independently restrict
    writes. ATTACH, PRAGMA, transactions, and extension loading are denied.
    """
    if (isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("timeout_seconds must be a positive finite number")
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
        raise ValueError("max_rows must be a positive integer")

    started = time.monotonic()
    connection: sqlite3.Connection | None = None
    timed_out = False
    result: dict[str, Any] = {
        "status": "error", "rows": None, "columns": [], "row_count": None,
        "error": None, "elapsed_seconds": 0.0,
    }

    def expired() -> bool:
        return time.monotonic() - started >= timeout_seconds

    def progress() -> int:
        nonlocal timed_out
        if expired():
            timed_out = True
            return 1
        return 0

    try:
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("SQL must be a non-empty string")
        path = Path(db_path).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError("Database path must name an existing file")
        connection = sqlite3.connect(
            path.as_uri() + "?mode=ro", uri=True, timeout=timeout_seconds,
        )
        connection.enable_load_extension(False)
        connection.execute("PRAGMA query_only = ON")
        # Millisecond SQLite lock timeout is separate from the VM callback.
        connection.execute(f"PRAGMA busy_timeout = {max(1, int(timeout_seconds * 1000))}")
        connection.set_authorizer(_authorize)
        connection.set_progress_handler(progress, 1000)
        cursor = connection.execute(sql)
        if cursor.description is None:
            raise ValueError("SQL must return a result set")
        result["columns"] = [column[0] for column in cursor.description]
        rows: list[list[Any]] = []
        result_bytes = len(json.dumps(result["columns"], ensure_ascii=True).encode("utf-8"))
        if result_bytes > max_result_bytes:
            result.update(status="result_limit", columns=[], error=(
                f"Columns exceed max_result_bytes={max_result_bytes}; result is not evaluable"))
            return result
        for row in cursor:
            if expired():
                timed_out = True
                raise TimeoutError("SQLite query exceeded the execution timeout")
            if len(rows) >= max_rows:
                result.update(
                    status="row_limit",
                    error=f"Query exceeds max_rows={max_rows}; result is not evaluable",
                    observed_rows=max_rows + 1,
                )
                break
            # Reject oversized cells before blob hex conversion / JSON copies.
            oversized = any(
                (isinstance(cell, bytes) and len(cell) * 2 > max_result_bytes)
                or (isinstance(cell, str) and len(cell) > max_result_bytes)
                for cell in row
            )
            converted = [] if oversized else [_json_cell(cell) for cell in row]
            row_bytes = (max_result_bytes + 1 if oversized else
                         len(json.dumps(converted, ensure_ascii=True).encode("utf-8")) + 2)
            result_bytes += row_bytes
            if result_bytes > max_result_bytes:
                result.update(status="result_limit", error=(
                    f"Query exceeds max_result_bytes={max_result_bytes}; result is not evaluable"))
                break
            rows.append(converted)
        else:
            if expired():
                timed_out = True
                raise TimeoutError("SQLite query exceeded the execution timeout")
            result.update(status="ok", rows=rows, row_count=len(rows))
    except (sqlite3.Error, sqlite3.Warning, OSError, ValueError, TypeError, TimeoutError) as exc:
        result["status"] = "timeout" if timed_out or expired() else "error"
        result["error"] = str(exc)[:2000]
    finally:
        if connection is not None:
            connection.close()
        result["elapsed_seconds"] = time.monotonic() - started
    return result


def execute_readonly(db_path: str | Path, sql: str, timeout_seconds: float = 5,
                     max_rows: int = 10000,
                     max_result_bytes: int = 8 * 1024 * 1024) -> dict[str, Any]:
    """Read one query in a subprocess; kill and reap it on hard timeout.

    The deadline includes interpreter startup, SQLite execution and result
    transfer. VM interruption is an additional guard inside the worker. Row
    and byte caps always return ``rows=None``, never truncated answers. Child
    output is one JSON object; SQL / rows are never printed as progress logs.
    Native memory allocation is not capped by this function.
    """
    if (isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("timeout_seconds must be a positive finite number")
    for name, value in (("max_rows", max_rows), ("max_result_bytes", max_result_bytes)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    started = time.monotonic()
    base: dict[str, Any] = {
        "status": "error", "rows": None, "columns": [], "row_count": None,
        "error": None, "elapsed_seconds": 0.0,
    }
    process = None
    try:
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("SQL must be a non-empty string")
        path = Path(db_path).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError("Database path must name an existing file")
        payload = json.dumps({"db_path": str(path), "sql": sql,
                              "timeout_seconds": timeout_seconds, "max_rows": max_rows,
                              "max_result_bytes": max_result_bytes}).encode("utf-8")
        process = subprocess.Popen(
            [sys.executable, "-m", "opd_sql.sqlite_tools", "--worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=(subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0),
        )
        remaining = max(0.001, timeout_seconds - (time.monotonic() - started))
        output, stderr = process.communicate(payload, timeout=remaining)
        if process.returncode != 0:
            raise RuntimeError(f"SQLite worker exited {process.returncode}: "
                               + stderr.decode("utf-8", errors="replace")[:2000])
        base = json.loads(output)
        base["worker_elapsed_seconds"] = base["elapsed_seconds"]
        base["hard_timeout"] = False
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        base.update(status="timeout", hard_timeout=True,
                    error=f"SQLite subprocess exceeded {timeout_seconds}s wall-clock deadline")
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        base["error"] = str(exc)[:2000]
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate()
        base["elapsed_seconds"] = time.monotonic() - started
    return base


def get_schema(db_path: str | Path) -> str:
    """Return CREATE TABLE/VIEW statements, including declared foreign keys."""
    result = execute_readonly(
        db_path,
        "SELECT sql FROM sqlite_schema "
        "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' "
        "AND sql IS NOT NULL ORDER BY type, name",
    )
    if result["status"] != "ok":
        raise RuntimeError(f"Cannot inspect SQLite schema: {result['error']}")
    return "\n\n".join(row[0].rstrip("; ") + ";" for row in result["rows"])


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("This module's CLI is reserved for the SQLite worker")
    arguments = json.loads(sys.stdin.buffer.read())
    sys.stdout.buffer.write(json.dumps(_execute_inprocess(**arguments),
                                       ensure_ascii=True, allow_nan=False).encode("utf-8"))
