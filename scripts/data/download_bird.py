"""Download BIRD's official OSS archives; partial files never count as ready."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

SOURCES = {split: f"https://bird-bench.oss-cn-beijing.aliyuncs.com/{split}.zip"
           for split in ("dev", "train")}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def digest_file(path: Path) -> tuple[str, str]:
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
            md5.update(chunk)
    return sha.hexdigest(), md5.hexdigest()


def download(root: Path, split: str) -> None:
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    url = SOURCES[split]
    final, partial = raw / f"{split}.zip", raw / f"{split}.zip.partial"
    status_path = root / "work" / f"{split}-download.json"
    headers = dict(urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60).headers)
    expected = int(headers["Content-Length"])
    expected_md5 = (base64.b64decode(headers["Content-MD5"]).hex()
                    if headers.get("Content-MD5") else None)
    status = {"split": split, "url": url, "source_kind": "official",
              "expected_bytes": expected, "official_headers": headers,
              "official_md5": expected_md5, "status": "downloading", "pid": os.getpid()}
    # Only promote the previously downloaded archive after the same checks below.
    existing = root / "work" / f"{split}.zip"
    if not final.exists() and not partial.exists() and existing.exists():
        existing.replace(partial)
    started = time.monotonic()
    try:
        candidate = final if final.exists() else partial
        size = candidate.stat().st_size if candidate.exists() else 0
        if size > expected:
            raise ValueError(f"Archive is larger than official object: {size} > {expected}")
        status.update(downloaded_bytes=size, checked_at=datetime.now(timezone.utc).isoformat())
        write_json(status_path, status)
        if size < expected:
            if final.exists():
                raise ValueError("A final archive is incomplete; refusing to append to it")
            req = urllib.request.Request(url, headers={"Range": f"bytes={size}-"} if size else {})
            with urllib.request.urlopen(req, timeout=60) as response:
                if size and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {size}-")):
                    raise ValueError("Server did not honor exact HTTP range; refusing unsafe resume")
                last_update = 0.0
                with partial.open("ab" if size else "wb") as handle:
                    while chunk := response.read(4 * 1024 * 1024):
                        handle.write(chunk)
                        size += len(chunk)
                        now = time.monotonic()
                        if now - last_update >= 5:
                            status.update(downloaded_bytes=size, elapsed_seconds=round(now - started, 2),
                                          checked_at=datetime.now(timezone.utc).isoformat())
                            write_json(status_path, status)
                            last_update = now
                    handle.flush()
                    os.fsync(handle.fileno())
            candidate = partial
        if candidate.stat().st_size != expected:
            raise ValueError("Downloaded archive length differs from official Content-Length")
        status.update(status="verifying", downloaded_bytes=expected)
        write_json(status_path, status)
        sha, md5 = digest_file(candidate)
        if expected_md5 and md5 != expected_md5:
            raise ValueError(f"Official MD5 mismatch: {md5}")
        with zipfile.ZipFile(candidate) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"ZIP CRC failed: {bad}")
            entries = len(archive.infolist())
        if candidate != final:
            candidate.replace(final)
        status.update(status="complete", sha256=sha, md5=md5, zip_crc_verified=True,
                      entries=entries, completed_at=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=round(time.monotonic() - started, 2))
        write_json(status_path, status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
    except Exception as exc:
        status.update(status="failed", error=repr(exc), checked_at=datetime.now(timezone.utc).isoformat())
        write_json(status_path, status)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=SOURCES, required=True)
    args = parser.parse_args()
    download(args.root.resolve(), args.split)
