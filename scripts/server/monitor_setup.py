"""Periodically record resource usage until the setup process exits."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

pid = int(sys.argv[1])
name = sys.argv[2] if len(sys.argv) > 2 else 'monitor'
root = Path('/root/autodl-tmp/opd-sql-agent/results/server')
while True:
    try:
        os.kill(pid, 0)
        process_state = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0]
        active = process_state != 'Z'
    except (ProcessLookupError, FileNotFoundError):
        active = False
    usage = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=20,
    )
    disk = os.statvfs('/root/autodl-tmp')
    record = dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  pid=pid, active=active, gpu=usage.stdout.strip(),
                  disk_free_bytes=disk.f_bavail * disk.f_frsize,
                  model_bytes=sum(path.stat().st_size for path in Path('/root/autodl-tmp/models').rglob('*') if path.is_file()))
    with (root / f'{name}.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    (root / f'{name}-latest.json').write_text(json.dumps(record, indent=2))
    if not active:
        break
    time.sleep(60)
