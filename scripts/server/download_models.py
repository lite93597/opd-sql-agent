"""Download pinned model snapshots directly into the persistent data disk."""
import json
import os
from pathlib import Path
import re
import subprocess
import time
import requests

from huggingface_hub import HfApi, snapshot_download

root = Path('/root/autodl-tmp')
report = root / 'opd-sql-agent/results/server/model-download.json'
state = {'status': 'running', 'models': []}
models = [
    ('Qwen/Qwen3.5-9B', 'c202236235762e1c871ad0ccb60c8ee5ba337b9a',
     'e8885939589b1e032d291a1ceefd21b775e6277e', 20_000_000_000),
    ('Qwen/Qwen3.8-27B', '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
     '1098534ab5d7220ea0f4a6b9f07bb03729a79c1d', 56_000_000_000),
]
try:
    for repository, revision, mirror_revision, required in models:
        usage = os.statvfs(root)
        assert usage.f_bavail * usage.f_frsize > required + 15_000_000_000, 'Insufficient disk headroom'
        destination = root / 'models' / repository.split('/')[-1]
        entry = dict(repository=repository, revision=revision, path=str(destination), status='downloading')
        state['models'].append(entry)
        report.write_text(json.dumps(state, indent=2))
        started = time.monotonic()
        snapshot_download(repository, revision=revision, local_dir=destination, max_workers=4,
                          allow_patterns=['*.json', '*.txt', '*.jinja', 'README.md', 'LICENSE'])
        index = json.loads((destination / 'model.safetensors.index.json').read_text())
        shards = sorted(set(index['weight_map'].values()))
        # The model-info endpoint can omit file metadata even with files_metadata=True.
        # The repository tree exposes the published LFS SHA256; xet_hash is distinct.
        siblings = {item.path: item for item in HfApi().list_repo_tree(
            repository, revision=revision, expand=True) if hasattr(item, 'lfs')}
        session = requests.Session()
        session.trust_env = False
        response = session.get(f'https://modelscope.cn/api/v1/models/{repository}/repo/files',
                               params={'Revision': mirror_revision, 'Recursive': 'true'}, timeout=30)
        response.raise_for_status()
        mirror = {item['Name']: item for item in response.json()['Data']['Files']}
        lines = []
        hashes = {}
        for name in shards:
            assert Path(name).name == name, 'Unexpected nested shard path'
            blob = siblings[name]
            assert blob.lfs and blob.lfs.sha256, f'Missing published LFS checksum: {name}'
            assert re.fullmatch(r'[0-9a-f]{64}', blob.lfs.sha256), 'Invalid published SHA256'
            assert mirror[name]['Sha256'] == blob.lfs.sha256 and mirror[name]['Size'] == blob.size, 'Mirror differs from pinned HF snapshot'
            hashes[name] = blob.lfs.sha256
            lines.extend([f'https://modelscope.cn/models/{repository}/resolve/{mirror_revision}/{name}',
                          f' out={name}', f' checksum=sha-256={blob.lfs.sha256}'])
        downloads = destination / '.cache/aria2-input.txt'
        downloads.parent.mkdir(exist_ok=True)
        downloads.write_text('\n'.join(lines) + '\n')
        entry.update(weight_transport='ModelScope', mirror_revision=mirror_revision,
                     mirror_matches_pinned_hf_size_and_sha256=True)
        report.write_text(json.dumps(state, indent=2))
        # HF metadata uses AutoDL's proxy; byte-identical domestic weight shards use direct transport.
        transport_env = {key: value for key, value in os.environ.items() if key.lower() not in ('http_proxy', 'https_proxy', 'all_proxy')}
        subprocess.run(['aria2c', f'--input-file={downloads}', f'--dir={destination}',
                        '--max-concurrent-downloads=4', '--max-connection-per-server=8', '--split=8',
                        '--min-split-size=8M', '--continue=true', '--auto-file-renaming=false',
                        '--allow-overwrite=true', '--file-allocation=none', '--summary-interval=60',
                        '--console-log-level=warn', '--download-result=hide', '--timeout=120',
                        '--connect-timeout=30', '--max-tries=10', '--retry-wait=5'], check=True, env=transport_env)
        assert all((destination / name).is_file() for name in shards), 'Missing weight shards'
        assert all((destination / name).stat().st_size == siblings[name].size for name in shards), 'Shard size mismatch'
        entry.update(status='complete', elapsed_seconds=round(time.monotonic() - started, 2),
                     shard_count=len(shards), weight_bytes=sum((destination/name).stat().st_size for name in shards),
                     sha256=hashes, checksums_verified_by='aria2')
        report.write_text(json.dumps(state, indent=2))
        print(json.dumps(entry), flush=True)
    state['status'] = 'complete'
except Exception as exc:
    state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    raise
finally:
    report.write_text(json.dumps(state, indent=2))
