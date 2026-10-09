#!/usr/bin/env bash
set -euo pipefail
source /root/autodl-tmp/opd-sql-agent/scripts/server/env.sh
if [ -f /etc/network_turbo ]; then
    source /etc/network_turbo > /dev/null 2>&1
fi
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1
status="$OPD_ROOT/results/server/setup-status.txt"
trap 'code=$?; printf "exit_code=%s\nfinished=%s\n" "$code" "$(date -Is)" >> "$status"' EXIT
printf 'phase=dependencies\nstarted=%s\n' "$(date -Is)" > "$status"
python "$OPD_ROOT/scripts/server/prepare_large_wheels.py"
aria2c --input-file="$OPD_STORAGE/wheels/aria2-input.txt" --dir="$OPD_STORAGE/wheels" \
  --max-concurrent-downloads=2 --max-connection-per-server=4 --split=4 \
  --min-split-size=8M --continue=true --auto-file-renaming=false \
  --allow-overwrite=true --file-allocation=none --summary-interval=60 \
  --console-log-level=warn --download-result=hide --timeout=120 --user-agent='Python-urllib/3.12' \
  --connect-timeout=30 --max-tries=5 --retry-wait=5
python - <<'PY'
import json, subprocess, sys
from pathlib import Path
root=Path('/root/autodl-tmp')
manifest=json.loads((root/'opd-sql-agent/results/server/large-wheels.json').read_text())
subprocess.run([sys.executable,'-m','pip','install','--no-deps',*[str(root/'wheels'/item['filename']) for item in manifest]],check=True)
PY
python -m pip install --timeout 120 --index-url https://mirrors.aliyun.com/pypi/simple -r "$OPD_ROOT/scripts/server/requirements.txt"
printf 'phase=cuda-toolkit\n' >> "$status"
python - <<'PY'
from pathlib import Path
import site
target = Path('/root/autodl-tmp/cuda-13')
prefix = Path(site.getsitepackages()[0]) / 'nvidia/cu13'
for name in ['bin/nvcc','include/cuda.h','nvvm/bin/cicc','nvvm/libdevice/libdevice.10.bc']:
    assert (prefix/name).exists(), f'Missing CUDA toolkit component: {name}'
for link, source in [(prefix/'lib64',Path('lib')),(prefix/'lib/libcudart.so',Path('libcudart.so.13'))]:
    if not link.exists():
        link.symlink_to(source)
if not target.exists():
    target.symlink_to(prefix, target_is_directory=True)
assert target.resolve() == prefix.resolve(), 'Unexpected existing CUDA_HOME'
print('cuda_home',target,'canonical',prefix)
PY
export CUDA_HOME="$OPD_STORAGE/cuda-13"
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version
printf 'phase=causal-conv1d\n' >> "$status"
python -m pip install --timeout 120 --index-url https://mirrors.aliyun.com/pypi/simple --no-build-isolation "causal-conv1d==1.7.0"
printf 'phase=verification\n' >> "$status"
python -m pip check
python -m pip freeze > "$OPD_ROOT/results/server/requirements.lock.txt"
python "$OPD_ROOT/scripts/server/verify_runtime.py" --output "$OPD_ROOT/results/server/runtime-verification.json"
printf 'phase=complete\n' >> "$status"
