#!/usr/bin/env bash
set -euo pipefail
source /root/autodl-tmp/opd-sql-agent/scripts/server/env.sh
case "${1:-student}" in
  student) model="$OPD_STORAGE/models/Qwen3.5-9B" ;;
  teacher) model="$OPD_STORAGE/models/Qwen3.8-27B" ;;
  *) exit 2 ;;
esac
python - "$model" "$OPD_ROOT/results/server/model-download.json" <<'PY'
import json, sys
from pathlib import Path
model=Path(sys.argv[1]); report=json.loads(Path(sys.argv[2]).read_text())
assert report['status']=='complete'
entry=next(e for e in report['models'] if Path(e['path']).resolve()==model.resolve())
assert entry['status']=='complete' and entry['sha256']
assert all((model/n).is_file() and not (model/(n+'.aria2')).exists() for n in entry['sha256'])
PY
export CUDA_VISIBLE_DEVICES=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
exec vllm serve "$model" \
  --host 127.0.0.1 --port 8001 \
  --tensor-parallel-size 1 --data-parallel-size 1 \
  --gpu-memory-utilization 0.75 --dtype bfloat16 \
  --max-model-len 16384 --max-num-seqs 1 \
  --model-impl vllm --language-model-only --enforce-eager \
  --generation-config vllm \
  --weight-transfer-config '{"backend":"nccl"}' \
  --logprobs-mode processed_logprobs --max-logprobs -1
