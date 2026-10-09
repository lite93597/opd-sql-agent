#!/usr/bin/env bash
set -euo pipefail
source /root/autodl-tmp/opd-sql-agent/scripts/server/env.sh
student="${1:-$OPD_STORAGE/models/Qwen3.5-9B}"
python - "$student" "$OPD_ROOT/results/server/model-download.json" <<'PY'
import json, sys
from pathlib import Path
model, report = Path(sys.argv[1]), Path(sys.argv[2])
state = json.loads(report.read_text())
assert any(Path(item['path']).resolve() == model.resolve() and item['status'] == 'complete'
           for item in state['models']), 'Student download and checksum verification have not completed'
index = json.loads((model/'model.safetensors.index.json').read_text())
for name in set(index['weight_map'].values()):
    assert (model/name).is_file() and not (model/(name+'.aria2')).exists(), f'Incomplete shard: {name}'
PY
export CUDA_VISIBLE_DEVICES=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
exec vllm serve "$student" \
  --host 127.0.0.1 --port 8001 \
  --tensor-parallel-size 1 --data-parallel-size 1 \
  --gpu-memory-utilization 0.70 --dtype bfloat16 \
  --max-model-len "${OPD_ROLLOUT_MAX_MODEL_LEN:-4096}" --max-num-seqs 4 \
  --model-impl vllm --language-model-only --enforce-eager \
  --generation-config vllm \
  --weight-transfer-config '{"backend":"nccl"}' \
  --logprobs-mode processed_logprobs --max-logprobs -1
