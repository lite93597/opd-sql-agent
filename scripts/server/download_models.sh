#!/usr/bin/env bash
set -euo pipefail
source /root/autodl-tmp/opd-sql-agent/scripts/server/env.sh
if [ -f /etc/network_turbo ]; then
    source /etc/network_turbo > /dev/null 2>&1
fi
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=30
while ! python -c 'import huggingface_hub' > /dev/null 2>&1; do
    if grep -q 'exit_code=[1-9]' "$OPD_ROOT/results/server/setup-status.txt"; then
        printf 'Environment setup failed; model download stopped.\n' >&2
        exit 1
    fi
    sleep 10
done
exec python "$OPD_ROOT/scripts/server/download_models.py"
