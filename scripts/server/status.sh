#!/usr/bin/env bash
set -u
root=/root/autodl-tmp/opd-sql-agent/results/server
for report in setup-status.txt model-download.json deployment-verification.json; do
    printf '\n%s\n' "$report"
    cat "$root/$report" 2>/dev/null || true
done
printf '\nGPU usage (index, memory MiB, utilization)\n'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
