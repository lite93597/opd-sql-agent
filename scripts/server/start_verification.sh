#!/usr/bin/env bash
set -euo pipefail
source /root/autodl-tmp/opd-sql-agent/scripts/server/env.sh
if pgrep -f '^(/root/autodl-tmp/envs/opd/bin/)?python .*scripts/server/verify_deployment.py$' > /dev/null; then
    printf 'Deployment verification is already active\n'
    exit 0
fi
nohup python "$OPD_ROOT/scripts/server/verify_deployment.py" > "$OPD_ROOT/results/server/deployment-verification.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$OPD_ROOT/results/server/deployment-verification.pid"
nohup /root/miniconda3/bin/python "$OPD_ROOT/scripts/server/monitor_setup.py" "$pid" deployment-monitor > "$OPD_ROOT/results/server/deployment-monitor.log" 2>&1 < /dev/null &
printf 'Deployment verification pid=%s; periodic monitor active\n' "$pid"
