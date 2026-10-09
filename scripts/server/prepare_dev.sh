#!/usr/bin/env bash
set -euo pipefail
source /root/autodl-tmp/opd-sql-agent/scripts/server/env.sh
python "$OPD_ROOT/scripts/data/download_bird.py" --root "$OPD_STORAGE/datasets/BIRD" --split dev
python "$OPD_ROOT/scripts/data/prepare_bird.py" --root "$OPD_STORAGE/datasets/BIRD"
