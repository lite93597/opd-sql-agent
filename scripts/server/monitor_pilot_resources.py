#!/usr/bin/env python3
"""Record both GPUs while the bounded pilot pipeline lives, then exit."""
import json
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

root=Path('/root/autodl-tmp/opd-sql-agent/results/server')
with (root/'pilot-resources.jsonl').open('a',encoding='utf-8') as handle:
    while True:
        if not (root/'pilot-status.json').exists():
            time.sleep(1)
            continue
        state=json.loads((root/'pilot-status.json').read_text())
        output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,power.draw',
            '--format=csv,noheader,nounits'],text=True)
        gpus=[]
        for line in output.strip().splitlines():
            index,memory,utilization,power=[field.strip() for field in line.split(',')]
            gpus.append({'index':int(index),'memory_mib':float(memory),'utilization_percent':float(utilization),
                         'power_watts':float(power) if power!='[N/A]' else None})
        handle.write(json.dumps({'time':datetime.now(timezone.utc).isoformat(),'phase':state.get('phase','initializing'),
            'child_phase':state.get('child_status',{}).get('phase'),
            'step':state.get('child_status',{}).get('step'),'gpus':gpus})+'\n')
        handle.flush()
        if state['status'] in ('complete','failed'):
            break
        time.sleep(30)
