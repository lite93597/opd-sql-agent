#!/usr/bin/env python3
"""Bounded BIRD baseline, length probes, 10+10 resumed OPD updates, and reevaluation."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import requests

ROOT = Path('/root/autodl-tmp/opd-sql-agent')
STORAGE = Path('/root/autodl-tmp')
RESULTS = ROOT/'results/server'
BASELINE = STORAGE/'runs/baseline-20261004-v2'
STATE = RESULTS/'pilot-status.json'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def update(state, phase, **kwargs):
    state.update(phase=phase, updated_at=stamp(), **kwargs)
    save(STATE, state)
    print(json.dumps({'phase':phase,**kwargs}, ensure_ascii=False), flush=True)


def child(command, log_path, state, phase, report=None, env=None):
    with log_path.open('a', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        update(state, phase, child_pid=process.pid, child_log=str(log_path))
        while process.poll() is None:
            if report and report.is_file():
                state['child_status'] = json.loads(report.read_text())
            state['updated_at'] = stamp()
            save(STATE, state)
            time.sleep(10)
        if report and report.is_file():
            state['child_status'] = json.loads(report.read_text())
        state.pop('child_pid', None)
        save(STATE, state)
        if process.returncode:
            raise RuntimeError(f'{phase} exited {process.returncode}; see {log_path}')


def wait_file(path, state, phase, timeout=3600):
    update(state, phase)
    deadline = time.monotonic()+timeout
    while True:
        if path.is_file():
            data = json.loads(path.read_text())
            if data.get('status') in ('failed','fail'):
                raise RuntimeError(f'{path}: {data.get("error")}')
            if data.get('status') in ('complete','pass'):
                return data
        if time.monotonic()>deadline:
            raise TimeoutError(f'Waiting for {path} exceeded {timeout} seconds')
        time.sleep(15)


def start_rollout(state, context=4096):
    with (RESULTS/'pilot-vllm.log').open('a', encoding='utf-8') as log:
        process = subprocess.Popen(['bash',str(ROOT/'scripts/server/start_rollout.sh')],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    env=dict(os.environ,OPD_ROLLOUT_MAX_MODEL_LEN=str(context)))
    update(state,'starting_rollout',rollout_pid=process.pid)
    deadline = time.monotonic()+1200
    while True:
        if process.poll() is not None:
            raise RuntimeError(f'Rollout engine exited {process.returncode}')
        try:
            if requests.get('http://127.0.0.1:8001/health',timeout=3).ok:
                return process
        except requests.RequestException:
            pass
        if time.monotonic()>deadline:
            stop_rollout(process)
            raise TimeoutError('Rollout startup exceeded 20 minutes')
        time.sleep(3)


def stop_rollout(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid,signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL)
        process.wait(timeout=10)


def train_stage(config, state, phase, resume=None):
    process = start_rollout(state)
    try:
        command = [sys.executable,'-m','opd_sql.onpolicy','--config',str(config)]
        if resume:
            command += ['--resume',str(resume)]
        cfg = json.loads(config.read_text())
        child(command,RESULTS/f'{phase}.log',state,phase,Path(cfg['output_dir'])/'status.json',
              dict(os.environ,CUDA_VISIBLE_DEVICES='0'))
    finally:
        stop_rollout(process)
        state.pop('rollout_pid',None)
        save(STATE,state)


def main():
    state={'status':'running','started_at':stamp(),'pid':os.getpid(),'scope':'length probes and 20 OPD updates only'}
    save(STATE,state)
    try:
        baseline = wait_file(RESULTS/'baseline-status.json',state,'waiting_baseline')
        if not baseline.get('engines_stopped'):
            raise RuntimeError('Baseline engines have not released the GPUs')
        paired = BASELINE/'paired-evaluation.json'
        if not paired.exists():
            child([sys.executable,'-m','opd_sql.bird_evaluation','--records',str(BASELINE/'records.jsonl'),
                '--predictions',str(BASELINE/'student-predictions.jsonl'),
                '--comparison-predictions',str(BASELINE/'teacher-predictions.jsonl'),
                '--output',str(paired)],RESULTS/'baseline-evaluation.log',state,'baseline_evaluation')
        state['baseline_evaluation'] = json.loads(paired.read_text())['comparison']
        wait_file(STORAGE/'datasets/BIRD/work/train-download.json',state,'waiting_train_download')
        child([sys.executable,str(ROOT/'scripts/data/prepare_bird.py'),'--root',str(STORAGE/'datasets/BIRD')],
              RESULTS/'train-preparation.log',state,'preparing_train')
        manifest = json.loads((STORAGE/'datasets/BIRD/processed/manifest.json').read_text())
        if not manifest['split_disjointness_verified']:
            raise RuntimeError('Train/dev database disjointness has not passed')
        state['dataset_counts'] = {k:v['count'] for k,v in manifest['splits'].items()}
        synthetic = STORAGE/'datasets/BIRD/work/memory-probe.jsonl'
        synthetic.write_text(json.dumps({'id':'synthetic:memory','db_id':'synthetic','source_split':'synthetic',
            'question':'Return a query to count all rows in t.', 'schema':'CREATE TABLE t(id INTEGER, label TEXT);',
            'evidence':'','gold_sql':'SELECT COUNT(*) FROM t;','db_path':''})+'\n')
        base = json.loads((ROOT/'configs/server/opd_smoke.json').read_text())
        configs = ROOT/'configs/server'
        probe = dict(base,mode='probe',max_steps=3,max_seq_length=4096,max_new_tokens=128,
                     probe_lengths=[512,2048,4096],train_file=str(synthetic),
                     output_dir=str(STORAGE/'runs/opd-memory-probe-v1'),save_steps=3)
        save(configs/'opd_probe_runtime.json',probe)
        train_stage(configs/'opd_probe_runtime.json',state,'memory_probe')
        probe_metrics=[json.loads(line) for line in (Path(probe['output_dir'])/'metrics.jsonl').read_text().splitlines()]
        actual = [r['prompt_tokens'][0]+r['completion_lengths'][0] for r in probe_metrics]
        if actual != [512,2048,4096]:
            raise RuntimeError(f'Memory probe actual lengths differ: {actual}')
        state['length_probes'] = [{'sequence_tokens':n,'memory':r['gpu0_memory']} for n,r in zip(actual,probe_metrics)]
        pilot = dict(base,mode='train',max_steps=10,train_file=str(STORAGE/'datasets/BIRD/processed/train.jsonl'),
                     output_dir=str(STORAGE/'runs/opd-pilot-20-v1'))
        save(configs/'opd_pilot_runtime.json',pilot)
        train_stage(configs/'opd_pilot_runtime.json',state,'pilot_first_10')
        run=Path(pilot['output_dir'])
        previous=(run/'sync-10.json').read_bytes()
        (run/'sync-10-before-resume.json').write_bytes(previous)
        before=json.loads(previous)['effective_weight_sha256']
        pilot['max_steps']=20
        save(configs/'opd_pilot_runtime.json',pilot)
        train_stage(configs/'opd_pilot_runtime.json',state,'pilot_resume_to_20',run/'checkpoint-10')
        resumed=json.loads((run/'sync-10.json').read_text())['effective_weight_sha256']
        if before!=resumed:
            raise RuntimeError('Checkpoint restore did not reproduce the exact BF16 merged policy')
        metrics=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
        if [r['step'] for r in metrics]!=list(range(1,21)):
            raise RuntimeError('Pilot metrics have missing or duplicate updates')
        state['checkpoint_restore_verified']=True
        state['pilot_final_checkpoint']=str(run/'checkpoint-20')
        update(state,'pilot_training_complete')
        post=STORAGE/'runs/opd-pilot-20-v1-evaluation'
        process=start_rollout(state,context=16384)
        try:
            child([sys.executable,str(ROOT/'scripts/server/evaluate_adapter.py'),'--baseline',str(BASELINE),
                '--checkpoint',str(run/'checkpoint-20'),'--output',str(post)],RESULTS/'student-reevaluation.log',
                state,'student_reevaluation',post/'status.json',dict(os.environ,CUDA_VISIBLE_DEVICES='0'))
        finally:
            stop_rollout(process)
            state.pop('rollout_pid',None)
        post_report=post/'paired-evaluation.json'
        child([sys.executable,'-m','opd_sql.bird_evaluation','--records',str(BASELINE/'records.jsonl'),
            '--predictions',str(post/'student-predictions.jsonl'),
            '--comparison-predictions',str(BASELINE/'teacher-predictions.jsonl'),'--output',str(post_report)],
            RESULTS/'post-evaluation.log',state,'post_execution_evaluation')
        before_report=json.loads(paired.read_text())
        after_report=json.loads(post_report.read_text())
        before_by_id={r['id']:r for r in before_report['results']}
        gained=lost=0
        for r in after_report['results']:
            old=before_by_id[r['id']]['bird_correct']
            gained+=int(r['bird_correct'] and not old)
            lost+=int(old and not r['bird_correct'])
        total=after_report['summary']['total']
        state['accuracy_comparison']={
            'total':total,'student_before':before_report['summary']['bird_execution_accuracy'],
            'teacher':before_report['teacher_report']['summary']['bird_execution_accuracy'],
            'student_after_20_steps':after_report['summary']['bird_execution_accuracy'],
            'paired_gained':gained,'paired_lost':lost,
            'accuracy_change_pp':100*(gained-lost)/total,
            'notes':'120 fixed dev questions; bounded 20-update pilot, not a full training efficacy claim.'}
        state.update(status='complete',phase='complete',completed_at=stamp())
        save(STATE,state)
    except Exception as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),updated_at=stamp())
        save(STATE,state)
        traceback.print_exc()
        raise


if __name__=='__main__':
    main()
