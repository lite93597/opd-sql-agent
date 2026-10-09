#!/usr/bin/env python3
"""A bounded, validation-selected SFT/OPD comparison on frozen BIRD splits."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import traceback

import requests

ROOT = Path('/root/autodl-tmp/opd-sql-agent')
STORAGE = Path('/root/autodl-tmp')
RUN = STORAGE/'runs/effect-experiment-v1'
DATA = STORAGE/'datasets/BIRD/processed/experiment-v1'
STATE = ROOT/'results/server/effect-status.json'
CONFIGS = ROOT/'configs/server/effect-v1'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def stop(process):
    if process is None:
        return
    try:
        os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL); process.wait(timeout=10)
    try:
        os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:
        pass


def recover_training(run, target):
    """Restore a completed boundary; archive every discarded nondurable log row."""
    checkpoints=[]
    for checkpoint in run.glob('checkpoint-*'):
        if not re.fullmatch(r'checkpoint-\d+',checkpoint.name):
            continue
        complete=checkpoint/'complete.json'
        if complete.is_file():
            info=json.loads(complete.read_text())
            if info['step']<=target:
                checkpoints.append((info['step'],checkpoint,info))
    if not checkpoints:
        if (run/'metrics.jsonl').exists():
            raise RuntimeError('Interrupted before first durable checkpoint; preserve run and create a new attempt explicitly')
        return None
    step,checkpoint,info=max(checkpoints,key=lambda item:item[0])
    for name,digest in info['sha256'].items():
        assert sha(checkpoint/name)==digest, f'Checkpoint SHA mismatch: {name}'
    archived={}
    archive=run/'recovery-evidence'/f'after-checkpoint-{step}-{time.time_ns()}'
    for name in ('metrics.jsonl','trajectories.jsonl'):
        path=run/name
        if not path.exists(): continue
        rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        if rows and rows[-1]['step']>step:
            archive.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,archive/name); archived[name]=sha(archive/name)
            retained=[row for row in rows if row['step']<=step]
            tmp=path.with_suffix('.recovery.tmp')
            tmp.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in retained)); tmp.replace(path)
    if archived:
        for path in run.glob('sync-*.json'):
            match=re.fullmatch(r'sync-(\d+)\.json',path.name)
            if match and int(match.group(1))>step:
                shutil.copy2(path,archive/path.name)
        for name in ('status.json','config.json'):
            if (run/name).exists(): shutil.copy2(run/name,archive/name)
        save(archive/'manifest.json',{'checkpoint':str(checkpoint),'step':step,'full_original_logs_sha256':archived,
            'reason':'Updates after last durable checkpoint cannot be restored. Original logs retained; replay resumes from checkpoint RNG/optimizer/cursor.'})
    for path in run.glob('checkpoint-*.incomplete'):
        if path.is_dir():
            archive.mkdir(parents=True,exist_ok=True)
            path.rename(archive/path.name)
    return checkpoint


class Pipeline:
    def __init__(self, resume=False):
        if STATE.exists():
            if not resume:
                raise FileExistsError('Existing effect experiment: use --resume, never launch a duplicate')
            self.state=json.loads(STATE.read_text())
            old_pid=self.state.get('pid')
            if old_pid and old_pid!=os.getpid():
                try:
                    os.kill(old_pid,0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError(f'Existing effect pipeline PID {old_pid} is alive')
            if self.state['status']=='complete':
                raise ValueError('Completed experiment must not be repeated')
            for name in ('child_pid','rollout_pid'):
                if self.state.get(name):
                    try:
                        os.kill(self.state[name],0)
                    except ProcessLookupError:
                        pass
                    else:
                        raise RuntimeError(f'Residual {name}={self.state[name]} must be inspected before resuming')
            save(STATE.with_name(f'effect-status-before-resume-{time.time_ns()}.json'),self.state)
            for name in ('error','traceback','failed_at','child_pid','rollout_pid'):
                self.state.pop(name,None)
        else:
            self.state={'started_at':stamp(),'completed_stages':{},'scope':'SFT warm-up, matched continued SFT and fresh on-policy KL branches; final test after validation selection',
                        'max_wall_hours':8,'target':'Improve held-out student execution accuracy and measure OPD contribution honestly'}
        self.deadline=time.monotonic()+8*3600
        self.child_process=None
        self.rollout_process=None
        self.last_resource=0
        self.state.update(status='running',pid=os.getpid(),resumed_at=stamp() if resume else None)
        self.update('initializing')

    def update(self, phase, **kwargs):
        self.state.update(phase=phase,updated_at=stamp(),**kwargs)
        save(STATE,self.state)

    def resources(self):
        if time.monotonic()-self.last_resource<30:
            return
        self.last_resource=time.monotonic()
        output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True)
        row={'time':stamp(),'phase':self.state['phase'],'child_phase':self.state.get('child_status',{}).get('phase'),
             'step':self.state.get('child_status',{}).get('step'),'gpu_csv':output.strip()}
        with (ROOT/'results/server/effect-resources.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        self.state['last_resource']=row

    def child(self, key, command, report=None, identity_command=None):
        identity_command=identity_command or command
        if key in self.state['completed_stages']:
            saved=self.state['completed_stages'][key]
            if saved['command']!=identity_command:
                raise ValueError(f'Completed stage command changed: {key}')
            return
        log=RUN/'logs'/f'{key}.log'; log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as stream:
            self.child_process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1'),start_new_session=True)
            self.update(key,child_pid=self.child_process.pid,child_log=str(log),child_report=str(report) if report else None,child_status={})
            while self.child_process.poll() is None:
                if time.monotonic()>self.deadline:
                    stop(self.child_process)
                    raise TimeoutError('8-hour round limit reached; completed checkpoints preserved')
                if self.rollout_process and self.rollout_process.poll() is not None:
                    stop(self.child_process)
                    raise RuntimeError('Rollout engine exited during active stage')
                if report and report.exists():
                    self.state['child_status']=json.loads(report.read_text())
                self.resources(); save(STATE,self.state)
                time.sleep(5)
            if report and report.exists():
                self.state['child_status']=json.loads(report.read_text())
            code=self.child_process.returncode
            self.child_process=None
            self.state.pop('child_pid',None)
            if code:
                save(STATE,self.state)
                raise RuntimeError(f'{key} failed with exit {code}; see {log}')
        self.state['completed_stages'][key]={'command':identity_command,'executed_command':command,'completed_at':stamp(),'report':str(report) if report else None}
        save(STATE,self.state)

    @contextmanager
    def engine(self, role='student'):
        log=RUN/'logs'/f'engine-{role}.log'; log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as stream:
            self.rollout_process=subprocess.Popen(['bash',str(ROOT/'scripts/server/start_experiment_rollout.sh'),role],
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        self.update('starting_'+role+'_engine',rollout_pid=self.rollout_process.pid)
        try:
            start=time.monotonic()
            while True:
                if self.rollout_process.poll() is not None:
                    raise RuntimeError(f'{role} engine exited; see {log}')
                if time.monotonic()-start>1200 or time.monotonic()>self.deadline:
                    raise TimeoutError('Engine startup/time budget expired')
                try:
                    if requests.get('http://127.0.0.1:8001/health',timeout=3).ok:
                        break
                except requests.RequestException:
                    pass
                self.resources(); time.sleep(3)
            yield
        finally:
            stop(self.rollout_process); self.rollout_process=None
            self.state.pop('rollout_pid',None); save(STATE,self.state)

    def evaluate(self, key, records, role='student', checkpoint=None, reference=None):
        out=RUN/key
        if key not in self.state['completed_stages']:
            with self.engine(role):
                command=[sys.executable,str(ROOT/'scripts/server/evaluate_experiment.py'),'--records',str(records),
                         '--role',role,'--output',str(out)]
                if checkpoint: command+=['--checkpoint',str(checkpoint)]
                if reference: command+=['--reference-run',str(reference)]
                self.child(key,command,out/'status.json')
        report=json.loads((out/'report.json').read_text())
        summary=report['summary']
        if any(summary[name] for name in ('missing_predictions','duplicate_predictions','generation_failures','unknown_prediction_ids')):
            raise RuntimeError(f'{key}: evaluation has missing/duplicate/generation/unknown failures')
        return summary['bird_execution_accuracy']

    def train(self, key, cfg, kind, resume=None, stop_after=None):
        path=CONFIGS/f'{key}.json'; save(path,cfg)
        command=[sys.executable,'-m',f'opd_sql.{kind}','--config',str(path)]
        if resume: command+=['--resume',str(resume)]
        if stop_after: command+=['--stop-after',str(stop_after)]
        planned=list(command)
        run=Path(cfg['output_dir']); report=run/'status.json'; target=stop_after or cfg['max_steps']
        if key not in self.state['completed_stages'] and report.exists():
            status=json.loads(report.read_text())
            if status.get('status')=='pass' and status.get('step')==target and (run/f'checkpoint-{target}/complete.json').exists():
                recover_training(run,target)
                self.state['completed_stages'][key]={'command':planned,'executed_command':None,'completed_at':stamp(),
                    'report':str(report),'recovered_completion':True}
                save(STATE,self.state); return
            checkpoint=recover_training(run,target)
            if checkpoint:
                checkpoint_step=int(checkpoint.name.rsplit('-',1)[1])
                if checkpoint_step==target:
                    verify_steps(run,target)
                    self.state['completed_stages'][key]={'command':planned,'executed_command':None,'completed_at':stamp(),
                        'report':str(report),'recovered_completion':'target checkpoint and continuous logs verified'}
                    save(STATE,self.state); return
                if '--resume' in command: command[command.index('--resume')+1]=str(checkpoint)
                else: command+=['--resume',str(checkpoint)]
        if kind=='onpolicy' and key not in self.state['completed_stages']:
            with self.engine(): self.child(key,command,report,planned)
        else:
            self.child(key,command,report,planned)


def common_pool():
    from transformers import AutoTokenizer
    from opd_sql.supervised import encode_supervised
    from opd_sql.prompts import format_messages
    derived=DATA/'eligible-8k-v1'; manifest=derived/'manifest.json'
    if manifest.exists():
        m=json.loads(manifest.read_text())
        assert m['input_sha256']==sha(DATA/'internal_train.jsonl')
        assert m['output_sha256']==sha(derived/'eligible_train.jsonl')
        return derived/'eligible_train.jsonl'
    tokenizer=AutoTokenizer.from_pretrained(STORAGE/'models/Qwen3.5-9B',local_files_only=True)
    rows=[json.loads(x) for x in (DATA/'internal_train.jsonl').read_text().splitlines()]
    eligible=[]; filtered=[]
    for record in rows:
        example=encode_supervised(tokenizer,record)
        reason='prompt_length' if len(example['prompt_ids'])>7680 else 'gold_completion_length' if example['completion_tokens']>512 else None
        if reason: filtered.append({'id':record['id'],'reason':reason,'prompt_tokens':len(example['prompt_ids']),'completion_tokens':example['completion_tokens']})
        else: eligible.append(record)
    derived.mkdir()
    with (derived/'eligible_train.jsonl').open('x') as stream:
        for row in eligible: stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    # Construct a long PROMPT, with a <=512-token forced completion, rather than
    # thousands of synthetic output tokens unlike SQL trajectories.
    probe={'id':'synthetic:8k','db_id':'synthetic','source_split':'synthetic','split':'synthetic',
           'question':'Return only SELECT COUNT(*) FROM t;','schema':'CREATE TABLE t(id INTEGER);', 'evidence':'','gold_sql':'SELECT COUNT(*) FROM t;','db_path':''}
    filler='\n-- unused context field description: integer identifier with example value 12345.'
    count=300
    while True:
        probe['schema']='CREATE TABLE t(id INTEGER);'+filler*count
        prompt=tokenizer.encode(tokenizer.apply_chat_template(format_messages(probe),tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
        if 7680<=len(prompt)<8192: break
        count+=max(1,(7680-len(prompt))//16) if len(prompt)<7680 else -1
        if count<=0: raise RuntimeError('Cannot construct length probe')
    (derived/'long-prompt-probe.jsonl').write_text(json.dumps(probe)+'\n')
    save(manifest,{'input_sha256':sha(DATA/'internal_train.jsonl'),'output_sha256':sha(derived/'eligible_train.jsonl'),
        'read':len(rows),'accepted':len(eligible),'filtered':filtered,'max_prompt_tokens':7680,'max_completion_tokens':512,
        'same_pool_for_sft_and_opd':True,'schema_truncated':False,'probe_prompt_tokens':len(prompt)})
    return derived/'eligible_train.jsonl'


def choose(pipeline, name, run, candidates, validation, reference):
    scores=[]
    for step in candidates:
        checkpoint=run/f'checkpoint-{step}'
        score=pipeline.evaluate(f'val-{name}-{step}',validation,checkpoint=checkpoint,reference=reference)
        scores.append({'step':step,'checkpoint':str(checkpoint),'accuracy':score})
    best=max(scores,key=lambda x:(x['accuracy'],-x['step']))
    pipeline.state.setdefault('validation_selection',{})[name]={'candidates':scores,'best':best,'selection_split':'internal_validation'}
    save(STATE,pipeline.state)
    return best


def verify_steps(run, last):
    metrics=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
    assert [x['step'] for x in metrics]==list(range(1,last+1)), 'Missing or duplicate optimizer updates'
    return metrics


def execute(pipeline):
    RUN.mkdir(parents=True,exist_ok=True)
    selection_path=RUN/'frozen-selection.json'
    if selection_path.exists():
        # No training, checkpoint selection or LR retry once the test is opened.
        return final_test(pipeline,json.loads(selection_path.read_text()))
    if not (DATA/'manifest.json').exists():
        pipeline.child('prepare_splits',[sys.executable,str(ROOT/'scripts/data/prepare_experiment.py'),'--root',str(STORAGE/'datasets/BIRD'),
            '--baseline-records',str(STORAGE/'runs/baseline-20261004-v2/records.jsonl')])
    data=json.loads((DATA/'manifest.json').read_text()); assert data['status']=='complete'
    validation=DATA/'internal-validation-120.jsonl'; test=DATA/'dev-heldout-test-300.jsonl'
    pipeline.update('preparing_common_training_pool')
    train=common_pool(); pool=json.loads((train.parent/'manifest.json').read_text())
    pipeline.state['dataset']={'train':pool['accepted'],'filtered':len(pool['filtered']),
         'internal_validation':120,'heldout_test':300,'same_train_pool':True,'manifest_sha256':sha(DATA/'manifest.json')}
    base=json.loads((ROOT/'configs/server/opd_smoke.json').read_text())
    base.update(experiment=True,max_seq_length=8192,gradient_accumulation_steps=4,train_file=str(train),save_steps=100,
                max_steps=300,learning_rate=1e-5,lr_schedule='linear',schedule_total_steps=300,warmup_ratio=.05)
    probe=dict(base,mode='probe',gradient_accumulation_steps=1,max_steps=1,lr_schedule='constant',
        max_new_tokens=512,probe_lengths=[8192],train_file=str(train.parent/'long-prompt-probe.jsonl'),
        output_dir=str(RUN/'memory-probe-8192'),save_steps=1)
    pipeline.train('probe_8192',probe,'onpolicy')
    probe_metric=verify_steps(Path(probe['output_dir']),1)[0]
    assert probe_metric['prompt_tokens'][0]+probe_metric['completion_lengths'][0]==8192
    pipeline.state['length_8192_verified']=probe_metric['gpu0_memory']
    val_base=pipeline.evaluate('val-base',validation)
    val_teacher=pipeline.evaluate('val-teacher',validation,'teacher',reference=RUN/'val-base')
    pipeline.state['validation_initial']={'student':val_base,'teacher':val_teacher}
    sft=dict(base,mode='sft',learning_rate=5e-5,ce_chunk_tokens=8,output_dir=str(RUN/'sft-warmup'))
    for key in ('experiment','lr_schedule','schedule_total_steps','probe_lengths'):
        sft.pop(key,None)
    pipeline.train('sft_warm_first3',sft,'supervised',stop_after=3)
    pipeline.train('sft_warm_to300',sft,'supervised',resume=RUN/'sft-warmup/checkpoint-3')
    warm_metrics=verify_steps(RUN/'sft-warmup',300)
    pipeline.state['sft_resume_verified']=True
    warm=choose(pipeline,'warm',RUN/'sft-warmup',(100,200,300),validation,RUN/'val-base')
    parent=warm['checkpoint']
    control=dict(sft,seed=43,initial_adapter=parent,learning_rate=2e-5,output_dir=str(RUN/'continued-sft'))
    pipeline.train('continued_sft_300',control,'supervised')
    control_best=choose(pipeline,'continued-sft',RUN/'continued-sft',(100,200,300),validation,RUN/'val-base')
    opd=dict(base,mode='train',seed=43,initial_adapter=parent,output_dir=str(RUN/'opd-1e-5'))
    first=dict(opd,max_steps=2,save_steps=100)
    pipeline.train('opd_first2_accum4',first,'onpolicy')
    sync2=RUN/'opd-1e-5/sync-2.json'
    before=RUN/'opd-1e-5/sync-2-before-resume.json'
    if not before.exists(): before.write_bytes(sync2.read_bytes())
    pipeline.train('opd_resume_to300',opd,'onpolicy',resume=RUN/'opd-1e-5/checkpoint-2')
    assert json.loads(before.read_text())['effective_weight_sha256']==json.loads(sync2.read_text())['effective_weight_sha256']
    metrics=verify_steps(RUN/'opd-1e-5',300)
    assert all(m['policy_version_sampled']==m['step']-1 and m['policy_version_synced']==m['step'] for m in metrics)
    pipeline.state['opd_resume_verified']=True
    opd_best=choose(pipeline,'opd-1e-5',RUN/'opd-1e-5',(100,200,300),validation,RUN/'val-base')
    # A predeclared second LR is tried ONLY on internal validation, not on dev.
    if 'try_lower_lr' not in pipeline.state:
        pipeline.state['try_lower_lr']=opd_best['accuracy']<=max(warm['accuracy'],control_best['accuracy']) and pipeline.deadline-time.monotonic()>5*3600
        save(STATE,pipeline.state)
    if pipeline.state['try_lower_lr']:
        long_control=dict(control,max_steps=600,output_dir=str(RUN/'continued-sft-600'))
        pipeline.train('matched_continued_sft_600',long_control,'supervised')
        long_control_best=choose(pipeline,'continued-sft-600',RUN/'continued-sft-600',(200,400,600),validation,RUN/'val-base')
        lower=dict(opd,learning_rate=5e-6,max_steps=600,schedule_total_steps=600,output_dir=str(RUN/'opd-5e-6'))
        pipeline.train('opd_lowerlr_600',lower,'onpolicy')
        other=choose(pipeline,'opd-5e-6',RUN/'opd-5e-6',(200,400,600),validation,RUN/'val-base')
        if other['accuracy']>opd_best['accuracy']:
            opd_best=other; control_best=long_control_best
    # Freeze every selected checkpoint BEFORE evaluating any final-test score.
    selection={'created_at':stamp(),'source_split':'internal_validation','test_untouched_before_selection':True,
        'warm':warm,'continued_sft':control_best,'opd':opd_best,'test_records_sha256':sha(test)}
    selection['matched_branch_budget_steps']=600 if 'opd-5e-6' in opd_best['checkpoint'] else 300
    if selection_path.exists():
        frozen=json.loads(selection_path.read_text())
        assert all(frozen[k]==selection[k] for k in ('warm','continued_sft','opd','test_records_sha256'))
    else: save(selection_path,selection)
    return final_test(pipeline,selection)


def final_test(pipeline,selection):
    test=DATA/'dev-heldout-test-300.jsonl'
    assert sha(test)==selection['test_records_sha256'], 'Frozen final test changed'
    warm=selection['warm']; control_best=selection['continued_sft']; opd_best=selection['opd']
    test_base=pipeline.evaluate('test-base',test)
    test_teacher=pipeline.evaluate('test-teacher',test,'teacher',reference=RUN/'test-base')
    test_warm=pipeline.evaluate('test-warm',test,checkpoint=Path(warm['checkpoint']),reference=RUN/'test-base')
    test_control=pipeline.evaluate('test-continued-sft',test,checkpoint=Path(control_best['checkpoint']),reference=RUN/'test-base')
    test_opd=pipeline.evaluate('test-opd',test,checkpoint=Path(opd_best['checkpoint']),reference=RUN/'test-base')
    gap=test_teacher-test_base
    outcome={'total':300,'student_base':test_base,'teacher':test_teacher,'sft_warm':test_warm,
             'continued_sft':test_control,'opd':test_opd,'opd_vs_base_pp':100*(test_opd-test_base),
             'opd_vs_warm_pp':100*(test_opd-test_warm),'opd_vs_continued_sft_pp':100*(test_opd-test_control),
             'teacher_gap_closed_fraction':(test_opd-test_base)/gap if gap>0 else None,
             'matched_branch_budget_steps':selection['matched_branch_budget_steps'],
             'student_improved':test_opd>test_base,'opd_increment_observed':test_opd>max(test_warm,test_control),
             'notes':'Single seed, fixed held-out 300 subset, validation-selected checkpoints. No statistical significance or full-dev claim.'}
    save(RUN/'outcome.json',outcome)
    pipeline.state.update(status='complete',completed_at=stamp(),outcome=outcome,
        goal_status='observed_improvement' if outcome['student_improved'] and outcome['opd_increment_observed'] else 'needs_further_research')
    pipeline.update('complete'); pipeline.resources()


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(); pipeline=Pipeline(args.resume)
    try:
        execute(pipeline)
    except Exception as error:
        stop(pipeline.child_process); stop(pipeline.rollout_process)
        pipeline.state.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc(),failed_at=stamp())
        save(STATE,pipeline.state); raise
    finally:
        stop(pipeline.child_process); stop(pipeline.rollout_process)
        pipeline.state.pop('child_pid',None); pipeline.state.pop('rollout_pid',None)
        pipeline.state['gpu_processes_stopped']=True; save(STATE,pipeline.state)


if __name__=='__main__': main()
