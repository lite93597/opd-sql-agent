#!/usr/bin/env python3
"""Two fixed BIRD baselines with one vLLM model per GPU and identical prompts."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import requests

from opd_sql.prompts import extract_sql, format_messages


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def start_engine(model, gpu, port, context, log_path):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), VLLM_WORKER_MULTIPROC_METHOD='spawn')
    command = [str(Path(sys.executable).parent/'vllm'), 'serve', str(model), '--host', '127.0.0.1',
               '--port', str(port), '--tensor-parallel-size', '1', '--data-parallel-size', '1',
               '--dtype', 'bfloat16', '--gpu-memory-utilization', '0.75',
               '--max-model-len', str(context), '--max-num-seqs', '4',
               '--model-impl', 'vllm', '--language-model-only', '--enforce-eager',
               '--generation-config', 'vllm']
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    return process


def stop_engine(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def generate(record, prompt, url, model, maximum, context):
    begin = time.monotonic()
    result = {'id': record['id'], 'db_id': record['db_id'], 'model': model,
              'input_tokens': len(prompt), 'output_tokens': 0, 'sql': '', 'raw_output': '',
              'finish_reason': None, 'error': None}
    if len(prompt) + maximum > context:
        result.update(error='context_limit: full schema was retained', finish_reason='context_limit')
    else:
        try:
            response = requests.post(url+'/v1/completions', json={
                'model': model, 'prompt': prompt, 'max_tokens': maximum, 'temperature': 0.0,
                'top_p': 1.0, 'top_k': -1, 'min_p': 0.0, 'repetition_penalty': 1.0,
                'presence_penalty': 0.0, 'frequency_penalty': 0.0, 'seed': 42,
            }, timeout=(10, 600))
            response.raise_for_status()
            data = response.json()
            result['raw_output'] = data['choices'][0]['text']
            result['sql'] = extract_sql(result['raw_output'])
            result['finish_reason'] = data['choices'][0]['finish_reason']
            result['output_tokens'] = data['usage']['completion_tokens']
            if data['usage']['prompt_tokens'] != len(prompt):
                raise RuntimeError('Server prompt token count differs from the fixed input')
        except Exception as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
            result['sql'] = ''
    result['generation_seconds'] = time.monotonic() - begin
    return result


def run(args, state):
    from transformers import AutoTokenizer
    record_bytes = args.records.read_bytes()
    records = [json.loads(line) for line in record_bytes.decode('utf-8-sig').splitlines() if line.strip()]
    ids = [r['id'] for r in records]
    if not records or len(ids) != len(set(ids)):
        raise ValueError('Baseline requires nonempty records with unique IDs')
    if any(r.get('source_split') != 'dev' for r in records):
        raise ValueError('This baseline must use the fixed official dev subset')
    tokenizer = AutoTokenizer.from_pretrained(args.student, local_files_only=True)
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher, local_files_only=True)
    if tokenizer.get_vocab() != teacher_tok.get_vocab():
        raise ValueError('Teacher and student token IDs differ')
    prompts = [tokenizer.encode(tokenizer.apply_chat_template(format_messages(r), tokenize=False,
               add_generation_prompt=True, enable_thinking=False), add_special_tokens=False) for r in records]
    if not all(isinstance(p, list) and p and all(isinstance(t, int) for t in p) for p in prompts):
        raise TypeError('Canonical prompts must be nonempty lists of token IDs')
    (args.output/'records.jsonl').write_bytes(record_bytes)
    manifest = {'started_at': stamp(), 'records_sha256': hashlib.sha256(record_bytes).hexdigest(),
                'record_ids': ids, 'seed': 42, 'prompt_template': 'student, enable_thinking=False',
                'temperature': 0, 'max_new_tokens': args.max_new_tokens, 'max_model_len': args.context,
                'repair_attempts': 0, 'oracle_schema_selection': False,
                'prompt_tokens': {'min': min(map(len, prompts)), 'median': statistics.median(map(len, prompts)),
                                  'max': max(map(len, prompts))},
                'prompt_ids_sha256': hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
                'models': {'student': str(args.student), 'teacher': str(args.teacher)}}
    save(args.output/'manifest.json', manifest)
    state.update(phase='starting_vllm', total=len(records)*2, completed=0, engines={}, models={})
    save(args.state, state)
    engines = []
    try:
        for role, model, gpu, port in [('student', args.student, 1, 8001), ('teacher', args.teacher, 0, 8002)]:
            url = f'http://127.0.0.1:{port}'
            try:
                response = requests.get(url+'/health', timeout=2)
                if response.ok:
                    raise RuntimeError(f'Port {port} is already used; preserve that service and stop this run')
            except requests.ConnectionError:
                pass
            process = start_engine(model, gpu, port, args.context, args.output/f'{role}-vllm.log')
            engines.append((role, model, process, url))
            state['engines'][role] = {'pid': process.pid, 'physical_gpu': gpu, 'port': port}
            save(args.state, state)
        deadline = time.monotonic() + 1200
        for role, model, process, url in engines:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'{role} vLLM exited {process.returncode}; see {role}-vllm.log')
                try:
                    if requests.get(url+'/health', timeout=3).ok:
                        break
                except requests.RequestException:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'{role} vLLM startup exceeded 20 minutes')
                time.sleep(3)
        state.update(phase='generating', inference_started_at=stamp())
        save(args.state, state)
        handles = {}
        outputs = {role: [] for role, *_ in engines}
        try:
            for role, *_ in engines:
                handles[role] = (args.output/f'{role}-predictions.jsonl').open('x', encoding='utf-8')
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                jobs = {}
                for record, prompt in zip(records, prompts):
                    for role, model, _, url in engines:
                        future = pool.submit(generate, record, prompt, url, str(model), args.max_new_tokens, args.context)
                        jobs[future] = role
                for future in concurrent.futures.as_completed(jobs):
                    role = jobs[future]
                    result = future.result()
                    handles[role].write(json.dumps(result, ensure_ascii=False)+'\n')
                    handles[role].flush()
                    outputs[role].append(result)
                    state['completed'] += 1
                    state['updated_at'] = stamp()
                    state['models'][role] = {'completed': len(outputs[role]),
                        'errors': sum(bool(r['error']) for r in outputs[role]),
                        'truncated': sum(r['finish_reason']=='length' for r in outputs[role])}
                    save(args.state, state)
                    print(json.dumps({'role':role,'completed':state['completed'],'total':state['total'],
                                      'id':result['id'],'error':result['error']}, ensure_ascii=False), flush=True)
        finally:
            for handle in handles.values():
                handle.close()
        for role, results in outputs.items():
            latency = [r['generation_seconds'] for r in results if not r['error']]
            save(args.output/f'{role}-generation-summary.json', {'model':str(dict((r,m) for r,m,*_ in engines)[role]),
                'total':len(results), 'errors':sum(bool(r['error']) for r in results),
                'truncated':sum(r['finish_reason']=='length' for r in results),
                'total_completion_tokens':sum(r['output_tokens'] for r in results),
                'mean_generation_seconds':statistics.mean(latency) if latency else None,
                'median_generation_seconds':statistics.median(latency) if latency else None,
                'latency_note':'Per-request latency includes queue waiting; 4 shared client workers, at most 4 sequences per model.'})
        state.update(status='complete', phase='generation_complete', completed_at=stamp())
    finally:
        for role, _, process, _ in engines:
            stop_engine(process)
        state['engines_stopped'] = True
        save(args.state, state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--student', type=Path, default=Path('/root/autodl-tmp/models/Qwen3.5-9B'))
    p.add_argument('--teacher', type=Path, default=Path('/root/autodl-tmp/models/Qwen3.8-27B'))
    p.add_argument('--context', type=int, default=16384)
    p.add_argument('--max-new-tokens', type=int, default=512)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Preserve existing run: {args.output}')
    args.output.mkdir(parents=True)
    state = {'status':'running','started_at':stamp(),'pid':os.getpid(),'output':str(args.output)}
    try:
        run(args, state)
    except Exception as exc:
        state.update(status='failed', phase='failed', error=f'{type(exc).__name__}: {exc}', updated_at=stamp())
        save(args.state, state)
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
