#!/usr/bin/env python3
"""Serial, resumable baseline / adapter evaluation with frozen prompt identity."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback
from urllib.parse import urlparse

from opd_sql.prompts import format_messages
from run_baselines import generate, save, stamp


ROOT = Path('/root/autodl-tmp/opd-sql-agent')
MODELS = {
    'student': Path('/root/autodl-tmp/models/Qwen3.5-9B'),
    'teacher': Path('/root/autodl-tmp/models/Qwen3.8-27B'),
}
ENGINE_SETTINGS = {
    'physical_gpu': 1, 'tensor_parallel_size': 1, 'dtype': 'bfloat16',
    'model_impl': 'vllm', 'language_model_only': True, 'enforce_eager': True,
    'max_model_len': 16384, 'max_num_seqs': 1, 'gpu_memory_utilization': 0.75,
    'generation_config': 'vllm', 'weight_transfer_backend': 'nccl',
}
GENERATION_SETTINGS = {
    'max_new_tokens': 512, 'temperature': 0.0, 'seed': 42, 'top_p': 1.0,
    'top_k': -1, 'min_p': 0.0, 'repetition_penalty': 1.0,
    'presence_penalty': 0.0, 'frequency_penalty': 0.0,
    'http_workers': 1, 'repair_attempts': 0,
}
IDENTITY_KEYS = ('records_sha256', 'record_ids', 'prompt_ids_sha256',
                 'generation_settings', 'engine_settings', 'launcher_sha256')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_records(records: list[dict]) -> list[str]:
    ids = [row.get('id') for row in records]
    if not ids or any(not isinstance(value, str) or not value for value in ids):
        raise ValueError('Nonempty records with nonempty string IDs required')
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate reference IDs')
    return ids


def canonical_prompts(records: list[dict], role: str):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODELS['student'], local_files_only=True)
    if role == 'teacher':
        teacher = AutoTokenizer.from_pretrained(MODELS['teacher'], local_files_only=True)
        if teacher.get_vocab() != tokenizer.get_vocab():
            raise ValueError('Teacher and student token mappings differ')
    prompts = [tokenizer.encode(tokenizer.apply_chat_template(format_messages(record),
        tokenize=False, add_generation_prompt=True, enable_thinking=False),
        add_special_tokens=False) for record in records]
    if any(not tokens or not isinstance(tokens, list)
           or any(not isinstance(token, int) for token in tokens) for tokens in prompts):
        raise ValueError('Canonical prompts must be nonempty flat token ID lists')
    return tokenizer, prompts


def verify_checkpoint(path: Path) -> dict:
    complete_path = path / 'complete.json'
    complete = json.loads(complete_path.read_text(encoding='utf-8'))
    hashes = complete.get('sha256')
    if not isinstance(hashes, dict) or not {'adapter_config.json', 'adapter_model.safetensors'} <= hashes.keys():
        raise ValueError('Completed checkpoint must list adapter config and safetensors checksums')
    for filename, expected in hashes.items():
        if not isinstance(filename, str) or Path(filename).name != filename or filename in ('.', '..'):
            raise ValueError('Checkpoint manifest contains an unsafe filename')
        if sha256(path / filename) != expected:
            raise ValueError(f'Checkpoint checksum differs: {filename}')
    return {'path': str(path.resolve()), 'step': complete.get('step'),
            'complete_sha256': sha256(complete_path), 'file_sha256': hashes}


def inspect_engine(url: str, model: Path) -> dict:
    import requests
    response = requests.get(url + '/v1/models', timeout=15)
    response.raise_for_status()
    entries = response.json().get('data', [])
    matches = [entry for entry in entries if entry.get('id') == str(model)]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly the selected model {model} at {url}')
    observed_length = matches[0].get('max_model_len')
    if observed_length is not None and observed_length != ENGINE_SETTINGS['max_model_len']:
        raise ValueError('Server context differs from the frozen experiment configuration')
    return {'model_id': matches[0]['id'], 'max_model_len': observed_length,
            'configuration_source': 'frozen launcher SHA; model identity checked via /v1/models'}


def synchronize_checkpoint(path: Path, source: dict, tokenizer, url: str) -> dict:
    """Hash-verified adapter -> real merged BF16 vLLM policy; always close client."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('Adapter HF sender must see physical GPU 0 only')
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText
    from trl.generation.vllm_client import VLLMClient
    from opd_sql.onpolicy import SynchronousPolicy

    base = AutoModelForImageTextToText.from_pretrained(MODELS['student'], local_files_only=True,
        dtype=torch.bfloat16, device_map={'': 'cuda:0'}, attn_implementation='sdpa').requires_grad_(False)
    student = PeftModel.from_pretrained(base, path, is_trainable=False).eval()
    client = None
    policy = None
    try:
        client = VLLMClient(base_url=url, group_port=51216, connection_timeout=30)
        if client.get_world_size() != 1:
            raise ValueError('Expected a single TP1 rollout worker')
        client.init_communicator(device='cuda:0')
        policy = SynchronousPolicy(client, student, 42)
        prompt = tokenizer.encode('Write a SQLite query to count all rows in the users table.\nSQL:',
                                  add_special_tokens=False)
        result = policy.sync(0, prompt, 1)
        result['checkpoint_complete_sha256'] = source['complete_sha256']
        result['checkpoint_step'] = source['step']
        result['parent_training_sync_verified'] = False
        if source['step'] is not None:
            training_sync = path.parent / f"sync-{source['step']}.json"
            if training_sync.exists():
                expected = json.loads(training_sync.read_text(encoding='utf-8'))
                if expected['effective_weight_sha256'] != result['effective_weight_sha256']:
                    raise ValueError('Loaded checkpoint differs from its recorded merged training policy')
                result['parent_training_sync_verified'] = True
                result['parent_training_sync_sha256'] = sha256(training_sync)
        result['verification_scope'] = ('Real merged BF16 matrix digest and one greedy token. '
            'policy_version=0 is the evaluation client initial version; missing parent sync '
            'does not claim a training-time merged-policy verification.')
        return result
    finally:
        try:
            if client is not None:
                try:
                    client.close_communicator()
                finally:
                    client.session.close()
        finally:
            del policy, student, base
            torch.cuda.empty_cache()


def check_identity(actual: dict, expected: dict, *, resume=False) -> None:
    keys = IDENTITY_KEYS + (('role', 'model', 'adapter_source', 'model_metadata_sha256') if resume else ())
    for key in keys:
        if key not in expected or actual[key] != expected[key]:
            raise ValueError(f"{'Resume' if resume else 'Reference'} identity differs: {key}")


def completed_predictions(path: Path, ids: list[str]) -> list[dict]:
    if not path.exists():
        return []
    # A torn final JSONL line is deliberately rejected, rather than silently lost.
    rows = read_jsonl(path)
    completed = [row.get('id') for row in rows]
    if any(record_id not in ids for record_id in completed) or len(set(completed)) != len(completed):
        raise ValueError('Existing predictions contain unknown or duplicate IDs')
    return rows


def evaluate_sql(output: Path) -> dict:
    from opd_sql.bird_evaluation import main as evaluate
    predictions_sha = sha256(output / 'predictions.jsonl')
    report_path = output / 'report.json'
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding='utf-8'))
        files = report['input_files']
        if (files['predictions']['sha256'] != predictions_sha
                or files['records']['sha256'] != sha256(output / 'records.jsonl')):
            raise ValueError('Existing report does not match preserved input files')
        return report
    incomplete = output / 'report.incomplete.json'
    report = evaluate(['--records', str(output / 'records.jsonl'), '--predictions',
                       str(output / 'predictions.jsonl'), '--output', str(incomplete), '--overwrite'])
    # Only the disposable incomplete report can be replaced after interruption.
    # Completed predictions and a completed report are always preserved.
    incomplete.replace(report_path)
    return report


def run(args) -> dict:
    if args.role == 'teacher' and args.checkpoint is not None:
        raise ValueError('Adapter checkpoint is supported only for student evaluation')
    parsed = urlparse(args.url)
    if parsed.scheme != 'http' or parsed.hostname not in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('Evaluation requires an explicit local HTTP vLLM URL')
    args.url = args.url.rstrip('/')
    args.output.mkdir(parents=True, exist_ok=True)
    state = {'status': 'running', 'phase': 'initializing', 'role': args.role, 'started_at': stamp()}
    state_path = args.output / 'status.json'
    try:
        save(state_path, state)
        records_bytes = args.records.read_bytes()
        records = [json.loads(line) for line in records_bytes.decode('utf-8-sig').splitlines() if line.strip()]
        ids = validate_records(records)
        tokenizer, prompts = canonical_prompts(records, args.role)
        source = verify_checkpoint(args.checkpoint) if args.checkpoint else None
        launcher = ROOT / 'scripts/server/start_experiment_rollout.sh'
        manifest = {
            'role': args.role, 'model': str(MODELS[args.role]), 'adapter_source': source,
            'records_source': str(args.records.resolve()), 'records_sha256': hashlib.sha256(records_bytes).hexdigest(),
            'record_ids': ids, 'prompt_ids_sha256': hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
            'prompt_template': 'student canonical, enable_thinking=False', 'gold_used_for_prompt': False,
            'generation_settings': GENERATION_SETTINGS, 'engine_settings': ENGINE_SETTINGS,
            'launcher': str(launcher), 'launcher_sha256': sha256(launcher),
            'model_metadata_sha256': {name: sha256(MODELS[args.role] / name) for name in
                ('config.json', 'model.safetensors.index.json', 'tokenizer.json')},
        }
        if args.reference_run:
            reference = json.loads((args.reference_run / 'manifest.json').read_text(encoding='utf-8'))
            check_identity(manifest, reference)
            manifest['reference_run'] = str(args.reference_run.resolve())
        manifest_path = args.output / 'manifest.json'
        snapshot = args.output / 'records.jsonl'
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding='utf-8'))
            check_identity(manifest, existing, resume=True)
            if sha256(snapshot) != manifest['records_sha256']:
                raise ValueError('Existing record snapshot differs from the frozen manifest')
        else:
            if snapshot.exists() or (args.output / 'predictions.jsonl').exists():
                raise ValueError('Output has unbound records/predictions; choose a fresh output directory')
            snapshot.write_bytes(records_bytes)
            save(manifest_path, manifest)
        model = MODELS[args.role]
        results = completed_predictions(args.output / 'predictions.jsonl', ids)
        known = {row['id'] for row in results}
        state.update(total=len(records), completed=len(known), resumed_existing_ids=sorted(known),
            generation_errors=sum(bool(row.get('error')) or row.get('finish_reason') == 'error' for row in results),
            generation_length_limits=sum(row.get('finish_reason') == 'length' for row in results))
        if len(known) < len(records):
            observed = inspect_engine(args.url, model)
            save(args.output / 'engine-observation.json', observed)
        elif not (args.output / 'engine-observation.json').exists():
            raise ValueError('Completed predictions lack their recorded engine observation')
        if source and len(known) < len(records):
            state.update(phase='sync_checkpoint'); save(state_path, state)
            sync = synchronize_checkpoint(args.checkpoint, source, tokenizer, args.url)
            previous_sync = args.output / 'checkpoint-sync.json'
            if previous_sync.exists():
                previous = json.loads(previous_sync.read_text(encoding='utf-8'))
                if previous['effective_weight_sha256'] != sync['effective_weight_sha256']:
                    raise ValueError('Resumed evaluation loaded different merged adapter weights')
            save(previous_sync, sync)
        elif source and not (args.output / 'checkpoint-sync.json').exists():
            raise ValueError('Completed adapter predictions lack their real checkpoint synchronization report')
        state.update(phase='generating'); save(state_path, state)
        with (args.output / 'predictions.jsonl').open('a', encoding='utf-8') as stream:
            for record, prompt in zip(records, prompts):
                if record['id'] in known:
                    continue
                result = generate(record, prompt, args.url, str(model),
                                  GENERATION_SETTINGS['max_new_tokens'], ENGINE_SETTINGS['max_model_len'])
                if result.get('id') != record['id']:
                    raise ValueError('Generation returned a different reference ID')
                if source:
                    result['adapter_checkpoint'] = source['path']
                stream.write(json.dumps(result, ensure_ascii=False) + '\n'); stream.flush()
                results.append(result); known.add(record['id'])
                state.update(completed=len(known), updated_at=stamp(),
                    generation_errors=sum(bool(row.get('error')) or row.get('finish_reason') == 'error' for row in results),
                    generation_length_limits=sum(row.get('finish_reason') == 'length' for row in results))
                save(state_path, state)
        state.update(phase='execution_evaluation', generation_completed_at=stamp()); save(state_path, state)
        report = evaluate_sql(args.output)
        state.update(status='complete', phase='complete', completed_at=stamp(),
                     summary=report['summary'], predictions_sha256=sha256(args.output / 'predictions.jsonl'))
        save(state_path, state)
        return state
    except Exception as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc(), updated_at=stamp())
        save(state_path, state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--role', choices=('student', 'teacher'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--reference-run', type=Path)
    parser.add_argument('--url', default='http://127.0.0.1:8001')
    args = parser.parse_args(argv)
    return run(args)


if __name__ == '__main__':
    main()
