"""CPU protocol checks; no weights, CUDA context or HTTP server are started."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/server'))
spec = importlib.util.spec_from_file_location('experiment_evaluation', ROOT / 'scripts/server/evaluate_experiment.py')
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    project = tmp_path / 'project'
    launcher = project / 'scripts/server/start_experiment_rollout.sh'
    launcher.parent.mkdir(parents=True)
    launcher.write_text('fixed seq1 .75 16k launcher\n', encoding='utf-8')
    models = {}
    for role in ('student', 'teacher'):
        models[role] = tmp_path / role
        models[role].mkdir()
        for name in ('config.json', 'model.safetensors.index.json', 'tokenizer.json'):
            (models[role] / name).write_text('{}', encoding='utf-8')
    db = tmp_path / 'db.sqlite'
    with sqlite3.connect(db) as connection:
        connection.execute('CREATE TABLE t(x)')
    records = [{'id': f'q{index}', 'db_id': 'db', 'db_path': str(db), 'question': 'return seven',
                'schema': 'CREATE TABLE t(x);', 'gold_sql': 'SELECT 7', 'evidence': '',
                'source_split': 'train', 'split': 'internal_validation', 'difficulty': 'unknown'}
               for index in (1, 2)]
    data = tmp_path / 'records.jsonl'
    data.write_text(''.join(json.dumps(row) + '\n' for row in records), encoding='utf-8')
    monkeypatch.setattr(experiment, 'ROOT', project)
    monkeypatch.setattr(experiment, 'MODELS', models)
    monkeypatch.setattr(experiment, 'canonical_prompts', lambda rows, role:
                        (SimpleNamespace(encode=lambda *args, **kwargs: [1, 2]), [[10], [20]]))
    monkeypatch.setattr(experiment, 'inspect_engine', lambda url, model: {'model_id': str(model), 'max_model_len': 16384})
    return SimpleNamespace(root=tmp_path, records=records, data=data, models=models, launcher=launcher)


def args(fixture, name='run', role='student', checkpoint=None, reference=None):
    return argparse.Namespace(records=fixture.data, role=role, output=fixture.root / name,
                              checkpoint=checkpoint, reference_run=reference, url='http://localhost:8001')


def successful_generation(record, prompt, url, model, maximum, context):
    assert maximum == 512 and context == 16384
    return {'id': record['id'], 'db_id': record['db_id'], 'model': model,
            'input_tokens': len(prompt), 'output_tokens': 2, 'sql': 'SELECT 7',
            'raw_output': 'SELECT 7', 'finish_reason': 'stop', 'error': None,
            'generation_seconds': 0.01}


def checkpoint(fixture):
    path = fixture.root / 'checkpoint-4'
    path.mkdir()
    for name in ('adapter_model.safetensors', 'adapter_config.json'):
        (path / name).write_bytes(b'CPU fixture, not real weights')
    (path / 'complete.json').write_text(json.dumps({'step': 4, 'sha256': {
        name: experiment.sha256(path / name) for name in ('adapter_model.safetensors', 'adapter_config.json')}}))
    return path


def test_interrupted_generation_resumes_by_id_and_preserves_prior_bytes(fixture, monkeypatch):
    calls = []
    def interrupted(record, *rest):
        calls.append(record['id'])
        if record['id'] == 'q2':
            raise RuntimeError('simulated interruption')
        return successful_generation(record, *rest)
    monkeypatch.setattr(experiment, 'generate', interrupted)
    options = args(fixture)
    with pytest.raises(RuntimeError, match='interruption'):
        experiment.run(options)
    previous = (options.output / 'predictions.jsonl').read_bytes()
    status = json.loads((options.output / 'status.json').read_text())
    assert status['status'] == 'failed' and status['completed'] == 1
    calls.clear()
    def resumed(record, *rest):
        calls.append(record['id'])
        return successful_generation(record, *rest)
    monkeypatch.setattr(experiment, 'generate', resumed)
    result = experiment.run(options)
    assert calls == ['q2']
    assert (options.output / 'predictions.jsonl').read_bytes().startswith(previous)
    assert result['status'] == 'complete' and result['completed'] == 2
    assert result['summary']['bird_correct'] == 2
    # A completed run reuses its hash-bound report without regenerating or appending.
    final_bytes = (options.output / 'predictions.jsonl').read_bytes()
    calls.clear()
    def no_engine_needed(*args):
        raise RuntimeError('Completed runs must permit CPU-only report recovery')
    monkeypatch.setattr(experiment, 'inspect_engine', no_engine_needed)
    experiment.run(options)
    assert calls == [] and (options.output / 'predictions.jsonl').read_bytes() == final_bytes


def test_reference_allows_teacher_role_but_rejects_new_snapshot(fixture, monkeypatch):
    monkeypatch.setattr(experiment, 'generate', successful_generation)
    base = args(fixture)
    experiment.run(base)
    teacher = args(fixture, 'teacher-run', role='teacher', reference=base.output)
    experiment.run(teacher)
    student_manifest = json.loads((base.output / 'manifest.json').read_text())
    teacher_manifest = json.loads((teacher.output / 'manifest.json').read_text())
    assert student_manifest['model'] != teacher_manifest['model']
    assert student_manifest['prompt_ids_sha256'] == teacher_manifest['prompt_ids_sha256']
    fixture.data.write_text(fixture.data.read_text() + '\n')
    with pytest.raises(ValueError, match='records_sha256'):
        experiment.run(args(fixture, 'changed', reference=base.output))


def test_reference_rejects_launcher_or_decoding_changes(fixture):
    manifest = {key: 'frozen' for key in experiment.IDENTITY_KEYS}
    for key in ('launcher_sha256', 'generation_settings', 'prompt_ids_sha256'):
        with pytest.raises(ValueError, match=key):
            experiment.check_identity({**manifest, key: 'changed'}, manifest)


def test_checkpoint_checksums_and_real_sync_metadata_are_separate(fixture, monkeypatch):
    path = checkpoint(fixture)
    source = experiment.verify_checkpoint(path)
    assert source['step'] == 4
    calls = []
    def synced(given, verified, tokenizer, url):
        calls.append(given)
        assert verified['complete_sha256'] == source['complete_sha256']
        return {'effective_weight_sha256': 'real-mock-digest', 'parent_training_sync_verified': False}
    monkeypatch.setattr(experiment, 'synchronize_checkpoint', synced)
    monkeypatch.setattr(experiment, 'generate', successful_generation)
    options = args(fixture, checkpoint=path)
    experiment.run(options)
    assert calls == [path]
    saved = json.loads((options.output / 'checkpoint-sync.json').read_text())
    assert saved['parent_training_sync_verified'] is False
    assert saved['effective_weight_sha256'] == 'real-mock-digest'
    (path / 'adapter_model.safetensors').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum differs'):
        experiment.verify_checkpoint(path)


def test_existing_duplicate_unknown_or_torn_predictions_are_rejected(fixture):
    path = fixture.root / 'predictions.jsonl'
    for content in ('{"id":"q1"}\n{"id":"q1"}\n', '{"id":"unknown"}\n', '{"id":'):
        path.write_text(content)
        with pytest.raises((ValueError, json.JSONDecodeError)):
            experiment.completed_predictions(path, ['q1', 'q2'])


def test_canonical_prompt_uses_student_template_without_gold(monkeypatch):
    captured = []
    class Tokenizer:
        def get_vocab(self):
            return {'x': 1}
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs['enable_thinking'] is False
            captured.extend(messages)
            return json.dumps(messages)
        def encode(self, text, **kwargs):
            return [len(text)]
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoTokenizer=SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: Tokenizer())))
    record = {'id': 'q', 'question': 'question', 'schema': 'CREATE TABLE t(x);',
              'evidence': '', 'gold_sql': 'SECRET GOLD SQL'}
    experiment.canonical_prompts([record], 'teacher')
    assert 'SECRET GOLD SQL' not in json.dumps(captured)
    assert 'CREATE TABLE t(x);' in json.dumps(captured)


def test_sync_failure_closes_communicator_session_and_frees_sender(fixture, monkeypatch):
    cleaned = []
    class Model:
        def requires_grad_(self, value): return self
        def eval(self): return self
    class Client:
        def __init__(self, **kwargs):
            self.session = SimpleNamespace(close=lambda: cleaned.append('session'))
        def get_world_size(self): return 1
        def init_communicator(self, **kwargs): pass
        def close_communicator(self): cleaned.append('communicator')
    class Policy:
        def __init__(self, *args): pass
        def sync(self, *args): raise RuntimeError('transfer failed')
    factory = SimpleNamespace(from_pretrained=lambda *args, **kwargs: Model())
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(bfloat16='bf16', cuda=SimpleNamespace(
        empty_cache=lambda: cleaned.append('cache'))))
    monkeypatch.setitem(sys.modules, 'peft', SimpleNamespace(PeftModel=factory))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoModelForImageTextToText=factory))
    monkeypatch.setitem(sys.modules, 'trl.generation.vllm_client', SimpleNamespace(VLLMClient=Client))
    monkeypatch.setitem(sys.modules, 'opd_sql.onpolicy', SimpleNamespace(SynchronousPolicy=Policy))
    path = checkpoint(fixture)
    with pytest.raises(RuntimeError, match='transfer failed'):
        experiment.synchronize_checkpoint(path, experiment.verify_checkpoint(path),
            SimpleNamespace(encode=lambda *args, **kwargs: [1]), 'http://localhost:8001')
    assert cleaned == ['communicator', 'session', 'cache']


def test_actual_reused_http_request_matches_frozen_generation_settings(monkeypatch):
    import run_baselines
    sent = []
    class Response:
        def raise_for_status(self): pass
        def json(self):
            return {'choices': [{'text': 'SELECT 7', 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 2, 'completion_tokens': 2}}
    def post(url, **kwargs):
        sent.append(kwargs['json'])
        return Response()
    monkeypatch.setattr(run_baselines.requests, 'post', post)
    result = experiment.generate({'id': 'q', 'db_id': 'db'}, [1, 2],
        'http://localhost:8001', 'model', 512, 16384)
    assert result['sql'] == 'SELECT 7' and result['error'] is None
    for key in ('temperature', 'seed', 'top_p', 'top_k', 'min_p',
                'repetition_penalty', 'presence_penalty', 'frequency_penalty'):
        assert sent[0][key] == experiment.GENERATION_SETTINGS[key]
    assert sent[0]['max_tokens'] == experiment.GENERATION_SETTINGS['max_new_tokens']
