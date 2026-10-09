"""CPU-only numerical and frozen-evidence checks for effect reporting."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/analysis/summarize_effect.py'
spec = importlib.util.spec_from_file_location('effect_summary', SCRIPT)
effect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(effect)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def rows(values, databases=None):
    databases = databases or ['a', 'a', 'b', 'b']
    return [{'id': f'q{i}', 'db_id': db, 'bird_correct': correct}
            for i, (db, correct) in enumerate(zip(databases, values))]


def test_paired_gain_loss_sign_ids_and_zero_difference():
    target = rows([True, False, True, False])
    reference = rows([False, True, False, False])
    result = effect.paired_comparison(target, reference)
    assert result['gained'] == 2 and result['lost'] == 1
    assert result['gained_ids'] == ['q0', 'q2'] and result['lost_ids'] == ['q1']
    assert result['delta_pp'] == 25
    same = effect.paired_comparison(target, target)
    assert same['bootstrap']['ci95_pp'] == [0, 0]
    assert same['delta_pp'] == 0


def test_cluster_bootstrap_keeps_correlated_rows_and_question_weighting():
    # Two correlated gains in database a, no gain in b. Cluster resampling has
    # deltas {0, 2/3, 1}; an independent-row bootstrap would have a different mean.
    target = rows([True, True, False], ['a', 'a', 'b'])
    reference = rows([False, False, False], ['a', 'a', 'b'])
    result = effect.paired_comparison(target, reference)
    assert result['delta_pp'] == pytest.approx(200 / 3)
    assert result['bootstrap']['database_clusters'] == 2
    assert result['bootstrap']['ci95_pp'] == [0, 100]
    assert 55 < result['bootstrap']['bootstrap_mean_pp'] < 62
    assert result == effect.paired_comparison(target, reference)
    assert result['bootstrap']['seed'] == 20261004 and result['bootstrap']['replicates'] == 2000


def test_pair_membership_and_database_mismatch_rejected():
    with pytest.raises(ValueError, match='membership'):
        effect.paired_comparison(rows([True, False]), rows([True]))
    with pytest.raises(ValueError, match='database differs'):
        effect.paired_comparison(rows([True], ['a']), rows([True], ['b']))


@pytest.mark.parametrize('teacher', [.6, .5])
def test_nonpositive_teacher_gap_is_undefined(teacher):
    result = effect.gap_fractions({'base': .6, 'teacher': teacher, 'warm': .7, 'continued-sft': .7, 'opd': .8})
    assert result['total_pipeline_gap_closed_fraction'] is None
    assert result['opd_additional_vs_warm_gap_fraction'] is None
    assert result['undefined_reason'] is not None


def test_teacher_gap_separates_sft_and_opd_without_clipping():
    result = effect.gap_fractions({'base': .5, 'teacher': .7, 'warm': .6, 'continued-sft': .65, 'opd': .75})
    assert result['total_pipeline_gap_closed_fraction'] == pytest.approx(1.25)
    assert result['sft_warm_gap_closed_fraction'] == pytest.approx(.5)
    assert result['opd_additional_vs_warm_gap_fraction'] == pytest.approx(.75)
    assert result['opd_vs_continued_sft_gap_fraction'] == pytest.approx(.5)


@pytest.fixture
def bundle(tmp_path):
    run = tmp_path / 'effect-experiment-v1'
    run.mkdir()
    values = {'base': [True, True, False, False], 'teacher': [True] * 4,
              'warm': [True, True, True, False], 'continued-sft': [True, False, True, False], 'opd': [True] * 4}
    ids = [f'q{i}' for i in range(4)]
    records = ''.join(json.dumps({'id': record['id'], 'db_id': record['db_id'], 'source_split': 'dev',
                                 'split': 'dev_heldout_test', 'gold_sql': 'not read by analysis'}) + '\n'
                      for record in rows([True] * 4))
    record_sha = hashlib.sha256(records.encode()).hexdigest()
    identity = {'records_sha256': record_sha, 'record_ids': ids, 'prompt_ids_sha256': 'p' * 64,
                'generation_settings': {'seed': 42, 'temperature': 0, 'http_workers': 1},
                'engine_settings': {'max_num_seqs': 1}, 'launcher_sha256': 'l' * 64}
    selected = {}
    checkpoints = {}
    for arm, name, step, validation_accuracy in (('warm', 'sft-warmup', 2, .7),
                                                ('continued-sft', 'continued-sft', 3, .6),
                                                ('opd', 'opd-1e-5', 4, .8)):
        parent = run / name
        checkpoint = parent / f'checkpoint-{step}'
        checkpoint.mkdir(parents=True)
        for filename in ('adapter_config.json', 'adapter_model.safetensors'):
            (checkpoint / filename).write_bytes(f'{arm} CPU evidence fixture'.encode())
        config = {'max_steps': 4, 'gradient_accumulation_steps': 4, 'seed': 42 if arm == 'warm' else 43,
                  'learning_rate': 1e-5, 'max_seq_length': 8192, 'max_new_tokens': 512,
                  'lora_rank': 8, 'lora_alpha': 16, 'train_file': 'common-pool.jsonl'}
        if arm != 'warm': config['initial_adapter'] = str(checkpoints['warm'])
        parent_weights = effect.sha256(checkpoints['warm'] / 'adapter_model.safetensors') if arm != 'warm' else None
        training = {'data_sha256': 'd' * 64,
                    'algorithm': ('GKD reverse KL(student||teacher)' if arm == 'opd' else 'completion-only gold SQL cross entropy'),
                    'settings': {key: value for key, value in config.items() if arm != 'opd' or key != 'max_steps'}}
        if arm == 'warm': training['parent_adapter'] = None
        if arm == 'continued-sft':
            training['parent_adapter'] = {'adapter_weights_sha256': parent_weights, 'path': str(checkpoints['warm']),
                                          'adapter_config_sha256': effect.sha256(checkpoints['warm'] / 'adapter_config.json')}
        if arm == 'opd':
            training['initial_adapter'] = {'adapter_sha256': parent_weights, 'path': str(checkpoints['warm']),
                                           'step': 2, 'complete_sha256': effect.sha256(checkpoints['warm'] / 'complete.json')}
        training['manifest_hash'] = hashlib.sha256(json.dumps(training, sort_keys=True).encode()).hexdigest()
        write(parent / 'manifest.json', training)
        complete = {'step': step, 'manifest_hash': training['manifest_hash'], 'sha256': {
            filename: effect.sha256(checkpoint / filename) for filename in ('adapter_config.json', 'adapter_model.safetensors')}}
        write(checkpoint / 'complete.json', complete)
        write(parent / 'config.json', config)
        write(parent / 'status.json', {'status': 'pass', 'step': 4, 'elapsed_seconds': 1})
        metrics = [{'step': index, 'completion_tokens': index + 1, 'policy_version_sampled': index - 1,
                    'policy_version_synced': index} for index in range(1, 5)]
        (parent / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in metrics))
        if arm == 'opd': write(parent / f'sync-{step}.json', {'effective_weight_sha256': 'a' * 64})
        selected[arm.replace('-', '_')] = {'checkpoint': str(checkpoint), 'step': step, 'accuracy': validation_accuracy}
        checkpoints[arm] = checkpoint
        validation_name = 'warm' if arm == 'warm' else name
        val_dir = run / f'val-{validation_name}-{step}'
        write(val_dir / 'report.json', {'summary': {'bird_execution_accuracy': validation_accuracy},
                                      'input_files': {'records': {'sha256': 'v' * 64}}})
        write(val_dir / 'status.json', {'status': 'complete', 'completed_at': '2026-10-04T08:00:00+00:00'})
    selection = {**selected, 'created_at': '2026-10-04T09:00:00+00:00', 'source_split': 'internal_validation',
                 'test_untouched_before_selection': True, 'test_records_sha256': record_sha, 'matched_branch_budget_steps': 4}
    write(run / 'frozen-selection.json', selection)
    for arm in effect.ARMS:
        directory = run / ('test-' + arm)
        directory.mkdir()
        (directory / 'records.jsonl').write_bytes(records.encode())
        predictions = ''.join(json.dumps({'id': record_id, 'sql': 'fixture'}) + '\n' for record_id in ids)
        (directory / 'predictions.jsonl').write_text(predictions)
        decisions = [{**row, 'correct': row['bird_correct'],
                      'gold_execution': {'status': 'ok'}, 'prediction_execution': {'status': 'ok'},
                      'prediction_input_status': 'present', 'generation_failure_reason': None,
                      'error': None, 'finish_reason': 'stop'} for row in rows(values[arm])]
        correct = sum(values[arm])
        summary = {'total': 4, 'bird_correct': correct, 'bird_execution_accuracy': correct / 4,
                   'valid_gold': 4, 'invalid_gold': 0, 'gold_execution_status_counts': {'ok': 4},
                   'prediction_execution_status_counts': {'ok': 4}, 'generation_length_limits': 0,
                   **{name: 0 for name in ('missing_predictions', 'duplicate_predictions', 'generation_failures',
                                          'unknown_prediction_ids', 'malformed_predictions')}}
        report = {'results': decisions, 'summary': summary,
                  'reference_id_sha256': hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest(),
                  'additional_metric': {'name': 'bird_execution_set_equality', 'official_harness': False},
                  'execution_limits': {'timeout_seconds': 30, 'max_rows': 100000, 'max_result_bytes': 16777216},
                  'diagnostics': {'unknown_prediction_ids': [], 'duplicate_prediction_ids': [], 'malformed_prediction_indices': []},
                  'input_files': {'records': {'sha256': record_sha},
                                  'predictions': {'sha256': effect.sha256(directory / 'predictions.jsonl')}}}
        write(directory / 'report.json', report)
        source = None
        if arm in checkpoints:
            checkpoint = checkpoints[arm]
            complete = effect.read_json(checkpoint / 'complete.json')
            source = {'path': str(checkpoint), 'step': complete['step'],
                      'complete_sha256': effect.sha256(checkpoint / 'complete.json'), 'file_sha256': complete['sha256']}
            sync = {'checkpoint_complete_sha256': source['complete_sha256'],
                    'checkpoint_step': source['step'], 'effective_weight_sha256': 'a' * 64,
                    'parent_training_sync_verified': arm == 'opd'}
            if arm == 'opd': sync['parent_training_sync_sha256'] = effect.sha256(checkpoint.parent / f'sync-{source["step"]}.json')
            write(directory / 'checkpoint-sync.json', sync)
        write(directory / 'manifest.json', {**identity, 'role': 'teacher' if arm == 'teacher' else 'student',
                                            'gold_used_for_prompt': False, 'adapter_source': source})
        write(directory / 'status.json', {'status': 'complete', 'total': 4, 'completed': 4,
                                         'started_at': '2026-10-04T10:00:00+00:00'})
    scores = {arm: sum(value) / 4 for arm, value in values.items()}
    outcome = {'total': 4, 'student_base': scores['base'], 'teacher': scores['teacher'],
               'sft_warm': scores['warm'], 'continued_sft': scores['continued-sft'], 'opd': scores['opd'],
               'matched_branch_budget_steps': 4, 'teacher_gap_closed_fraction': 1,
               'opd_vs_base_pp': 50, 'opd_vs_warm_pp': 25, 'opd_vs_continued_sft_pp': 50,
               'student_improved': True, 'opd_increment_observed': True}
    write(run / 'outcome.json', outcome)
    return run


def test_complete_bundle_separates_effect_and_provenance(bundle):
    summary = effect.summarize(bundle)
    assert summary['protocol_verified'] and summary['total'] == 4
    assert summary['scores']['base']['correct'] == 2
    assert summary['scores']['opd']['correct'] == 4
    assert summary['scores']['opd']['input_and_generation_diagnostics']['generation_failures'] == 0
    assert summary['evaluation_protocol']['execution_limits']['timeout_seconds'] == 30
    assert summary['paired_comparisons']['opd_vs_warm']['delta_pp'] == 25
    assert summary['teacher_gap']['total_pipeline_gap_closed_fraction'] == 1
    assert summary['teacher_gap']['sft_warm_gap_closed_fraction'] == .5
    assert summary['matched_budget']['selected_checkpoint_steps_equal'] is False
    assert summary['training_lineage']['opd']['parent_training_sync_verified'] is True
    assert summary['training_lineage']['warm']['parent_training_sync_verified'] is False
    assert summary['conclusions']['opd_increment_over_warm_and_continued_sft'] is True


@pytest.mark.parametrize('filename,key,value,match', [
    ('test-opd/manifest.json', 'prompt_ids_sha256', 'changed', 'prompt_ids_sha256'),
    ('frozen-selection.json', 'matched_branch_budget_steps', 99, 'matched budget'),
    ('outcome.json', 'opd_vs_base_pp', 100, 'Outcome differs'),
    ('opd-1e-5/config.json', 'seed', 42, 'configuration differs'),
    ('test-opd/report.json', 'execution_limits', {'timeout_seconds': 25}, 'execution limits differ'),
    ('test-opd/checkpoint-sync.json', 'parent_training_sync_sha256', 'bad', 'synchronization evidence hash differs'),
])
def test_inconsistent_metadata_rejected(bundle, filename, key, value, match):
    path = bundle / filename
    changed = effect.read_json(path); changed[key] = value; write(path, changed)
    with pytest.raises(ValueError, match=match): effect.summarize(bundle)


def test_failed_gold_and_generation_diagnostics_prevent_effect_claim(bundle):
    path = bundle / 'test-opd/report.json'
    original = effect.read_json(path)
    changed = json.loads(json.dumps(original)); changed['summary']['generation_failures'] = 1; write(path, changed)
    with pytest.raises(ValueError, match='generation_failures'): effect.summarize(bundle)
    changed = json.loads(json.dumps(original)); changed['results'][0]['gold_execution']['status'] = 'timeout'; write(path, changed)
    with pytest.raises(ValueError, match='invalid gold'): effect.summarize(bundle)


def set_gold(bundle, arm, diagnostic, index=3):
    """Make a coherent synthetic report; preserve its full reference membership."""
    path = bundle / f'test-{arm}/report.json'
    report = effect.read_json(path)
    row = report['results'][index]
    row['gold_execution'] = dict(diagnostic)
    if diagnostic['status'] != 'ok':
        row['correct'] = row['bird_correct'] = False
    summary = report['summary']
    summary['valid_gold'] = sum(item['gold_execution']['status'] == 'ok' for item in report['results'])
    summary['invalid_gold'] = summary['total'] - summary['valid_gold']
    summary['gold_execution_status_counts'] = dict(Counter(item['gold_execution']['status'] for item in report['results']))
    summary['bird_correct'] = sum(item['bird_correct'] for item in report['results'])
    summary['bird_execution_accuracy'] = summary['bird_correct'] / summary['total']
    write(path, report)


def shared_invalid_gold(bundle, status='timeout'):
    diagnostic = {'status': status, 'columns': [], 'row_count': None,
                  'error': 'fixture execution failed', 'hard_timeout': status == 'timeout',
                  'elapsed_seconds': 30.01}
    if status == 'row_limit': diagnostic['observed_rows'] = 100001
    for arm in effect.ARMS:
        set_gold(bundle, arm, diagnostic)
    # q3 was previously correct only for teacher/OPD. All five arms retain four
    # reference questions; the invalid answer is now explicitly false for both.
    outcome = effect.read_json(bundle / 'outcome.json')
    outcome.update(teacher=.75, opd=.75, opd_vs_base_pp=25, opd_vs_warm_pp=0,
                   opd_vs_continued_sft_pp=25, opd_increment_observed=False)
    write(bundle / 'outcome.json', outcome)
    return diagnostic


@pytest.mark.parametrize('status', ['timeout', 'error', 'row_limit', 'result_limit'])
def test_shared_invalid_gold_keeps_full_denominator_and_reports_ids(bundle, status):
    shared_invalid_gold(bundle, status)
    summary = effect.summarize(bundle)
    assert summary['total'] == summary['gold_diagnostics']['main_denominator'] == 4
    assert summary['gold_diagnostics']['valid_gold'] == 3
    assert summary['gold_diagnostics']['invalid_gold'] == 1
    assert summary['gold_diagnostics']['invalid_gold_ids'] == ['q3']
    assert summary['gold_diagnostics']['status_counts'] == {'ok': 3, status: 1}
    assert summary['gold_diagnostics']['invalid_gold_records'][0]['execution']['status'] == status
    assert summary['scores']['teacher']['accuracy'] == .75
    assert summary['scores']['opd']['correct'] == 3
    assert all(row['total'] == 4 and row['invalid_gold'] == 1 for row in summary['scores'].values())
    assert summary['paired_comparisons']['opd_vs_base']['total'] == 4
    assert summary['paired_comparisons']['opd_vs_base']['delta_pp'] == 25
    assert summary['conclusions']['opd_increment_over_warm_and_continued_sft'] is False


def test_gold_execution_timer_changes_and_presence_do_not_change_comparison(bundle):
    diagnostic = shared_invalid_gold(bundle)
    for index, arm in enumerate(effect.ARMS):
        changed = {**diagnostic, 'elapsed_seconds': 30.01 + index,
                   'worker_elapsed_seconds': .02 + index}
        if arm == 'teacher': changed.pop('elapsed_seconds')
        set_gold(bundle, arm, changed)
    summary = effect.summarize(bundle)
    assert summary['gold_diagnostics']['all_arm_non_timing_diagnostics_identical']
    assert summary['gold_diagnostics']['comparison_excluded_fields'] == ['elapsed_seconds', 'worker_elapsed_seconds']
    assert 'elapsed_seconds' not in summary['gold_diagnostics']['invalid_gold_records'][0]['execution']


def test_equal_gold_failure_counts_at_different_ids_are_rejected(bundle):
    diagnostic = shared_invalid_gold(bundle)
    set_gold(bundle, 'teacher', {'status': 'ok'}, index=3)
    set_gold(bundle, 'teacher', diagnostic, index=2)
    with pytest.raises(ValueError, match='gold eligibility or non-timing diagnostics differ'):
        effect.summarize(bundle)


@pytest.mark.parametrize('key,value', [
    ('error', 'different failure'), ('hard_timeout', False), ('columns', ['changed']),
    ('row_count', 0), ('observed_rows', 100001), ('future_result_bytes', 123),
])
def test_gold_nontiming_diagnostic_changes_are_rejected(bundle, key, value):
    diagnostic = shared_invalid_gold(bundle)
    set_gold(bundle, 'opd', {**diagnostic, key: value})
    with pytest.raises(ValueError, match='gold eligibility or non-timing diagnostics differ'):
        effect.summarize(bundle)


@pytest.mark.parametrize('field', ['correct', 'bird_correct'])
def test_invalid_gold_cannot_score_as_correct_under_either_metric(bundle, field):
    shared_invalid_gold(bundle)
    path = bundle / 'test-opd/report.json'
    report = effect.read_json(path)
    report['results'][3][field] = True
    report['summary']['bird_correct'] = sum(row['bird_correct'] for row in report['results'])
    report['summary']['bird_execution_accuracy'] = report['summary']['bird_correct'] / 4
    write(path, report)
    with pytest.raises(ValueError, match='invalid gold execution was scored as correct'):
        effect.summarize(bundle)


def test_valid_gold_diagnostic_difference_also_prevents_comparison(bundle):
    set_gold(bundle, 'opd', {'status': 'ok', 'columns': ['different']}, index=0)
    with pytest.raises(ValueError, match='gold eligibility or non-timing diagnostics differ'):
        effect.summarize(bundle)


def test_gold_mismatch_audit_failure_preserves_raw_reports_and_frozen_selection(bundle):
    diagnostic = shared_invalid_gold(bundle)
    set_gold(bundle, 'teacher', {**diagnostic, 'error': 'different deadline'})
    paths = [bundle / f'test-{arm}/report.json' for arm in effect.ARMS]
    paths.extend([bundle / 'frozen-selection.json', bundle / 'outcome.json'])
    before = {path: path.read_bytes() for path in paths}
    with pytest.raises(SystemExit) as exc:
        effect.main(['--run-dir', str(bundle)])
    assert exc.value.code == 2
    result = effect.read_json(bundle / 'effect-summary.json')
    assert result['status'] == 'audit_failed' and not result['protocol_verified']
    assert 'gold eligibility or non-timing diagnostics differ' in result['error']
    assert all(path.read_bytes() == content for path, content in before.items())


def test_sql_failure_and_length_finish_are_reported_without_changing_denominator(bundle):
    path = bundle / 'test-base/report.json'
    changed = effect.read_json(path)
    changed['results'][3]['prediction_execution']['status'] = 'timeout'
    changed['results'][0]['finish_reason'] = 'length'
    changed['summary']['prediction_execution_status_counts'] = {'ok': 3, 'timeout': 1}
    changed['summary']['generation_length_limits'] = 1
    write(path, changed)
    summary = effect.summarize(bundle)
    assert summary['scores']['base']['correct'] == 2
    assert summary['scores']['base']['total'] == 4
    assert summary['scores']['base']['generation_length_limits'] == 1
    assert summary['scores']['base']['sql_execution_status_counts']['timeout'] == 1


def test_cli_default_output_is_complete_only_after_all_evidence_passes(bundle):
    result = effect.main(['--run-dir', str(bundle)])
    written = effect.read_json(bundle / 'effect-summary.json')
    assert written == result and written['status'] == 'complete'
    assert not (bundle / 'effect-summary.json.tmp').exists()


def test_stale_trainer_status_requires_a_verified_durable_budget_checkpoint(bundle):
    parent = bundle / 'continued-sft'
    write(parent / 'status.json', {'status': 'failed', 'step': 2})
    with pytest.raises(FileNotFoundError): effect.summarize(bundle)
    target = parent / 'checkpoint-4'
    target.mkdir()
    complete = effect.read_json(parent / 'checkpoint-3/complete.json')
    for filename in complete['sha256']:
        (target / filename).write_bytes((parent / 'checkpoint-3' / filename).read_bytes())
    complete['step'] = 4
    write(target / 'complete.json', complete)
    summary = effect.summarize(bundle)
    evidence = summary['training_lineage']['continued-sft']['completion_evidence']
    assert evidence['source'] == 'durable_target_checkpoint_and_continuous_optimizer_log'
    assert evidence['reported_status'] == 'failed'
    assert evidence['target_complete_sha256'] == effect.sha256(target / 'complete.json')


def test_cli_writes_audit_failure_instead_of_a_misleading_success(bundle):
    path = bundle / 'opd-1e-5/checkpoint-4/adapter_model.safetensors'
    path.write_bytes(b'corrupted after evaluation')
    output = bundle / 'effect-summary.json'
    with pytest.raises(SystemExit) as exc:
        effect.main(['--run-dir', str(bundle), '--output', str(output)])
    assert exc.value.code == 2
    summary = effect.read_json(output)
    assert summary['status'] == 'audit_failed' and summary['protocol_verified'] is False
    assert 'checkpoint bytes differ' in summary['error']
