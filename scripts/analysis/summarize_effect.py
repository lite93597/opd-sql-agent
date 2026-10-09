#!/usr/bin/env python3
"""Audit frozen five-arm BIRD results and report paired database bootstrap CIs.

This script uses only CPU / stdlib and never loads a model or executes SQL.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random


SEED = 20261004
REPLICATES = 2000
ARMS = ('base', 'teacher', 'warm', 'continued-sft', 'opd')
IDENTITY = ('records_sha256', 'record_ids', 'prompt_ids_sha256',
            'generation_settings', 'engine_settings', 'launcher_sha256')
GOLD_TIMING_FIELDS = {'elapsed_seconds', 'worker_elapsed_seconds'}


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_comparison(target: list[dict], reference: list[dict], *,
                      seed: int = SEED, replicates: int = REPLICATES) -> dict:
    """Question-weighted accuracy difference; resample paired database clusters."""
    require(replicates >= 1, 'At least one bootstrap replicate required')
    left = {row['id']: row for row in target}
    right = {row['id']: row for row in reference}
    require(len(left) == len(target) and len(right) == len(reference), 'Duplicate paired IDs')
    require(left.keys() == right.keys() and bool(left), 'Paired membership differs or is empty')
    clusters = defaultdict(lambda: {'total': 0, 'gained': 0, 'lost': 0})
    gained_ids, lost_ids = [], []
    for record_id, row in left.items():
        other = right[record_id]
        require(row['db_id'] == other['db_id'], f'Paired database differs: {record_id}')
        require(isinstance(row['bird_correct'], bool) and isinstance(other['bird_correct'], bool),
                'Paired correctness must be boolean')
        gained = row['bird_correct'] and not other['bird_correct']
        lost = other['bird_correct'] and not row['bird_correct']
        cluster = clusters[row['db_id']]
        cluster['total'] += 1; cluster['gained'] += int(gained); cluster['lost'] += int(lost)
        if gained: gained_ids.append(record_id)
        if lost: lost_ids.append(record_id)
    names = sorted(clusters)
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        selected = [clusters[rng.choice(names)] for _ in names]
        denominator = sum(cluster['total'] for cluster in selected)
        delta = sum(cluster['gained'] - cluster['lost'] for cluster in selected)
        draws.append(100 * delta / denominator)
    total, gained, lost = len(left), len(gained_ids), len(lost_ids)
    return {
        'total': total, 'gained': gained, 'lost': lost,
        'gained_ids': sorted(gained_ids), 'lost_ids': sorted(lost_ids),
        'delta_pp': 100 * (gained - lost) / total,
        'by_database': {name: {**value, 'delta_pp': 100 * (value['gained'] - value['lost']) / value['total']}
                        for name, value in sorted(clusters.items())},
        'bootstrap': {'method': 'paired database-cluster percentile bootstrap; question-weighted per draw',
                      'seed': seed, 'replicates': replicates, 'database_clusters': len(names),
                      'ci95_pp': [percentile(draws, .025), percentile(draws, .975)],
                      'bootstrap_mean_pp': sum(draws) / len(draws),
                      'scope': 'Resamples observed databases with replacement, preserving all within-database pairs. '
                               'Does not include training-seed, checkpoint-selection or backend numerical uncertainty.'},
    }


def gap_fractions(scores: dict[str, float]) -> dict:
    gap = scores['teacher'] - scores['base']
    valid = gap > 0
    return {
        'teacher_minus_base_pp': 100 * gap,
        'total_pipeline_gap_closed_fraction': (scores['opd'] - scores['base']) / gap if valid else None,
        'sft_warm_gap_closed_fraction': (scores['warm'] - scores['base']) / gap if valid else None,
        'opd_additional_vs_warm_gap_fraction': (scores['opd'] - scores['warm']) / gap if valid else None,
        'opd_vs_continued_sft_gap_fraction': (scores['opd'] - scores['continued-sft']) / gap if valid else None,
        'undefined_reason': None if valid else 'Teacher does not outperform raw base; nonpositive denominator',
        'notes': 'Total pipeline gain includes SFT warm-up and cannot be attributed to OPD alone. '
                 'Fractions are not clipped: negative values represent regression, values >1 exceed the observed teacher.',
    }


def gold_diagnostics(rows: list[dict]) -> dict:
    """Preserve every recorded gold diagnostic except execution timers."""
    return {row['id']: {key: value for key, value in row['gold_execution'].items()
                        if key not in GOLD_TIMING_FIELDS} for row in rows}


def audit_arm(run: Path, arm: str, reference: dict | None = None) -> dict:
    directory = run / ('test-' + arm)
    report = read_json(directory / 'report.json')
    manifest = read_json(directory / 'manifest.json')
    status = read_json(directory / 'status.json')
    rows = report['results']
    ids = [row['id'] for row in rows]
    require(bool(ids) and len(ids) == len(set(ids)), f'{arm}: empty or duplicate reference IDs')
    require(ids == manifest['record_ids'], f'{arm}: report IDs differ from frozen manifest order')
    expected_id_sha = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
    require(report['reference_id_sha256'] == expected_id_sha, f'{arm}: reference ID hash differs')
    files = report['input_files']
    require(files['records']['sha256'] == manifest['records_sha256'] == sha256(directory / 'records.jsonl'),
            f'{arm}: record snapshot hash differs')
    require(files['predictions']['sha256'] == sha256(directory / 'predictions.jsonl'),
            f'{arm}: prediction file hash differs')
    require(status['status'] == 'complete' and status['completed'] == status['total'] == len(rows),
            f'{arm}: evaluation has not completed every record')
    require(manifest['role'] == ('teacher' if arm == 'teacher' else 'student'), f'{arm}: wrong role')
    require(manifest.get('gold_used_for_prompt') is False, f'{arm}: prompt gold exclusion is unverified')
    require(report['additional_metric']['name'] == 'bird_execution_set_equality', f'{arm}: wrong metric')
    require(report['additional_metric'].get('official_harness') is False, f'{arm}: misleading official-harness label')
    summary = report['summary']
    correct = sum(row['bird_correct'] is True for row in rows)
    require(all(isinstance(row['bird_correct'], bool) for row in rows), f'{arm}: nonboolean correctness')
    require(summary['total'] == len(rows) and summary['bird_correct'] == correct,
            f'{arm}: summary count does not match per-record decisions')
    require(math.isclose(summary['bird_execution_accuracy'], correct / len(rows), abs_tol=1e-12),
            f'{arm}: summary accuracy differs')
    valid_gold = sum(row['gold_execution']['status'] == 'ok' for row in rows)
    require(summary['valid_gold'] == valid_gold and summary['invalid_gold'] == len(rows) - valid_gold,
            f'{arm}: invalid gold diagnostic counts differ from per-record executions')
    require(all(row.get('correct') is False and row['bird_correct'] is False
                for row in rows if row['gold_execution']['status'] != 'ok'),
            f'{arm}: invalid gold execution was scored as correct')
    require(summary['gold_execution_status_counts'] == dict(Counter(row['gold_execution']['status'] for row in rows)),
            f'{arm}: gold status registry differs')
    gold = gold_diagnostics(rows)
    actual_prediction_status = Counter(row['prediction_execution']['status'] if row['prediction_execution'] else 'not_executed'
                                       for row in rows)
    require(summary['prediction_execution_status_counts'] == dict(actual_prediction_status),
            f'{arm}: prediction SQL status registry differs')
    for key in ('missing_predictions', 'duplicate_predictions', 'generation_failures',
                'unknown_prediction_ids', 'malformed_predictions'):
        require(summary[key] == 0, f'{arm}: nonzero input/generation diagnostic {key}')
    diagnostics = report['diagnostics']
    require(not any(diagnostics.get(key) for key in ('unknown_prediction_ids', 'duplicate_prediction_ids',
                                                    'malformed_prediction_indices')), f'{arm}: invalid prediction IDs')
    require(all(row.get('prediction_input_status') == 'present' and not row.get('generation_failure_reason')
                and not row.get('error') and row.get('finish_reason') != 'error' for row in rows),
            f'{arm}: per-record missing/failed generation')
    require(summary['generation_length_limits'] == sum(row.get('finish_reason') == 'length' for row in rows),
            f'{arm}: generation length counter differs')
    if reference:
        for key in IDENTITY:
            require(manifest[key] == reference['manifest'][key], f'{arm}: reference identity differs: {key}')
        require(report['execution_limits'] == reference['report']['execution_limits'], f'{arm}: SQL execution limits differ')
        require(report['additional_metric'] == reference['report']['additional_metric'], f'{arm}: comparison protocol differs')
        require([(row['id'], row['db_id']) for row in rows] ==
                [(row['id'], row['db_id']) for row in reference['rows']], f'{arm}: database membership differs')
        require(gold == reference['gold_diagnostics'], f'{arm}: gold eligibility or non-timing diagnostics differ from base')
    return {'directory': directory, 'report': report, 'manifest': manifest, 'status': status,
            'gold_diagnostics': gold, 'valid_gold': valid_gold, 'invalid_gold': len(rows) - valid_gold,
            'rows': rows, 'correct': correct, 'total': len(rows), 'accuracy': correct / len(rows),
            'source_sha256': {name: sha256(directory / name) for name in
                              ('report.json', 'manifest.json', 'status.json')}}


def checkpoint_location(run: Path, source: str) -> Path:
    path = Path(source)
    if path.is_dir():
        return path
    # Allows a complete run/evidence copy to be audited on a different host.
    return run / path.parent.name / path.name


def audit_checkpoint(run: Path, arm: str, selected: dict, evaluation: dict, frozen_at: datetime) -> dict:
    path = checkpoint_location(run, selected['checkpoint'])
    complete = read_json(path / 'complete.json')
    source = evaluation['manifest']['adapter_source']
    require(source is not None and source['path'] == selected['checkpoint'], f'{arm}: checkpoint source differs from frozen selection')
    require(complete['step'] == selected['step'] == source['step'], f'{arm}: checkpoint step differs')
    require(source['complete_sha256'] == sha256(path / 'complete.json'), f'{arm}: completed checkpoint manifest hash differs')
    require(source['file_sha256'] == complete['sha256'], f'{arm}: checkpoint file registry differs')
    for filename, expected in complete['sha256'].items():
        require(Path(filename).name == filename, f'{arm}: unsafe checkpoint filename')
        require(sha256(path / filename) == expected, f'{arm}: checkpoint bytes differ: {filename}')
    training = read_json(path.parent / 'manifest.json')
    require(complete['manifest_hash'] == training['manifest_hash'], f'{arm}: checkpoint training manifest differs')
    payload = {key: value for key, value in training.items() if key != 'manifest_hash'}
    calculated = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    require(training['manifest_hash'] == calculated, f'{arm}: training manifest payload differs from its hash')
    config = read_json(path.parent / 'config.json')
    require(all(key in config and config[key] == value for key, value in training['settings'].items()),
            f'{arm}: configuration differs from its bound training manifest')
    train_status = read_json(path.parent / 'status.json')
    require(0 < selected['step'] <= config['max_steps'], f'{arm}: selected step exceeds its declared budget')
    metrics = [json.loads(line) for line in (path.parent / 'metrics.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    require([row['step'] for row in metrics] == list(range(1, config['max_steps'] + 1)), f'{arm}: optimizer history is not continuous')
    completion_evidence = {'source': 'trainer_report_pass', 'reported_status': train_status.get('status'),
                           'reported_step': train_status.get('step')}
    if not (train_status.get('status') == 'pass' and train_status.get('step') == config['max_steps']):
        # The runner can recover a durable final checkpoint saved immediately
        # before the process was interrupted writing its final status. Verify
        # that boundary ourselves; a stale status alone never proves completion.
        target = path.parent / f"checkpoint-{config['max_steps']}"
        terminal = read_json(target / 'complete.json')
        require(terminal['step'] == config['max_steps'] and terminal['manifest_hash'] == training['manifest_hash'],
                f'{arm}: durable training-budget boundary differs')
        require({'adapter_config.json', 'adapter_model.safetensors'} <= terminal['sha256'].keys(),
                f'{arm}: durable target checkpoint has no adapter registry')
        for filename, expected in terminal['sha256'].items():
            require(Path(filename).name == filename and sha256(target / filename) == expected,
                    f'{arm}: durable target checkpoint bytes differ: {filename}')
        completion_evidence.update(source='durable_target_checkpoint_and_continuous_optimizer_log',
                                   target_complete_sha256=sha256(target / 'complete.json'))
    sync = read_json(evaluation['directory'] / 'checkpoint-sync.json')
    require(sync['checkpoint_complete_sha256'] == source['complete_sha256'] and sync['checkpoint_step'] == selected['step'],
            f'{arm}: evaluation synchronization source differs')
    require(isinstance(sync.get('effective_weight_sha256'), str) and len(sync['effective_weight_sha256']) == 64,
            f'{arm}: real merged matrix hash is missing')
    if arm == 'opd':
        require(sync.get('parent_training_sync_verified') is True, 'opd: training/evaluation merged policy was not verified')
        original_sync = read_json(path.parent / f"sync-{selected['step']}.json")
        require(sync['parent_training_sync_sha256'] == sha256(path.parent / f"sync-{selected['step']}.json"),
                'opd: training synchronization evidence hash differs')
        require(sync['effective_weight_sha256'] == original_sync['effective_weight_sha256'], 'opd: merged policy differs from selected training checkpoint')
        require(all(row['policy_version_sampled'] == row['step'] - 1 and row['policy_version_synced'] == row['step']
                    for row in metrics), 'opd: stale or unacknowledged rollout version')
    validation_name = 'warm' if arm == 'warm' else path.parent.name
    validation_dir = run / f"val-{validation_name}-{selected['step']}"
    validation_report = read_json(validation_dir / 'report.json')
    validation_status = read_json(validation_dir / 'status.json')
    validation_score = validation_report['summary']['bird_execution_accuracy']
    require(math.isclose(selected['accuracy'], validation_score, abs_tol=1e-12), f'{arm}: frozen selection score differs from validation')
    require(validation_status['status'] == 'complete' and datetime.fromisoformat(validation_status['completed_at']) <= frozen_at,
            f'{arm}: validation completed after the final selection was frozen')
    validation_records_sha = validation_report['input_files']['records']['sha256']
    require(validation_records_sha != evaluation['manifest']['records_sha256'], f'{arm}: validation and test snapshots are identical')
    return {'source_checkpoint': selected['checkpoint'], 'selected_step': selected['step'],
            'training_budget_steps': config['max_steps'], 'effective_batch': config['gradient_accumulation_steps'],
            'training_seed': config['seed'], 'learning_rate': config['learning_rate'],
            'data_sha256': training['data_sha256'], 'config': config, 'training_manifest': training,
            'checkpoint_complete_sha256': source['complete_sha256'],
            'completion_evidence': completion_evidence,
            'effective_merged_weight_sha256': sync['effective_weight_sha256'],
            'parent_training_sync_verified': sync.get('parent_training_sync_verified', False),
            'selected_internal_validation_accuracy': validation_score,
            'internal_validation_records_sha256': validation_records_sha,
            'training_completion_tokens': sum(row['completion_tokens'] for row in metrics),
            'selected_checkpoint_completion_tokens': sum(row['completion_tokens'] for row in metrics if row['step'] <= selected['step']),
            'reported_last_training_process_elapsed_seconds': train_status.get('elapsed_seconds')}


def summarize(run: Path) -> dict:
    selection = read_json(run / 'frozen-selection.json')
    outcome = read_json(run / 'outcome.json')
    require(selection['source_split'] == 'internal_validation' and selection['test_untouched_before_selection'] is True,
            'Selection was not frozen using internal validation before the final test')
    arms = {}
    for name in ARMS:
        arms[name] = audit_arm(run, name, arms.get('base'))
    base = arms['base']
    require(base['manifest']['records_sha256'] == selection['test_records_sha256'], 'Final-test snapshot differs from frozen selection')
    require(all(arms[name]['manifest']['adapter_source'] is None for name in ('base', 'teacher')),
            'Raw base or teacher unexpectedly has an adapter')
    frozen_at = datetime.fromisoformat(selection['created_at'])
    require(all(datetime.fromisoformat(arm['status']['started_at']) >= frozen_at for arm in arms.values()),
            'A final-test generation started before checkpoint selection was frozen')
    lineage = {arm: audit_checkpoint(run, arm, selection[arm.replace('-', '_')], arms[arm], frozen_at)
               for arm in ('warm', 'continued-sft', 'opd')}
    warm_source = selection['warm']['checkpoint']
    budget = selection['matched_branch_budget_steps']
    for name in ('continued-sft', 'opd'):
        record = lineage[name]
        require(record['config'].get('initial_adapter') == warm_source, f'{name}: branch did not start from the frozen common SFT parent')
        require(record['training_budget_steps'] == budget, f'{name}: declared matched budget differs')
        require(record['data_sha256'] == lineage['warm']['data_sha256'], f'{name}: training pool differs from warm-up')
    control, opd = lineage['continued-sft'], lineage['opd']
    require(len({record['internal_validation_records_sha256'] for record in lineage.values()}) == 1,
            'Branches were selected on different internal validation snapshots')
    for key in ('gradient_accumulation_steps', 'max_seq_length', 'max_new_tokens', 'lora_rank', 'lora_alpha', 'seed', 'train_file'):
        require(control['config'][key] == opd['config'][key], f'Branches differ in controlled configuration: {key}')
    warm_weights = arms['warm']['manifest']['adapter_source']['file_sha256']['adapter_model.safetensors']
    warm_config = arms['warm']['manifest']['adapter_source']['file_sha256']['adapter_config.json']
    require(lineage['warm']['training_manifest']['parent_adapter'] is None, 'Warm-up unexpectedly has a parent adapter')
    require(control['training_manifest']['parent_adapter']['path'] == warm_source and
            control['training_manifest']['parent_adapter']['adapter_config_sha256'] == warm_config,
            'Continued SFT parent path or configuration differs from frozen warm-up')
    require(control['training_manifest']['parent_adapter']['adapter_weights_sha256'] == warm_weights,
            'Continued SFT source adapter bytes differ from frozen warm-up')
    require(opd['training_manifest']['initial_adapter']['adapter_sha256'] == warm_weights,
            'OPD source adapter bytes differ from frozen warm-up')
    require(opd['training_manifest']['initial_adapter']['path'] == warm_source and
            opd['training_manifest']['initial_adapter']['step'] == selection['warm']['step'] and
            opd['training_manifest']['initial_adapter']['complete_sha256'] == lineage['warm']['checkpoint_complete_sha256'],
            'OPD parent checkpoint identity differs from frozen warm-up')
    require('cross entropy' in control['training_manifest']['algorithm'] and 'reverse KL' in opd['training_manifest']['algorithm'],
            'Control / OPD objectives are mislabeled')
    scores = {name: arm['accuracy'] for name, arm in arms.items()}
    comparisons = {f'opd_vs_{name.replace("-", "_")}': paired_comparison(arms['opd']['rows'], arms[name]['rows'])
                   for name in ('base', 'warm', 'continued-sft')}
    gaps = gap_fractions(scores)
    expected = {'total': base['total'], 'student_base': scores['base'], 'teacher': scores['teacher'],
                'sft_warm': scores['warm'], 'continued_sft': scores['continued-sft'], 'opd': scores['opd'],
                'matched_branch_budget_steps': budget,
                'teacher_gap_closed_fraction': gaps['total_pipeline_gap_closed_fraction'],
                **{name + '_pp': value['delta_pp'] for name, value in comparisons.items()}}
    for key, value in expected.items():
        require(key in outcome, f'Outcome lacks required audited result: {key}')
        actual = outcome.get(key)
        valid = actual is None if value is None else isinstance(actual, (int, float)) and math.isclose(actual, value, abs_tol=1e-12)
        require(valid, f'Outcome differs from audited results: {key}')
    require(outcome['student_improved'] == (scores['opd'] > scores['base']) and
            outcome['opd_increment_observed'] == (scores['opd'] > max(scores['warm'], scores['continued-sft'])),
            'Outcome improvement flags differ from audited paired scores')
    db_counts = Counter(row['db_id'] for row in base['rows'])
    sanitized_lineage = {name: {key: value for key, value in info.items() if key not in ('config', 'training_manifest')}
                         for name, info in lineage.items()}
    return {
        'status': 'complete', 'protocol_verified': True,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'run_dir': str(run.resolve()), 'metric': 'bird_execution_set_equality',
        'official_harness': False, 'total': base['total'], 'database_count': len(db_counts),
        'database_question_counts': dict(sorted(db_counts.items())),
        'evaluation_protocol': {**{key: base['manifest'][key] for key in IDENTITY if key != 'record_ids'},
                                'reference_id_sha256': base['report']['reference_id_sha256'],
                                'execution_limits': base['report']['execution_limits']},
        'gold_diagnostics': {'valid_gold': base['valid_gold'], 'invalid_gold': base['invalid_gold'],
                             'status_counts': base['report']['summary']['gold_execution_status_counts'],
                             'invalid_gold_ids': [row['id'] for row in base['rows']
                                                  if row['gold_execution']['status'] != 'ok'],
                             'invalid_gold_records': [{'id': row['id'], 'db_id': row['db_id'],
                                                       'execution': base['gold_diagnostics'][row['id']]}
                                                      for row in base['rows'] if row['gold_execution']['status'] != 'ok'],
                             'main_denominator': base['total'],
                             'all_arm_non_timing_diagnostics_identical': True,
                             'comparison_excluded_fields': sorted(GOLD_TIMING_FIELDS),
                             'verification_scope': 'Recorded execution metadata; excludes timers. '
                                                   'Does not establish a shared SQL-result-row cache.'},
        'scores': {name: {'correct': arm['correct'], 'total': arm['total'], 'accuracy': arm['accuracy'],
                         'accuracy_percent': 100 * arm['accuracy'],
                         'valid_gold': arm['valid_gold'], 'invalid_gold': arm['invalid_gold'],
                         'gold_status_counts': arm['report']['summary']['gold_execution_status_counts'],
                         'input_and_generation_diagnostics': {key: arm['report']['summary'][key] for key in
                             ('missing_predictions', 'duplicate_predictions', 'generation_failures',
                              'unknown_prediction_ids', 'malformed_predictions')},
                         'generation_length_limits': arm['report']['summary']['generation_length_limits'],
                         'sql_execution_status_counts': arm['report']['summary']['prediction_execution_status_counts']}
                   for name, arm in arms.items()},
        'paired_comparisons': comparisons, 'teacher_gap': gaps,
        'training_lineage': sanitized_lineage,
        'matched_budget': {'optimizer_horizon_steps_per_branch': budget, 'effective_batch': control['effective_batch'],
                           'selected_checkpoint_steps_equal': control['selected_step'] == opd['selected_step'],
                           'notes': 'Both branches trained to the same declared horizon from the same SFT parent/pool. '
                                    'Validation may choose different checkpoint steps. Output tokens, wall time and '
                                    'total hyperparameter-search compute are not asserted equal.'},
        'conclusions': {'student_pipeline_improved_over_raw_base': scores['opd'] > scores['base'],
                        'opd_increment_over_warm_and_continued_sft': scores['opd'] > max(scores['warm'], scores['continued-sft']),
                        'opd_vs_control_ci95_excludes_zero_on_positive_side': comparisons['opd_vs_continued_sft']['bootstrap']['ci95_pp'][0] > 0,
                        'attribution': 'Raw-base to final OPD improvement includes SFT warm-up; OPD contribution '
                                       'must be reported separately against warm and matched continued SFT.'},
        'limits': [f'Single training seed per selected branch; only {len(db_counts)} observed final-test databases.',
                   'Fixed held-out subset, not a full BIRD dev or independent benchmark submission.',
                   'Percentile cluster intervals are descriptive with few clusters; they do not establish reproducibility across seeds.',
                   'Invalid gold executions count as incorrect for every arm in the fixed question denominator; '
                   'model correctness on those IDs is not measured.',
                   'Intervals omit checkpoint/LR-selection variability, greedy backend numerical effects and model-generation training variability.'],
        'input_sha256': {'frozen-selection.json': sha256(run / 'frozen-selection.json'),
                         'outcome.json': sha256(run / 'outcome.json'),
                         **{name: arm['source_sha256'] for name, arm in arms.items()}},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    output = args.output or args.run_dir / 'effect-summary.json'
    try:
        result = summarize(args.run_dir)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        result = {'status': 'audit_failed', 'protocol_verified': False, 'error': f'{type(exc).__name__}: {exc}'}
        code = 2
    else:
        code = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(output)
    print(json.dumps({'status': result['status'], 'output': str(output.resolve()), 'error': result.get('error')}))
    if code:
        raise SystemExit(code)
    return result


if __name__ == '__main__':
    main()
