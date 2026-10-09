import json
from pathlib import Path

import pytest

from opd_sql.onpolicy import scheduled_learning_rate, sha256, verify_initial_adapter


def test_linear_schedule_is_indexed_by_optimizer_update_with_fixed_resume_horizon():
    cfg = {'learning_rate': 1e-5, 'lr_schedule': 'linear', 'schedule_total_steps': 100, 'warmup_ratio': .05}
    assert scheduled_learning_rate(1, cfg) == pytest.approx(2e-6)
    assert scheduled_learning_rate(5, cfg) == 1e-5
    assert scheduled_learning_rate(6, cfg) == 1e-5
    assert scheduled_learning_rate(100, cfg) == pytest.approx(1e-5 / 95)
    uninterrupted = [scheduled_learning_rate(i, cfg) for i in range(1, 101)]
    resumed = [scheduled_learning_rate(i, cfg) for i in range(1, 51)] + [scheduled_learning_rate(i, cfg) for i in range(51, 101)]
    assert uninterrupted == resumed
    with pytest.raises(ValueError):
        scheduled_learning_rate(101, cfg)


def test_sft_initialization_binds_base_topology_and_adapter_checksum(tmp_path):
    base = tmp_path / 'base'
    base.mkdir()
    source = tmp_path / 'checkpoint-300'
    source.mkdir()
    (source / 'adapter_model.safetensors').write_bytes(b'adapter-content')
    (source / 'adapter_config.json').write_text(json.dumps({'r': 8, 'lora_alpha': 16, 'base_model_name_or_path': str(base)}))
    hashes = {p.name: sha256(p) for p in source.iterdir()}
    (source / 'complete.json').write_text(json.dumps({'step': 300, 'manifest_hash': 'source-manifest', 'sha256': hashes}))
    cfg = {'student_model': str(base), 'lora_rank': 8, 'lora_alpha': 16}
    assert verify_initial_adapter(source, cfg)['adapter_sha256'] == hashes['adapter_model.safetensors']
    (source / 'adapter_model.safetensors').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='SHA256'):
        verify_initial_adapter(source, cfg)
