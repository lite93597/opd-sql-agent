import importlib.util
import json
from pathlib import Path

import pytest

path=Path(__file__).resolve().parents[1]/'scripts/server/run_effect_experiment.py'
spec=importlib.util.spec_from_file_location('effect_pipeline',path)
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_restore_archives_nondurable_updates_instead_of_silently_overwriting_evidence(tmp_path):
    checkpoint=tmp_path/'checkpoint-100'; checkpoint.mkdir()
    (checkpoint/'adapter.safetensors').write_bytes(b'saved-at-step-100')
    info={'step':100,'sha256':{'adapter.safetensors':module.sha(checkpoint/'adapter.safetensors')}}
    (checkpoint/'complete.json').write_text(json.dumps(info))
    original=''.join(json.dumps({'step':i})+'\n' for i in range(1,151))
    (tmp_path/'metrics.jsonl').write_text(original)
    assert module.recover_training(tmp_path,300)==checkpoint
    assert json.loads((tmp_path/'metrics.jsonl').read_text().splitlines()[-1])['step']==100
    archives=list((tmp_path/'recovery-evidence').glob('*/metrics.jsonl'))
    assert len(archives)==1 and archives[0].read_text()==original


def test_frozen_final_test_resume_cannot_restart_training_or_checkpoint_selection(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'RUN',tmp_path)
    selection={'opd':{'checkpoint':'frozen-opd'},'test_records_sha256':'fixed-test'}
    (tmp_path/'frozen-selection.json').write_text(json.dumps(selection))
    called=[]
    monkeypatch.setattr(module,'common_pool',lambda:pytest.fail('Training preparation after opening test'))
    monkeypatch.setattr(module,'final_test',lambda pipeline,s:called.append(s))
    module.execute(object())
    assert called==[selection]


def test_temporary_checkpoint_with_complete_json_cannot_be_selected_for_restore(tmp_path):
    saved=tmp_path/'checkpoint-100'; saved.mkdir()
    (saved/'complete.json').write_text(json.dumps({'step':100,'sha256':{}}))
    temporary=tmp_path/'checkpoint-200.incomplete'; temporary.mkdir()
    (temporary/'complete.json').write_text(json.dumps({'step':200,'sha256':{}}))
    assert module.recover_training(tmp_path,300)==saved
    assert not temporary.exists()
    assert list((tmp_path/'recovery-evidence').glob('*/checkpoint-200.incomplete/complete.json'))
