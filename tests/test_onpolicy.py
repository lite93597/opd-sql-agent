import copy
from pathlib import Path
import sys
import random

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from opd_sql.onpolicy import (assert_training_record, completion_positions,
    checkpoint_state, exact_merged, plain_parameter_name, restore_state,
    reverse_kl_hidden_backward)


def test_completion_alignment_includes_first_and_eos_excludes_prompt_and_final_hidden():
    targets = [11, 12, 13, 20, 21, 99]  # three prompt tokens, two SQL + EOS
    positions = completion_positions(len(targets), 3)
    assert list(range(len(targets)))[positions] == [2, 3, 4]
    assert targets[positions.start + 1:positions.stop + 1] == [20, 21, 99]
    with pytest.raises(ValueError):
        completion_positions(3, 3)


@pytest.mark.parametrize("chunk_tokens", [1, 2, 9])
def test_chunk_backward_matches_dense_loss_and_backbone_gradient(chunk_tokens):
    torch.manual_seed(19)
    backbone = torch.nn.Linear(5, 7, bias=False).double()
    other = copy.deepcopy(backbone)
    inputs = torch.randn(1, 8, 5, dtype=torch.double)
    teacher_hidden = torch.randn(1, 8, 9, dtype=torch.double)
    student_head = torch.nn.Linear(7, 23, bias=False).double().requires_grad_(False)
    teacher_head = torch.nn.Linear(9, 23, bias=False).double().requires_grad_(False)
    positions = completion_positions(8, 3)
    dense_hidden = backbone(inputs)
    dense_hidden.retain_grad()
    student_logp = F.log_softmax(student_head(dense_hidden[:, positions]).float(), -1)
    teacher_logp = F.log_softmax(teacher_head(teacher_hidden[:, positions]).float(), -1)
    dense_loss = (student_logp.exp() * (student_logp - teacher_logp)).sum(-1).mean()
    dense_loss.backward()
    hidden = other(inputs)
    hidden.retain_grad()
    info = reverse_kl_hidden_backward(hidden, teacher_hidden, student_head, teacher_head, positions, 5, chunk_tokens)
    assert info["completion_tokens"] == 5
    assert info["kl_sum"] / 5 == pytest.approx(dense_loss.item(), abs=2e-7)
    torch.testing.assert_close(hidden.grad, dense_hidden.grad, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(other.weight.grad, backbone.weight.grad, rtol=2e-6, atol=2e-7)
    assert hidden.grad[:, :2].count_nonzero() == 0
    assert hidden.grad[:, -1].count_nonzero() == 0
    assert student_head.weight.grad is None and teacher_head.weight.grad is None


def test_unequal_length_accumulation_uses_total_tokens_not_mean_of_means():
    torch.manual_seed(41)
    backbone = torch.nn.Linear(3, 4, bias=False)
    other = copy.deepcopy(backbone)
    sh = torch.nn.Linear(4, 17, bias=False).requires_grad_(False)
    th = torch.nn.Linear(6, 17, bias=False).requires_grad_(False)
    data = [(torch.randn(1, n, 3), torch.randn(1, n, 6), completion_positions(n, 2)) for n in (4, 7)]
    total = 2 + 5
    dense_sum = 0
    for inputs, teacher, positions in data:
        sp = F.log_softmax(sh(backbone(inputs)[:, positions]).float(), -1)
        tp = F.log_softmax(th(teacher[:, positions]).float(), -1)
        dense_sum = dense_sum + (sp.exp() * (sp - tp)).sum()
    (dense_sum / total).backward()
    for inputs, teacher, positions in data:
        reverse_kl_hidden_backward(other(inputs), teacher, sh, th, positions, total, 2)
    torch.testing.assert_close(other.weight.grad, backbone.weight.grad, rtol=2e-6, atol=2e-7)


def test_train_only_requires_explicit_split_and_rejects_dev():
    record = {"id": "x", "db_id": "db", "question": "q", "schema": "s", "source_split": "train", "split": "internal_train"}
    assert_training_record(record, 1)
    for changed in ({"source_split": "dev"}, {"split": "internal_validation"}, {"source_split": None}):
        with pytest.raises(ValueError):
            assert_training_record({**record, **changed}, 1)
    synthetic = {**record, "source_split": "synthetic", "split": "synthetic"}
    assert_training_record(synthetic, 1, probe=True)
    with pytest.raises(ValueError):
        assert_training_record(synthetic, 1)


def test_server_names_are_base_names_only():
    assert plain_parameter_name("base_model.model.model.language_model.layers.0.self_attn.q_proj.base_layer.weight") == "model.language_model.layers.0.self_attn.q_proj.weight"
    with pytest.raises(ValueError):
        plain_parameter_name("base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight")


def test_merge_restores_base_exactly_even_if_transfer_raises():
    class FakeLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = torch.nn.Linear(2, 2, bias=False).to(torch.bfloat16)
            self.lora_A = torch.nn.ModuleDict()
            self.merged_adapters = []

        @property
        def merged(self):
            return bool(self.merged_adapters)

        def get_base_layer(self):
            return self.base_layer

    class FakeStudent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = FakeLayer()

        def merge_adapter(self, safe_merge):
            with torch.no_grad():
                self.layer.base_layer.weight.add_(0.03125)
            self.layer.merged_adapters.append("default")

    student = FakeStudent()
    original = student.layer.base_layer.weight.detach().clone()
    with pytest.raises(RuntimeError, match="transfer failed"):
        with exact_merged(student) as report:
            assert report["cpu_backup_bytes"] == 8
            assert not torch.equal(student.layer.base_layer.weight, original)
            raise RuntimeError("transfer failed")
    assert torch.equal(student.layer.base_layer.weight, original)
    assert not student.layer.merged


def test_checkpoint_restores_next_random_draws_and_optimizer_update(monkeypatch):
    import numpy as np
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda values: None)
    torch.manual_seed(17)
    random.seed(17)
    np.random.seed(17)
    sampler = random.Random(17)
    parameter = torch.nn.Parameter(torch.randn(4))
    optimizer = torch.optim.AdamW([parameter], lr=0.03)
    parameter.square().sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    saved_parameter = parameter.detach().clone()
    saved = copy.deepcopy(checkpoint_state(optimizer, 1, 7, [2, 0, 1], 2, sampler, "manifest"))

    def next_update(value, optim, rng):
        draws = (random.random(), np.random.rand(), rng.random())
        noise = torch.randn_like(value)
        (value - noise).square().sum().backward()
        optim.step()
        optim.zero_grad()
        return draws, value.detach().clone()

    expected_draws, expected_parameter = next_update(parameter, optimizer, sampler)
    replacement = torch.nn.Parameter(saved_parameter)
    replacement_optimizer = torch.optim.AdamW([replacement], lr=0.03)
    replacement_sampler = random.Random(1234)
    restore_state(saved, replacement_optimizer, replacement_sampler, "manifest")
    actual_draws, actual_parameter = next_update(replacement, replacement_optimizer, replacement_sampler)
    assert actual_draws == expected_draws
    torch.testing.assert_close(actual_parameter, expected_parameter, rtol=0, atol=0)
    assert saved["order"][saved["cursor"]] == 1
    with pytest.raises(ValueError):
        restore_state(saved, replacement_optimizer, replacement_sampler, "different-data")
