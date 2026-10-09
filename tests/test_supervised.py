import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from opd_sql.onpolicy import completion_positions
from opd_sql.supervised import (adapter_provenance, assert_internal_train, cross_entropy_hidden_backward,
    encode_supervised, linear_warmup_decay, validate_adapter_topology, validate_resume_manifest)


@pytest.mark.parametrize("chunk_tokens", [1, 2, 7])
def test_supervised_chunk_ce_matches_dense_hidden_and_backbone_gradients(chunk_tokens):
    torch.manual_seed(57)
    backbone = torch.nn.Linear(4, 6, bias=False).double()
    other = copy.deepcopy(backbone)
    head = torch.nn.Linear(6, 19, bias=False).double().requires_grad_(False)
    inputs = torch.randn(1, 8, 4, dtype=torch.double)
    positions = completion_positions(8, 3)
    targets = torch.tensor([[2, 5, 7, 12, 18]])  # final target represents EOS
    dense_hidden = backbone(inputs)
    dense_hidden.retain_grad()
    dense_logits = head(dense_hidden[:, positions]).float()
    dense_loss = F.cross_entropy(dense_logits.reshape(-1, 19), targets.reshape(-1))
    dense_loss.backward()
    hidden = other(inputs)
    hidden.retain_grad()
    info = cross_entropy_hidden_backward(hidden, head, targets, positions, 5, chunk_tokens)
    assert info["ce_sum"] / 5 == pytest.approx(dense_loss.item(), abs=3e-7)
    torch.testing.assert_close(hidden.grad, dense_hidden.grad, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(other.weight.grad, backbone.weight.grad, rtol=2e-6, atol=2e-7)
    assert hidden.grad[:, :2].count_nonzero() == 0
    assert hidden.grad[:, -1].count_nonzero() == 0
    assert head.weight.grad is None


def test_supervised_unequal_length_accumulation_matches_one_token_normalized_loss():
    torch.manual_seed(66)
    first = torch.nn.Linear(3, 4, bias=False)
    second = copy.deepcopy(first)
    head = torch.nn.Linear(4, 13, bias=False).requires_grad_(False)
    examples = [(torch.randn(1, length, 3), torch.randint(0, 13, (1, length - 2))) for length in (4, 7)]
    total_tokens = 7
    dense_sum = 0
    for inputs, targets in examples:
        selected = first(inputs)[:, completion_positions(inputs.shape[1], 2)]
        dense_sum = dense_sum + F.cross_entropy(head(selected).reshape(-1, 13), targets.reshape(-1), reduction="sum")
    (dense_sum / total_tokens).backward()
    for inputs, targets in examples:
        cross_entropy_hidden_backward(second(inputs), head, targets,
            completion_positions(inputs.shape[1], 2), total_tokens, 2)
    torch.testing.assert_close(second.weight.grad, first.weight.grad, rtol=2e-6, atol=2e-7)


class NativeTemplateStub:
    eos_token_id = 99

    def __init__(self, break_token_prefix=False):
        self.break_token_prefix = break_token_prefix
        self.saw_answer_in_prompt = False
        self.settings = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        self.settings.append((tokenize, enable_thinking))
        if add_generation_prompt:
            self.saw_answer_in_prompt = any(message.get("content") == "SELECT 1;" for message in messages)
            return "PREFIX"
        assert messages[-1] == {"role": "assistant", "content": "SELECT 1;"}
        return "PREFIXSQL<EOS>\n"

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        if text == "PREFIX":
            return [1, 2, 3]
        if self.break_token_prefix:
            return [1, 2, 30, 4, 5, 99, 198]
        return [1, 2, 3, 4, 5, 99, 198]


def test_supervised_native_prefix_keeps_gold_only_in_completion_and_includes_eos():
    tokenizer = NativeTemplateStub()
    record = {"id": "train:1", "db_id": "db", "question": "Return one", "schema": "CREATE TABLE t(id INT);", "gold_sql": " SELECT 1; "}
    result = encode_supervised(tokenizer, record)
    assert result["prompt_ids"] == [1, 2, 3]
    assert result["input_ids"] == [1, 2, 3, 4, 5, 99]
    assert result["completion_tokens"] == 3
    assert not tokenizer.saw_answer_in_prompt
    assert all(setting == (False, False) for setting in tokenizer.settings)


def test_supervised_checks_token_prefix_even_if_strings_have_same_prefix():
    record = {"id": "train:1", "question": "q", "db_id": "db", "schema": "s", "gold_sql": "SELECT 1;"}
    with pytest.raises(ValueError, match="token prefix"):
        encode_supervised(NativeTemplateStub(break_token_prefix=True), record)


def test_supervised_requires_explicit_internal_train_and_rejects_validation():
    record = {"id": "x", "db_id": "db", "question": "q", "schema": "s", "gold_sql": "SELECT 1;", "source_split": "train", "split": "internal_train"}
    assert_internal_train(record, 1)
    for patch in ({"source_split": "dev"}, {"split": "internal_validation"}, {"split": None}, {"gold_sql": ""}):
        with pytest.raises(ValueError):
            assert_internal_train({**record, **patch}, 1)


def test_warmup_scheduler_restore_preserves_next_lr_and_update():
    schedule = lambda step: linear_warmup_decay(step, 2, 10)
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.02)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    for _ in range(4):
        parameter.square().sum().backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    saved_parameter = parameter.detach().clone()
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    saved_scheduler = copy.deepcopy(scheduler.state_dict())
    expected_lr = optimizer.param_groups[0]["lr"]
    parameter.square().sum().backward()
    optimizer.step()
    scheduler.step()
    replacement = torch.nn.Parameter(saved_parameter)
    resumed_optimizer = torch.optim.AdamW([replacement], lr=0.02)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, schedule)
    resumed_optimizer.load_state_dict(saved_optimizer)
    resumed_scheduler.load_state_dict(saved_scheduler)
    assert resumed_optimizer.param_groups[0]["lr"] == expected_lr
    replacement.square().sum().backward()
    resumed_optimizer.step()
    resumed_scheduler.step()
    torch.testing.assert_close(replacement, parameter, rtol=0, atol=0)
    assert resumed_scheduler.last_epoch == scheduler.last_epoch
    assert resumed_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    assert schedule(0) == 0 and schedule(2) == 1 and schedule(10) == 0


def test_initial_adapter_must_reference_the_same_fixed_base_model_path(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    base_model = tmp_path / "fixed-model"
    base_model.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"test-adapter-hash")
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": str(base_model)}))
    assert adapter_provenance(adapter, base_model)["base_model_path"] == str(base_model.resolve())
    with pytest.raises(ValueError, match="base model path"):
        adapter_provenance(adapter, tmp_path / "other-model")


class TinyLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList([
            torch.nn.ModuleDict({name: torch.nn.Linear(2, 2) for name in
                ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj")})
            for _ in range(4)])
        self.model.visual = torch.nn.Linear(2, 2)


def test_peft_saved_compressed_targets_resume_checks_actual_injected_paths(tmp_path):
    from peft import LoraConfig, PeftModel, get_peft_model

    base = TinyLanguageModel()
    targets = [name for name, module in base.named_modules()
               if name.startswith("model.language_model.layers.") and isinstance(module, torch.nn.Linear)]
    student = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                                            target_modules=targets, bias="none"))
    student.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "adapter_config.json").read_text())
    assert len(saved["target_modules"]) < len(targets)  # PEFT's real optimization triggered.
    resumed = PeftModel.from_pretrained(TinyLanguageModel(), tmp_path, is_trainable=True)
    validate_adapter_topology(resumed, targets)
    with pytest.raises(ValueError, match="different set"):
        validate_adapter_topology(resumed, targets[:-1])
    first_layer = resumed.get_base_model().get_submodule(targets[0])
    first_layer.r["default"] = 4
    with pytest.raises(ValueError, match="per-module"):
        validate_adapter_topology(resumed, targets)


def migration_fixture(tmp_path):
    import opd_sql.supervised as supervised

    before = tmp_path / "before-supervised.py"
    after = tmp_path / "after-supervised.py"
    before.write_text("# original supervised source archived before repair\n")
    after.write_bytes(Path(supervised.__file__).read_bytes())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    old_sha, new_sha = digest(before), digest(after)
    manifest = {"data_sha256": "same-data", "settings": {"max_steps": 300},
                "code_sha256": {"supervised.py": new_sha, "onpolicy.py": "same-onpolicy", "prompts.py": "same-prompts"}}
    legacy = copy.deepcopy(manifest)
    legacy["code_sha256"]["supervised.py"] = old_sha
    legacy_hash = hashlib.sha256(json.dumps(legacy, sort_keys=True).encode()).hexdigest()
    completed = {"manifest_hash": legacy_hash, "step": 3, "sha256": {"training-state.pt": "same-state"}}
    receipt = {"version": 1, "original_manifest_hash": legacy_hash,
               "previous_supervised_sha256": old_sha, "current_supervised_sha256": new_sha,
               "before_source": {"path": str(before.resolve()), "sha256": old_sha},
               "after_source": {"path": str(after.resolve()), "sha256": new_sha},
               "reason": "Validate actual LoRA paths after equivalent PEFT suffix compression"}
    (tmp_path / "code-migration.json").write_text(json.dumps(receipt))
    return manifest, completed, receipt


def test_receipted_source_migration_preserves_original_checkpoint_and_new_future_manifest(tmp_path):
    manifest, completed, _ = migration_fixture(tmp_path)
    original = copy.deepcopy(completed)
    restored_hash, audit = validate_resume_manifest(manifest, completed, tmp_path)
    assert restored_hash == completed["manifest_hash"]
    assert audit["checkpoint_step"] == 3 and audit["checkpoint_sha256"] == completed["sha256"]
    assert audit["status"] == "manifest_compatibility_verified" and audit["restoration_verified"] is False
    assert completed == original  # The checkpoint's state/weights/hash are never rewritten.
    future = {**completed, "manifest_hash": audit["current_manifest_hash"], "step": 100}
    assert validate_resume_manifest(manifest, future, tmp_path) == (future["manifest_hash"], None)


@pytest.mark.parametrize("change", ["data", "settings", "other_code", "archive"])
def test_source_migration_refuses_other_lineage_or_unverified_archives(tmp_path, change):
    manifest, completed, receipt = migration_fixture(tmp_path)
    if change == "data":
        manifest["data_sha256"] = "different-data"
    elif change == "settings":
        manifest["settings"]["max_steps"] = 600
    elif change == "other_code":
        manifest["code_sha256"]["onpolicy.py"] = "different-onpolicy"
    else:
        Path(receipt["before_source"]["path"]).write_text("modified archive")
    with pytest.raises(ValueError, match="more than|archived source"):
        validate_resume_manifest(manifest, completed, tmp_path)


def test_source_migration_requires_explicit_receipt(tmp_path):
    manifest, completed, _ = migration_fixture(tmp_path)
    (tmp_path / "code-migration.json").unlink()
    with pytest.raises(ValueError, match="lineage/code differ"):
        validate_resume_manifest(manifest, completed, tmp_path)
