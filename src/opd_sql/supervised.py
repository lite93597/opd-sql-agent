"""Language-LoRA SFT warmup/continuation with completion-only chunked CE.

This stage uses training gold SQL explicitly. OPD remains a separate stage and
continues to use only student trajectories and teacher conditional distributions.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time
import traceback
from typing import Any

import torch
import torch.nn.functional as F

from opd_sql.onpolicy import (TARGETS, atomic_json, checkpoint_state,
    completion_positions, memory, native_hidden, restore_state, save_checkpoint, sha256)
from opd_sql.prompts import format_messages


def assert_internal_train(record: dict[str, Any], line: int) -> None:
    if record.get("source_split") != "train" or record.get("split") != "internal_train":
        raise ValueError(f"Line {line}: SFT requires source_split=train AND explicit split=internal_train")
    missing = {"id", "db_id", "question", "schema", "gold_sql"} - record.keys()
    if missing or not isinstance(record.get("gold_sql"), str) or not record["gold_sql"].strip():
        raise ValueError(f"Line {line}: missing/empty SFT fields: {sorted(missing)}")


def encode_supervised(tokenizer: Any, record: dict[str, Any]) -> dict[str, Any]:
    """Use one native template for rollout prefix and supervised SQL + EOS.

    Check token-prefix equality, not just string equality: BPE can merge across
    a boundary. The native template's newline AFTER EOS is not a generated
    completion target and is removed; no schema or SQL token is truncated.
    """
    messages = format_messages(record)
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    full_text = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": record["gold_sql"].strip()}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False)
    prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    full = tokenizer.encode(full_text, add_special_tokens=False)
    if not prompt or not isinstance(prompt, list) or full[:len(prompt)] != prompt:
        raise ValueError(f"{record['id']}: native supervised token prefix differs from student rollout prompt")
    eos = tokenizer.eos_token_id
    if not isinstance(eos, int):
        raise ValueError("Tokenizer must define one assistant EOS token ID")
    completion = full[len(prompt):]
    eos_indices = [i for i, token in enumerate(completion) if token == eos]
    if len(eos_indices) != 1 or eos_indices[0] == 0:
        raise ValueError(f"{record['id']}: expected nonempty SQL followed by one native assistant EOS")
    completion = completion[:eos_indices[0] + 1]
    return {"id": str(record["id"]), "db_id": record["db_id"], "prompt_ids": prompt,
            "input_ids": prompt + completion, "completion_tokens": len(completion)}


def cross_entropy_hidden_backward(hidden: torch.Tensor, head: Any, targets: torch.Tensor,
                                  positions: slice, denominator: int, chunk_tokens: int) -> dict[str, float | int]:
    """Full-vocabulary CE, bounded LM-head graph, then one backbone backward.

    targets [B,C] correspond one-to-one to hidden[:, positions]. Teacher and
    prompt-token probabilities are not used in this SFT objective.
    """
    if hidden.ndim != 3 or targets.ndim != 2 or denominator < 1 or chunk_tokens < 1:
        raise ValueError("Expected [B,L,H] hidden, [B,C] targets and positive normalizer/chunk")
    if any(parameter.requires_grad for parameter in head.parameters()):
        raise ValueError("Memory-bounded SFT requires a frozen LM head")
    selected = hidden[:, positions, :]
    if selected.shape[:2] != targets.shape or targets.numel() == 0:
        raise ValueError("Supervised hidden/target positions are misaligned or empty")
    gradient = torch.zeros_like(hidden)
    selected_grad = gradient[:, positions, :]
    loss_sum = 0.0
    for start in range(0, targets.shape[1], chunk_tokens):
        end = min(start + chunk_tokens, targets.shape[1])
        leaf = selected[:, start:end].detach().requires_grad_(True)
        logits = head(leaf)
        losses = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                 targets[:, start:end].reshape(-1), reduction="sum")
        loss = losses / denominator
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite chunked supervised CE")
        loss_sum += losses.detach().double().item()
        loss.backward()
        if leaf.grad is None or not torch.isfinite(leaf.grad).all():
            raise FloatingPointError("Missing/nonfinite supervised hidden gradient")
        selected_grad[:, start:end].copy_(leaf.grad)
        del leaf, logits, losses, loss
    hidden.backward(gradient)
    return {"ce_sum": loss_sum, "completion_tokens": targets.numel()}


def linear_warmup_decay(step: int, warmup_steps: int, total_steps: int) -> float:
    if total_steps < 1 or not 0 <= warmup_steps < total_steps:
        raise ValueError("Need 0 <= warmup_steps < total_steps")
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    return max(0.0, (total_steps - step) / (total_steps - warmup_steps))


def prepare_examples(path: Path, tokenizer: Any, cfg: dict[str, Any], out: Path) -> list[dict[str, Any]]:
    examples, filtered, seen = [], [], set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            assert_internal_train(record, line_number)
            if str(record["id"]) in seen:
                raise ValueError(f"Duplicate SFT ID: {record['id']}")
            seen.add(str(record["id"]))
            example = encode_supervised(tokenizer, record)
            reason = None
            if len(example["prompt_ids"]) > cfg["max_seq_length"] - cfg["max_new_tokens"]:
                reason = "prompt_too_long"
            elif example["completion_tokens"] > cfg["max_new_tokens"]:
                reason = "gold_completion_too_long"
            if reason is not None:
                filtered.append({"id": record["id"], "prompt_tokens": len(example["prompt_ids"]),
                    "completion_tokens": example["completion_tokens"], "reason": reason,
                    "schema_truncated": False, "sql_truncated": False})
            else:
                examples.append(example)
    atomic_json(out / "length-filter.json", {"read": len(seen), "accepted": len(examples),
                "filtered": filtered, "data_sha256": sha256(path), "schema_truncated": False, "sql_truncated": False})
    if not examples:
        raise ValueError("No internal-train SFT examples fit; inspect length-filter.json")
    return examples


def adapter_provenance(path: Path, model_path: Path) -> dict[str, Any]:
    if not (path / "adapter_config.json").is_file() or not (path / "adapter_model.safetensors").is_file():
        raise ValueError(f"Expected a saved PEFT adapter at {path}")
    adapter_config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    base_reference = adapter_config.get("base_model_name_or_path")
    if not base_reference or Path(base_reference).resolve() != model_path.resolve():
        raise ValueError("Parent adapter references a different/unpinned base model path")
    return {"path": str(path.resolve()), "adapter_config_sha256": sha256(path / "adapter_config.json"),
            "adapter_weights_sha256": sha256(path / "adapter_model.safetensors"),
            "base_model_path": str(model_path.resolve())}


def validate_adapter_topology(student: Any, targets: list[str]) -> None:
    """Compare injected modules, since PEFT may compress target names to suffixes."""
    from peft.tuners.lora.layer import LoraLayer

    settings = student.peft_config["default"]
    if settings.r != 8 or settings.lora_alpha != 16 or settings.lora_dropout != 0:
        raise ValueError("Parent/resumed adapter must use the same rank/alpha/dropout/language targets")
    actual = {name: module for name, module in student.get_base_model().named_modules()
              if isinstance(module, LoraLayer) and "default" in module.r}
    if set(actual) != set(targets):
        raise ValueError("Parent/resumed adapter injected a different set of language targets")
    for module in actual.values():
        dropout = module.lora_dropout["default"]
        if (module.r["default"] != 8 or module.lora_alpha["default"] != 16
                or (isinstance(dropout, torch.nn.Dropout) and dropout.p != 0)):
            raise ValueError("Parent/resumed adapter has a per-module rank/alpha/dropout mismatch")


def validate_resume_manifest(manifest: dict[str, Any], completed: dict[str, Any],
                             run_dir: Path) -> tuple[str, dict[str, Any] | None]:
    """Permit only a receipted supervised-source repair; all other lineage stays fixed."""
    current_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if completed["manifest_hash"] == current_hash:
        return current_hash, None
    receipt_path = run_dir / "code-migration.json"
    if not receipt_path.is_file():
        raise ValueError("SFT checkpoint data/models/settings/scheduler/lineage/code differ")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    old_sha = receipt["previous_supervised_sha256"]
    new_sha = receipt["current_supervised_sha256"]
    if (receipt.get("version") != 1 or not str(receipt.get("reason", "")).strip()
            or receipt["original_manifest_hash"] != completed["manifest_hash"]
            or new_sha != manifest["code_sha256"]["supervised.py"]
            or new_sha != sha256(Path(__file__)) or old_sha == new_sha):
        raise ValueError("Invalid or mismatched supervised code-migration receipt")
    for key, expected in (("before_source", old_sha), ("after_source", new_sha)):
        source = receipt[key]
        archived = Path(source["path"])
        if (not archived.is_absolute() or not archived.is_file()
                or source["sha256"] != expected or sha256(archived) != expected):
            raise ValueError(f"Code-migration archived source failed SHA256: {key}")
    legacy = {**manifest, "code_sha256": {**manifest["code_sha256"], "supervised.py": old_sha}}
    legacy_hash = hashlib.sha256(json.dumps(legacy, sort_keys=True).encode()).hexdigest()
    if legacy_hash != completed["manifest_hash"]:
        raise ValueError("Code migration changes more than the supervised source hash")
    audit = {"version": 1, "status": "manifest_compatibility_verified", "restoration_verified": False,
             "receipt_sha256": sha256(receipt_path),
             "original_manifest_hash": legacy_hash, "current_manifest_hash": current_hash,
             "previous_supervised_sha256": old_sha, "current_supervised_sha256": new_sha,
             "checkpoint_step": completed["step"], "checkpoint_sha256": completed["sha256"],
             "before_source": receipt["before_source"], "after_source": receipt["after_source"],
             "reason": receipt["reason"]}
    return legacy_hash, audit


def run(cfg: dict[str, Any], resume: Path | None = None, stop_after: int | None = None) -> dict[str, Any]:
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or torch.cuda.device_count() != 1:
        raise ValueError("SFT must see only physical GPU 0")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 GPU required")
    for name, expected in {"transformers": "5.18.0", "peft": "0.21.2"}.items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"Pinned {name} {expected} required")
    if cfg.get("mode") != "sft" or cfg["max_steps"] < 2:
        raise ValueError("This entry requires mode=sft, max_steps>=2")
    if not 0 <= cfg["warmup_ratio"] < 1 or not 0 < cfg["max_new_tokens"] < cfg["max_seq_length"] <= 16384:
        raise ValueError("Invalid warmup ratio or sequence length")
    if min(cfg["gradient_accumulation_steps"], cfg["ce_chunk_tokens"], cfg["save_steps"]) < 1:
        raise ValueError("Positive accumulation/chunk/save counts required")
    if cfg["lora_rank"] != 8 or cfg["lora_alpha"] != 16 or cfg["learning_rate"] <= 0:
        raise ValueError("Current supervised stage pins r=8, alpha=16, positive learning rate")
    final_step = cfg["max_steps"] if stop_after is None else stop_after
    if not 1 <= final_step <= cfg["max_steps"]:
        raise ValueError("stop_after must be within the fixed complete scheduler budget")
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.jsonl").exists() and resume is None:
        raise FileExistsError("Existing SFT run requires explicit resume or a fresh output directory")
    set_seed(cfg["seed"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["student_model"], local_files_only=True)
    examples = prepare_examples(Path(cfg["train_file"]), tokenizer, cfg, out)
    initial_adapter = Path(cfg["initial_adapter"]) if cfg.get("initial_adapter") else None
    if initial_adapter is not None and resume is not None:
        # A resumed branch retains initial_adapter in its manifest for lineage;
        # resume itself loads the later checkpoint, not the parent adapter twice.
        adapter_provenance(initial_adapter, Path(cfg["student_model"]))
    manifest = {"stage": "sft", "algorithm": "completion-only full-vocabulary gold SQL cross entropy",
        "data_sha256": sha256(Path(cfg["train_file"])), "student_model": cfg["student_model"],
        "parent_adapter": adapter_provenance(initial_adapter, Path(cfg["student_model"])) if initial_adapter else None,
        "model_metadata_sha256": {name: sha256(Path(cfg["student_model"]) / name)
            for name in ("config.json", "tokenizer.json", "model.safetensors.index.json")},
        "template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "code_sha256": {"supervised.py": sha256(Path(__file__)),
            "onpolicy.py": sha256(Path(__file__).with_name("onpolicy.py")),
            "prompts.py": sha256(Path(__file__).with_name("prompts.py"))},
        "settings": {key: value for key, value in cfg.items() if key not in {"output_dir", "save_steps"}},
        "scheduler": "linear warmup then linear decay; fixed total max_steps",
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")}}
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    restore_manifest_hash, migration = manifest_hash, None
    if resume is not None:
        completed = json.loads((resume / "complete.json").read_text(encoding="utf-8"))
        for filename, expected in completed["sha256"].items():
            if sha256(resume / filename) != expected:
                raise ValueError(f"Checkpoint SHA256 mismatch: {filename}")
        restore_manifest_hash, migration = validate_resume_manifest(manifest, completed, out)
        if migration is not None:
            migration["checkpoint"] = str(resume.resolve())
            atomic_json(out / "resume-code-migration.json", migration)
    atomic_json(out / "manifest.json", {**manifest, "manifest_hash": manifest_hash})
    atomic_json(out / "config.json", cfg)
    report = {"status": "running", "phase": "student_load", "step": 0, "stage": "sft",
        "training_started": False, "manifest_hash": manifest_hash, "parent_adapter": manifest["parent_adapter"]}
    if migration is not None:
        report["resume_code_migration"] = migration
    atomic_json(out / "status.json", report)
    try:
        base = AutoModelForImageTextToText.from_pretrained(cfg["student_model"], local_files_only=True,
            dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa").requires_grad_(False)
        if base.config.model_type != "qwen3_5":
            raise ValueError("Expected native Qwen3.5-family model")
        targets = [name for name, module in base.named_modules() if name.startswith("model.language_model.layers.")
            and name.rsplit(".", 1)[-1] in TARGETS and isinstance(module, torch.nn.Linear)]
        if {name.rsplit(".", 1)[-1] for name in targets} != TARGETS:
            raise ValueError("Missing pinned language-attention LoRA target types")
        adapter = resume or initial_adapter
        if adapter is None:
            student = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                target_modules=targets, task_type="CAUSAL_LM", bias="none"))
        else:
            student = PeftModel.from_pretrained(base, adapter, is_trainable=True)
            validate_adapter_topology(student, targets)
        for name, parameter in student.named_parameters():
            if parameter.requires_grad:
                if "lora_" not in name or ".language_model.layers." not in name:
                    raise ValueError(f"Unexpected trainable base/vision parameter: {name}")
                parameter.data = parameter.data.float()
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        student.enable_input_require_grads()
        student.config.use_cache = False
        for module in student.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        student.train()
        head = student.get_output_embeddings()
        trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=cfg["learning_rate"], weight_decay=0.0, foreach=False)
        warmup_steps = min(cfg["max_steps"] - 1, math.ceil(cfg["warmup_ratio"] * cfg["max_steps"]))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
            lambda step: linear_warmup_decay(step, warmup_steps, cfg["max_steps"]))
        sampler_rng = random.Random(cfg["seed"])
        order, cursor, initial_step, total_tokens = list(range(len(examples))), len(examples), 0, 0
        if resume is not None:
            state = torch.load(resume / "training-state.pt", map_location="cuda:0", weights_only=False)
            state["torch_rng"] = state["torch_rng"].cpu()
            state["cuda_rng"] = [item.cpu() for item in state["cuda_rng"]]
            restore_state(state, optimizer, sampler_rng, restore_manifest_hash)
            scheduler.load_state_dict(state["scheduler"])
            order, cursor, initial_step, total_tokens = state["order"], state["cursor"], state["step"], state["total_tokens"]
            if sorted(order) != list(range(len(examples))) or not 0 <= cursor <= len(order):
                raise ValueError("Invalid SFT sampler state")
            if scheduler.last_epoch != initial_step:
                raise ValueError("Scheduler update count does not match checkpoint step")
            for filename in ("metrics.jsonl",):
                if (out / filename).is_file():
                    rows = [json.loads(line) for line in (out / filename).read_text(encoding="utf-8").splitlines() if line.strip()]
                    if rows and rows[-1]["step"] != initial_step:
                        raise ValueError("SFT log/checkpoint step differ; resume into a fresh output directory")
        if initial_step >= final_step:
            raise ValueError("Resume/stop_after target must be after the saved step")
        if migration is not None:
            migration.update(status="restoration_verified", restoration_verified=True,
                             resumed_step=initial_step, cursor=cursor, total_tokens=total_tokens,
                             scheduler_last_epoch=scheduler.last_epoch)
            atomic_json(out / "resume-code-migration.json", migration)
        report.update(step=initial_step, scheduler_total_steps=cfg["max_steps"], warmup_steps=warmup_steps,
            trainable_parameters=sum(parameter.numel() for parameter in trainable), target_module_count=len(targets))
        start = time.perf_counter()
        for step in range(initial_step + 1, final_step + 1):
            torch.cuda.reset_peak_memory_stats(0)
            selected = []
            for _ in range(cfg["gradient_accumulation_steps"]):
                if cursor == len(order):
                    sampler_rng.shuffle(order)
                    cursor = 0
                selected.append(examples[order[cursor]])
                cursor += 1
            denominator = sum(example["completion_tokens"] for example in selected)
            optimizer.zero_grad(set_to_none=True)
            ce_sum = forward_seconds = backward_seconds = 0.0
            report.update(phase="supervised_forward_backward", training_started=True)
            atomic_json(out / "status.json", report)
            for example in selected:
                ids = torch.tensor([example["input_ids"]], dtype=torch.long, device="cuda:0")
                prompt_length = len(example["prompt_ids"])
                tick = time.perf_counter()
                hidden = native_hidden(student, ids)
                torch.cuda.synchronize(0)
                forward_seconds += time.perf_counter() - tick
                tick = time.perf_counter()
                info = cross_entropy_hidden_backward(hidden, head, ids[:, prompt_length:],
                    completion_positions(ids.shape[1], prompt_length), denominator, cfg["ce_chunk_tokens"])
                torch.cuda.synchronize(0)
                backward_seconds += time.perf_counter() - tick
                ce_sum += info["ce_sum"]
                del ids, hidden
            for name, parameter in student.named_parameters():
                if parameter.requires_grad:
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f"Missing/nonfinite LoRA gradient: {name}")
                elif parameter.grad is not None:
                    raise RuntimeError(f"Frozen student parameter received a gradient: {name}")
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg["max_grad_norm"])
            if not torch.isfinite(grad_norm) or grad_norm.item() == 0:
                raise FloatingPointError("Nonfinite/zero total supervised gradient")
            learning_rate_used = optimizer.param_groups[0]["lr"]
            tick = time.perf_counter()
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize(0)
            optimizer_seconds = time.perf_counter() - tick
            total_tokens += denominator
            item = {"stage": "sft", "step": step, "loss": ce_sum / denominator,
                "ce_sum": ce_sum, "completion_tokens": denominator, "total_completion_tokens": total_tokens,
                "ids": [example["id"] for example in selected],
                "prompt_tokens": [len(example["prompt_ids"]) for example in selected],
                "completion_lengths": [example["completion_tokens"] for example in selected],
                "grad_norm": grad_norm.item(), "learning_rate_used": learning_rate_used,
                "learning_rate_next": optimizer.param_groups[0]["lr"], "scheduler_step": scheduler.last_epoch,
                "student_forward_seconds": forward_seconds, "ce_and_backward_seconds": backward_seconds,
                "optimizer_seconds": optimizer_seconds, "gpu0_memory": memory(),
                "elapsed_seconds": time.perf_counter() - start}
            with (out / "metrics.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
            if step % cfg["save_steps"] == 0 or step == final_step:
                state = checkpoint_state(optimizer, step, total_tokens, order, cursor, sampler_rng, manifest_hash)
                state.update(stage="sft", scheduler=scheduler.state_dict(), scheduler_total_steps=cfg["max_steps"],
                             scheduler_name="linear_warmup_decay")
                save_checkpoint(student, tokenizer, optimizer, out / f"checkpoint-{step}", state)
            report.update(step=step, phase="update_complete", last_update=item)
            atomic_json(out / "status.json", report)
            print(json.dumps({"stage": "sft", "step": step, "loss": item["loss"],
                "learning_rate": learning_rate_used, "tokens": denominator, "memory": item["gpu0_memory"]}), flush=True)
        report.update(status="pass", phase="complete" if final_step == cfg["max_steps"] else "checkpoint_stop",
            elapsed_seconds=time.perf_counter() - start)
    except Exception as error:
        report.update(status="fail", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        atomic_json(out / "status.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int, help="Stop/checkpoint early without changing the fixed scheduler budget")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    run(cfg, args.resume, args.stop_after)


if __name__ == "__main__":
    main()
