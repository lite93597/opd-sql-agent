"""Synchronous, full-vocabulary reverse-KL OPD for the pinned Qwen 27B/9B pair.

GPU 0 owns frozen teacher + language LoRA student. GPU 1 owns a native vLLM
server. Each optimizer update consumes only fresh trajectories from one fully
acknowledged weight version. This pilot never uses SQL answers or task rewards.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
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


TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def completion_positions(sequence_length: int, prompt_length: int) -> slice:
    """Hidden state at prompt_length - 1 predicts the FIRST completion token.

    The final sequence hidden state has no next-token target and is excluded.
    EOS, when present in the returned trajectory, is an ordinary included target.
    """
    if not 1 <= prompt_length < sequence_length:
        raise ValueError("Need a nonempty prompt and completion")
    return slice(prompt_length - 1, sequence_length - 1)


def reverse_kl_hidden_backward(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    student_head: Any,
    teacher_head: Any,
    positions: slice,
    denominator: int,
    chunk_tokens: int,
) -> dict[str, float | int]:
    """Exact full-vocabulary KL with bounded vocabulary-sized autograd storage.

    Backpropagate each LM-head chunk into an independent hidden leaf; discard
    that chunk graph immediately. Then propagate the collected hidden gradient
    through the student backbone ONCE. Neither LM head may be trainable.
    denominator is the total completion-token count for the whole update,
    making unequal-length gradient accumulation token-normalized.
    """
    if student_hidden.ndim != 3 or teacher_hidden.ndim != 3 or student_hidden.shape[:2] != teacher_hidden.shape[:2]:
        raise ValueError("Expected aligned [batch, sequence, hidden] states")
    if denominator < 1 or chunk_tokens < 1:
        raise ValueError("Positive denominator and chunk_tokens required")
    if any(p.requires_grad for head in (student_head, teacher_head) for p in head.parameters()):
        raise ValueError("This memory-bounded path requires frozen LM heads")
    if teacher_hidden.requires_grad:
        raise ValueError("Teacher hidden states must be detached")
    selected_student = student_hidden[:, positions, :]
    selected_teacher = teacher_hidden[:, positions, :]
    count = selected_student.shape[0] * selected_student.shape[1]
    if count < 1:
        raise ValueError("No selected completion positions")
    gradient = torch.zeros_like(student_hidden)
    selected_grad = gradient[:, positions, :]
    loss_sum = 0.0
    entropy_sum = 0.0
    # Microbatch size is one in the real trainer; batching here also supports tests.
    for start in range(0, selected_student.shape[1], chunk_tokens):
        end = min(start + chunk_tokens, selected_student.shape[1])
        leaf = selected_student[:, start:end].detach().requires_grad_(True)
        with torch.no_grad():
            teacher_logits = teacher_head(selected_teacher[:, start:end])
            teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
            del teacher_logits
        student_logits = student_head(leaf)
        if student_logits.shape != teacher_logp.shape:
            raise ValueError("Teacher/student output vocabularies differ")
        student_logp = F.log_softmax(student_logits.float(), dim=-1)
        student_p = student_logp.exp()
        token_kl = (student_p * (student_logp - teacher_logp)).sum(-1)
        loss = token_kl.sum() / denominator
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite reverse KL")
        loss_sum += token_kl.detach().double().sum().item()
        entropy_sum += (-(student_p * student_logp).sum(-1)).detach().double().sum().item()
        loss.backward()
        if leaf.grad is None or not torch.isfinite(leaf.grad).all():
            raise FloatingPointError("Missing/nonfinite hidden gradient")
        selected_grad[:, start:end].copy_(leaf.grad)
        del leaf, teacher_logp, student_logits, student_logp, student_p, token_kl, loss
    student_hidden.backward(gradient)
    return {"kl_sum": loss_sum, "entropy_sum": entropy_sum, "completion_tokens": count}


def plain_parameter_name(name: str) -> str:
    if name.startswith("base_model.model."):
        name = name[len("base_model.model."):]
    name = name.replace(".base_layer.", ".")
    if "lora_" in name or ".modules_to_save." in name:
        raise ValueError(f"Adapter parameter cannot be a server base weight: {name}")
    return name


def base_parameters(student: Any) -> list[tuple[str, torch.Tensor]]:
    entries = [(plain_parameter_name(name), p) for name, p in student.named_parameters() if "lora_" not in name]
    names = [name for name, _ in entries]
    if len(names) != len(set(names)) or any(p.requires_grad for _, p in entries):
        raise ValueError("Duplicate server names or trainable base weights")
    return entries


@contextmanager
def exact_merged(student: Any):
    """Temporarily merge LoRA; restore base bits rather than subtracting BF16 delta.

    CPU backups cover only adapted matrices. No 9B GPU clone or weight-file
    export is made. finally runs even if NCCL transfer/cache reset fails.
    """
    layers = [(name, module) for name, module in student.named_modules()
              if hasattr(module, "lora_A") and hasattr(module, "get_base_layer")]
    if not layers or any(module.merged for _, module in layers):
        raise ValueError("Expected at least one unmerged LoRA layer")
    snapshots = []
    restored = False
    try:
        for name, module in layers:
            base = module.get_base_layer()
            if getattr(base, "bias", None) is not None:
                raise ValueError(f"Bias-bearing LoRA target unsupported: {name}")
            snapshots.append((module, base.weight.detach().cpu().clone()))
        with torch.no_grad():
            student.merge_adapter(safe_merge=True)
        yield {"cpu_backup_bytes": sum(saved.numel() * saved.element_size() for _, saved in snapshots),
               "merged_layers": len(snapshots)}
    finally:
        with torch.no_grad():
            for module, saved in snapshots:
                module.get_base_layer().weight.copy_(saved, non_blocking=False)
                # Deliberately bypass unmerge's subtract-and-round operation.
                module.merged_adapters.clear()
        restored = all(not module.merged for _, module in layers)
        if not restored:
            raise RuntimeError("LoRA merge state failed to restore")


def native_hidden(model: Any, input_ids: torch.Tensor) -> torch.Tensor:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    # Keep the native VL wrapper's text position computation, but never run lm_head
    # over the full sequence. With no image input, the visual module is unused.
    return base.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                      use_cache=False, return_dict=True).last_hidden_state


def greedy_hf(student: Any, prompt_ids: list[int], tokens: int) -> list[int]:
    sequence = torch.tensor([prompt_ids], dtype=torch.long, device="cuda:0")
    result = []
    with torch.no_grad():
        for _ in range(tokens):
            hidden = native_hidden(student, sequence)
            logits = student.get_output_embeddings()(hidden[:, -1:, :])
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite HF greedy logits")
            token = logits[0, -1].argmax().item()
            result.append(token)
            sequence = torch.cat((sequence, sequence.new_tensor([[token]])), dim=1)
    return result


class SynchronousPolicy:
    def __init__(self, client: Any, student: Any, seed: int):
        self.client, self.student, self.seed = client, student, seed
        self.version: int | None = None
        self.last_digest: str | None = None

    def sync(self, version: int, verification_prompt: list[int], verify_tokens: int) -> dict[str, Any]:
        started = time.perf_counter()
        self.version = None  # A failed sync invalidates sampling, including old weights.
        entries = base_parameters(self.student)
        metadata = [(name, str(p.dtype).removeprefix("torch."), list(p.shape)) for name, p in entries]
        merge_start = time.perf_counter()
        with exact_merged(self.student) as merge:
            merge_seconds = time.perf_counter() - merge_start
            transfer_start = time.perf_counter()
            with torch.no_grad():
                self.client.update_named_params(metadata, ((name, p.data) for name, p in entries))
            torch.cuda.synchronize(0)
            transfer_seconds = time.perf_counter() - transfer_start
            cache_start = time.perf_counter()
            self.client.reset_prefix_cache()
            cache_seconds = time.perf_counter() - cache_start
            # Hash all effective adapted matrices: distinguish a real update from
            # an acknowledgement that silently retransmitted the initial base.
            digest = hashlib.sha256()
            for name, module in self.student.named_modules():
                if hasattr(module, "lora_A"):
                    digest.update(name.encode())
                    digest.update(module.get_base_layer().weight.detach().cpu().view(torch.uint8).numpy().tobytes())
            effective_digest = digest.hexdigest()
            verification_start = time.perf_counter()
            hf_ids = greedy_hf(self.student, verification_prompt, verify_tokens)
            response = self.client.generate([verification_prompt], n=1, temperature=0.0,
                repetition_penalty=1.0, top_p=1.0, top_k=-1, min_p=0.0,
                max_tokens=verify_tokens, logprobs=0,
                generation_kwargs={"ignore_eos": True, "seed": self.seed})
            if response["prompt_ids"] != [verification_prompt] or response["completion_ids"] != [hf_ids]:
                raise RuntimeError(f"Merged HF/vLLM greedy mismatch: HF={hf_ids}, server={response['completion_ids']}")
            verification_seconds = time.perf_counter() - verification_start
        changed = self.last_digest is not None and effective_digest != self.last_digest
        if version > 0 and self.last_digest is not None and not changed:
            raise RuntimeError("Optimizer update did not change any BF16 merged policy matrix")
        self.last_digest, self.version = effective_digest, version
        return {"policy_version": version, "sync_seconds": time.perf_counter() - started,
                "merge_seconds": merge_seconds, "transfer_seconds": transfer_seconds,
                "cache_reset_seconds": cache_seconds, "greedy_verification_seconds": verification_seconds,
                "sync_bytes": sum(p.numel() * p.element_size() for _, p in entries),
                "effective_weight_sha256": effective_digest, "effective_weights_changed": changed,
                "greedy_ids": hf_ids, "base_restored_exactly": True, **merge}

    def sample(self, prompts: list[list[int]], version: int, max_tokens: int, probe: bool = False) -> list[list[int]]:
        if self.version != version:
            raise RuntimeError(f"Stale/unacknowledged rollout: requested {version}, acknowledged {self.version}")
        kwargs = {"seed": self.seed + version, "ignore_eos": probe}
        if probe:
            kwargs["min_tokens"] = max_tokens
        response = self.client.generate(prompts, n=1, temperature=1.0, repetition_penalty=1.0,
            top_p=1.0, top_k=-1, min_p=0.0, max_tokens=max_tokens, logprobs=0, generation_kwargs=kwargs)
        if response["prompt_ids"] != prompts or len(response["completion_ids"]) != len(prompts):
            raise RuntimeError("Rollout changed prompt IDs or response count")
        completions = response["completion_ids"]
        if any(not ids or len(ids) > max_tokens for ids in completions):
            raise RuntimeError("Empty/oversized completion; no zero-token update is allowed")
        return completions


def assert_training_record(record: dict[str, Any], line: int, probe: bool = False) -> None:
    required = {"id", "db_id", "question", "schema"}
    if required - record.keys():
        raise ValueError(f"Training line {line}: missing {sorted(required - record.keys())}")
    synthetic_probe = probe and record.get("source_split") == "synthetic" and record.get("split", "synthetic") == "synthetic"
    train = record.get("source_split") == "train" and record.get("split", "train") in {"train", "internal_train"}
    if not train and not synthetic_probe:
        raise ValueError(f"Training line {line}: explicitly train-only split metadata required")


def prepare_examples(path: Path, tokenizer: Any, cfg: dict[str, Any], out: Path) -> list[dict[str, Any]]:
    from opd_sql.prompts import format_messages

    examples, filtered = [], []
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            assert_training_record(record, line_number, probe=cfg["mode"] == "probe")
            if cfg.get("experiment") and cfg["mode"] == "train" and record.get("split") != "internal_train":
                raise ValueError("Effect experiment must use explicit internal_train records only")
            if str(record["id"]) in seen:
                raise ValueError(f"Duplicate training ID: {record['id']}")
            seen.add(str(record["id"]))
            text = tokenizer.apply_chat_template(format_messages(record), tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not isinstance(ids, list) or any(not isinstance(token, int) for token in ids):
                raise ValueError("Expected a flat list of student-template token IDs")
            if cfg["mode"] == "probe":
                limit = min(cfg.get("probe_lengths") or [cfg["max_seq_length"]]) - 1
            else:
                limit = cfg["max_seq_length"] - cfg["max_new_tokens"]
            if not ids or len(ids) > limit:
                filtered.append({"id": record["id"], "prompt_tokens": len(ids), "reason": "prompt_too_long",
                                 "max_prompt_tokens": limit, "schema_truncated": False})
                continue
            examples.append({"id": str(record["id"]), "db_id": record["db_id"], "prompt_ids": ids})
    atomic_json(out / "length-filter.json", {"read": len(seen), "accepted": len(examples), "filtered": filtered,
                "schema_truncated": False, "input_sha256": sha256(path)})
    if not examples:
        raise ValueError("No train-only prompts fit; inspect length-filter.json. No schema is silently truncated.")
    return examples


def memory() -> dict[str, float]:
    return {name: round(function(0) / 2**30, 4) for name, function in (
        ("allocated_gib", torch.cuda.memory_allocated), ("reserved_gib", torch.cuda.memory_reserved),
        ("peak_allocated_gib", torch.cuda.max_memory_allocated), ("peak_reserved_gib", torch.cuda.max_memory_reserved))}


def checkpoint_state(optimizer: Any, step: int, total_tokens: int, order: list[int], cursor: int,
                     sampler_rng: random.Random, manifest_hash: str) -> dict[str, Any]:
    import numpy as np
    return {"step": step, "total_tokens": total_tokens, "order": order, "cursor": cursor,
            "sampler_rng": sampler_rng.getstate(), "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(), "optimizer": optimizer.state_dict(),
            "manifest_hash": manifest_hash, "policy_version": step}


def save_checkpoint(student: Any, tokenizer: Any, optimizer: Any, target: Path, state: dict[str, Any]) -> None:
    temporary = target.with_name(target.name + ".incomplete")
    if target.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {target}")
    temporary.mkdir(parents=True)
    student.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    torch.save(state, temporary / "training-state.pt")
    hashes = {path.name: sha256(path) for path in temporary.iterdir() if path.is_file()}
    if "training-state.pt" not in hashes or "adapter_model.safetensors" not in hashes:
        raise RuntimeError("Incomplete checkpoint write")
    atomic_json(temporary / "complete.json", {"step": state["step"], "policy_version": state["policy_version"],
                "manifest_hash": state["manifest_hash"], "optimizer": "AdamW", "scheduler": state.get("scheduler_name", "constant_lr"), "sha256": hashes})
    temporary.rename(target)


def restore_state(state: dict[str, Any], optimizer: Any, sampler_rng: random.Random, manifest_hash: str) -> None:
    import numpy as np
    if state["manifest_hash"] != manifest_hash or state["step"] != state["policy_version"]:
        raise ValueError("Checkpoint dataset/models/training settings differ, or policy version is inconsistent")
    optimizer.load_state_dict(state["optimizer"])
    sampler_rng.setstate(state["sampler_rng"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])


def scheduled_learning_rate(step: int, cfg: dict[str, Any]) -> float:
    """Deterministic update-index schedule; its total horizon is manifest-bound."""
    schedule = cfg.get("lr_schedule", "constant")
    base_lr = float(cfg["learning_rate"])
    if schedule == "constant":
        return base_lr
    if schedule != "linear" or not 0 < step <= cfg["schedule_total_steps"]:
        raise ValueError("Expected linear schedule with update inside fixed horizon")
    total = int(cfg["schedule_total_steps"])
    warmup = max(1, math.ceil(total * float(cfg.get("warmup_ratio", 0.05))))
    if not 0 < warmup < total:
        raise ValueError("Warmup must be shorter than the fixed schedule horizon")
    scale = step / warmup if step <= warmup else (total - step + 1) / (total - warmup)
    return base_lr * scale


def verify_initial_adapter(path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Initialize a new OPD branch from SFT weights, never its optimizer/RNG."""
    completed = json.loads((path / "complete.json").read_text(encoding="utf-8"))
    for filename, expected in completed["sha256"].items():
        if sha256(path / filename) != expected:
            raise ValueError(f"Initial adapter failed SHA256: {filename}")
    adapter = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    if adapter["r"] != cfg["lora_rank"] or adapter["lora_alpha"] != cfg["lora_alpha"]:
        raise ValueError("Initial adapter topology differs from OPD configuration")
    origin = Path(adapter["base_model_name_or_path"]).resolve()
    if origin != Path(cfg["student_model"]).resolve():
        raise ValueError("Initial adapter belongs to another student base")
    return {"path": str(path.resolve()), "step": completed["step"],
            "manifest_hash": completed["manifest_hash"], "complete_sha256": sha256(path / "complete.json"),
            "adapter_sha256": completed["sha256"]["adapter_model.safetensors"]}


def run(cfg: dict[str, Any], resume: Path | None = None) -> dict[str, Any]:
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed
    from trl.generation.vllm_client import VLLMClient

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or torch.cuda.device_count() != 1:
        raise ValueError("Trainer must see only physical GPU 0: CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable GPU required")
    for name, version in {"transformers": "5.18.0", "peft": "0.21.2", "trl": "1.14.1", "vllm": "0.20.1"}.items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f"Pinned {name} {version} required")
    experiment = cfg.get("experiment", False)
    step_limit, context_limit = (2000, 16384) if experiment else (50, 4096)
    if cfg["mode"] not in {"train", "probe"} or not 1 <= cfg["max_steps"] <= step_limit:
        raise ValueError(f"This configuration allows train/probe and 1–{step_limit} updates")
    if not 0 < cfg["max_new_tokens"] < cfg["max_seq_length"] <= context_limit:
        raise ValueError(f"Need 0 < max_new_tokens < max_seq_length <= {context_limit}")
    if cfg.get("lr_schedule", "constant") != "constant":
        if not experiment or cfg["max_steps"] > cfg.get("schedule_total_steps", 0):
            raise ValueError("Experimental schedule requires a fixed horizon covering all updates")
        scheduled_learning_rate(1, cfg)
    if cfg["gradient_accumulation_steps"] < 1 or cfg["kl_chunk_tokens"] < 1 or cfg["save_steps"] < 1:
        raise ValueError("Positive accumulation and chunk size required")
    if cfg["mode"] == "probe" and cfg["gradient_accumulation_steps"] != 1:
        raise ValueError("Memory probe uses exactly one generated trajectory per optimizer update")
    if cfg.get("probe_lengths"):
        lengths = cfg["probe_lengths"]
        if cfg["mode"] != "probe" or len(lengths) != cfg["max_steps"] or lengths != sorted(set(lengths)):
            raise ValueError("probe_lengths must be increasing, unique, and contain exactly max_steps lengths")
        if any(not 1 < length <= cfg["max_seq_length"] for length in lengths):
            raise ValueError("Each probe length must be at most max_seq_length")
    set_seed(cfg["seed"])
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.jsonl").exists() and resume is None:
        raise FileExistsError("Existing run: choose a fresh output directory or explicit checkpoint resume")
    tokenizer = AutoTokenizer.from_pretrained(cfg["student_model"], local_files_only=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(cfg["teacher_model"], local_files_only=True)
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("Teacher/student token mappings differ")
    if not tokenizer.chat_template:
        raise ValueError("Student chat template required")
    examples = prepare_examples(Path(cfg["train_file"]), tokenizer, cfg, out)
    manifest = {"data_sha256": sha256(Path(cfg["train_file"])),
        "student_model": cfg["student_model"], "teacher_model": cfg["teacher_model"],
        "model_metadata_sha256": {f"{which}/{name}": sha256(Path(cfg[f"{which}_model"]) / name)
            for which in ("student", "teacher") for name in ("config.json", "model.safetensors.index.json", "tokenizer.json")},
        "template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "algorithm": "GKD reverse KL(student||teacher), full vocabulary, beta=1,T=1,lambda=1",
        "policy_precision_note": "Rollout uses BF16 merged base; training uses BF16 base plus FP32 LoRA. Merge rounding means logits are not numerically identical. Greedy sync checks compare the merged HF policy with vLLM.",
        "code_sha256": {name: sha256(Path(__file__).with_name(name)) for name in ("onpolicy.py", "prompts.py")},
        "settings": {key: value for key, value in cfg.items() if key not in {"output_dir", "max_steps", "save_steps"}},
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "trl", "vllm")}}
    download_path = Path(__file__).resolve().parents[2] / "results/server/model-download.json"
    downloads = json.loads(download_path.read_text(encoding="utf-8"))
    if downloads.get("status") != "complete":
        raise ValueError("Both model downloads must be fully SHA256 verified before training")
    model_records = []
    for which in ("student", "teacher"):
        match = next((entry for entry in downloads["models"] if Path(entry["path"]).resolve() == Path(cfg[f"{which}_model"]).resolve()), None)
        if match is None or match.get("status") != "complete":
            raise ValueError(f"No completed model checksum manifest for {which}")
        model_records.append(match)
    manifest["verified_model_download_records"] = model_records
    if cfg.get("initial_adapter"):
        manifest["initial_adapter"] = verify_initial_adapter(Path(cfg["initial_adapter"]), cfg)
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if resume is not None:
        if not (resume / "complete.json").is_file():
            raise ValueError("Only an atomically completed checkpoint can be resumed")
        completed = json.loads((resume / "complete.json").read_text(encoding="utf-8"))
        if completed["manifest_hash"] != manifest_hash:
            raise ValueError("Checkpoint manifest differs from current settings/data/models")
        for filename, expected in completed["sha256"].items():
            if sha256(resume / filename) != expected:
                raise ValueError(f"Checkpoint file failed SHA256: {filename}")
    atomic_json(out / "manifest.json", {**manifest, "manifest_hash": manifest_hash})
    atomic_json(out / "config.json", cfg)
    teacher = AutoModelForImageTextToText.from_pretrained(cfg["teacher_model"], local_files_only=True,
        dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa").requires_grad_(False).eval()
    base = AutoModelForImageTextToText.from_pretrained(cfg["student_model"], local_files_only=True,
        dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa").requires_grad_(False)
    if base.config.model_type != "qwen3_5" or teacher.config.model_type != "qwen3_5":
        raise ValueError("Expected native Qwen3.5-family VL models")
    targets = [name for name, module in base.named_modules() if name.startswith("model.language_model.layers.")
               and name.rsplit(".", 1)[-1] in TARGETS and isinstance(module, torch.nn.Linear)]
    if {name.rsplit(".", 1)[-1] for name in targets} != TARGETS:
        raise ValueError("Pinned model is missing one of the six language-attention LoRA target types")
    if resume is None and cfg.get("initial_adapter"):
        student = PeftModel.from_pretrained(base, cfg["initial_adapter"], is_trainable=True)
    elif resume is None:
        student = get_peft_model(base, LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
            lora_dropout=0.0, target_modules=targets, task_type="CAUSAL_LM", bias="none"))
    else:
        student = PeftModel.from_pretrained(base, resume, is_trainable=True)
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
    student_head, teacher_head = student.get_output_embeddings(), teacher.get_output_embeddings()
    if student_head.weight.shape[0] != teacher_head.weight.shape[0]:
        raise ValueError("Output vocabulary dimensions differ")
    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg["learning_rate"], weight_decay=0.0, foreach=False)
    sampler_rng = random.Random(cfg["seed"])
    order, cursor, initial_step, total_tokens = list(range(len(examples))), len(examples), 0, 0
    if resume is not None:
        state = torch.load(resume / "training-state.pt", map_location="cuda:0", weights_only=False)
        # RNG byte tensors must be on CPU; optimizer tensors remain CUDA.
        state["torch_rng"] = state["torch_rng"].cpu()
        state["cuda_rng"] = [item.cpu() for item in state["cuda_rng"]]
        restore_state(state, optimizer, sampler_rng, manifest_hash)
        order, cursor, initial_step, total_tokens = state["order"], state["cursor"], state["step"], state["total_tokens"]
        if sorted(order) != list(range(len(examples))) or not 0 <= cursor <= len(order):
            raise ValueError("Checkpoint sampler permutation/cursor invalid")
    if initial_step >= cfg["max_steps"]:
        raise ValueError("max_steps is an absolute final update count; increase it to resume")
    if resume is not None:
        for filename in ("metrics.jsonl", "trajectories.jsonl"):
            if (out / filename).is_file():
                rows = [json.loads(line) for line in (out / filename).read_text(encoding="utf-8").splitlines() if line.strip()]
                if rows and rows[-1]["step"] != initial_step:
                    raise ValueError("Run log extends beyond/differs from checkpoint; resume into a fresh output directory")
    # This plain diagnostic prompt is short and identical across every sync.
    verification_prompt = tokenizer.encode("Write a SQLite query to count all rows in the users table.\nSQL:", add_special_tokens=False)
    client = VLLMClient(base_url=cfg["server_url"], group_port=cfg["group_port"], connection_timeout=30)
    report = {"status": "running", "phase": "initial_sync", "step": initial_step, "training_started": False,
              "mode": cfg["mode"], "scope": "Bounded effect experiment" if experiment else "Bounded pilot; not an accuracy result", "manifest_hash": manifest_hash}
    started = time.perf_counter()
    atomic_json(out / "status.json", report)
    try:
        if client.get_world_size() != 1:
            raise ValueError("Expected one TP1 rollout worker")
        client.init_communicator(device="cuda:0")
        policy = SynchronousPolicy(client, student, cfg["seed"])
        sync = policy.sync(initial_step, verification_prompt, cfg.get("initial_verify_greedy_tokens", 8))
        atomic_json(out / f"sync-{initial_step}.json", sync)
        for step in range(initial_step + 1, cfg["max_steps"] + 1):
            torch.cuda.reset_peak_memory_stats(0)
            selected = []
            for _ in range(cfg["gradient_accumulation_steps"]):
                if cursor == len(order):
                    sampler_rng.shuffle(order)
                    cursor = 0
                selected.append(examples[order[cursor]])
                cursor += 1
            if cfg["mode"] == "probe":
                sequence_limit = (cfg.get("probe_lengths") or [cfg["max_seq_length"]] * cfg["max_steps"])[step - 1]
                max_tokens = sequence_limit - len(selected[0]["prompt_ids"])
            else:
                sequence_limit = cfg["max_seq_length"]
                max_tokens = cfg["max_new_tokens"]
            report.update(phase="rollout", step=step - 1)
            atomic_json(out / "status.json", report)
            rollout_start = time.perf_counter()
            completions = policy.sample([entry["prompt_ids"] for entry in selected], step - 1, max_tokens, cfg["mode"] == "probe")
            rollout_seconds = time.perf_counter() - rollout_start
            denominator = sum(map(len, completions))
            optimizer.zero_grad(set_to_none=True)
            kl_sum = entropy_sum = 0.0
            teacher_seconds = student_seconds = backward_seconds = 0.0
            for entry, completion in zip(selected, completions):
                ids = torch.tensor([entry["prompt_ids"] + completion], dtype=torch.long, device="cuda:0")
                if ids.shape[1] > sequence_limit:
                    raise RuntimeError("Rollout exceeded sequence length; no loss-time truncation permitted")
                if cfg["mode"] == "probe" and ids.shape[1] != sequence_limit:
                    raise RuntimeError("Memory probe did not generate the requested exact sequence length")
                positions = completion_positions(ids.shape[1], len(entry["prompt_ids"]))
                report.update(phase="teacher_student_backward", training_started=True)
                atomic_json(out / "status.json", report)
                timestamp = time.perf_counter()
                with torch.no_grad():
                    teacher_hidden = native_hidden(teacher, ids)
                torch.cuda.synchronize(0)
                teacher_seconds += time.perf_counter() - timestamp
                timestamp = time.perf_counter()
                student_hidden = native_hidden(student, ids)
                torch.cuda.synchronize(0)
                student_seconds += time.perf_counter() - timestamp
                timestamp = time.perf_counter()
                info = reverse_kl_hidden_backward(student_hidden, teacher_hidden, student_head, teacher_head,
                    positions, denominator, cfg["kl_chunk_tokens"])
                torch.cuda.synchronize(0)
                backward_seconds += time.perf_counter() - timestamp
                kl_sum += info["kl_sum"]
                entropy_sum += info["entropy_sum"]
                del ids, student_hidden, teacher_hidden
            for name, parameter in student.named_parameters():
                if parameter.requires_grad:
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f"Missing/nonfinite adapter gradient: {name}")
                elif parameter.grad is not None:
                    raise RuntimeError(f"Frozen base received a gradient: {name}")
            if any(p.grad is not None for p in teacher.parameters()):
                raise RuntimeError("Teacher received gradients")
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg["max_grad_norm"])
            if not torch.isfinite(grad_norm) or grad_norm.item() == 0:
                raise FloatingPointError("Nonfinite or zero total LoRA gradient")
            optimizer_start = time.perf_counter()
            for group in optimizer.param_groups:
                group["lr"] = scheduled_learning_rate(step, cfg)
            optimizer.step()
            torch.cuda.synchronize(0)
            optimizer_seconds = time.perf_counter() - optimizer_start
            total_tokens += denominator
            report.update(phase="sync_updated_policy")
            atomic_json(out / "status.json", report)
            sync = policy.sync(step, verification_prompt, cfg["verify_greedy_tokens"])
            atomic_json(out / f"sync-{step}.json", sync)
            item = {"step": step, "policy_version_sampled": step - 1, "policy_version_synced": step,
                "sequence_limit": sequence_limit, "mode": cfg["mode"],
                "loss": kl_sum / denominator, "entropy": entropy_sum / denominator,
                "grad_norm": grad_norm.item(), "learning_rate": optimizer.param_groups[0]["lr"],
                "completion_tokens": denominator, "total_completion_tokens": total_tokens,
                "ids": [entry["id"] for entry in selected],
                "prompt_tokens": [len(entry["prompt_ids"]) for entry in selected],
                "completion_lengths": list(map(len, completions)),
                "eos_ended": [bool(ids and ids[-1] in tokenizer.all_special_ids) for ids in completions],
                "at_generation_limit": [len(ids) == max_tokens for ids in completions],
                "rollout_seconds": rollout_seconds, "teacher_seconds": teacher_seconds,
                "student_forward_seconds": student_seconds, "kl_and_backward_seconds": backward_seconds,
                "optimizer_seconds": optimizer_seconds, "elapsed_seconds": time.perf_counter() - started,
                "gpu0_memory": memory(), "sync": sync}
            with (out / "metrics.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
            with (out / "trajectories.jsonl").open("a", encoding="utf-8") as log:
                for entry, completion in zip(selected, completions):
                    log.write(json.dumps({"step": step, "policy_version": step - 1, "id": entry["id"],
                        "prompt_ids": entry["prompt_ids"], "completion_ids": completion,
                        "text": tokenizer.decode(completion, skip_special_tokens=False)}, ensure_ascii=False) + "\n")
            if step % cfg["save_steps"] == 0 or step == cfg["max_steps"]:
                state = checkpoint_state(optimizer, step, total_tokens, order, cursor, sampler_rng, manifest_hash)
                state["scheduler_name"] = cfg.get("lr_schedule", "constant")
                save_checkpoint(student, tokenizer, optimizer, out / f"checkpoint-{step}", state)
            report.update(step=step, phase="update_complete", last_update=item)
            atomic_json(out / "status.json", report)
            print(json.dumps({"step": step, "loss": item["loss"], "tokens": denominator,
                "memory": item["gpu0_memory"], "sync_seconds": sync["sync_seconds"]}), flush=True)
        report.update(status="pass", phase="complete", elapsed_seconds=time.perf_counter() - started)
    except Exception as error:
        report.update(status="fail", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        try:
            client.close_communicator()
            client.session.close()
            report["communicator_closed"] = True
        except Exception as error:
            report.update(status="fail", communicator_closed=False, close_error=f"{type(error).__name__}: {error}")
            raise
        finally:
            atomic_json(out / "status.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-dir")
    parser.add_argument("--train-file")
    parser.add_argument("--mode", choices=("train", "probe"))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--probe-lengths", help="Increasing comma-separated exact lengths; one optimizer update at each length")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("output_dir", "train_file", "mode", "max_steps", "max_seq_length", "max_new_tokens", "save_steps"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if args.probe_lengths:
        cfg["probe_lengths"] = [int(value) for value in args.probe_lengths.split(",")]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    run(cfg, args.resume)


if __name__ == "__main__":
    main()
