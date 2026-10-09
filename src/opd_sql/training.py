"""SFT and single-turn, student-rollout on-policy distillation.

For every valid completion position t on a freshly sampled student trajectory,
forward KL is D_KL(p_teacher(. | x, y_<t) || p_student(. | x, y_<t));
reverse KL reverses those arguments. The loss is the mean over completion
tokens, with prompt and padding excluded. Sampling is detached: gradients flow
through the student's conditional distributions, not through sampled actions.

This is a local token-distribution distillation objective, not an unbiased
gradient of an expected sequence-level KL. No reward or SQL execution signal
is used here; execution-feedback repair belongs to the inference/evaluation
pipeline. Every microbatch is sampled again from the current student.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def masked_token_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    direction: str = "forward",
    temperature: float = 1.0,
    chunk_tokens: int = 16,
) -> torch.Tensor:
    """Exact full-vocabulary KL, averaged over mask, scaled by temperature².

    Inputs are *already aligned* next-token logits [batch, time, vocabulary]
    and the corresponding target-token mask [batch, time]. Teacher is detached
    even if its caller accidentally enabled gradients. Float32 log-softmax is
    evaluated in token chunks, avoiding full-sequence float32 probability
    tensors. The model forward still materializes full-sequence logits.
    """
    if direction not in {"forward", "reverse"}:
        raise ValueError("direction must be 'forward' or 'reverse'")
    if temperature <= 0 or chunk_tokens < 1:
        raise ValueError("temperature and chunk_tokens must be positive")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Student and teacher logits must have identical shapes")
    if student_logits.ndim != 3 or mask.shape != student_logits.shape[:-1]:
        raise ValueError("Expected logits [B,T,V] and mask [B,T]")
    selected = mask.to(device=student_logits.device, dtype=torch.bool).reshape(-1)
    indices = selected.nonzero(as_tuple=False).flatten()
    if not indices.numel():
        raise ValueError("No completion tokens available for KL loss")
    student_flat = student_logits.reshape(-1, student_logits.shape[-1])
    teacher_flat = teacher_logits.detach().reshape(-1, teacher_logits.shape[-1])
    loss = student_logits.new_zeros((), dtype=torch.float32)
    for start in range(0, indices.numel(), chunk_tokens):
        idx = indices[start : start + chunk_tokens]
        student_logp = F.log_softmax(student_flat.index_select(0, idx).float() / temperature, dim=-1)
        teacher_chunk = teacher_flat.index_select(0, idx.to(teacher_flat.device)).to(student_logits.device)
        teacher_logp = F.log_softmax(teacher_chunk.float() / temperature, dim=-1)
        if direction == "forward":
            token_kl = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(-1)
        else:
            token_kl = (student_logp.exp() * (student_logp - teacher_logp)).sum(-1)
        loss = loss + token_kl.sum()
    return loss * (temperature**2 / indices.numel())


def make_completion_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor, prompt_length: int) -> torch.Tensor:
    """Mask for next-token logits [:, :-1]; include the first completion token."""
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask must have identical shapes")
    if prompt_length < 1 or prompt_length >= input_ids.shape[1]:
        raise ValueError("Need at least one prompt token and one completion token")
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    return ((positions >= prompt_length).unsqueeze(0) & attention_mask.bool())[:, 1:]


def encode_prompt(tokenizer: Any, record: dict[str, Any]) -> list[int]:
    from opd_sql.prompts import format_messages

    messages = format_messages(record)
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)


def encode_sft(tokenizer: Any, record: dict[str, Any]) -> tuple[list[int], list[int]]:
    """Keep assistant completion only in labels, including its end token."""
    from opd_sql.prompts import format_messages

    messages = format_messages(record)
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": record["gold_sql"]}],
        tokenize=True,
        add_generation_prompt=False,
    )
    if full[: len(prompt)] != prompt:
        raise ValueError("Chat template does not preserve the generation prompt as the assistant prefix")
    if len(full) <= len(prompt):
        raise ValueError("SFT completion is empty")
    return full, [-100] * len(prompt) + full[len(prompt) :]


def validate_tokenizers(student_tokenizer: Any, teacher_tokenizer: Any) -> None:
    """A shared vocabulary size alone does not establish token ID alignment."""
    if student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("OPD requires identical student/teacher token-to-ID vocabularies")
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if getattr(student_tokenizer, key) != getattr(teacher_tokenizer, key):
            raise ValueError(f"Student/teacher {key} mismatch")
    if student_tokenizer.chat_template != teacher_tokenizer.chat_template:
        raise ValueError("Student/teacher chat templates differ; use an explicitly compatible pair")


def _autocast(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


def sft_microbatch(model: Any, example: dict[str, Any], device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, dict[str, Any]]:
    ids = torch.tensor([example["input_ids"]], dtype=torch.long, device=device)
    labels = torch.tensor([example["labels"]], dtype=torch.long, device=device)
    with _autocast(device, dtype):
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False)
    return output.loss, {"completion_tokens": int((labels != -100).sum()), "prompt_tokens": int((labels == -100).sum())}


def opd_microbatch(
    student: Any,
    teacher: Any,
    tokenizer: Any,
    example: dict[str, Any],
    cfg: dict[str, Any],
    student_device: torch.device,
    teacher_device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, Any]]:
    prompt = torch.tensor([example["prompt_ids"]], dtype=torch.long, device=student_device)
    prompt_length = prompt.shape[1]
    # eval disables dropout while sampling; the sampled actions have no gradient.
    student.eval()
    try:
        with torch.no_grad(), _autocast(student_device, dtype):
            sequences = student.generate(
                input_ids=prompt,
                attention_mask=torch.ones_like(prompt),
                max_new_tokens=int(cfg.get("max_new_tokens", 384)),
                do_sample=True,
                temperature=float(cfg.get("sampling_temperature", 1.0)),
                top_p=float(cfg.get("top_p", 1.0)),
                top_k=int(cfg.get("top_k", 0)),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
    finally:
        student.train()
    attention = torch.ones_like(sequences)
    mask = make_completion_mask(sequences, attention, prompt_length)
    # Teacher stays frozen and sees precisely the student-generated prefixes.
    teacher.eval()
    with torch.no_grad(), _autocast(teacher_device, dtype):
        teacher_logits = teacher(
            input_ids=sequences.to(teacher_device),
            attention_mask=attention.to(teacher_device),
            use_cache=False,
        ).logits[:, :-1, :]
    with _autocast(student_device, dtype):
        student_logits = student(input_ids=sequences, attention_mask=attention, use_cache=False).logits[:, :-1, :]
        loss = masked_token_kl(
            student_logits, teacher_logits, mask,
            direction=cfg.get("kl_direction", "forward"),
            temperature=float(cfg.get("distillation_temperature", 1.0)),
            chunk_tokens=int(cfg.get("kl_chunk_tokens", 16)),
        )
    return loss, {
        "completion_tokens": int(mask.sum()),
        "prompt_tokens": prompt_length,
        "generated_sql": tokenizer.decode(sequences[0, prompt_length:].tolist(), skip_special_tokens=True),
    }


def _resolve(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _check_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"Requested {name}, but CUDA is not available")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"Requested {name}, but only {torch.cuda.device_count()} CUDA device(s) are available")
    return device


def _load_model(model_id: str, cfg: dict[str, Any], device: torch.device, dtype: torch.dtype, teacher: bool = False):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    key = "teacher_load_in_4bit" if teacher else "load_in_4bit"
    quantized = bool(cfg.get(key, False))
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": False,
        "cache_dir": cfg.get("cache_dir"),
        "local_files_only": cfg.get("local_files_only", False),
        "attn_implementation": cfg.get("attn_implementation", "sdpa"),
    }
    if quantized:
        if device.type != "cuda":
            raise ValueError("This training runner supports 4-bit loading only on CUDA")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        kwargs["device_map"] = {"": str(device)}
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not quantized:
        model.to(device)
    if teacher:
        model.requires_grad_(False)
        model.eval()
        return model
    lora = cfg.get("lora", {"enabled": True})
    if quantized and not lora.get("enabled", True):
        raise ValueError("4-bit student training requires LoRA")
    if lora.get("enabled", True):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if quantized:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.get("gradient_checkpointing", True))
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", r=int(lora.get("r", 16)),
            lora_alpha=int(lora.get("alpha", 32)), lora_dropout=float(lora.get("dropout", 0.0)),
            target_modules=lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]), bias="none",
        ))
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.config.use_cache = False
    # Sampling and loss must use the same dropout-free conditionals for OPD.
    if cfg.get("mode") == "opd":
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        for key in ("attention_dropout", "hidden_dropout", "hidden_dropout_prob", "attention_probs_dropout_prob"):
            if hasattr(model.config, key):
                setattr(model.config, key, 0.0)
    return model


def _prepare_examples(cfg: dict[str, Any], tokenizer: Any, path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    examples: list[dict[str, Any]] = []
    counts = {"read": 0, "accepted": 0, "too_long": 0}
    required = {"id", "db_id", "question", "evidence", "gold_sql", "db_path", "schema"}
    max_prompt = int(cfg.get("max_prompt_tokens", 1536))
    max_completion = int(cfg.get("max_new_tokens", 384))
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = required - record.keys()
            if missing:
                raise ValueError(f"{path}:{line_number}: missing keys {sorted(missing)}")
            counts["read"] += 1
            prompt = encode_prompt(tokenizer, record)
            example: dict[str, Any] = {"id": record["id"], "prompt_ids": prompt}
            completion_length = 0
            if cfg["mode"] == "sft":
                full, labels = encode_sft(tokenizer, record)
                completion_length = len(full) - len(prompt)
                example.update(input_ids=full, labels=labels)
            if len(prompt) > max_prompt or completion_length > max_completion:
                counts["too_long"] += 1
                continue
            examples.append(example)
            counts["accepted"] += 1
            if cfg.get("max_samples") and len(examples) >= int(cfg["max_samples"]):
                break
    if not examples:
        raise ValueError(f"No usable examples in {path}; length filtering summary: {counts}")
    return examples, counts


def _save_checkpoint(student: Any, tokenizer: Any, optimizer: Any, step: int, target: Path, cfg: dict[str, Any]) -> None:
    target.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(target)
    tokenizer.save_pretrained(target)
    torch.save({"step": step, "optimizer": optimizer.state_dict(), "torch_rng_state": torch.get_rng_state()}, target / "training_state.pt")
    (target / "training_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def run_training(config_path: str | Path) -> dict[str, Any]:
    from transformers import AutoTokenizer, set_seed

    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("mode") not in {"sft", "opd"}:
        raise ValueError("mode must be 'sft' or 'opd'")
    if int(cfg.get("batch_size", 1)) != 1:
        raise ValueError("This reference runner uses batch_size=1; use gradient_accumulation_steps")
    steps = int(cfg.get("max_steps", 100))
    accumulation = int(cfg.get("gradient_accumulation_steps", 1))
    if steps < 1 or accumulation < 1:
        raise ValueError("max_steps and gradient_accumulation_steps must be positive")
    if cfg.get("mode") == "opd" and not cfg.get("teacher_model"):
        raise ValueError("OPD requires teacher_model")
    if float(cfg.get("sampling_temperature", 1.0)) <= 0:
        raise ValueError("sampling_temperature must be positive")
    output_dir = _resolve(cfg["output_dir"], config_path.parent)
    if (output_dir / "metrics.jsonl").exists():
        raise FileExistsError(f"Existing run at {output_dir}; select a new output_dir to preserve results")
    train_file = _resolve(cfg["train_file"], config_path.parent)
    if not train_file.is_file():
        raise FileNotFoundError(f"Training JSONL not found: {train_file}")
    student_device = _check_device(cfg.get("student_device", "cuda:0"))
    teacher_device = _check_device(cfg.get("teacher_device", str(student_device))) if cfg["mode"] == "opd" else student_device
    dtype_name = cfg.get("dtype", "bfloat16")
    if dtype_name not in {"float32", "float16", "bfloat16"}:
        raise ValueError("dtype must be float32, float16, or bfloat16")
    dtype = getattr(torch, dtype_name)
    if student_device.type == "cpu" and dtype != torch.float32:
        raise ValueError("Use dtype=float32 for the CPU smoke test")
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    tokenizer_args = {"trust_remote_code": False, "cache_dir": cfg.get("cache_dir"), "local_files_only": cfg.get("local_files_only", False)}
    tokenizer = AutoTokenizer.from_pretrained(cfg["student_model"], **tokenizer_args)
    if not tokenizer.chat_template:
        raise ValueError("Student tokenizer must define a chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    teacher_tokenizer = None
    if cfg["mode"] == "opd":
        teacher_tokenizer = AutoTokenizer.from_pretrained(cfg["teacher_model"], **tokenizer_args)
        if teacher_tokenizer.pad_token_id is None:
            teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
        validate_tokenizers(tokenizer, teacher_tokenizer)
    examples, counts = _prepare_examples(cfg, tokenizer, train_file)
    student = _load_model(cfg["student_model"], cfg, student_device, dtype)
    if cfg.get("student_adapter"):
        # Train a new LoRA run initialized from SFT; adapters must use the same base model/config.
        from peft import PeftModel

        if isinstance(student, PeftModel):
            # Avoid nesting a second adapter wrapper created by _load_model.
            student = student.unload()
        student = PeftModel.from_pretrained(student, str(_resolve(cfg["student_adapter"], config_path.parent)), is_trainable=True)
        if cfg.get("gradient_checkpointing", True):
            student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            student.enable_input_require_grads()
        if cfg["mode"] == "opd":
            for module in student.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.0
    teacher = _load_model(cfg["teacher_model"], cfg, teacher_device, dtype, teacher=True) if cfg["mode"] == "opd" else None
    if teacher is not None:
        if student.get_output_embeddings().weight.shape[0] != teacher.get_output_embeddings().weight.shape[0]:
            raise ValueError("Teacher/student output vocabulary dimensions differ")
    trainable = [p for p in student.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("Student has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=float(cfg.get("learning_rate", 2e-5)), weight_decay=float(cfg.get("weight_decay", 0.0)))
    scaler = torch.amp.GradScaler("cuda", enabled=student_device.type == "cuda" and dtype == torch.float16)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_cfg = dict(cfg, train_file=str(train_file), output_dir=str(output_dir))
    (output_dir / "config.json").write_text(json.dumps(resolved_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "data_summary.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    rng = random.Random(seed)
    order = list(range(len(examples)))
    cursor = len(order)
    student.train()
    start = time.perf_counter()
    total_tokens = 0
    metrics: dict[str, Any] = {}
    with (output_dir / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            step_tokens = 0
            for _ in range(accumulation):
                if cursor >= len(order):
                    rng.shuffle(order)
                    cursor = 0
                example = examples[order[cursor]]
                cursor += 1
                if cfg["mode"] == "sft":
                    loss, info = sft_microbatch(student, example, student_device, dtype)
                else:
                    loss, info = opd_microbatch(student, teacher, tokenizer, example, cfg, student_device, teacher_device, dtype)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at step {step}: {loss.item()}")
                scaler.scale(loss / accumulation).backward()
                step_loss += float(loss.detach()) / accumulation
                step_tokens += int(info["completion_tokens"])
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(cfg.get("max_grad_norm", 1.0)))
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite gradient norm at step {step}")
            scaler.step(optimizer)
            scaler.update()
            total_tokens += step_tokens
            metrics = {
                "step": step, "loss": step_loss, "grad_norm": float(grad_norm),
                "completion_tokens": step_tokens, "total_completion_tokens": total_tokens,
                "elapsed_seconds": time.perf_counter() - start,
                "mode": cfg["mode"], "last_example_id": example["id"],
            }
            if cfg["mode"] == "opd":
                metrics.update(kl_direction=cfg.get("kl_direction", "forward"), last_generated_sql=info["generated_sql"])
            cuda_devices = {d for d in (student_device, teacher_device) if d.type == "cuda"}
            metrics["peak_cuda_allocated_bytes"] = {str(d): torch.cuda.max_memory_allocated(d) for d in cuda_devices}
            log.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            log.flush()
            print(json.dumps({k: v for k, v in metrics.items() if k != "last_generated_sql"}, ensure_ascii=False), flush=True)
            if cfg.get("save_steps", 0) and step % int(cfg["save_steps"]) == 0:
                _save_checkpoint(student, tokenizer, optimizer, step, output_dir / f"checkpoint-{step}", resolved_cfg)
        _save_checkpoint(student, tokenizer, optimizer, steps, output_dir / "final", resolved_cfg)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON config; data/output/adapter paths are relative to its directory")
    args = parser.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
