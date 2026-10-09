#!/usr/bin/env python3
"""One offline BF16 teacher/student CUDA:0 check; no optimizer or model writes.

This <=32-token, microbatch-1 check does not establish 4096-token stability.
Run in the pinned server environment, after model-download.json is complete.
"""
import argparse
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback


def save(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


@contextmanager
def phase(report, output, name):
    item = {"name": name, "status": "running"}
    report["stages"].append(item)
    save(report, output)
    started = time.monotonic()
    try:
        yield item
        item["status"] = "pass"
    except Exception:
        item["status"] = "fail"
        raise
    finally:
        item["elapsed_seconds"] = round(time.monotonic() - started, 3)
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            item["cuda0_memory_gib"] = {
                "allocated": round(torch.cuda.memory_allocated(0) / 2**30, 3),
                "reserved": round(torch.cuda.memory_reserved(0) / 2**30, 3),
                "peak_allocated": round(torch.cuda.max_memory_allocated(0) / 2**30, 3),
                "peak_reserved": round(torch.cuda.max_memory_reserved(0) / 2**30, 3),
            }
            report["cuda0_memory_gib"] = item["cuda0_memory_gib"]
        save(report, output)


def check_downloads(paths, download_report):
    if not download_report.is_file():
        raise RuntimeError(f"Model download report missing: {download_report}; finish downloads first")
    manifest = json.loads(download_report.read_text(encoding="utf-8"))
    entries = manifest.get("models", [])
    if not entries or any(entry.get("status") != "complete" for entry in entries):
        raise RuntimeError(f"Incomplete model-download.json entries: {entries}; finish downloads first")
    for path in paths:
        entry = next((entry for entry in entries if Path(entry.get("path", "")).resolve() == path.resolve()), None)
        if entry is None:
            raise RuntimeError(f"No completed download entry for {path} in {download_report}")
        partial = next(path.rglob("*.aria2"), None)
        if partial is not None:
            raise RuntimeError(f"Incomplete model download: {partial}; finish downloads first")
        for filename in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors.index.json"):
            if not (path / filename).is_file():
                raise RuntimeError(f"Incomplete model snapshot: missing {path / filename}")
        index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
        shards = set(index.get("weight_map", {}).values())
        if not shards or any(not (path / shard).is_file() or (path / shard).stat().st_size == 0 for shard in shards):
            raise RuntimeError(f"Incomplete model weight shards under {path}")
    return {"report": str(download_report), "models": [str(path) for path in paths]}


def check_bf16_gpu0(model, item):
    import torch

    policies = {key: sorted(getattr(model, key, None) or [])
                for key in ("_keep_in_fp32_modules", "_keep_in_fp32_modules_strict")}
    declared = set().union(*policies.values())
    counts, fp32 = {}, []
    for name, parameter in model.named_parameters():
        assert parameter.device == torch.device("cuda:0"), f"Parameter is not on CUDA:0: {name}"
        dtype = str(parameter.dtype)
        counts[dtype] = counts.get(dtype, 0) + parameter.numel()
        if parameter.dtype == torch.bfloat16:
            continue
        gdn_scalar = (name.startswith("model.language_model.layers.") and ".linear_attn." in name
                      and name.rsplit(".", 1)[-1] in ("A_log", "dt_bias"))
        allowed = gdn_scalar or any(part in declared for part in name.split("."))
        assert parameter.dtype == torch.float32 and parameter.ndim < 2 and allowed, f"Unexpected base parameter dtype: {name} ({dtype})"
        fp32.append({"name": name, "elements": parameter.numel()})
    item.update(official_fp32_policies=policies, native_fp32_parameters=fp32,
                dtype_parameter_elements=counts, main_matrices_bf16=True, all_parameters_cuda0=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-model", type=Path, default=Path("/root/autodl-tmp/models/Qwen3.8-27B"))
    parser.add_argument("--student-model", type=Path, default=Path("/root/autodl-tmp/models/Qwen3.5-9B"))
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--output", type=Path, default=root / "results/server/teacher-verification.json")
    args = parser.parse_args()
    report = {"status": "running", "pass": False, "passed": False, "stages": [], "device": "cuda:0",
              "scope": "One <=32-token microbatch, no cache, no optimizer; 4096-token stability unverified"}
    try:
        with phase(report, args.output, "downloads") as item:
            item.update(check_downloads((args.teacher_model, args.student_model), root / "results/server/model-download.json"))
        with phase(report, args.output, "environment") as item:
            versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")}
            report["versions"] = versions
            for name, expected in (("torch", "2.11.0"), ("transformers", "5.18.0"), ("peft", "0.21.2")):
                assert versions[name].split("+")[0] == expected, f"Expected {name} {expected}, found {versions[name]}"
            import torch
            import torch.nn.functional as functional
            from peft import LoraConfig, get_peft_model
            from transformers import AutoModelForImageTextToText, AutoTokenizer

            report["versions"].update(cuda_runtime=torch.version.cuda, torch_build=torch.__version__)
            assert torch.cuda.is_available() and torch.version.cuda == "13.0", "CUDA 13.0 GPU environment required"
            torch.cuda.set_device(0)
            assert torch.cuda.is_bf16_supported(), "GPU 0 must support BF16"
            torch.cuda.reset_peak_memory_stats(0)
            torch.manual_seed(123)
            item.update(gpu=torch.cuda.get_device_name(0), versions=report["versions"],
                        CUDA_VISIBLE_DEVICES=os.getenv("CUDA_VISIBLE_DEVICES"))
        with phase(report, args.output, "teacher_load") as item:
            teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, local_files_only=True)
            teacher = AutoModelForImageTextToText.from_pretrained(
                args.teacher_model, local_files_only=True, dtype=torch.bfloat16,
                device_map={"": "cuda:0"}, attn_implementation="sdpa")
            assert teacher.config.model_type == "qwen3_5", "Teacher must use native qwen3_5 implementation"
            teacher.requires_grad_(False).eval()
            check_bf16_gpu0(teacher, item)
            gate_fields = {scope: {key: value for key, value in config.to_dict().items() if "gate" in key}
                           for scope, config in (("root", teacher.config), ("text_config", teacher.config.text_config))}
            item.update(parameter_count=sum(p.numel() for p in teacher.parameters()), gate_fields=gate_fields,
                        gate_semantics="output_gate_type controls GDN output RMSNorm; swish equals SiLU x*sigmoid(x). Full attention uses sigmoid.",
                        implementation=type(teacher).__name__, weights_modified=False)
        with phase(report, args.output, "teacher_forward") as item:
            prompt, completion = "Translate to SQL: select one.\nSQL:", " SELECT 1;"
            prompt_ids = teacher_tokenizer.encode(prompt, add_special_tokens=False)
            completion_ids = teacher_tokenizer.encode(completion, add_special_tokens=False)
            assert prompt_ids and completion_ids and len(prompt_ids + completion_ids) <= 32, "Short check must use 1-32 tokens"
            ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device="cuda:0")
            attention_mask = torch.ones_like(ids)
            positions = slice(len(prompt_ids) - 1, ids.shape[1] - 1)
            with torch.no_grad():
                teacher_logits = teacher(input_ids=ids, attention_mask=attention_mask, use_cache=False).logits
                assert torch.isfinite(teacher_logits).all(), "Teacher produced nonfinite logits"
                teacher_logp = functional.log_softmax(teacher_logits[:, positions, :].float(), dim=-1)
            item.update(finite_logits=True, logits_shape=list(teacher_logits.shape), prompt=prompt,
                        completion=completion, prompt_ids=prompt_ids, sequence_tokens=ids.shape[1], completion_tokens=len(completion_ids))
            del teacher_logits
            torch.cuda.synchronize(0)
        with phase(report, args.output, "student_load_and_vocab") as item:
            student_tokenizer = AutoTokenizer.from_pretrained(args.student_model, local_files_only=True)
            vocab_equal = teacher_tokenizer.get_vocab() == student_tokenizer.get_vocab()
            prompt_equal = prompt_ids == student_tokenizer.encode(prompt, add_special_tokens=False)
            completion_equal = completion_ids == student_tokenizer.encode(completion, add_special_tokens=False)
            item.update(vocabulary_equal=vocab_equal, prompt_ids_equal=prompt_equal, completion_ids_equal=completion_equal,
                        tokenizer_vocabulary_size=len(teacher_tokenizer.get_vocab()))
            assert vocab_equal and prompt_equal and completion_equal, "Teacher/student token mappings or fixed token IDs differ"
            student = AutoModelForImageTextToText.from_pretrained(
                args.student_model, local_files_only=True, dtype=torch.bfloat16,
                device_map={"": "cuda:0"}, attn_implementation="sdpa")
            assert student.config.model_type == "qwen3_5", "Student must use native qwen3_5 implementation"
            check_bf16_gpu0(student, item)
            item.update(parameter_count=sum(p.numel() for p in student.parameters()), implementation=type(student).__name__)
        with phase(report, args.output, "language_lora") as item:
            targets = [name for name, module in student.named_modules()
                       if name.startswith("model.language_model.layers.")
                       and name.rsplit(".", 1)[-1] in ("q_proj", "in_proj_qkv") and isinstance(module, torch.nn.Linear)]
            assert targets and {name.rsplit(".", 1)[-1] for name in targets} == {"q_proj", "in_proj_qkv"}, "Missing language attention LoRA targets"
            student.requires_grad_(False)
            student = get_peft_model(student, LoraConfig(
                r=4, lora_alpha=8, lora_dropout=0, target_modules=targets, task_type="CAUSAL_LM"))
            student.train()
            assert all("lora_" in name and ".language_model.layers." in name
                       for name, p in student.named_parameters() if p.requires_grad), "Unexpected trainable base/vision parameter"
            item.update(target_modules=targets, trainable_parameters=sum(p.numel() for p in student.parameters() if p.requires_grad))
        with phase(report, args.output, "reverse_kl_backward") as item:
            student_logits = student(input_ids=ids, attention_mask=attention_mask, use_cache=False).logits
            assert torch.isfinite(student_logits).all(), "Student produced nonfinite logits"
            assert student_logits.shape[-1] == teacher_logp.shape[-1], "Model output vocabulary sizes differ"
            student_logp = functional.log_softmax(student_logits[:, positions, :].float(), dim=-1)
            loss = (student_logp.exp() * (student_logp - teacher_logp)).sum(-1).mean()
            assert torch.isfinite(loss), "Nonfinite full-vocabulary reverse KL"
            loss.backward()
            norms = {}
            for name, parameter in student.named_parameters():
                if parameter.requires_grad:
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), f"Missing/nonfinite LoRA gradient: {name}"
                    if "lora_B" in name:
                        norm = parameter.grad.float().norm().item()
                        assert norm > 0, f"Zero LoRA_B gradient: {name}"
                        norms[name] = norm
                else:
                    assert parameter.grad is None, f"Frozen student base gradient: {name}"
            assert norms and len(norms) == len(targets), "Missing LoRA_B gradients"
            assert all(not p.requires_grad and p.grad is None for p in teacher.parameters()), "Teacher accumulated gradients"
            assert all(p.device == torch.device("cuda:0") for model in (teacher, student) for p in model.parameters()), "Models no longer share CUDA:0"
            torch.cuda.synchronize(0)
            item.update(loss=loss.item(), reverse_kl="KL(student || teacher), temperature=1, full vocabulary, completion positions only",
                        output_vocabulary_size=student_logits.shape[-1], lora_B_gradient_norms=norms,
                        teacher_gradient_free=True, student_base_gradient_free=True, same_gpu_resident=True)
        report.update(status="pass", passed=True, **{"pass": True})
    except Exception as exc:
        report.update(status="fail", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    save(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
