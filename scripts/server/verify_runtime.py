#!/usr/bin/env python3
"""Offline, small-tensor checks for a two-GPU OPD training environment.

Run with the training environment's Python, without torchrun:
    python scripts/server/verify_runtime.py --output results/runtime.json

No pretrained model, tokenizer, dataset, or network download is used. The
DistillationTrainer private loss API below was checked against TRL v1.14.1;
Qwen3.5's text configuration was checked against Transformers v5.18.0.
NCCL uses two spawned processes and a temporary file rendezvous. Exit 1 means
at least one check failed; unavailable PCIe P2P alone is informational.
"""

import argparse
import importlib.metadata
import inspect
import json
import os
import platform
import sys
import tempfile
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path


def gpu_inventory():
    import torch

    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        with torch.cuda.device(index):
            bf16 = torch.cuda.is_bf16_supported()
        devices.append({
            "index": index, "name": props.name,
            "compute_capability": list(torch.cuda.get_device_capability(index)),
            "memory_gib": round(props.total_memory / 2**30, 2), "bf16": bf16,
        })
    return {
        "status": "pass" if len(devices) >= 2 else "fail",
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "nccl": list(torch.cuda.nccl.version()) if torch.distributed.is_nccl_available() else None,
        "compiled_architectures": torch.cuda.get_arch_list(), "devices": devices,
        "required_visible_gpus": 2, "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
    }


def bf16_compute():
    import torch

    assert torch.cuda.device_count() >= 2, "Two visible CUDA GPUs are required"
    results = []
    for index in (0, 1):
        with torch.cuda.device(index):
            assert torch.cuda.is_bf16_supported(), f"GPU {index} does not support BF16"
            x = torch.randn(32, 64, device=f"cuda:{index}", dtype=torch.bfloat16, requires_grad=True)
            weight = torch.randn(64, 32, device=x.device, dtype=x.dtype, requires_grad=True)
            output = x @ weight
            reference = x.detach().float() @ weight.detach().float()
            relative_error = (output.float() - reference).norm() / reference.norm()
            assert relative_error.item() < 0.02, "BF16 matmul differs excessively from FP32"
            loss = output.float().square().mean()
            loss.backward()
            assert torch.isfinite(loss), "Nonfinite BF16 loss"
            for grad in (x.grad, weight.grad):
                assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
            torch.cuda.synchronize(index)
            results.append({"gpu": index, "loss": loss.item(), "relative_error": relative_error.item()})
    return {"devices": results}


def peer_copy():
    import torch

    assert torch.cuda.device_count() >= 2, "Two visible CUDA GPUs are required"
    results = []
    for source, destination in ((0, 1), (1, 0)):
        peer = torch.cuda.can_device_access_peer(source, destination)
        expected = torch.arange(64, dtype=torch.float32)
        copied = expected.to(f"cuda:{source}").to(f"cuda:{destination}")
        torch.cuda.synchronize(destination)
        torch.testing.assert_close(copied.cpu(), expected, rtol=0, atol=0)
        results.append({"source": source, "destination": destination, "p2p_supported": peer,
                        "copy_verified": True})
    return {"directions": results, "note": "Copy correctness does not prove the transfer used a direct PCIe route."}


def nccl_worker(rank, rendezvous, directory):
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=45), device_id=device)
    try:
        values = torch.full((1024,), float(rank + 1), device=device)
        dist.all_reduce(values)
        torch.testing.assert_close(values, torch.full_like(values, 3), rtol=0, atol=0)
        parameter = torch.full((16, 16), 7 if rank == 0 else 0, device=device, dtype=torch.bfloat16)
        dist.broadcast(parameter, src=0)
        torch.testing.assert_close(parameter, torch.full_like(parameter, 7), rtol=0, atol=0)
        torch.cuda.synchronize(rank)
        Path(directory, f"rank{rank}.json").write_text(json.dumps({
            "rank": rank, "all_reduce": "pass", "bf16_broadcast": "pass",
        }), encoding="utf-8")
    finally:
        dist.destroy_process_group()


def nccl_collectives():
    import torch
    import torch.multiprocessing as mp

    assert torch.cuda.device_count() >= 2, "Two visible CUDA GPUs are required"
    assert torch.distributed.is_nccl_available(), "This PyTorch build does not provide NCCL"
    with tempfile.TemporaryDirectory(prefix="opd-nccl-") as directory:
        rendezvous = Path(directory, "rendezvous").as_uri()
        context = mp.spawn(nccl_worker, args=(rendezvous, directory), nprocs=2, join=False)
        deadline = time.monotonic() + 90
        try:
            while not context.join(timeout=1):
                if time.monotonic() >= deadline:
                    raise TimeoutError("NCCL smoke check exceeded 90 seconds")
        finally:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            for process in context.processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
        return {"ranks": [json.loads(Path(directory, f"rank{rank}.json").read_text()) for rank in (0, 1)]}


def qwen_tiny(device="cuda:0"):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5

    torch.manual_seed(123)
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    config = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=32, linear_value_head_dim=32, linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"], max_position_embeddings=64,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 1.0, "mrope_section": [6, 5, 5]},
        pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=False, dtype=dtype,
    )
    config._attn_implementation = "sdpa"
    model = Qwen3_5ForCausalLM(config).to(device=device, dtype=dtype).train()
    assert len(model.model.layers) == 2
    count = sum(parameter.numel() for parameter in model.parameters())
    assert count < 1_000_000, "Smoke configuration unexpectedly created a large model"
    tokens = torch.randint(3, config.vocab_size, (2, 16), device=device)
    mask = torch.ones_like(tokens)
    mask[0, -3:] = 0
    tokens[0, -3:] = 0
    labels = tokens.clone()
    labels[mask == 0] = -100
    loss = model(input_ids=tokens, attention_mask=mask, labels=labels, use_cache=False).loss
    assert torch.isfinite(loss), "Nonfinite Qwen3.5 loss"
    loss.backward()
    suffixes = ("linear_attn.in_proj_qkv.weight", "self_attn.q_proj.weight")
    base_grad_norms = {}
    for suffix in suffixes:
        parameter = next(parameter for name, parameter in model.named_parameters() if name.endswith(suffix))
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0, f"No gradient for {suffix}"
        base_grad_norms[suffix] = parameter.grad.float().norm().item()
    base_loss = loss.item()
    linear = model.model.layers[0].linear_attn
    chunk = getattr(linear, "chunk_gated_delta_rule", modeling_qwen3_5.torch_chunk_gated_delta_rule)
    backend = f"{chunk.__module__}.{chunk.__name__}"
    # Transformers 5.18 wraps the torch reference name even when the installed FLA function is selected.
    selected = chunk
    wrapped = chunk
    while inspect.isfunction(wrapped):
        implementation = inspect.getclosurevars(wrapped).nonlocals.get("implementation")
        if implementation is not None:
            selected = implementation
            break
        wrapped = getattr(wrapped, "__wrapped__", None)
    selected_backend = f"{selected.__module__}.{selected.__name__}"
    hub_enabled = bool(model.use_kernels)
    model.zero_grad(set_to_none=True)
    model = get_peft_model(model, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0, target_modules=["q_proj", "in_proj_qkv"], task_type="CAUSAL_LM",
    ))
    lora_loss = model(input_ids=tokens, attention_mask=mask, labels=labels, use_cache=False).loss
    assert torch.isfinite(lora_loss), "Nonfinite LoRA loss"
    lora_loss.backward()
    lora_grad_norms = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert "lora_" in name, f"Unexpected unfrozen base parameter: {name}"
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            # LoRA B starts at zero; its first backward must be nonzero, while A can correctly have zero gradient.
            if "lora_B" in name:
                assert parameter.grad.abs().sum() > 0, f"Zero first-step LoRA B gradient: {name}"
                lora_grad_norms[name] = parameter.grad.float().norm().item()
        else:
            assert parameter.grad is None, f"Frozen parameter accumulated a gradient: {name}"
    assert len(lora_grad_norms) == 2, "LoRA must cover both full and linear attention"
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    return {"device": str(device), "dtype": str(dtype), "parameter_count": count,
            "layer_types": config.layer_types, "full_attention_backend": config._attn_implementation,
            "linear_chunk_callable": backend,
            "linear_chunk_implementation": selected_backend, "hub_kernels_enabled": hub_enabled,
            "base_loss": base_loss, "base_grad_norms": base_grad_norms,
            "lora_loss": lora_loss.item(), "lora_B_grad_norms": lora_grad_norms}


def trl_chunked_kl(device="cuda:0"):
    import torch
    import torch.nn.functional as functional
    from trl.trainer.distillation_trainer import DistillationTrainer, _chunked_divergence_loss

    torch.manual_seed(456)
    # Different hidden widths are intentional: only the student/teacher vocabulary must agree.
    tensors = [torch.randn(shape, device=device) * 0.2 for shape in ((2, 7, 8), (2, 7, 12), (17, 8), (17, 12))]
    mask = torch.tensor([[0, 1, 1, 1, 0, 1, 0], [1, 1, 0, 1, 1, 0, 1]], device=device)
    cases = []
    for beta, temperature, empty, normalization in (
        (0.0, 1.0, False, None), (1.0, 0.7, False, None),
        (0.0, 0.7, False, 18), (0.0, 1.0, True, None),
    ):
        valid = torch.zeros_like(mask) if empty else mask
        hs, ht, ws, wt = [value.detach().clone().requires_grad_() for value in tensors]
        loss, entropy, count = _chunked_divergence_loss(
            student_hidden_states=hs, teacher_hidden_states=ht,
            student_lm_head_weight=ws, teacher_lm_head_weight=wt,
            completion_mask=valid, beta=beta, chunk_size=4,
            num_items_in_batch=normalization, temperature=temperature,
        )
        gradients = torch.autograd.grad(loss, (hs, ws, ht, wt), allow_unused=True)
        assert gradients[2] is None and gradients[3] is None, "Teacher must receive no gradients"

        dense_hs = hs.detach().clone().requires_grad_()
        dense_ws = ws.detach().clone().requires_grad_()
        student_logp = functional.log_softmax((dense_hs @ dense_ws.t()).float() / temperature, dim=-1)
        teacher_logp = functional.log_softmax((ht.detach() @ wt.detach().t()).float() / temperature, dim=-1)
        if beta == 0:
            per_token = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(-1)
        else:
            per_token = (student_logp.exp() * (student_logp - teacher_logp)).sum(-1)
        denominator = normalization if normalization is not None else valid.sum().clamp(min=1)
        dense_loss = (per_token * valid).sum() / denominator
        dense_entropy = (-(student_logp.exp() * student_logp).sum(-1) * valid).sum()
        dense_gradients = torch.autograd.grad(dense_loss, (dense_hs, dense_ws))
        torch.testing.assert_close(loss, dense_loss, rtol=2e-4, atol=2e-6)
        torch.testing.assert_close(entropy, dense_entropy, rtol=2e-4, atol=2e-6)
        assert count.item() == valid.sum().item()
        max_error = 0.0
        for actual, expected in zip(gradients[:2], dense_gradients):
            assert actual is not None and torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-6)
            max_error = max(max_error, (actual - expected).abs().max().item())
        assert torch.count_nonzero(gradients[0][valid == 0]).item() == 0, "Masked positions received gradients"
        cases.append({"beta": beta, "temperature": temperature, "valid_tokens": count.item(),
                      "num_items_in_batch": normalization, "loss": loss.item(),
                      "dense_loss": dense_loss.item(), "gradient_max_abs_error": max_error})
    return {"trainer": DistillationTrainer.__name__, "loss_signature": str(inspect.signature(_chunked_divergence_loss)),
            "device": str(device), "dtype": "float32", "cases": cases}


def run_check(function):
    started = time.monotonic()
    try:
        result = {"status": "pass", **function()}
    except Exception as error:
        result = {"status": "fail", "error": f"{type(error).__name__}: {error}",
                  "traceback": traceback.format_exc(limit=8)}
    result["seconds"] = round(time.monotonic() - started, 3)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, help="Optional JSON report path; JSON is also printed to stdout")
    args = parser.parse_args()
    if os.getenv("WORLD_SIZE", "1") != "1":
        parser.error("Run this script with ordinary python; it spawns its own two NCCL ranks")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    packages = {}
    for name in ("torch", "transformers", "peft", "trl", "accelerate", "vllm", "flash-linear-attention", "causal-conv1d"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
              "python_executable": sys.executable, "platform": platform.platform(), "packages": packages, "checks": {}}
    for function in (gpu_inventory, bf16_compute, peer_copy, nccl_collectives, qwen_tiny, trl_chunked_kl):
        print(f"Checking {function.__name__}...", file=sys.stderr, flush=True)
        report["checks"][function.__name__] = run_check(function)
    report["passed"] = all(result["status"] == "pass" for result in report["checks"].values())
    encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
