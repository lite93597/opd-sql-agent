#!/usr/bin/env python3
"""Verify HF-to-vLLM synchronization with a reversible in-memory LM-head change.

Start scripts/server/start_rollout.sh separately on physical GPU 1, then run:
    CUDA_VISIBLE_DEVICES=0 python scripts/server/verify_vllm_sync.py \
        --output results/vllm-sync.json

Requires local Qwen3.5-9B weights, Transformers 5.18.0, TRL 1.14.1 and
vLLM 0.20.1. Uses the TRL VLLMClient API, including vLLM's pre-0.21 weight
update lifecycle. Three full transfers cover baseline, perturbed and restored
weights. No training, adapter mutation, weight-file write or model download is
performed. Baseline logits/logprobs are diagnostics; the forced token and each
implementation's restored baseline first token must match exactly.
"""

import argparse
import importlib.metadata
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


def phase(report, name):
    report["phase"] = name
    print(f"{name}...", file=sys.stderr, flush=True)


def sync_model(client, model, report, name):
    import torch

    phase(report, f"sync_{name}")
    entry = {"name": name, "status": "running", "prefix_cache_reset": False}
    report.setdefault("syncs", []).append(entry)
    started = time.monotonic()
    try:
        with torch.no_grad():
            client.update_model_params(model)
        torch.cuda.synchronize(0)
        client.reset_prefix_cache()
        entry.update(status="pass", prefix_cache_reset=True)
    except Exception as error:
        entry.update(status="fail", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        entry["seconds"] = time.monotonic() - started


def vllm_greedy(client, prompt_ids, length):
    response = client.generate(
        [prompt_ids], n=1, temperature=0.0, repetition_penalty=1.0,
        top_p=1.0, top_k=0, min_p=0.0, max_tokens=length, logprobs=5,
        generation_kwargs={"ignore_eos": True, "min_tokens": length, "seed": 0},
    )
    if response["prompt_ids"] != [prompt_ids]:
        raise RuntimeError("The vLLM response used different prompt token IDs")
    if len(response["completion_ids"][0]) != length:
        raise RuntimeError(f"Expected {length} vLLM tokens, got {len(response['completion_ids'][0])}")
    return response


def hf_first_token(model, prompt_ids):
    import torch

    inputs = torch.tensor([prompt_ids], device="cuda:0", dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                       use_cache=False, logits_to_keep=1).logits[0, -1].float()
        if not torch.isfinite(logits).all():
            raise RuntimeError("HF next-token logits contain NaN or infinity")
        return logits.argmax().item()


def verify(args, report):
    import requests
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from trl.generation.vllm_client import VLLMClient

    if os.environ["CUDA_VISIBLE_DEVICES"].strip() != "0":
        raise ValueError("Run the HF sender with CUDA_VISIBLE_DEVICES=0; the rollout server uses physical GPU 1")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("The HF sender must see exactly one CUDA GPU")
    torch.cuda.set_device(0)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The sender GPU does not support BF16")
    report["gpu"] = {"name": torch.cuda.get_device_name(0),
                     "compute_capability": list(torch.cuda.get_device_capability(0))}
    if not args.model_path.is_dir() or not (args.model_path / "config.json").is_file():
        raise FileNotFoundError(f"Expected an existing local model directory: {args.model_path}")

    phase(report, "http_health")
    health = requests.get(f"{args.server_url}/health", timeout=10)
    health.raise_for_status()
    report["http_health_status"] = health.status_code
    client = None
    try:
        client = VLLMClient(base_url=args.server_url, group_port=args.group_port, connection_timeout=5)
        report["served_model"] = client.model
        workers = client.get_world_size()
        if workers != 1:
            raise RuntimeError(f"Expected one rollout worker on GPU 1, got {workers}")

        phase(report, "load_local_hf_model")
        model = AutoModelForImageTextToText.from_pretrained(
            str(args.model_path), local_files_only=True, dtype=torch.bfloat16,
            device_map={"": "cuda:0"}, attn_implementation="sdpa",
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
        if model.config.model_type != "qwen3_5":
            raise ValueError(f"Expected Qwen3.5 multimodal architecture, got {model.config.model_type}")
        parameter_count = 0
        transfer_bytes = 0
        vision_count = 0
        dtypes = {}
        for name, parameter in model.named_parameters():
            if parameter.device != torch.device("cuda:0"):
                raise RuntimeError(f"Parameter is not on sender GPU 0: {name}: {parameter.device}")
            if "lora_" in name:
                raise RuntimeError("This check expects a plain HF model without an attached adapter")
            parameter_count += parameter.numel()
            transfer_bytes += parameter.numel() * parameter.element_size()
            vision_count += parameter.numel() if name.startswith("model.visual.") else 0
            dtype = str(parameter.dtype).removeprefix("torch.")
            dtypes[dtype] = dtypes.get(dtype, 0) + parameter.numel()
        report["hf_model"] = {"class": type(model).__name__, "parameter_count": parameter_count,
                              "transfer_gib": transfer_bytes / 2**30, "parameter_dtypes": dtypes,
                              "vision_parameters_streamed_but_skipped_by_language_only_server": vision_count}

        phase(report, "init_nccl_weight_transfer")
        client.init_communicator(device="cuda:0")
        report["communicator_initialized"] = True
        sync_model(client, model, report, "baseline")

        prompt = "Write a SQLite query to count all rows in the users table.\nSQL:"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not prompt_ids:
            raise RuntimeError("Tokenizer returned an empty prompt")
        report["prompt"] = prompt
        report["prompt_ids"] = prompt_ids
        phase(report, "hf_greedy_eight_tokens")
        # A short uncached loop avoids generation-config EOS defaults and retains raw first-token logits.
        sequence = torch.tensor([prompt_ids], device="cuda:0", dtype=torch.long)
        hf_tokens = []
        first_logits = None
        first_hidden = None
        with torch.inference_mode():
            for index in range(8):
                output = model(input_ids=sequence, attention_mask=torch.ones_like(sequence),
                               use_cache=False, logits_to_keep=1, output_hidden_states=index == 0)
                logits = output.logits[0, -1].float()
                if not torch.isfinite(logits).all():
                    raise RuntimeError("HF next-token logits contain NaN or infinity")
                if index == 0:
                    first_logits = logits.cpu()
                    if not output.hidden_states:
                        raise RuntimeError("HF did not expose the final normalized hidden states")
                    first_hidden = output.hidden_states[-1][0, -1].detach().clone()
                next_token = logits.argmax().reshape(1, 1)
                hf_tokens.append(next_token.item())
                sequence = torch.cat((sequence, next_token), dim=1)
        report["hf_completion_ids"] = hf_tokens
        report["hf_completion_text"] = tokenizer.decode(hf_tokens, skip_special_tokens=False)

        phase(report, "vllm_greedy_eight_tokens")
        response = vllm_greedy(client, prompt_ids, 8)
        vllm_tokens = response["completion_ids"][0]
        report["vllm_completion_ids"] = vllm_tokens
        report["vllm_completion_text"] = tokenizer.decode(vllm_tokens, skip_special_tokens=False)
        equal_prefix = 0
        for hf_token, vllm_token in zip(hf_tokens, vllm_tokens):
            if hf_token != vllm_token:
                break
            equal_prefix += 1
        logp = first_logits.log_softmax(dim=-1)
        top = first_logits.topk(5)
        report["comparison"] = {
            "first_token_equal": hf_tokens[0] == vllm_tokens[0],
            "all_eight_tokens_equal": hf_tokens == vllm_tokens, "matching_prefix_tokens": equal_prefix,
            "hf_next_token_top5": [{"token_id": token, "logit": value, "logprob": logp[token].item()}
                                    for token, value in zip(top.indices.tolist(), top.values.tolist())],
            "note": "The HTTP API exposes processed logprobs, not raw logits; differences are recorded without asserting equality.",
        }
        if response.get("logprobs") and response.get("logprob_token_ids"):
            entries = []
            for token, server_logp in zip(response["logprob_token_ids"][0][0], response["logprobs"][0][0]):
                if server_logp is None or not math.isfinite(server_logp):
                    raise RuntimeError("vLLM returned a nonfinite next-token logprob")
                if not 0 <= token < len(first_logits):
                    raise RuntimeError(f"vLLM returned an out-of-vocabulary token: {token}")
                hf_logp = logp[token].item()
                entries.append({"token_id": token, "hf_logit": first_logits[token].item(),
                                "hf_logprob": hf_logp, "vllm_processed_logprob": server_logp,
                                "logprob_difference": server_logp - hf_logp})
            report["comparison"]["next_token_logprob_comparison"] = entries
        else:
            report["comparison"]["next_token_logprob_comparison"] = None

        phase(report, "prepare_memory_only_perturbation")
        if model.config.text_config.tie_word_embeddings:
            raise RuntimeError("The reversible LM-head check requires untied output embeddings")
        head = model.get_output_embeddings()
        target = None
        for text in ("1", "2"):
            candidates = tokenizer.encode(text, add_special_tokens=False)
            if (len(candidates) == 1 and candidates[0] not in tokenizer.all_special_ids
                    and candidates[0] not in (hf_tokens[0], vllm_tokens[0])
                    and 0 <= candidates[0] < head.weight.shape[0]):
                target = candidates[0]
                break
        if target is None:
            raise RuntimeError("Neither '1' nor '2' supplies a distinct ordinary single token")
        if first_hidden.shape != head.weight[target].shape or not torch.isfinite(first_hidden).all():
            raise RuntimeError("Final hidden state is incompatible with the LM-head row")
        if first_hidden.float().norm().item() == 0:
            raise RuntimeError("Cannot force a token from a zero hidden state")
        saved_row = head.weight[target].detach().clone()
        changed_row = head.weight[target]
        check = {"memory_only": True, "weight_files_written": False, "target_token_id": target,
                 "target_token_text": tokenizer.decode([target], skip_special_tokens=False),
                 "multiplier": 8.0, "perturbation_status": "running", "restoration_status": "pending"}
        report["perturbation_restoration"] = check
        try:
            phase(report, "perturb_lm_head_row")
            with torch.no_grad():
                changed_row.copy_(first_hidden.to(changed_row.dtype) * 8)
            sync_model(client, model, report, "perturbed")
            phase(report, "verify_forced_first_token")
            forced_hf = hf_first_token(model, prompt_ids)
            forced_vllm = vllm_greedy(client, prompt_ids, 1)["completion_ids"][0][0]
            check.update(perturbed_hf_first_token=forced_hf, perturbed_vllm_first_token=forced_vllm)
            if forced_hf != target or forced_vllm != target:
                raise RuntimeError(f"Perturbed token must be {target}; HF={forced_hf}, vLLM={forced_vllm}")
            check["perturbation_status"] = "pass"
        except Exception as error:
            check.update(perturbation_status="fail", perturbation_error=f"{type(error).__name__}: {error}")
            report["failed_phase"] = report["phase"]
            raise
        finally:
            # Restore the local row before any potentially failing server operation.
            phase(report, "restore_original_lm_head_row")
            with torch.no_grad():
                changed_row.copy_(saved_row)
            check["local_row_restored_exactly"] = torch.equal(changed_row, saved_row)
            check["restoration_status"] = "running"
            try:
                sync_model(client, model, report, "restored")
                phase(report, "verify_restored_baseline_first_tokens")
                restored_hf = hf_first_token(model, prompt_ids)
                restored_vllm = vllm_greedy(client, prompt_ids, 1)["completion_ids"][0][0]
                check.update(restored_hf_first_token=restored_hf, restored_vllm_first_token=restored_vllm)
                if restored_hf != hf_tokens[0] or restored_vllm != vllm_tokens[0]:
                    raise RuntimeError("Restored HF/vLLM first tokens do not match their respective baselines")
                check["restoration_status"] = "pass"
            except Exception as error:
                check.update(restoration_status="fail", restoration_error=f"{type(error).__name__}: {error}")
                raise
        report["status"] = "pass"
    except Exception:
        report.setdefault("failed_phase", report["phase"])
        raise
    finally:
        if client is not None:
            phase(report, "close_communicator")
            client.close_communicator()
            report["communicator_closed"] = client.communicator is None
            client.session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", type=Path, default=Path("/root/autodl-tmp/models/Qwen3.5-9B"))
    parser.add_argument("--server-url", default="http://127.0.0.1:8001")
    parser.add_argument("--group-port", type=int, default=51216)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.server_url = args.server_url.rstrip("/")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": "fail",
              "model_path": str(args.model_path), "server_url": args.server_url,
              "group_port": args.group_port, "packages": {}}
    for name in ("torch", "transformers", "trl", "vllm"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    started = time.monotonic()
    try:
        verify(args, report)
    except Exception as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc(limit=8)
    report["seconds"] = round(time.monotonic() - started, 3)
    encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
