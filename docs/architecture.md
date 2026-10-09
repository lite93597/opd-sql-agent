# Implementation guide

This guide describes the recorded Qwen3.5-9B / Qwen3.8-27B experiment. Older reference configurations are separately identified in `configs/README.md`.

## One update

1. Require the rollout replica to acknowledge policy version `k`.
2. Generate one completion for each of four fresh prompts with the current student.
3. Recompute teacher and student hidden states on identical prompt/completion IDs. The teacher runs under `no_grad`; only language-attention LoRA parameters receive gradients.
4. Compute full-vocabulary `KL(student || teacher)` on completion prediction positions. Normalize all microbatches by their combined completion-token count.
5. Clip the accumulated gradient and update LoRA with AdamW.
6. Synchronize version `k+1`, reset prefix cache, check finite behavior, and acknowledge the version before the next generation.

There is no replay buffer or hard-label SQL term in the OPD loss. Discrete generation is not differentiated. This is conditional token-distribution distillation, without a REINFORCE score-function term for the sampled prefix distribution.

## Exact token-position chunking

`reverse_kl_hidden_backward` in `src/opd_sql/onpolicy.py` avoids keeping every completion position's large output-head graph alive together:

- Run the native backbones without expanding full-sequence LM-head logits.
- Select hidden positions `[prompt_length - 1 : sequence_length - 1]` to predict the completion IDs.
- For each block of eight positions, make a differentiable hidden leaf, compute FP32 full-vocabulary log probabilities, and immediately backward to the leaf.
- Store `dL/dH`, release that block's graph, and finally pass the accumulated hidden gradient through the original student backbone.

The chain rule connects both backward stages. Heads must remain frozen; this path is not sufficient for a trainable head. Chunking does not truncate the vocabulary and is checked against dense loss/gradient calculations in CPU tests. The output-head dimension in this experiment is 248320; tokenizer mappings agree on 248077 defined tokens.

SFT uses an analogous completion-only cross-entropy path against gold SQL. Prompt positions are excluded from the target loss but still condition predictions and the gradient computation.

## LoRA and memory

The adapter targets `q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, and `out_proj` in language attention. It uses rank 8, alpha 16, no dropout or bias: 80 adapted modules and 5,898,240 trainable parameters. The vision encoder is not executed for these text-only inputs; its resident HF weights are not removed.

Model weights are BF16, LoRA and KL probability calculations are FP32. Non-reentrant gradient checkpointing, microbatch 1 / accumulation 4, and disabled training KV cache reduce memory. FLA handles linear attention/GDN; it is not the same mechanism as full-attention FlashAttention.

An 8192-token probe covered teacher forward, full-vocabulary KL backward, optimizer update, and full synchronization. Real training's observed maximum is 7137 tokens. GPU 0's recorded PyTorch peak allocated and reserved memory were about 73.38 and 94.10 GiB respectively; reserved includes allocator cache and must not be added to allocated.

## Rollout synchronization

Training keeps a frozen BF16 base plus FP32 LoRA. Before exporting merged weights, it clones the adapted base matrices to CPU. It then merges LoRA, streams full base parameters through TRL VLLMClient / NCCL, resets vLLM prefix cache, and compares fixed short greedy outputs. The original BF16 matrices are restored from the backup, avoiding repeated merge/unmerge rounding drift.

The full parameter stream is approximately 17.53 GiB per update, and the adapted-matrix backup is 3.125 GiB. This implementation does not claim adapter-only synchronization. Sender-side adapted-matrix hashes and finite greedy checks are diagnostics, not independent receiver checksums of every weight or proof of equality on all prompts. The unmerged FP32 adapter and merged BF16 rollout weights also have rounding differences.

## Checkpoints and audit boundaries

Complete checkpoints include the adapter, tokenizer, optimizer, step/version, sample permutation and cursor, RNG states, cumulative token counts, scheduler information, and manifest/file hashes. Saving uses an `.incomplete` directory and final rename; incomplete checkpoints are excluded from recovery.

Restoration checks model/data/template/config/source lineage before reuse. Rollout weights are synchronized before restored training generates more data. A recorded SFT compatibility repair replaced literal PEFT `target_modules` string comparison with verification of the actual injected modules and per-module adapter settings; original checkpoint files were retained.

Public artifacts contain aggregate derivatives and original source hashes. They do not bundle checkpoint weights or raw databases, and therefore are insufficient by themselves to rerun every original checkpoint-file audit.

See [results](results.md) for the measured effects and [reproduction](reproduction.md) for environment preparation.
