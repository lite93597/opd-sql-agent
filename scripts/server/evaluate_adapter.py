#!/usr/bin/env python3
"""Evaluate the real merged pilot checkpoint on the exact baseline prompt snapshot."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer
from trl.generation.vllm_client import VLLMClient

from opd_sql.onpolicy import SynchronousPolicy
from opd_sql.prompts import format_messages
from run_baselines import generate, save, stamp


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':
        raise ValueError('HF sender must see physical GPU0 only')
    if args.output.exists():
        raise FileExistsError(f'Preserve existing output: {args.output}')
    args.output.mkdir(parents=True)
    manifest=json.loads((args.baseline/'manifest.json').read_text())
    checkpoint=json.loads((args.checkpoint/'complete.json').read_text())
    for filename,expected in checkpoint['sha256'].items():
        actual=hashlib.sha256((args.checkpoint/filename).read_bytes()).hexdigest()
        if actual!=expected:
            raise ValueError(f'Checkpoint checksum differs: {filename}')
    records_bytes=(args.baseline/'records.jsonl').read_bytes()
    if hashlib.sha256(records_bytes).hexdigest()!=manifest['records_sha256']:
        raise ValueError('Baseline record snapshot checksum differs')
    records=[json.loads(line) for line in records_bytes.decode().splitlines() if line.strip()]
    tokenizer=AutoTokenizer.from_pretrained(manifest['models']['student'],local_files_only=True)
    prompts=[tokenizer.encode(tokenizer.apply_chat_template(format_messages(r),tokenize=False,
        add_generation_prompt=True,enable_thinking=False),add_special_tokens=False) for r in records]
    if hashlib.sha256(json.dumps(prompts).encode()).hexdigest()!=manifest['prompt_ids_sha256']:
        raise ValueError('Post-training prompts differ from baseline token IDs')
    base=AutoModelForImageTextToText.from_pretrained(manifest['models']['student'],local_files_only=True,
        dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa').requires_grad_(False)
    student=PeftModel.from_pretrained(base,args.checkpoint,is_trainable=False).eval()
    client=VLLMClient(base_url='http://127.0.0.1:8001',group_port=51216,connection_timeout=30)
    state={'status':'running','phase':'sync_checkpoint','started_at':stamp(),'step':checkpoint['step']}
    save(args.output/'status.json',state)
    try:
        client.init_communicator(device='cuda:0')
        policy=SynchronousPolicy(client,student,42)
        prompt=tokenizer.encode('Write a SQLite query to count all rows in the users table.\nSQL:',add_special_tokens=False)
        sync=policy.sync(0,prompt,1)
        trained=json.loads((args.checkpoint.parent/f"sync-{checkpoint['step']}.json").read_text())
        if sync['effective_weight_sha256']!=trained['effective_weight_sha256']:
            raise RuntimeError('Inference checkpoint policy differs from the final training policy')
        save(args.output/'checkpoint-sync.json',sync)
        save(args.output/'manifest.json',{'baseline':str(args.baseline),'checkpoint':str(args.checkpoint),
            'same_prompt_ids_verified':True,'same_generation_settings':True,'max_new_tokens':manifest['max_new_tokens'],
            'records_sha256':manifest['records_sha256'],'checkpoint_step':checkpoint['step']})
        state.update(phase='generating',completed=0,total=len(records))
        with (args.output/'student-predictions.jsonl').open('x',encoding='utf-8') as handle:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(generate,r,p,'http://127.0.0.1:8001',manifest['models']['student'],
                    manifest['max_new_tokens'],manifest['max_model_len']) for r,p in zip(records,prompts)]
                for future in concurrent.futures.as_completed(futures):
                    result=future.result()
                    result['adapter_checkpoint']=str(args.checkpoint)
                    handle.write(json.dumps(result,ensure_ascii=False)+'\n'); handle.flush()
                    state.update(completed=state['completed']+1,updated_at=stamp())
                    save(args.output/'status.json',state)
        state.update(status='complete',phase='generation_complete',completed_at=stamp())
    finally:
        client.close_communicator()
        client.session.close()
        save(args.output/'status.json',state)


if __name__=='__main__':
    main()
