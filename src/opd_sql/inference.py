"""Run a local HF model, or a supplied OpenAI-compatible inference endpoint."""

import argparse
import json
import os
from pathlib import Path
import time
import urllib.request

from .prompts import extract_sql, format_messages
from .sqlite_tools import execute_readonly


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class HFBackend:
    def __init__(self, config):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
        self.torch = torch
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"], local_files_only=config.get("local_files_only", True))
        factory = {"causal": AutoModelForCausalLM, "image_text": AutoModelForImageTextToText}[config.get("model_class", "causal")]
        dtype = getattr(torch, config.get("dtype", "bfloat16"))
        self.device = config.get("device", "cuda")
        self.model = factory.from_pretrained(
            config["model"], dtype=dtype, local_files_only=config.get("local_files_only", True),
            attn_implementation=config.get("attn_implementation", "sdpa"),
        ).to(self.device).eval()
        self.max_input_tokens = config.get("max_input_tokens", 8192)

    def generate(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.config.get("enable_thinking", False),
        )
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_tokens = inputs["input_ids"].shape[1]
        if input_tokens > self.max_input_tokens:
            raise ValueError(f"Prompt has {input_tokens} tokens, exceeds {self.max_input_tokens}; schema was not silently truncated")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.config.get("max_new_tokens", 384),
                do_sample=False, use_cache=True, pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        completion = generated[0, input_tokens:]
        return {"text": self.tokenizer.decode(completion, skip_special_tokens=True),
                "input_tokens": input_tokens, "output_tokens": len(completion)}


class EndpointBackend:
    def __init__(self, config):
        self.config = config
        if not config.get("base_url"):
            raise ValueError("endpoint backend requires an explicitly configured base_url")

    def generate(self, messages):
        cfg = self.config
        payload = {"model": cfg["model"], "messages": messages, "temperature": 0,
                   "max_tokens": cfg.get("max_new_tokens", 384)}
        headers = {"Content-Type": "application/json"}
        token = os.environ.get(cfg.get("api_key_env", "OPD_SQL_API_KEY"))
        if token:
            headers["Authorization"] = "Bearer " + token
        req = urllib.request.Request(cfg["base_url"].rstrip("/") + "/chat/completions",
                                     json.dumps(payload).encode("utf-8"), headers)
        with urllib.request.urlopen(req, timeout=cfg.get("request_timeout_seconds", 180)) as response:
            data = json.load(response)
        return {"text": data["choices"][0]["message"]["content"],
                "input_tokens": data.get("usage", {}).get("prompt_tokens"),
                "output_tokens": data.get("usage", {}).get("completion_tokens")}


def run_record(record, backend, config):
    attempts, feedback = [], []
    start = time.perf_counter()
    sql = ""
    for attempt_index in range(config.get("max_repairs", 0) + 1):
        step_start = time.perf_counter()
        try:
            generation = backend.generate(format_messages(record, feedback))
            sql = extract_sql(generation["text"])
            execution = execute_readonly(record["db_path"], sql,
                                         timeout_seconds=config.get("sql_timeout_seconds", 5),
                                         max_rows=config.get("max_result_rows", 10000))
            attempt = {"attempt": attempt_index, "sql": sql, "raw_output": generation["text"],
                       "input_tokens": generation["input_tokens"], "output_tokens": generation["output_tokens"],
                       "status": execution["status"], "execution_error": execution.get("error"),
                       "latency_seconds": time.perf_counter() - step_start}
            attempts.append(attempt)
            # A successfully executed query can still be semantically wrong.
            # No oracle SQL/result is exposed to the inference repair loop.
            if execution["status"] == "ok":
                break
            feedback.append({"sql": sql, "error": execution.get("error") or execution["status"]})
        except Exception as exc:
            attempts.append({"attempt": attempt_index, "sql": sql, "status": "generation_error",
                             "execution_error": f"{type(exc).__name__}: {exc}",
                             "latency_seconds": time.perf_counter() - step_start})
            break
    return {"id": str(record["id"]), "db_id": record["db_id"], "sql": sql,
            "attempts": attempts, "latency_seconds": time.perf_counter() - start}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    records = read_jsonl(args.data)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        records = records[:args.limit]
    ids = [str(r["id"]) for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate record IDs")
    if not records:
        raise ValueError("No records to infer")
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; use a new run path or --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    backend = {"hf": HFBackend, "endpoint": EndpointBackend}[config.get("backend", "hf")](config)
    manifest = {"config": config, "input": str(Path(args.data).resolve()), "record_ids": ids,
                "purpose": config.get("purpose", "experiment"),
                "repair_feedback": "execution errors only; no reference SQL or oracle correctness"}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with output.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            result = run_record(record, backend, config)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"done": index + 1, "total": len(records), "id": result["id"],
                              "status": result["attempts"][-1]["status"],
                              "latency_seconds": round(result["latency_seconds"], 3)}), flush=True)


if __name__ == "__main__":
    main()
