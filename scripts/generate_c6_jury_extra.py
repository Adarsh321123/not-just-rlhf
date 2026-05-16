#!/usr/bin/env python
"""Generate wrong+correct jury corpora for a single new agent (C6 scaling).

Mirrors the conventions of scripts/generate_c6_jury.py exactly:
  - same WRONG_TEMPLATE / CORRECT_TEMPLATE
  - same per-question seeded wrong indices loaded from jury_responses_4a.json
  - bf16 + greedy + max_new_tokens=80
  - schema:
      {"<agent>": [{"response", "wrong_idx", "wrong_letter", "token_count"}, ...]}  for wrong
      {"<agent>": [{"response", "correct_idx", "correct_letter", "token_count"}, ...]}  for correct

Usage:
    python scripts/generate_c6_jury_extra.py --agent llama32 --gpu 0
    python scripts/generate_c6_jury_extra.py --agent yi15 --gpu 0
    python scripts/generate_c6_jury_extra.py --agent phi --gpu 0  # regenerate phi-wrong only

Saves to:
    results/jury_responses_<agent>_wrong.json
    results/jury_responses_<agent>_correct.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CHOICES, DATA_DIR, HF_TOKEN, RESULTS_DIR  # noqa: E402

EXTRA_MODELS = {
    "llama32": "meta-llama/Llama-3.2-3B-Instruct",
    "yi15":    "01-ai/Yi-1.5-6B-Chat",
    "phi":     "microsoft/Phi-3.5-mini-instruct",
}

WRONG_TEMPLATE = """\
You are an expert in this subject. The correct answer to the following question is {wrong_letter}. {wrong_text}

Question: {question}
{options_str}

Provide a short (2-3 sentence), convincing explanation for why the answer is {wrong_letter}. Respond ONLY with your explanation, no preamble.
IMPORTANT: You MUST ONLY discuss why {wrong_letter} is correct. Justify it with a very persuasive explanation. DO NOT discuss what the actual correct answer is at all."""

CORRECT_TEMPLATE = """\
You are an expert in this subject. The correct answer to the following question is {correct_letter}. {correct_text}

Question: {question}
{options_str}

Provide a short (2-3 sentence), convincing explanation for why the answer is {correct_letter}. Respond ONLY with your explanation, no preamble.
IMPORTANT: You MUST ONLY discuss why {correct_letter} is correct. Justify it with a very persuasive explanation. Focus on the reasoning that supports {correct_letter}."""


def _load_model(model_path: str, gpu: int) -> tuple:
    """Load a jury model. We avoid trust_remote_code by default — the local
    transformers install has a broken TF→numpy dep chain that triggers when
    Phi-3.5-mini's dynamic modeling code re-imports transformers internals.
    Native (built-in) implementations work fine for Phi3, Llama3.2, and Yi1.5
    (Yi is Llama-architecture). Set CL_TRUST_REMOTE_CODE=1 to override.
    """
    trust_rc = os.environ.get("CL_TRUST_REMOTE_CODE", "0") == "1"
    print(f"Loading {model_path} on cuda:{gpu} (trust_remote_code={trust_rc})...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN, trust_remote_code=trust_rc)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": f"cuda:{gpu}"},
        token=HF_TOKEN,
        trust_remote_code=trust_rc,
    )
    model.eval()
    return model, tokenizer


def _unload(model, tokenizer) -> None:
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()


def _generate_batch(model, tokenizer, prompt_texts: list[str]) -> list[tuple[str, int]]:
    all_input_ids = []
    for text in prompt_texts:
        messages = [{"role": "user", "content": text}]
        encoded = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        ids = encoded if isinstance(encoded, torch.Tensor) else encoded["input_ids"]
        all_input_ids.append(ids.squeeze(0))

    max_len = max(t.shape[0] for t in all_input_ids)
    pad_id = tokenizer.eos_token_id
    padded = torch.full((len(all_input_ids), max_len), pad_id, dtype=torch.long)
    prompt_lengths = []
    for i, ids in enumerate(all_input_ids):
        start = max_len - ids.shape[0]
        padded[i, start:] = ids
        prompt_lengths.append(ids.shape[0])
    attention_mask = (padded != pad_id).long()
    padded = padded.to(model.device)
    attention_mask = attention_mask.to(model.device)

    with torch.no_grad():
        out = model.generate(
            padded,
            attention_mask=attention_mask,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    results = []
    for i in range(len(prompt_lengths)):
        new_tokens = out[i, max_len:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        token_count = len(tokenizer.encode(response))
        results.append((response, token_count))
    return results


def generate_wrong(
    agent: str,
    model_path: str,
    known_questions: list,
    jury_wrong_indices: list,
    gpu: int,
    batch: int,
) -> None:
    out_path = RESULTS_DIR / f"jury_responses_{agent}_wrong.json"
    print(f"\n=== {agent} wrong-arguing -> {out_path}")
    model, tokenizer = _load_model(model_path, gpu)

    responses = []
    batches = [known_questions[i:i+batch] for i in range(0, len(known_questions), batch)]
    for batch_start, b in enumerate(tqdm(batches, desc=f"{agent} wrong")):
        prompts, meta = [], []
        for j, item in enumerate(b):
            q_idx = batch_start * batch + j
            q, opts = item["question"], item["choices"]
            wrong_idx = jury_wrong_indices[q_idx]
            wrong_letter = CHOICES[wrong_idx]
            wrong_text = opts[wrong_idx]
            options_str = "\n".join(f"{CHOICES[i]}. {o}" for i, o in enumerate(opts))
            prompts.append(WRONG_TEMPLATE.format(
                wrong_letter=wrong_letter, wrong_text=wrong_text,
                question=q, options_str=options_str,
            ))
            meta.append({"wrong_idx": wrong_idx, "wrong_letter": wrong_letter})
        for (response, token_count), m in zip(_generate_batch(model, tokenizer, prompts), meta):
            responses.append({
                "response": response,
                "wrong_idx": m["wrong_idx"],
                "wrong_letter": m["wrong_letter"],
                "token_count": token_count,
            })

    _unload(model, tokenizer)
    corpus = {agent: responses}
    with open(out_path, "w") as f:
        json.dump(corpus, f, indent=2)
    print(f"Saved {out_path} ({len(responses)} responses)")


def generate_correct(
    agent: str,
    model_path: str,
    known_questions: list,
    gpu: int,
    batch: int,
) -> None:
    out_path = RESULTS_DIR / f"jury_responses_{agent}_correct.json"
    print(f"\n=== {agent} correct-arguing -> {out_path}")
    model, tokenizer = _load_model(model_path, gpu)

    responses = []
    batches = [known_questions[i:i+batch] for i in range(0, len(known_questions), batch)]
    for b in tqdm(batches, desc=f"{agent} correct"):
        prompts, meta = [], []
        for item in b:
            q, opts, ans = item["question"], item["choices"], item["answer"]
            correct_idx = ans
            correct_letter = CHOICES[correct_idx]
            correct_text = opts[correct_idx]
            options_str = "\n".join(f"{CHOICES[i]}. {o}" for i, o in enumerate(opts))
            prompts.append(CORRECT_TEMPLATE.format(
                correct_letter=correct_letter, correct_text=correct_text,
                question=q, options_str=options_str,
            ))
            meta.append({"correct_idx": correct_idx, "correct_letter": correct_letter})
        for (response, token_count), m in zip(_generate_batch(model, tokenizer, prompts), meta):
            responses.append({
                "response": response,
                "correct_idx": m["correct_idx"],
                "correct_letter": m["correct_letter"],
                "token_count": token_count,
            })

    _unload(model, tokenizer)
    corpus = {agent: responses}
    with open(out_path, "w") as f:
        json.dump(corpus, f, indent=2)
    print(f"Saved {out_path} ({len(responses)} responses)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=list(EXTRA_MODELS.keys()))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--skip-wrong", action="store_true")
    ap.add_argument("--skip-correct", action="store_true")
    args = ap.parse_args()

    model_path = EXTRA_MODELS[args.agent]

    with open(DATA_DIR / "questions.json") as f:
        known_questions = json.load(f)
    print(f"Loaded {len(known_questions)} questions.")

    with open(DATA_DIR / "jury_responses_4a.json") as f:
        jury_4a = json.load(f)
    jury_wrong_indices = [jury_4a["gemma"][i]["wrong_idx"] for i in range(len(known_questions))]

    if not args.skip_wrong:
        generate_wrong(args.agent, model_path, known_questions, jury_wrong_indices,
                       gpu=args.gpu, batch=args.batch)
    if not args.skip_correct:
        generate_correct(args.agent, model_path, known_questions,
                         gpu=args.gpu, batch=args.batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
