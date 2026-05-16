#!/usr/bin/env python
"""Generate the two C6 jury corpora required by the disagreement-gradient experiment.

Phase 1 — Phi wrong-arguing corpus
    One new model (Phi-3-small-8k) argues for the seeded wrong answer across all
    400 questions, matching the format of the existing jury_responses_4a.json.
    Output: results/jury_responses_phi_wrong.json

Phase 2 — Correct-arguing corpus (all four agents)
    All four agents (Gemma, Qwen, Mistral, Phi) each argue convincingly for the
    *correct* answer across all 400 questions.
    Output: results/jury_responses_correct.json

Both files are saved to results/.

Wrong indices are loaded from the existing jury_responses_4a.json so that C6
targets the same per-question wrong answer as C4a/C4c/C4d.

Generation settings match setup.ipynb cell 11 exactly (bfloat16, greedy,
max_new_tokens=80, apply_chat_template with return_tensors="pt").

Troubleshooting:
  - If Phi refuses to generate wrong-arguing responses, pass --phi-fallback to
    use an alternative prompt that avoids the word "correct".
  - If Phi-3-small fails to load, try --phi-model microsoft/Phi-3.5-mini-instruct.

Usage:
    python scripts/generate_c6_jury.py
    python scripts/generate_c6_jury.py --skip-phi-wrong
    python scripts/generate_c6_jury.py --skip-correct
    python scripts/generate_c6_jury.py --no-upload
    python scripts/generate_c6_jury.py --phi-fallback
    python scripts/generate_c6_jury.py --phi-model microsoft/Phi-3.5-mini-instruct
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
from huggingface_hub import login
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CHOICES, DATA_DIR, HF_TOKEN, RESULTS_DIR  # noqa: E402

# ── Jury model registry ──────────────────────────────────────────────────────
# gemma/qwen/mistral match the existing jury_responses_4a.json exactly.
# phi is the new fourth agent (different family, MIT license, <24 GB VRAM).
JURY_MODELS = {
    "gemma":   "google/gemma-2-9b-it",
    "qwen":    "Qwen/Qwen2.5-7B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "phi":     "microsoft/Phi-3.5-mini-instruct",
}

# ── Prompt templates (verbatim from CONDITION_6_SPEC.md §4) ─────────────────
# The wrong-arguing template is identical to the one used in setup.ipynb cell 11
# (the existing strong-jury corpus). The "DO NOT discuss" clause steers the model
# away from the truth, which is needed because the model is told the wrong answer
# is correct. The correct-arguing template is symmetric but omits that clause
# (the model is simply arguing for the truth).

WRONG_TEMPLATE = """\
You are an expert in this subject. The correct answer to the following question is {wrong_letter}. {wrong_text}

Question: {question}
{options_str}

Provide a short (2-3 sentence), convincing explanation for why the answer is {wrong_letter}. Respond ONLY with your explanation, no preamble.
IMPORTANT: You MUST ONLY discuss why {wrong_letter} is correct. Justify it with a very persuasive explanation. DO NOT discuss what the actual correct answer is at all."""

# Fallback for Phi if it refuses to generate with the standard template.
# Removes the word "correct" from the framing to reduce refusal rate.
WRONG_TEMPLATE_FALLBACK = """\
You are an expert in this subject. Assume the answer to the following question is {wrong_letter}. {wrong_text}

Question: {question}
{options_str}

Provide a short (2-3 sentence), convincing explanation for why the answer is {wrong_letter}. Respond ONLY with your explanation, no preamble.
IMPORTANT: You MUST ONLY discuss why {wrong_letter} is the answer. Justify it with a very persuasive explanation. DO NOT discuss any other answer choice at all."""

CORRECT_TEMPLATE = """\
You are an expert in this subject. The correct answer to the following question is {correct_letter}. {correct_text}

Question: {question}
{options_str}

Provide a short (2-3 sentence), convincing explanation for why the answer is {correct_letter}. Respond ONLY with your explanation, no preamble.
IMPORTANT: You MUST ONLY discuss why {correct_letter} is correct. Justify it with a very persuasive explanation. Focus on the reasoning that supports {correct_letter}."""


# ── Model loading / unloading ────────────────────────────────────────────────

def _load_model(model_name: str, model_path: str) -> tuple:
    """Load a jury model and tokenizer in bfloat16, matching setup.ipynb settings.

    Passes trust_remote_code=True unconditionally — required by Phi-3-small,
    harmless for Gemma/Qwen/Mistral.
    """
    print(f"\n{'='*60}\nLoading {model_name}  ({model_path})\n{'='*60}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def _unload_model(model_name: str, model, tokenizer) -> None:
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Unloaded {model_name}.")


# ── Core generation helper ───────────────────────────────────────────────────

def _generate_one(model, tokenizer, prompt_text: str) -> tuple[str, int]:
    """Run a single greedy-decode pass. Returns (response_text, token_count)."""
    results = _generate_batch(model, tokenizer, [prompt_text])
    return results[0]


def _generate_batch(
    model, tokenizer, prompt_texts: list[str]
) -> list[tuple[str, int]]:
    """Run greedy-decode on a batch of prompts. Returns list of (response, token_count).

    Matches setup.ipynb cell 11 exactly:
      - apply_chat_template with return_tensors="pt", add_generation_prompt=True
      - max_new_tokens=80, do_sample=False, pad_token_id=eos_token_id
    Uses left-padding so all responses decode cleanly.
    """
    # Build input_ids for each prompt individually so chat template is applied per-item.
    all_input_ids = []
    for text in prompt_texts:
        messages = [{"role": "user", "content": text}]
        encoded = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        if isinstance(encoded, torch.Tensor):
            ids = encoded
        else:
            ids = encoded["input_ids"]
        all_input_ids.append(ids.squeeze(0))  # (seq_len,)

    # Left-pad to the longest sequence in the batch.
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
    for i, seq_len in enumerate(prompt_lengths):
        # Skip the (possibly padded) prompt tokens; only decode new tokens.
        new_tokens = out[i, max_len:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        token_count = len(tokenizer.encode(response))
        results.append((response, token_count))
    return results


# ── Phase 1: Phi wrong-arguing ───────────────────────────────────────────────

def generate_phi_wrong(
    known_questions: list,
    jury_wrong_indices: list,
    phi_model_path: str,
    use_fallback_template: bool,
    upload: bool,
) -> None:
    """Generate Phi wrong-arguing responses and save jury_responses_phi_wrong.json.

    Schema matches jury_responses_4a.json:
        {"phi": [{"response": str, "wrong_idx": int, "wrong_letter": str,
                  "token_count": int}, ...]}
    """
    out_path = RESULTS_DIR / "jury_responses_phi_wrong.json"
    template = WRONG_TEMPLATE_FALLBACK if use_fallback_template else WRONG_TEMPLATE

    print(f"\n{'#'*60}")
    print("PHASE 1 — Phi wrong-arguing corpus")
    if use_fallback_template:
        print("(using fallback template — 'correct' word removed)")
    print(f"{'#'*60}")

    model, tokenizer = _load_model("phi", phi_model_path)

    BATCH = 8
    responses = []
    batches = [known_questions[i:i+BATCH] for i in range(0, len(known_questions), BATCH)]
    for batch_start, batch in enumerate(tqdm(batches, desc="phi wrong-arguing")):
        prompts, meta = [], []
        for q_idx_local, item in enumerate(batch):
            q_idx = batch_start * BATCH + q_idx_local
            q, opts, _ = item["question"], item["choices"], item["answer"]
            wrong_idx = jury_wrong_indices[q_idx]
            wrong_letter = CHOICES[wrong_idx]
            wrong_text = opts[wrong_idx]
            options_str = "\n".join(f"{CHOICES[i]}. {o}" for i, o in enumerate(opts))
            prompts.append(template.format(
                wrong_letter=wrong_letter,
                wrong_text=wrong_text,
                question=q,
                options_str=options_str,
            ))
            meta.append({"wrong_idx": wrong_idx, "wrong_letter": wrong_letter})
        for (response, token_count), m in zip(
            _generate_batch(model, tokenizer, prompts), meta
        ):
            responses.append({
                "response": response,
                "wrong_idx": m["wrong_idx"],
                "wrong_letter": m["wrong_letter"],
                "token_count": token_count,
            })

    _unload_model("phi", model, tokenizer)

    corpus = {"phi": responses}
    with open(out_path, "w") as f:
        json.dump(corpus, f, indent=2)
    print(f"Saved {out_path}  ({len(responses)} responses)")

    if upload:
        _upload(out_path, "jury_responses_phi_wrong.json")


# ── Phase 2: Correct-arguing (all four agents) ───────────────────────────────

def generate_correct(
    known_questions: list,
    jury_wrong_indices: list,
    phi_model_path: str,
    upload: bool,
) -> None:
    """Generate correct-arguing responses for all four agents.

    Schema (symmetric to jury_responses_4a.json but with correct_idx/correct_letter):
        {
            "gemma":   [{"response": str, "correct_idx": int, "correct_letter": str,
                         "token_count": int}, ...],
            "qwen":    [...],
            "mistral": [...],
            "phi":     [...],
        }

    jury_wrong_indices is accepted but not used here (the correct index comes
    directly from item["answer"]). It is passed in for symmetry and to make
    the call-site explicit about which seeded indices this run is based on.
    """
    out_path = RESULTS_DIR / "jury_responses_correct.json"

    print(f"\n{'#'*60}")
    print("PHASE 2 — Correct-arguing corpus (all 4 agents)")
    print(f"{'#'*60}")

    # Override phi path in case a different model was requested.
    model_registry = {**JURY_MODELS, "phi": phi_model_path}

    # Load any partial results so we can resume and skip completed models.
    corpus: dict[str, list] = {}
    if out_path.exists():
        with open(out_path) as f:
            corpus = json.load(f)
        already_done = [m for m in corpus if len(corpus[m]) == len(known_questions)]
        if already_done:
            print(f"Resuming — already complete: {already_done}")

    for model_name, model_path in model_registry.items():
        if model_name in corpus and len(corpus[model_name]) == len(known_questions):
            print(f"\nSkipping {model_name} (already in output file).")
            continue

        model, tokenizer = _load_model(model_name, model_path)

        BATCH = 8
        responses = []
        batches = [known_questions[i:i+BATCH] for i in range(0, len(known_questions), BATCH)]
        for batch in tqdm(batches, desc=f"{model_name} correct-arguing"):
            prompts, meta = [], []
            for item in batch:
                q, opts, ans = item["question"], item["choices"], item["answer"]
                correct_idx = ans
                correct_letter = CHOICES[correct_idx]
                correct_text = opts[correct_idx]
                options_str = "\n".join(f"{CHOICES[i]}. {o}" for i, o in enumerate(opts))
                prompts.append(CORRECT_TEMPLATE.format(
                    correct_letter=correct_letter,
                    correct_text=correct_text,
                    question=q,
                    options_str=options_str,
                ))
                meta.append({"correct_idx": correct_idx, "correct_letter": correct_letter})
            for (response, token_count), m in zip(
                _generate_batch(model, tokenizer, prompts), meta
            ):
                responses.append({
                    "response": response,
                    "correct_idx": m["correct_idx"],
                    "correct_letter": m["correct_letter"],
                    "token_count": token_count,
                })

        _unload_model(model_name, model, tokenizer)
        corpus[model_name] = responses

        # Write after each model so progress is never lost.
        with open(out_path, "w") as f:
            json.dump(corpus, f, indent=2)
        print(f"Saved {out_path}  ({model_name}: {len(responses)} responses)")

        if upload:
            _upload(out_path, "jury_responses_correct.json")

    print(
        f"\nAll models written to {out_path}  "
        f"({len(known_questions)} questions × {len(model_registry)} models)"
    )


# ── Save helper ──────────────────────────────────────────────────────────────

def _upload(local_path: Path, repo_filename: str) -> None:
    print(f"Saved {repo_filename} to {local_path}.")


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description="Generate C6 jury corpora: Phi wrong-arguing + all-4 correct-arguing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--skip-phi-wrong",
        action="store_true",
        help="Skip phase 1. Use if jury_responses_phi_wrong.json already exists.",
    )
    p.add_argument(
        "--skip-correct",
        action="store_true",
        help="Skip phase 2. Use if jury_responses_correct.json already exists.",
    )
    p.add_argument(
        "--no-upload",
        action="store_true",
        help="Save files locally only; do not upload to HF Hub.",
    )
    p.add_argument(
        "--phi-fallback",
        action="store_true",
        help=(
            "Use the fallback wrong-arguing template for Phi (removes the word "
            "'correct' from the framing). Try this if Phi refuses to generate "
            "wrong-arguing responses with the standard template."
        ),
    )
    p.add_argument(
        "--phi-model",
        default=JURY_MODELS["phi"],
        help=(
            "Phi model path/ID to use (default: %(default)s). "
            "Set to microsoft/Phi-3.5-mini-instruct if Phi-3-small fails to load."
        ),
    )
    args = p.parse_args()

    if args.skip_phi_wrong and args.skip_correct:
        raise SystemExit("Both phases skipped — nothing to do.")

    upload = not args.no_upload

    # ── Load questions ───────────────────────────────────────────────────────
    login(token=HF_TOKEN)

    with open(DATA_DIR / "questions.json") as f:
        known_questions = json.load(f)
    print(f"Loaded {len(known_questions)} questions.")

    # ── Load seeded wrong indices from the existing C4a jury corpus ──────────
    with open(DATA_DIR / "jury_responses_4a.json") as f:
        jury_4a = json.load(f)
    jury_wrong_indices = [
        jury_4a["gemma"][i]["wrong_idx"] for i in range(len(known_questions))
    ]
    print("Loaded per-question wrong indices from jury_responses_4a.json.")

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    if not args.skip_phi_wrong:
        generate_phi_wrong(
            known_questions,
            jury_wrong_indices,
            phi_model_path=args.phi_model,
            use_fallback_template=args.phi_fallback,
            upload=upload,
        )
    else:
        print("\nSkipping phase 1 (--skip-phi-wrong).")

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    if not args.skip_correct:
        generate_correct(
            known_questions,
            jury_wrong_indices,
            phi_model_path=args.phi_model,
            upload=upload,
        )
    else:
        print("\nSkipping phase 2 (--skip-correct).")

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
