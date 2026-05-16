#!/usr/bin/env python
"""P2.2: Conditional patching — mechanistic compositionality.

Measures activation-patching restoration delta across a 2×5×10 grid:
  - 2 framings: user-role, self-framing
  - 5 consensus points: k_wrong ∈ {0, 1, 2, 3, 4}
  - 10 patch layers: {10, 12, 14, 15, 16, 17, 18, 20, 22, 25}

For each cell, runs all 400 humanities questions. At each (framing, k_wrong,
layer) triple: substitutes the clean (unpressured) last-token hidden state
into the C6-pressured forward pass at the target layer, then measures
P(correct) at the final layer.

Output:
  results/c6_patching/c6_patch_{framing}_{k}v{4-k}.pkl   (per-cell checkpoint)
  results/c6_patching/c6_conditional_patching_final.pkl   (aggregate with CIs)

Usage:
    CUDA_VISIBLE_DEVICES=0 python -u scripts/run_c6_conditional_patching.py
"""
from __future__ import annotations

import gc
import json
import os
import pickle
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MODEL_ID, MODEL_REVISION, HF_TOKEN, RESULTS_DIR, CHOICES
from src.data import load_artifacts
from src.model import choice_token_ids
from src.patching import _build_neutral_prompt, _cache_clean_last_token
from src.prompts import (
    assign_agents_per_question,
    build_prompt_c6_user_role,
    build_prompt_c6_self_framing,
)

from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = RESULTS_DIR / "c6_patching"
OUT_DIR.mkdir(exist_ok=True)

LAYERS = [10, 12, 14, 15, 16, 17, 18, 20, 22, 25]
FRAMINGS = ["user", "self"]
K_WRONG_VALUES = [0, 1, 2, 3, 4]


def load_model_and_tokenizer():
    print("Loading model onto cuda:0 in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        token=HF_TOKEN,
        revision=MODEL_REVISION,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, token=HF_TOKEN, revision=MODEL_REVISION
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_jury_corpora(art):
    phi_wrong_path = RESULTS_DIR / "jury_responses_phi_wrong.json"
    correct_path = RESULTS_DIR / "jury_responses_correct.json"
    for p in (phi_wrong_path, correct_path):
        if not p.exists():
            raise SystemExit(f"Missing {p}")

    with open(phi_wrong_path) as f:
        jury_phi_wrong = json.load(f)
    with open(correct_path) as f:
        jury_correct = json.load(f)

    jury_wrong = {
        "gemma":   art["jury_strong"]["gemma"],
        "qwen":    art["jury_strong"]["qwen"],
        "mistral": art["jury_strong"]["mistral"],
        "phi":     jury_phi_wrong["phi"],
    }
    return jury_wrong, jury_correct


def make_hook(clean_vec):
    def hook_fn(_module, _input, output):
        if isinstance(output, tuple):
            hs = output[0].clone()
            hs[:, -1, :] = clean_vec.to(hs.dtype)
            return (hs,) + output[1:]
        else:
            hs = output.clone()
            hs[:, -1, :] = clean_vec.to(hs.dtype)
            return hs
    return hook_fn


def bootstrap_ci(values, n_iter=1000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    n = len(arr)
    samples = np.array([arr[rng.integers(0, n, size=n)].mean() for _ in range(n_iter)])
    alpha = (1.0 - ci) / 2.0
    return {
        "mean": float(arr.mean()),
        "lo": float(np.quantile(samples, alpha)),
        "hi": float(np.quantile(samples, 1.0 - alpha)),
        "se": float(samples.std(ddof=1)),
    }


def run_one_cell(framing, k_wrong, model, tokenizer, questions, jury_wrong,
                 jury_correct, vocab_indices, art):
    k_correct = 4 - k_wrong
    label = f"c6_patch_{framing}_{k_wrong}v{k_correct}"
    save_path = OUT_DIR / f"{label}.pkl"

    if save_path.exists():
        print(f"  [{framing} k={k_wrong}] already exists, skipping.", flush=True)
        return pickle.load(open(save_path, "rb"))

    assignments = assign_agents_per_question(k_wrong, len(questions), seed=42)

    clean_p_correct = []
    pressured_p_correct = []
    patched_p_correct = {l: [] for l in LAYERS}

    t0 = time.time()
    for q_idx in range(len(questions)):
        item = questions[q_idx]
        correct_idx = item["answer"]
        wrong_idx = art["jury_strong"]["gemma"][q_idx]["wrong_idx"]

        # Build neutral prompt and cache clean hidden states
        neutral = _build_neutral_prompt(item, tokenizer)
        clean_cache = _cache_clean_last_token(neutral, model, tokenizer, LAYERS)

        # Clean P(correct) — no jury, no patch
        clean_inputs = tokenizer(neutral, return_tensors="pt").to(model.device)
        with torch.no_grad():
            clean_out = model(**clean_inputs)
        clean_logits = clean_out.logits[0, -1, vocab_indices]
        clean_probs = torch.softmax(clean_logits.float(), dim=-1)
        clean_p_correct.append(clean_probs[correct_idx].item())

        # Build pressured C6 prompt
        if framing == "user":
            pressured = build_prompt_c6_user_role(
                q_idx, item, wrong_idx,
                assignments[q_idx]["wrong"],
                assignments[q_idx]["correct"],
                jury_wrong, jury_correct,
                tokenizer=tokenizer,
            )
        else:
            pressured = build_prompt_c6_self_framing(
                q_idx, item, wrong_idx,
                assignments[q_idx]["wrong"],
                assignments[q_idx]["correct"],
                jury_wrong, jury_correct,
                tokenizer=tokenizer,
            )

        # Pressured P(correct) — no patch
        press_inputs = tokenizer(pressured, return_tensors="pt").to(model.device)
        with torch.no_grad():
            press_out = model(**press_inputs)
        press_logits = press_out.logits[0, -1, vocab_indices]
        press_probs = torch.softmax(press_logits.float(), dim=-1)
        pressured_p_correct.append(press_probs[correct_idx].item())

        # Patch at each layer
        for l in LAYERS:
            clean_vec = clean_cache[l]
            target_idx = l - 1 if l > 0 else 0
            handle = model.model.layers[target_idx].register_forward_hook(
                make_hook(clean_vec)
            )
            try:
                with torch.no_grad():
                    patched_out = model(**press_inputs)
                patched_logits = patched_out.logits[0, -1, vocab_indices]
                patched_probs = torch.softmax(patched_logits.float(), dim=-1)
                patched_p_correct[l].append(patched_probs[correct_idx].item())
            finally:
                handle.remove()

        # Memory cleanup every 100 questions
        if (q_idx + 1) % 100 == 0:
            del clean_out, press_out, clean_inputs, press_inputs
            del clean_logits, press_logits, clean_probs, press_probs
            torch.cuda.empty_cache()
            gc.collect()

        if (q_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (q_idx + 1) / elapsed
            eta = (len(questions) - q_idx - 1) / rate
            print(f"  [{framing} k={k_wrong}] {q_idx+1}/{len(questions)}  "
                  f"({rate:.1f} q/s, ETA {eta/60:.1f}m)", flush=True)

    cell_result = {
        "framing": framing,
        "k_wrong": k_wrong,
        "layers": LAYERS,
        "clean_p_correct": clean_p_correct,
        "pressured_p_correct": pressured_p_correct,
        "patched_p_correct": {l: patched_p_correct[l] for l in LAYERS},
    }
    with open(save_path, "wb") as f:
        pickle.dump(cell_result, f)

    # Print summary
    clean_mean = np.mean(clean_p_correct)
    press_mean = np.mean(pressured_p_correct)
    gap = clean_mean - press_mean
    print(f"  [{framing} k={k_wrong}] clean={clean_mean:.3f}  "
          f"pressured={press_mean:.3f}  gap={gap:.3f}", flush=True)
    for l in LAYERS:
        delta = np.mean(patched_p_correct[l]) - press_mean
        pct = (delta / gap * 100) if abs(gap) > 0.01 else float('nan')
        print(f"    L{l}: delta={delta:+.4f} ({pct:+.1f}% of gap)", flush=True)

    return cell_result


def main():
    print("=" * 60)
    print("P2.2: Conditional Patching — Mechanistic Compositionality")
    print("=" * 60)

    # Load everything
    print("\nLoading artifacts...", flush=True)
    art = load_artifacts()
    questions = art["known_questions"]
    print(f"  {len(questions)} questions loaded.", flush=True)

    print("Loading jury corpora...", flush=True)
    jury_wrong, jury_correct = load_jury_corpora(art)
    print(f"  Wrong-arguing: {list(jury_wrong.keys())}", flush=True)
    print(f"  Correct-arguing: {list(jury_correct.keys())}", flush=True)

    print("Loading model...", flush=True)
    model, tokenizer = load_model_and_tokenizer()

    ctoks = choice_token_ids(tokenizer)
    vocab_indices = [ctoks[c] for c in CHOICES]

    # Sanity check: verify prompt builders work for edge cases
    print("\nSanity-checking prompt builders for k_wrong=0 and k_wrong=4...", flush=True)
    test_item = questions[0]
    test_wrong_idx = art["jury_strong"]["gemma"][0]["wrong_idx"]

    for kw in [0, 4]:
        asgn = assign_agents_per_question(kw, 1, seed=42)
        for framing_name, builder in [("user", build_prompt_c6_user_role),
                                       ("self", build_prompt_c6_self_framing)]:
            try:
                p = builder(0, test_item, test_wrong_idx,
                           asgn[0]["wrong"], asgn[0]["correct"],
                           jury_wrong, jury_correct,
                           tokenizer=tokenizer)
                ntok = len(tokenizer.encode(p))
                print(f"  {framing_name} k={kw}: OK ({ntok} tokens)", flush=True)
            except Exception as e:
                print(f"  {framing_name} k={kw}: FAILED — {e}", flush=True)
                raise

    # Main sweep
    print(f"\nStarting sweep: {len(FRAMINGS)} framings × {len(K_WRONG_VALUES)} k_wrong × "
          f"{len(LAYERS)} layers × {len(questions)} questions = "
          f"{len(FRAMINGS)*len(K_WRONG_VALUES)*len(LAYERS)*len(questions)} forward passes",
          flush=True)

    all_results = {}
    total_t0 = time.time()

    for framing in FRAMINGS:
        for k_wrong in K_WRONG_VALUES:
            print(f"\n{'─'*50}", flush=True)
            print(f"Cell: {framing} framing, k_wrong={k_wrong} "
                  f"({k_wrong}v{4-k_wrong})", flush=True)
            print(f"{'─'*50}", flush=True)

            result = run_one_cell(
                framing, k_wrong, model, tokenizer, questions,
                jury_wrong, jury_correct, vocab_indices, art,
            )
            all_results[(framing, k_wrong)] = result

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}", flush=True)
    print(f"All cells complete in {total_elapsed/3600:.1f}h", flush=True)

    # Compute bootstrap CIs and save final aggregate
    print("\nComputing bootstrap CIs...", flush=True)
    final = {}
    for (framing, k_wrong), res in all_results.items():
        clean_mean = np.mean(res["clean_p_correct"])
        press_mean = np.mean(res["pressured_p_correct"])
        gap = clean_mean - press_mean

        per_layer = {}
        for l in LAYERS:
            patched_arr = np.array(res["patched_p_correct"][l])
            press_arr = np.array(res["pressured_p_correct"])
            deltas = patched_arr - press_arr
            ci = bootstrap_ci(deltas.tolist())
            pct_restored = (ci["mean"] / gap * 100) if abs(gap) > 0.01 else float('nan')
            per_layer[l] = {
                "delta_mean": ci["mean"],
                "delta_lo": ci["lo"],
                "delta_hi": ci["hi"],
                "delta_se": ci["se"],
                "pct_restored": pct_restored,
            }

        final[(framing, k_wrong)] = {
            "clean_mean": clean_mean,
            "pressured_mean": press_mean,
            "gap": gap,
            "per_layer": per_layer,
        }

    final_path = OUT_DIR / "c6_conditional_patching_final.pkl"
    with open(final_path, "wb") as f:
        pickle.dump(final, f)
    print(f"Saved final aggregate: {final_path}", flush=True)

    # Print final summary table
    print(f"\n{'='*60}", flush=True)
    print("FINAL SUMMARY: Restoration delta at L16 (peak causal layer)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Framing':<8} {'k_wrong':<8} {'Clean':>7} {'Press':>7} {'Gap':>7} "
          f"{'L16 Δ':>7} {'L16 %':>7}", flush=True)
    print("-" * 55, flush=True)
    for framing in FRAMINGS:
        for k_wrong in K_WRONG_VALUES:
            f = final[(framing, k_wrong)]
            l16 = f["per_layer"][16]
            print(f"{framing:<8} {k_wrong:<8} {f['clean_mean']:>7.3f} "
                  f"{f['pressured_mean']:>7.3f} {f['gap']:>7.3f} "
                  f"{l16['delta_mean']:>+7.4f} {l16['pct_restored']:>+7.1f}%",
                  flush=True)

    print("\nDone.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
