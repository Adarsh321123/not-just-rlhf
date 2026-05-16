#!/usr/bin/env python
"""Attention-head ablation at the commitment position (Agent 1, GPU 0).

Pipeline:
  Phase 1 — identify top question-attending heads (20 questions)
  Phase 2 — run baseline P(correct) + ablation conditions on all 100 questions
  Phase 3 — save summaries and figures

"Commit position" in this script is the **precommit** sequence index
(``seq_len - 2`` when the letter is the last input token). This is the position
whose LM-head output *predicts the letter itself*; ablating heads there is
the causally relevant test for whether attention routing drives letter
commitment. Per-head question-attention mass is computed at this same position.

Outputs:
  results/attention_ablation/top_heads.json
  results/attention_ablation/summary.json
  results/attention_ablation/per_question_predictions.json
  figures/attention_ablation_effect.png
  figures/attention_ablation_layer_specific.png
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import gc
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CHOICES, FIGURES_DIR, HF_TOKEN, MODEL_ID, MODEL_REVISION, RESULTS_DIR
from src.data import load_artifacts
from src.prompts import build_prompt_user_role_jury
from src.attention_ablation import ablate_heads, compute_per_head_question_mass

from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda:0"
OUT_DIR = RESULTS_DIR / "attention_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ATTENTION_LAYERS = [16, 20, 25, 30]
N_ATTENTION_QUESTIONS = 20
SEED = 0


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Commit regex (identical to run_nonlinear_recovery) ────────────────────
def find_commitment(text: str) -> tuple[int, str | None]:
    patterns = [
        r"(?i)(?:the\s+)?(?:correct\s+)?answer\s+is\s+([A-D])",
        r"(?i)I\s+(?:believe|think)\s+(?:the\s+)?(?:correct\s+)?answer\s+(?:is|would\s+be)\s+([A-D])",
        r"(?i)therefore[,\s]+(?:the\s+)?(?:correct\s+)?answer\s+(?:is|would\s+be)\s+([A-D])",
        r"(?i)(?:would|should|must)\s+be\s+([A-D])\b",
        r"(?i)(?:it(?:'s| is))\s+([A-D])\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.start(1), m.group(1)
    matches = list(re.finditer(r"\b([A-D])\b", text))
    if matches:
        last = matches[-1]
        return last.start(1), last.group(1)
    return max(0, len(text) - 1), None


def _find_token_spans(full_text, prompt, tokenizer) -> dict:
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids[0]
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    prompt_len = int(prompt_ids.shape[0])
    full_len = int(full_ids.shape[0])
    jury_char = prompt.find("Before you answer")
    if jury_char < 0:
        return {
            "question_range": (0, prompt_len),
            "jury_range": (prompt_len, prompt_len),
            "generation_range": (prompt_len, full_len),
        }
    pre_jury_ids = tokenizer(prompt[:jury_char], return_tensors="pt").input_ids[0]
    pre_jury_len = int(pre_jury_ids.shape[0])
    return {
        "question_range": (0, pre_jury_len),
        "jury_range": (pre_jury_len, prompt_len),
        "generation_range": (prompt_len, full_len),
    }


def reconstruct_prompt(questions, jury_strong, rec, tokenizer) -> str:
    item = questions[rec["q_idx"]]
    prompt = build_prompt_user_role_jury(0, item, rec["wrong_idx"], jury_strong, tokenizer)
    return prompt.removesuffix("The correct answer is (")


def build_inputs(rec, questions, jury_strong, tokenizer, device):
    """Build forward-pass inputs for a single commitment record.

    Returns ``(inputs, prompt, gen_up_to_commit, spans, precommit_pos, correct_letter_id)``.
    ``precommit_pos`` is ``seq_len - 2`` — the position whose LM-head output
    should predict the letter token.
    """
    prompt = reconstruct_prompt(questions, jury_strong, rec, tokenizer)
    gen_up_to_commit = rec["gen_up_to_commit"]
    full_text = prompt + gen_up_to_commit
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    seq_len = int(inputs["input_ids"].shape[1])
    spans = _find_token_spans(full_text, prompt, tokenizer)
    precommit_pos = seq_len - 2
    return inputs, prompt, gen_up_to_commit, spans, precommit_pos, seq_len


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    _log(f"loading artifacts and model on {DEVICE}")
    art = load_artifacts()
    questions = art["known_questions"]
    jury_strong = art["jury_strong"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
        device_map={"": DEVICE},
        attn_implementation="eager",  # needed for output_attentions
    )
    model.eval()
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    n_heads = cfg.num_attention_heads
    _log(f"  model loaded. hidden={cfg.hidden_size} n_heads={n_heads} head_dim={head_dim}")

    with open(RESULTS_DIR / "commitment_probes" / "c4a_commitment.pkl", "rb") as f:
        commitment = pickle.load(f)
    records = commitment["results"]
    _log(f"  loaded {len(records)} commitment records, committed_correct_rate={commitment['committed_correct']/commitment['n_total']:.3f}")

    # Letter-token ids
    choice_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in CHOICES]
    _log(f"  choice_ids (A,B,C,D) = {choice_ids}")

    # ─── Phase 1: identify top question-attending heads ───────────────────
    _log("=== Phase 1: per-head question-mass on 20 questions ===")
    # Collect per-head question mass at PRECOMMIT position for each layer.
    # Shape: [n_questions, len(layers), n_heads]
    n_Q = N_ATTENTION_QUESTIONS
    per_head_mass_precommit = np.zeros((n_Q, len(ATTENTION_LAYERS), n_heads), dtype=np.float32)
    per_head_mass_letter = np.zeros((n_Q, len(ATTENTION_LAYERS), n_heads), dtype=np.float32)
    per_head_top1_in_q_precommit = np.zeros((n_Q, len(ATTENTION_LAYERS), n_heads), dtype=bool)

    for i in range(n_Q):
        rec = records[i]
        inputs, prompt, gen_up, spans, precommit_pos, seq_len = build_inputs(
            rec, questions, jury_strong, tokenizer, DEVICE
        )
        q_lo, q_hi = spans["question_range"]
        with torch.no_grad():
            out = model(**inputs, output_attentions=True, use_cache=False)
        for li, L in enumerate(ATTENTION_LAYERS):
            attn = out.attentions[L]  # [1, n_heads, S, S]
            # precommit position
            mass_pre = compute_per_head_question_mass(attn, precommit_pos, (q_lo, q_hi)).numpy()
            per_head_mass_precommit[i, li] = mass_pre
            # letter position
            mass_let = compute_per_head_question_mass(attn, seq_len - 1, (q_lo, q_hi)).numpy()
            per_head_mass_letter[i, li] = mass_let
            # top-1 in question at precommit
            a_pre = attn[0, :, precommit_pos, :]
            top1 = a_pre.argmax(dim=-1).cpu().numpy()
            per_head_top1_in_q_precommit[i, li] = (top1 >= q_lo) & (top1 < q_hi)
        del out
        if (i + 1) % 5 == 0:
            torch.cuda.empty_cache()
            _log(f"  attention extracted [{i+1}/{n_Q}]")

    # Mean per (layer, head) across questions
    mean_mass_pre = per_head_mass_precommit.mean(axis=0)   # [L, H]
    mean_mass_let = per_head_mass_letter.mean(axis=0)      # [L, H]
    frac_top1_in_q = per_head_top1_in_q_precommit.mean(axis=0)  # [L, H]

    # Rank ALL (layer, head) pairs by precommit-position question mass
    flat = []
    for li, L in enumerate(ATTENTION_LAYERS):
        for h in range(n_heads):
            flat.append({
                "layer": int(L),
                "head": int(h),
                "question_mass_precommit": float(mean_mass_pre[li, h]),
                "question_mass_letter": float(mean_mass_let[li, h]),
                "frac_top1_in_q_precommit": float(frac_top1_in_q[li, h]),
            })
    flat.sort(key=lambda d: -d["question_mass_precommit"])
    top_pairs = flat[:40]  # store top-40 so we can sweep k up to 40

    with open(OUT_DIR / "top_heads.json", "w") as f:
        json.dump({
            "layers_scanned": ATTENTION_LAYERS,
            "n_heads_per_layer": n_heads,
            "head_dim": head_dim,
            "n_questions_for_ranking": n_Q,
            "ranking_metric": "mean attention mass to question region at precommit position (seq_len-2)",
            "top_40": top_pairs,
            "all_heads": flat,
        }, f, indent=2)
    _log(f"  saved top_heads.json; top-5 heads:")
    for p in top_pairs[:5]:
        _log(f"    L{p['layer']}-H{p['head']}: q_mass_precommit={p['question_mass_precommit']:.3f} (letter={p['question_mass_letter']:.3f})")

    # ─── Phase 2: run baseline + ablation experiments on all 100 questions ─
    _log("=== Phase 2: baseline and ablation conditions ===")

    def measure_p_correct(hook_spec: list | None, pos_mode: str = "precommit") -> dict:
        """Run forward pass on all 100 records. Returns dict with per-record preds
        and aggregate P(correct) computed at the PRECOMMIT position via
        ABCD-restricted LM head.

        ``hook_spec`` is a list of (layer, head) pairs to ablate at the chosen
        position. ``pos_mode``: 'precommit' or 'letter'.
        """
        preds = []
        corrects = []
        committed = []
        p_correct_per = []
        for idx, rec in enumerate(records):
            inputs, prompt, gen_up, spans, precommit_pos, seq_len = build_inputs(
                rec, questions, jury_strong, tokenizer, DEVICE
            )
            pos = precommit_pos if pos_mode == "precommit" else seq_len - 1
            ctx = (
                ablate_heads(model, hook_spec, position=pos, head_dim=head_dim)
                if hook_spec else _null_context()
            )
            with ctx, torch.no_grad():
                out = model(**inputs, output_hidden_states=False, use_cache=False)
            # LM head logits at precommit position
            logits = out.logits[0, precommit_pos, :].float().cpu().numpy()
            abcd = logits[choice_ids]
            abcd_exp = np.exp(abcd - abcd.max())
            abcd_probs = abcd_exp / abcd_exp.sum()
            pred_letter_idx = int(np.argmax(abcd))
            correct_idx = int(rec["correct_idx"])
            committed_letter = rec.get("committed_answer")
            preds.append(pred_letter_idx)
            corrects.append(correct_idx)
            committed.append(committed_letter)
            p_correct_per.append(float(abcd_probs[correct_idx]))
            del out
            if (idx + 1) % 25 == 0:
                torch.cuda.empty_cache()
        preds = np.array(preds)
        corrects = np.array(corrects)
        acc = float((preds == corrects).mean())
        p_mean = float(np.mean(p_correct_per))
        return {
            "p_correct": acc,
            "mean_p_correct_softmax": p_mean,
            "preds": preds.tolist(),
            "p_correct_per": p_correct_per,
        }

    results_by_condition: dict[str, dict] = {}

    # Baseline
    _log("  baseline (no ablation)")
    t0 = time.time()
    results_by_condition["baseline"] = measure_p_correct(None)
    _log(f"    P(correct)={results_by_condition['baseline']['p_correct']:.3f}  ({time.time()-t0:.1f}s)")

    # Top-k sweep
    ks = [5, 10, 20, 40]
    for k in ks:
        heads = [(p["layer"], p["head"]) for p in top_pairs[:k]]
        _log(f"  ablate top-{k} question-attending heads")
        t0 = time.time()
        results_by_condition[f"topk_question_attending_k{k}"] = measure_p_correct(heads)
        _log(f"    k={k} P(correct)={results_by_condition[f'topk_question_attending_k{k}']['p_correct']:.3f}  ({time.time()-t0:.1f}s)")

    # Random-head control (same layers, same k distribution as top-20)
    rng = np.random.default_rng(SEED)
    # Count how many heads per layer are in top-20
    top20_by_layer = {L: 0 for L in ATTENTION_LAYERS}
    for p in top_pairs[:20]:
        top20_by_layer[p["layer"]] += 1
    _log(f"  random control (matched layer distribution): {top20_by_layer}")
    random_heads = []
    for L, n in top20_by_layer.items():
        if n <= 0:
            continue
        # exclude heads already in top-20 at that layer
        top20_heads_at_L = {p["head"] for p in top_pairs[:20] if p["layer"] == L}
        available = [h for h in range(n_heads) if h not in top20_heads_at_L]
        picks = rng.choice(available, size=n, replace=False)
        random_heads += [(L, int(h)) for h in picks]
    t0 = time.time()
    results_by_condition["random_k20"] = measure_p_correct(random_heads)
    results_by_condition["random_k20"]["head_specs"] = random_heads
    _log(f"    random-20 P(correct)={results_by_condition['random_k20']['p_correct']:.3f}  ({time.time()-t0:.1f}s)")

    # Layer-specific: L25 only, L30 only, L16 only, L20 only — top heads at each layer
    for L in ATTENTION_LAYERS:
        layer_heads = [(p["layer"], p["head"]) for p in flat if p["layer"] == L][:20]
        _log(f"  ablate top-20 heads within L{L} only (n={len(layer_heads)})")
        t0 = time.time()
        results_by_condition[f"layer_only_L{L}_top20"] = measure_p_correct(layer_heads)
        _log(f"    L{L}-only P(correct)={results_by_condition[f'layer_only_L{L}_top20']['p_correct']:.3f}  ({time.time()-t0:.1f}s)")

    # ─── Phase 3: save and summarize ───────────────────────────────────────
    _log("=== Phase 3: summary ===")
    summary = {
        "reference": {
            "committed_correct_rate_generation": commitment["committed_correct"] / commitment["n_total"],
            "n_questions": commitment["n_total"],
            "position": "precommit (seq_len - 2)",
            "metric": "P(correct) = fraction where ABCD-restricted LM-head top-1 equals correct letter",
        },
        "conditions": {
            name: {
                "p_correct": d["p_correct"],
                "mean_p_correct_softmax": d["mean_p_correct_softmax"],
            }
            for name, d in results_by_condition.items()
        },
        "ks_sweep": [
            {"k": 0, "p_correct": results_by_condition["baseline"]["p_correct"]},
        ] + [
            {"k": k, "p_correct": results_by_condition[f"topk_question_attending_k{k}"]["p_correct"]}
            for k in ks
        ],
        "random_control": {
            "k": 20,
            "p_correct": results_by_condition["random_k20"]["p_correct"],
            "matched_layer_distribution": top20_by_layer,
        },
        "layer_specific": {
            f"L{L}": results_by_condition[f"layer_only_L{L}_top20"]["p_correct"]
            for L in ATTENTION_LAYERS
        },
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _log(f"  saved summary.json")

    # Per-question predictions (for reproducibility)
    per_q = {
        "correct_idx": [int(r["correct_idx"]) for r in records],
        "committed_letter": [r.get("committed_answer") for r in records],
        "conditions": {
            name: {
                "preds": d["preds"],
                "p_correct_per": d["p_correct_per"],
            }
            for name, d in results_by_condition.items()
        },
    }
    with open(OUT_DIR / "per_question_predictions.json", "w") as f:
        json.dump(per_q, f, indent=2)
    _log(f"  saved per_question_predictions.json")

    _log("done — now run generate figures script")


# ── tiny null context manager used when hook_spec is None ─────────────────
from contextlib import contextmanager
@contextmanager
def _null_context():
    yield


if __name__ == "__main__":
    main()
