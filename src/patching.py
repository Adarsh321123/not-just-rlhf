"""E10: activation patching causal check.

For each question we cache the clean hidden states on a neutral prompt, run
the forward pass on the corresponding C4a (pressured) prompt with a forward
hook that substitutes the clean hidden state at the last-token position of
the target layer, then read out the downstream logit-lens trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import CHOICES
from .data import load_artifacts
from .logit_lens import run_logit_lens
from .model import choice_token_ids
from .prompts import build_prompt_user_role_jury, format_question


def _build_neutral_prompt(item: dict, tokenizer) -> str:
    """Clean neutral prompt identical in spirit to the setup.ipynb high-confidence check."""
    q, opts = item["question"], item["choices"]
    user_content = format_question(q, opts) + "Please provide the correct answer."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt += "The correct answer is ("
    return prompt


def _cache_clean_last_token(
    neutral_prompt: str,
    model,
    tokenizer,
    layers: list[int],
) -> dict[int, torch.Tensor]:
    """Run the neutral prompt once, returning last-token hidden states at each requested layer."""
    inputs = tokenizer(neutral_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # outputs.hidden_states is a tuple of length num_layers+1 (embedding + per-layer).
    # Layer index `l` here refers to the post-layer-l hidden state, matching how
    # the logit lens indexes into outputs.hidden_states.
    return {
        l: outputs.hidden_states[l][:, -1, :].detach().clone()
        for l in layers
    }


def _read_final_layer_probs(
    text: str, model, tokenizer, correct_idx: int, wrong_idx: int
) -> tuple[float, float]:
    """Convenience: final-layer P(correct), P(wrong_target) via logit lens."""
    truth, syco, _ = run_logit_lens(text, correct_idx, wrong_idx, model, tokenizer)
    return truth[-1], syco[-1]


@dataclass
class PatchResult:
    layer: int
    mean_clean_truth: float
    mean_pressured_truth: float
    mean_patched_truth: float
    mean_clean_syco: float
    mean_pressured_syco: float
    mean_patched_syco: float
    delta: float  # patched - pressured (positive = restoration)


def run_activation_patching(
    model,
    tokenizer,
    layers: list[int],
    n_questions: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the full patching sweep.

    For each ``layer`` in ``layers``:
        1. Cache clean last-token hidden state at that layer (neutral prompt).
        2. Forward the C4a pressured prompt with a hook that overwrites
           ``hidden_states[layer][:, -1, :]`` with the cached clean state.
        3. Read the final-layer logit-lens P(correct) / P(wrong).

    Returns a dict with per-layer aggregates + per-question raw arrays.
    """
    art = load_artifacts()
    jury_strong = art["jury_strong"]
    questions = art["known_questions"]

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(questions), size=n_questions, replace=False)

    # Per-question baselines (clean + pressured)
    clean_truth_base = np.zeros(n_questions)
    clean_syco_base = np.zeros(n_questions)
    pressured_truth_base = np.zeros(n_questions)
    pressured_syco_base = np.zeros(n_questions)

    # Per-(layer, question) patched readouts
    patched_truth = {l: np.zeros(n_questions) for l in layers}
    patched_syco = {l: np.zeros(n_questions) for l in layers}

    for i, q_idx in enumerate(
        tqdm(idx.tolist(), desc=f"patching ({n_questions}q × {len(layers)}L)")
    ):
        item = questions[q_idx]
        correct_idx = item["answer"]
        wrong_idx = jury_strong["gemma"][q_idx]["wrong_idx"]

        neutral = _build_neutral_prompt(item, tokenizer)
        pressured = build_prompt_user_role_jury(
            q_idx, item, wrong_idx, jury_strong, tokenizer
        )

        # 1) Clean baseline + cache
        cache = _cache_clean_last_token(neutral, model, tokenizer, layers)
        ct, cs = _read_final_layer_probs(
            neutral, model, tokenizer, correct_idx, wrong_idx
        )
        clean_truth_base[i] = ct
        clean_syco_base[i] = cs

        # 2) Pressured baseline (no patch)
        pt, ps = _read_final_layer_probs(
            pressured, model, tokenizer, correct_idx, wrong_idx
        )
        pressured_truth_base[i] = pt
        pressured_syco_base[i] = ps

        # 3) Patched runs: one per target layer
        pressured_inputs = tokenizer(pressured, return_tensors="pt").to(model.device)
        vocab_indices = [
            choice_token_ids(tokenizer)[c] for c in CHOICES
        ]
        for l in layers:
            clean_vec = cache[l]  # (1, d_model)

            # Hook: replace the last-token position at the output of layer `l-1`
            # (i.e. the input to layer `l`) so that layer `l` sees the clean state.
            # ``model.model.layers[l-1]`` output is exactly
            # ``outputs.hidden_states[l]`` in the logit-lens indexing used above.
            target_layer = l - 1 if l > 0 else 0

            def hook_fn(_module, _input, output, clean_vec=clean_vec):
                if isinstance(output, tuple):
                    hs = output[0]
                    hs = hs.clone()
                    hs[:, -1, :] = clean_vec.to(hs.dtype)
                    return (hs,) + output[1:]
                else:
                    hs = output.clone()
                    hs[:, -1, :] = clean_vec.to(hs.dtype)
                    return hs

            handle = model.model.layers[target_layer].register_forward_hook(hook_fn)
            try:
                with torch.no_grad():
                    out = model(**pressured_inputs)
                logits = out.logits[0, -1, :]
                mc = torch.softmax(logits[vocab_indices], dim=-1)
                patched_truth[l][i] = mc[correct_idx].item()
                patched_syco[l][i] = mc[wrong_idx].item()
            finally:
                handle.remove()

    per_layer: dict[int, PatchResult] = {}
    for l in layers:
        per_layer[l] = PatchResult(
            layer=l,
            mean_clean_truth=float(clean_truth_base.mean()),
            mean_pressured_truth=float(pressured_truth_base.mean()),
            mean_patched_truth=float(patched_truth[l].mean()),
            mean_clean_syco=float(clean_syco_base.mean()),
            mean_pressured_syco=float(pressured_syco_base.mean()),
            mean_patched_syco=float(patched_syco[l].mean()),
            delta=float(patched_truth[l].mean() - pressured_truth_base.mean()),
        )

    return {
        "question_indices": idx.tolist(),
        "layers": layers,
        "clean_truth_base": clean_truth_base,
        "pressured_truth_base": pressured_truth_base,
        "clean_syco_base": clean_syco_base,
        "pressured_syco_base": pressured_syco_base,
        "patched_truth": patched_truth,
        "patched_syco": patched_syco,
        "per_layer": per_layer,
    }
