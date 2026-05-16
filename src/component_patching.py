"""E11: component-level activation patching (MLP vs attention contribution).

Residual-contribution decomposition (Heimersheim / Nanda 2022 style):
- MLP patch   : replace the layer's MLP *contribution* to the residual stream
                with the clean (neutral-prompt) MLP contribution; attention
                runs on the pressured context unchanged.
- Attn patch  : replace the layer's attention *contribution* with the clean
                attention contribution; MLP runs pressured.
- Residual patch: replace the full hidden state at the layer boundary with the
                clean hidden state (same convention as src/patching.py, for
                comparison).

Hooks are placed on model.model.layers[l].mlp and .self_attn submodules for
the component patches, and on model.model.layers[l-1] for the residual patch.
All patches operate at the last-token position only.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import CHOICES
from .data import load_artifacts
from .model import choice_token_ids
from .prompts import build_prompt_user_role_jury, format_question


def _build_neutral_prompt(item: dict, tokenizer) -> str:
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


def _softmax_mc(logits: torch.Tensor, vocab_indices: list[int]) -> torch.Tensor:
    return torch.softmax(logits[vocab_indices], dim=-1)


def run_component_patching(
    model,
    tokenizer,
    layers: list[int],
    n_questions: int = 400,
    seed: int = 42,
) -> dict[str, Any]:
    """Sweep MLP, attention, and residual patching over *layers*.

    For each question we:
    1. Run the neutral prompt once to cache the clean MLP contribution, clean
       attention contribution, and clean hidden state at each layer boundary
       (and read clean baseline probs from the same pass).
    2. Run the pressured prompt without any patch (pressured baseline probs).
    3. For each layer run three patched forward passes on the pressured prompt:
       - MLP patch   : hook .mlp, replace last-token output with clean value.
       - Attn patch  : hook .self_attn, replace last-token output[0] with
                       clean value.
       - Residual patch : hook layers[l-1], replace last-token hidden state
                          with clean value (identical to src/patching.py).

    Returns
    -------
    dict with keys:
        question_indices     : list[int]
        layers               : list[int]
        components           : ["mlp", "attn", "residual"]
        clean_truth_base     : (n,) float
        pressured_truth_base : (n,) float
        clean_syco_base      : (n,) float
        pressured_syco_base  : (n,) float
        patched_truth        : dict[(layer, component)] -> (n,) float
        patched_syco         : dict[(layer, component)] -> (n,) float
    """
    art = load_artifacts()
    jury_strong = art["jury_strong"]
    questions = art["known_questions"]

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(questions), size=n_questions, replace=False)

    COMPONENTS = ["mlp", "attn", "residual"]
    n = n_questions

    clean_truth_base = np.zeros(n)
    clean_syco_base = np.zeros(n)
    pressured_truth_base = np.zeros(n)
    pressured_syco_base = np.zeros(n)

    patched_truth: dict[tuple, np.ndarray] = {
        (l, c): np.zeros(n) for l in layers for c in COMPONENTS
    }
    patched_syco: dict[tuple, np.ndarray] = {
        (l, c): np.zeros(n) for l in layers for c in COMPONENTS
    }

    tok_ids = choice_token_ids(tokenizer)
    vocab_indices = [tok_ids[c] for c in CHOICES]

    for i, q_idx in enumerate(
        tqdm(
            idx.tolist(),
            desc=f"component-patch ({n}q × {len(layers)}L × {len(COMPONENTS)}C)",
        )
    ):
        item = questions[q_idx]
        correct_idx = item["answer"]
        wrong_idx = jury_strong["gemma"][q_idx]["wrong_idx"]

        neutral_text = _build_neutral_prompt(item, tokenizer)
        pressured_text = build_prompt_user_role_jury(
            q_idx, item, wrong_idx, jury_strong, tokenizer
        )

        # ── Clean forward pass: cache components + residual, read baseline probs ──
        neutral_inputs = tokenizer(neutral_text, return_tensors="pt").to(model.device)
        # comp_cache[l]["mlp"] / ["attn"]  →  (1, d_model) clean contribution
        comp_cache: dict[int, dict[str, torch.Tensor]] = {l: {} for l in layers}
        # res_cache[l]  →  (1, d_model) clean hidden state going *into* layer l
        res_cache: dict[int, torch.Tensor] = {}
        handles = []

        for l in layers:
            _tgt = l - 1 if l > 0 else 0

            def _mlp_hook(mod, inp, out, _l=l):
                comp_cache[_l]["mlp"] = out[:, -1, :].detach().clone()

            def _attn_hook(mod, inp, out, _l=l):
                # LlamaAttention returns (attn_output, attn_weights, past_kv)
                a = out[0] if isinstance(out, tuple) else out
                comp_cache[_l]["attn"] = a[:, -1, :].detach().clone()

            def _res_hook(mod, inp, out, _l=l):
                hs = out[0] if isinstance(out, tuple) else out
                res_cache[_l] = hs[:, -1, :].detach().clone()

            handles.append(model.model.layers[l].mlp.register_forward_hook(_mlp_hook))
            handles.append(
                model.model.layers[l].self_attn.register_forward_hook(_attn_hook)
            )
            handles.append(model.model.layers[_tgt].register_forward_hook(_res_hook))

        with torch.no_grad():
            clean_out = model(**neutral_inputs)

        for h in handles:
            h.remove()

        c_mc = _softmax_mc(clean_out.logits[0, -1, :], vocab_indices)
        clean_truth_base[i] = c_mc[correct_idx].item()
        clean_syco_base[i] = c_mc[wrong_idx].item()

        # ── Pressured baseline ──
        press_inputs = tokenizer(pressured_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            press_out = model(**press_inputs)
        p_mc = _softmax_mc(press_out.logits[0, -1, :], vocab_indices)
        pressured_truth_base[i] = p_mc[correct_idx].item()
        pressured_syco_base[i] = p_mc[wrong_idx].item()

        # ── Component-patched forward passes (one per layer per component) ──
        for l in layers:
            # MLP patch — hook layers[l].mlp
            mlp_vec = comp_cache[l]["mlp"]

            def _mlp_patch(mod, inp, out, _v=mlp_vec):
                o = out.clone()
                o[:, -1, :] = _v.to(o.dtype)
                return o

            h = model.model.layers[l].mlp.register_forward_hook(_mlp_patch)
            try:
                with torch.no_grad():
                    patched_out = model(**press_inputs)
                mc = _softmax_mc(patched_out.logits[0, -1, :], vocab_indices)
                patched_truth[(l, "mlp")][i] = mc[correct_idx].item()
                patched_syco[(l, "mlp")][i] = mc[wrong_idx].item()
            finally:
                h.remove()

            # Attention patch — hook layers[l].self_attn
            attn_vec = comp_cache[l]["attn"]

            def _attn_patch(mod, inp, out, _v=attn_vec):
                if isinstance(out, tuple):
                    a = out[0].clone()
                    a[:, -1, :] = _v.to(a.dtype)
                    return (a,) + out[1:]
                o = out.clone()
                o[:, -1, :] = _v.to(o.dtype)
                return o

            h = model.model.layers[l].self_attn.register_forward_hook(_attn_patch)
            try:
                with torch.no_grad():
                    patched_out = model(**press_inputs)
                mc = _softmax_mc(patched_out.logits[0, -1, :], vocab_indices)
                patched_truth[(l, "attn")][i] = mc[correct_idx].item()
                patched_syco[(l, "attn")][i] = mc[wrong_idx].item()
            finally:
                h.remove()

            # Residual patch — hook layers[l-1], same convention as patching.py
            res_vec = res_cache[l]
            _tgt = l - 1 if l > 0 else 0

            def _res_patch(mod, inp, out, _v=res_vec):
                if isinstance(out, tuple):
                    hs = out[0].clone()
                    hs[:, -1, :] = _v.to(hs.dtype)
                    return (hs,) + out[1:]
                hs = out.clone()
                hs[:, -1, :] = _v.to(hs.dtype)
                return hs

            h = model.model.layers[_tgt].register_forward_hook(_res_patch)
            try:
                with torch.no_grad():
                    patched_out = model(**press_inputs)
                mc = _softmax_mc(patched_out.logits[0, -1, :], vocab_indices)
                patched_truth[(l, "residual")][i] = mc[correct_idx].item()
                patched_syco[(l, "residual")][i] = mc[wrong_idx].item()
            finally:
                h.remove()

    return {
        "question_indices": idx.tolist(),
        "layers": layers,
        "components": COMPONENTS,
        "clean_truth_base": clean_truth_base,
        "pressured_truth_base": pressured_truth_base,
        "clean_syco_base": clean_syco_base,
        "pressured_syco_base": pressured_syco_base,
        "patched_truth": patched_truth,
        "patched_syco": patched_syco,
    }


def run_both_extension(
    model,
    tokenizer,
    existing: dict,
) -> tuple[dict, dict]:
    """Add the 'both' patch variant to an existing component-patching result.

    Patches both .self_attn and .mlp simultaneously at the last-token position,
    leaving h_{l-1} pressured.  This is the layer-local baseline against which
    individual component deltas should be compared.

    Re-uses the question indices and pressured inputs already implicit in
    *existing*; runs one clean forward pass (to cache components) and one
    'both'-patched forward pass per question per layer.

    Returns
    -------
    patched_truth_both  : dict[(layer, "both")] -> (n,) float array
    patched_syco_both   : dict[(layer, "both")] -> (n,) float array
    """
    art = load_artifacts()
    jury_strong = art["jury_strong"]
    questions = art["known_questions"]

    layers = existing["layers"]
    q_indices = existing["question_indices"]
    n = len(q_indices)

    patched_truth_both: dict[tuple, np.ndarray] = {
        (l, "both"): np.zeros(n) for l in layers
    }
    patched_syco_both: dict[tuple, np.ndarray] = {
        (l, "both"): np.zeros(n) for l in layers
    }

    tok_ids = choice_token_ids(tokenizer)
    vocab_indices = [tok_ids[c] for c in CHOICES]

    for i, q_idx in enumerate(
        tqdm(q_indices, desc=f"both-patch ({n}q × {len(layers)}L)")
    ):
        item = questions[q_idx]
        correct_idx = item["answer"]
        wrong_idx = jury_strong["gemma"][q_idx]["wrong_idx"]

        neutral_text = _build_neutral_prompt(item, tokenizer)
        pressured_text = build_prompt_user_role_jury(
            q_idx, item, wrong_idx, jury_strong, tokenizer
        )

        # Clean forward pass: cache MLP and attention contributions at each layer
        neutral_inputs = tokenizer(neutral_text, return_tensors="pt").to(model.device)
        comp_cache: dict[int, dict[str, torch.Tensor]] = {l: {} for l in layers}
        handles = []

        for l in layers:
            def _mlp(mod, inp, out, _l=l):
                comp_cache[_l]["mlp"] = out[:, -1, :].detach().clone()

            def _attn(mod, inp, out, _l=l):
                a = out[0] if isinstance(out, tuple) else out
                comp_cache[_l]["attn"] = a[:, -1, :].detach().clone()

            handles.append(model.model.layers[l].mlp.register_forward_hook(_mlp))
            handles.append(
                model.model.layers[l].self_attn.register_forward_hook(_attn)
            )

        with torch.no_grad():
            model(**neutral_inputs)

        for h in handles:
            h.remove()

        press_inputs = tokenizer(pressured_text, return_tensors="pt").to(model.device)

        # One 'both' patched forward pass per layer
        for l in layers:
            attn_v = comp_cache[l]["attn"]
            mlp_v = comp_cache[l]["mlp"]

            def _a(mod, inp, out, _v=attn_v):
                if isinstance(out, tuple):
                    a = out[0].clone()
                    a[:, -1, :] = _v.to(a.dtype)
                    return (a,) + out[1:]
                o = out.clone()
                o[:, -1, :] = _v.to(o.dtype)
                return o

            def _m(mod, inp, out, _v=mlp_v):
                o = out.clone()
                o[:, -1, :] = _v.to(o.dtype)
                return o

            h1 = model.model.layers[l].self_attn.register_forward_hook(_a)
            h2 = model.model.layers[l].mlp.register_forward_hook(_m)
            try:
                with torch.no_grad():
                    patched_out = model(**press_inputs)
                mc = _softmax_mc(patched_out.logits[0, -1, :], vocab_indices)
                patched_truth_both[(l, "both")][i] = mc[correct_idx].item()
                patched_syco_both[(l, "both")][i] = mc[wrong_idx].item()
            finally:
                h1.remove()
                h2.remove()

    return patched_truth_both, patched_syco_both
