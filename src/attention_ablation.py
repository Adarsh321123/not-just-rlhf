"""Attention-head ablation hooks for LlamaAttention.

Zeros the per-head output contribution of specified ``(layer, head)`` pairs at a
specified sequence position. Hooks attach to ``model.model.layers[L].self_attn.o_proj``
as ``forward_pre_hook``s; the input to ``o_proj`` has shape ``(bsz, seq, n_heads*head_dim)``
and contains the per-head concatenated attention outputs. Zeroing
``input[:, pos, h*head_dim:(h+1)*head_dim]`` removes head ``h``'s contribution to
position ``pos``'s hidden state without touching other positions or heads.

This module is standalone — it does not modify any existing ``src/`` file.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import torch


def _make_hook(head_indices: list[int], pos: int, head_dim: int):
    """Build a forward_pre_hook that zeros given heads at sequence index ``pos``.

    ``pos`` may be a non-negative absolute index or a negative offset from the
    end (``-1`` = last token, ``-2`` = predecessor, etc.).
    """
    def hook(module, args):
        # args is a tuple; the first arg is the input tensor: (B, S, H*d)
        x = args[0]
        # Resolve negative pos relative to the current input's seq dim
        S = x.shape[1]
        p = pos if pos >= 0 else S + pos
        if p < 0 or p >= S:
            return args
        # Clone to avoid modifying a tensor that might be reused upstream
        x = x.clone()
        for h in head_indices:
            x[:, p, h * head_dim:(h + 1) * head_dim] = 0.0
        return (x,) + args[1:]
    return hook


@contextmanager
def ablate_heads(
    model,
    head_specs: Iterable[tuple[int, int]],
    position: int,
    head_dim: int = 128,
):
    """Context manager: register forward_pre_hooks zeroing given (layer, head)
    pairs at the given position. Hooks are removed on exit.

    Parameters
    ----------
    model : HuggingFace LlamaForCausalLM (or equivalent with ``model.model.layers``)
    head_specs : iterable of (layer_idx, head_idx) tuples
        Which heads to ablate.
    position : int
        Sequence position at which to zero. Negative values count from the end
        (``-1`` = last input token).
    head_dim : int
        Per-head dimension (128 for Llama-3.1-8B).
    """
    # Group by layer
    by_layer: dict[int, list[int]] = {}
    for L, h in head_specs:
        by_layer.setdefault(int(L), []).append(int(h))

    handles = []
    try:
        for L, heads in by_layer.items():
            o_proj = model.model.layers[L].self_attn.o_proj
            handle = o_proj.register_forward_pre_hook(_make_hook(heads, position, head_dim))
            handles.append(handle)
        yield
    finally:
        for h in handles:
            h.remove()


def compute_per_head_question_mass(
    attn_layer: torch.Tensor,   # [1, n_heads, seq, seq]
    target_pos: int,
    question_range: tuple[int, int],
) -> torch.Tensor:
    """Return per-head attention mass to the question token range at target_pos.

    Output shape: [n_heads], on CPU as float32.
    """
    if target_pos < 0:
        target_pos = attn_layer.shape[2] + target_pos
    q_lo, q_hi = question_range
    q_lo = max(q_lo, 0)
    q_hi = min(q_hi, attn_layer.shape[3])
    if q_hi <= q_lo:
        return torch.zeros(attn_layer.shape[1])
    mass = attn_layer[0, :, target_pos, q_lo:q_hi].sum(dim=-1).float().cpu()
    return mass
