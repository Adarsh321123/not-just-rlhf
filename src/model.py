"""Subject model + tokenizer loader.

Usage:
    from src.model import get_model_and_tokenizer
    model, tokenizer = get_model_and_tokenizer()
"""
from __future__ import annotations

from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import CHOICES, MODEL_ID, MODEL_REVISION, HF_TOKEN


@lru_cache(maxsize=1)
def get_tokenizer():
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, token=HF_TOKEN, revision=MODEL_REVISION
    )
    tok.pad_token = tok.eos_token
    return tok


@lru_cache(maxsize=1)
def get_model():
    print(
        f"Loading {MODEL_ID}@{MODEL_REVISION[:10] if MODEL_REVISION else 'main'} "
        f"in bf16 onto available devices..."
    )
    # Use ``torch_dtype`` (works in both transformers 4.46 and 4.57) instead of
    # ``dtype`` which is 4.55+ only. E18 runs this code under the pinned env.
    import transformers as _tfs  # noqa: F401 — version probe comment
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
        revision=MODEL_REVISION,
    )
    model.eval()
    return model


def get_model_and_tokenizer():
    return get_model(), get_tokenizer()


def choice_token_ids(tokenizer=None) -> dict[str, int]:
    """Return a dict mapping each choice letter ('A', 'B', ...) to its token id."""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    return {c: tokenizer.encode(c, add_special_tokens=False)[0] for c in CHOICES}
