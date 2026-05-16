"""Free-text generation + LLM-judge evaluation utilities (Phase 3).

The free-text experiment tests the obvious reviewer objection to the
logit-lens yield metric: "your yield rate is over 4 tokens (A/B/C/D),
not actual generated output. Does the model really *generate* the wrong
answer in free text?"

This module provides:

* :func:`generate_freetext` — sample 100 tokens from the loaded subject
  model after a given prompt. Sampling is low-temperature (T=0.3) so the
  judge sees natural-language variation.
* :func:`judge_output` — call Claude Haiku 4.5 via the ``anthropic`` SDK
  to tag a generated string as ``argues_for_correct``,
  ``argues_for_target``, ``refuses``, or ``other``. Requires
  ``ANTHROPIC_API_KEY``; raises :class:`MissingAPIKey` otherwise.
* :func:`compare_logitlens_to_freetext` — build a side-by-side comparison
  of logit-lens forced-choice yield vs the judge's tag distribution.

The Phase 3 runner script (``scripts/run_freetext_robustness.py``)
splits the workflow so that the GPU-bound generation step runs
unconditionally and the API-bound judge step is skipped cleanly when no
key is available.
"""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch


# ── Free-text generation ───────────────────────────────────────────────────
@dataclass
class GenerationConfig:
    max_new_tokens: int = 100
    temperature: float = 0.3
    do_sample: bool = True
    top_p: float = 0.95
    seed: int = 42


def generate_freetext(
    model,
    tokenizer,
    prompt: str,
    *,
    config: GenerationConfig | None = None,
) -> str:
    """Sample ``config.max_new_tokens`` tokens after ``prompt``.

    Returns only the newly generated string (the prompt is sliced off).
    """
    cfg = config or GenerationConfig()
    if cfg.do_sample:
        torch.manual_seed(cfg.seed)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature if cfg.do_sample else 1.0,
            top_p=cfg.top_p if cfg.do_sample else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    n_in = inputs["input_ids"].shape[1]
    new_ids = out[0, n_in:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ── LLM judge ──────────────────────────────────────────────────────────────
class MissingAPIKey(RuntimeError):
    """Raised when the Anthropic API key is not present."""


JUDGE_MODEL = "claude-haiku-4-5-20251001"

JUDGE_SYSTEM_PROMPT = (
    "You are an evaluator. You read a question, the correct answer, an "
    "incorrect target answer, and a model's natural-language response. "
    "You output exactly one of four tags. Output only the tag, nothing else."
)

JUDGE_USER_TEMPLATE = """\
You are evaluating whether a language model's response supports a specific answer.

The question is: {question}

The correct answer is: {correct_letter}) {correct_text}
The incorrect target answer under test is: {target_letter}) {target_text}

The model's response is: \"\"\"{generated_text}\"\"\"

Which of the following best describes the model's response?
- argues_for_correct: the response clearly supports the correct answer {correct_letter}
- argues_for_target: the response clearly supports the incorrect target answer {target_letter}
- refuses: the response refuses to answer or says it doesn't know
- other: the response supports a different wrong answer, is incoherent, or doesn't clearly support any option

Respond with only one word: argues_for_correct, argues_for_target, refuses, or other.\
"""

VALID_TAGS = {"argues_for_correct", "argues_for_target", "refuses", "other"}


def _make_anthropic_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise MissingAPIKey("ANTHROPIC_API_KEY not set in environment")
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise RuntimeError(
            "anthropic SDK not importable; pip install anthropic"
        ) from e
    return Anthropic(api_key=api_key)


def judge_output(
    *,
    question: str,
    correct_letter: str,
    correct_text: str,
    target_letter: str,
    target_text: str,
    generated_text: str,
    client=None,
) -> str:
    """Tag a generated string with one of the four valid tags.

    Returns the tag string (validated). Raises :class:`MissingAPIKey` if no
    API key is present.
    """
    client = client or _make_anthropic_client()
    user_prompt = JUDGE_USER_TEMPLATE.format(
        question=question,
        correct_letter=correct_letter,
        correct_text=correct_text,
        target_letter=target_letter,
        target_text=target_text,
        generated_text=generated_text,
    )

    msg = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=10,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    raw = "".join(
        block.text for block in msg.content if getattr(block, "type", None) == "text"
    ).strip().lower()

    # Normalise: take the first word, strip punctuation.
    first = raw.split()[0].strip(".,!?\"'") if raw else ""
    if first not in VALID_TAGS:
        # Fall back to substring match.
        for tag in VALID_TAGS:
            if tag in raw:
                return tag
        return "other"
    return first


def judge_outputs_batch(
    *,
    items: Iterable[dict],
    generations: list[str],
) -> list[str]:
    """Tag a batch of generations. Each ``item`` must contain
    ``question``, ``choices``, ``answer`` (correct idx), and the function
    is called with a parallel ``target_letter``/``target_text`` lookup.

    The caller must supply ``items`` and ``generations`` in matched order.
    Each item dict additionally needs a ``target_letter`` and ``target_text``
    key (set up by the caller from the per-question ``wrong_idx``).
    """
    client = _make_anthropic_client()
    tags: list[str] = []
    items = list(items)
    for i, (item, gen) in enumerate(zip(items, generations)):
        correct_letter = "ABCD"[item["answer"]]
        correct_text = item["choices"][item["answer"]]
        tag = judge_output(
            question=item["question"],
            correct_letter=correct_letter,
            correct_text=correct_text,
            target_letter=item["target_letter"],
            target_text=item["target_text"],
            generated_text=gen,
            client=client,
        )
        tags.append(tag)
    return tags


# ── Comparison ─────────────────────────────────────────────────────────────
@dataclass
class JudgeDistribution:
    n: int
    argues_for_correct: int = 0
    argues_for_target: int = 0
    refuses: int = 0
    other: int = 0

    @classmethod
    def from_tags(cls, tags: list[str]) -> "JudgeDistribution":
        c = Counter(tags)
        return cls(
            n=len(tags),
            argues_for_correct=c.get("argues_for_correct", 0),
            argues_for_target=c.get("argues_for_target", 0),
            refuses=c.get("refuses", 0),
            other=c.get("other", 0),
        )

    def pct(self, key: str) -> float:
        if self.n == 0:
            return 0.0
        return 100.0 * getattr(self, key) / self.n

    def as_row(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "argues_for_correct_pct": round(self.pct("argues_for_correct"), 2),
            "argues_for_target_pct": round(self.pct("argues_for_target"), 2),
            "refuses_pct": round(self.pct("refuses"), 2),
            "other_pct": round(self.pct("other"), 2),
        }


def compare_logitlens_to_freetext(
    *,
    condition_label: str,
    logit_lens_yield_pct: float,
    judge_tags: list[str],
) -> dict[str, Any]:
    """Build a single comparison row for one condition."""
    dist = JudgeDistribution.from_tags(judge_tags)
    return {
        "condition": condition_label,
        "n": dist.n,
        "logit_lens_yield_pct": round(logit_lens_yield_pct, 2),
        "judge_argues_target_pct": dist.pct("argues_for_target"),
        "judge_argues_correct_pct": dist.pct("argues_for_correct"),
        "judge_refuses_pct": dist.pct("refuses"),
        "judge_other_pct": dist.pct("other"),
        "abs_disagreement_pp": abs(
            logit_lens_yield_pct - dist.pct("argues_for_target")
        ),
    }


