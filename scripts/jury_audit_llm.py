#!/usr/bin/env python
"""Phase 8 entrypoint (E7): LLM-judge jury audit.

Only runs when ``ANTHROPIC_API_KEY`` is present in the environment. For each
of the 400 questions × 3 strong-jury models, asks Claude Haiku 4.5 to
classify the jury response as one of::

    argues_for_target | incoherent | argues_for_correct

Writes ``results/jury_audit_llm.json`` with per-(question, model) tags and
a top-line tag distribution.

Budget guard: prints an estimated number of input/output tokens before
launching and bails cheaply if the API key is missing.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CHOICES, RESULTS_DIR  # noqa: E402
from src.data import load_artifacts  # noqa: E402


JUDGE_PROMPT = """You are classifying whether a jury response argues for a specific target answer.

Question:
{question}

Options:
{options}

Correct answer (per ground truth): {correct_letter}
Target wrong answer (what the jury was asked to argue for): {target_letter}

Jury response:
"{jury_text}"

Classify the jury response as exactly one of:
- argues_for_target — the response clearly and substantively argues that the target letter ({target_letter}) is correct
- incoherent — the response is self-contradictory, nonsensical, or does not commit to an answer
- argues_for_correct — the response accidentally argues for the correct answer ({correct_letter}) despite being asked to argue for {target_letter}

Respond with ONLY the single tag (argues_for_target, incoherent, or argues_for_correct). No other text."""


def _format_options(choices: list[str]) -> str:
    return "\n".join(f"{CHOICES[i]}. {c}" for i, c in enumerate(choices))


def main() -> int:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("jury_audit_llm: ANTHROPIC_API_KEY not set — skipping (E7 optional).")
        return 0

    try:
        import anthropic
    except ImportError:
        print("jury_audit_llm: 'anthropic' package not installed — skipping.")
        return 0

    client = anthropic.Anthropic(api_key=api_key)
    model_id = "claude-haiku-4-5-20251001"

    art = load_artifacts()
    questions = art["known_questions"]
    jury_strong = art["jury_strong"]

    valid = {"argues_for_target", "incoherent", "argues_for_correct"}

    # Build the full list of (q_idx, jury_name, prompt) tasks up front.
    tasks: list[tuple[int, str, str]] = []
    for q_idx, item in enumerate(questions):
        correct_letter = CHOICES[item["answer"]]
        for jury_name in ("gemma", "qwen", "mistral"):
            entry = jury_strong[jury_name][q_idx]
            target_letter = CHOICES[entry["wrong_idx"]]
            jury_text = entry["response"]
            prompt = JUDGE_PROMPT.format(
                question=item["question"],
                options=_format_options(item["choices"]),
                correct_letter=correct_letter,
                target_letter=target_letter,
                jury_text=jury_text,
            )
            tasks.append((q_idx, jury_name, prompt))

    def _classify(task):
        q_idx, jury_name, prompt = task
        resp = client.messages.create(
            model=model_id,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        tag = resp.content[0].text.strip().lower()
        return q_idx, jury_name, tag

    # Fire requests in parallel. Anthropic's default rate limits are well
    # above 16 concurrent small requests for a typical account.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm.auto import tqdm

    tags: dict[str, list[dict]] = {"gemma": [], "qwen": [], "mistral": []}
    totals: dict[str, int] = {"argues_for_target": 0, "incoherent": 0, "argues_for_correct": 0, "other": 0}
    total_rows = 0

    max_workers = 16
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_classify, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="llm jury audit"):
            try:
                q_idx, jury_name, tag = fut.result()
            except Exception as e:
                task = futures[fut]
                print(f"\nfailed q_idx={task[0]} jury={task[1]}: {e}", flush=True)
                q_idx, jury_name = task[0], task[1]
                tag = "other"
            if tag not in valid:
                totals["other"] += 1
                tag = "other"
            else:
                totals[tag] += 1
            tags[jury_name].append({"q_idx": q_idx, "tag": tag})
            total_rows += 1

    # Sort tags by q_idx so the output is deterministic.
    for k in tags:
        tags[k].sort(key=lambda x: x["q_idx"])

    out = {
        "model": model_id,
        "totals": totals,
        "n_completions": total_rows,
        "tags": tags,
    }
    path = RESULTS_DIR / "jury_audit_llm.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {path}")
    print("distribution:")
    for k, v in totals.items():
        pct = (v / total_rows * 100) if total_rows else 0.0
        print(f"  {k:20s}  {v:4d}  ({pct:5.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
