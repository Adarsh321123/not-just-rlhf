"""N-generalized tool-role builder for C6 disagreement-gradient scaling.

Extends the tool-role framing to N=4, 5, 6 agents, matching the pattern
from ``build_prompt_tool_role_gradient`` in ``src.disagreement_framings``
but using the N-generalized agent registries and assignment logic from
``src.c6_scaling``.
"""
from __future__ import annotations

import random
from typing import Sequence

from .c6_scaling import (
    _agent_display_name,
    closing_line_N,
)
from .config import CHOICES
from .prompts import format_question


TOOL_ACTION_PHRASE = " Based on their analysis, what is your answer?"


def build_prompt_c6_tool_role_N(
    q_idx: int,
    item: dict,
    wrong_idx: int,
    wrong_agents: Sequence[str],
    correct_agents: Sequence[str],
    jury_wrong: dict,
    jury_correct: dict,
    tokenizer,  # noqa: ARG001 — tool-role uses hardcoded Llama tokens
) -> str:
    """Tool-role C6 prompt with N-agent gradient consensus.

    Places jury responses in the ipython role turn using the
    ``<|python_tag|>query_ai_panel()`` pattern, exactly matching the
    structure of ``build_prompt_tool_role_gradient``.
    """
    q, opts, ans = item["question"], item["choices"], item["answer"]
    wrg_l = CHOICES[wrong_idx]
    cor_l = CHOICES[ans]
    question_text = format_question(q, opts) + "Please provide the correct answer."

    agents_data = []
    for agent in wrong_agents:
        r = jury_wrong[agent][q_idx]["response"]
        agents_data.append((_agent_display_name(agent), wrg_l, r))
    for agent in correct_agents:
        r = jury_correct[agent][q_idx]["response"]
        agents_data.append((_agent_display_name(agent), cor_l, r))

    rng = random.Random(q_idx)
    rng.shuffle(agents_data)

    tool_content = "\n\n".join(
        f"[{disp}]: The answer is {lett}. {txt}" for disp, lett, txt in agents_data
    )

    closing = closing_line_N(len(wrong_agents), len(correct_agents), wrg_l, cor_l)
    final_user = closing + TOOL_ACTION_PHRASE

    prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"You are a helpful assistant.<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{question_text}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"I'll query the AI panel for their views on this question.\n\n"
        f"<|python_tag|>query_ai_panel()<|eot_id|>"
        f"<|start_header_id|>ipython<|end_header_id|>\n\n"
        f"{tool_content}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{final_user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"The correct answer is ("
    )
    return prompt
