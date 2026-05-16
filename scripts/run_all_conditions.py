#!/usr/bin/env python
"""Phase 4 entrypoint: run one or more conditions on a single GPU.

Usage::

    CUDA_VISIBLE_DEVICES=0 python scripts/run_all_conditions.py \
        --conditions c4a,c3,c4c,c4c_matched,c4d,c4e

    CUDA_VISIBLE_DEVICES=1 python scripts/run_all_conditions.py \
        --conditions c1,c5a,c5c,c5c_matched,c5d,c5e

    # C6 disagreement-gradient sweep (primary: user-role)
    CUDA_VISIBLE_DEVICES=0 python scripts/run_all_conditions.py \
        --conditions c6 --framing user --device 0

    # C6 secondary: self-framing
    CUDA_VISIBLE_DEVICES=1 python scripts/run_all_conditions.py \
        --conditions c6 --framing self --device 1

Each condition is pickled to ``results/<name>.pkl``. C3 depends on C4a's
per-question token counts; this script detects that dependency and either
runs C4a first (if it's in the list) or loads cached token counts from
``results/c4a_token_counts.npy`` on disk.

C6 is a special token — it is not in CONDITION_REGISTRY. When ``c6`` appears
in ``--conditions``, the script runs the full 5-point gradient sweep
(0v4 … 4v0) under the framing specified by ``--framing`` (``user`` or
``self``). Output files: ``results/c6_{u|s}_{k}v{4-k}.pkl`` for each point.
C6 requires ``results/jury_responses_phi_wrong.json`` and
``results/jury_responses_correct.json`` to exist (generate them via the
corpus-generation notebook or script first).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import RESULTS_DIR  # noqa: E402
from src.data import load_artifacts  # noqa: E402
from src.experiment import run_experiment  # noqa: E402
from src.lda import CleanLDA  # noqa: E402
from src.model import get_model_and_tokenizer  # noqa: E402
from src.prompts import CONDITION_REGISTRY, set_c4a_token_counts  # noqa: E402


C4A_TOKENS_PATH = RESULTS_DIR / "c4a_token_counts.npy"
C6_GRADIENT_POINTS = [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]


def _condition_order(names: list[str]) -> list[str]:
    """C3 must run after C4a if both are requested; otherwise preserve order."""
    if "c3" in names and "c4a" in names:
        # Promote c4a before c3 (stable otherwise)
        out = [n for n in names if n not in ("c4a", "c3")]
        out_head = []
        for n in names:
            if n == "c4a":
                out_head.append("c4a")
                break
        i4a = out_head.index("c4a")
        rest = [n for n in names if n != "c4a"]
        if "c3" in rest:
            rest.remove("c3")
            # insert c3 right after c4a
            out = ["c4a", "c3"] + [n for n in names if n not in ("c4a", "c3")]
            return out
    return list(names)


def _save_result(name: str, result: dict, suffix: str = "") -> Path:
    fname = f"{name}{suffix}.pkl"
    path = RESULTS_DIR / fname
    with open(path, "wb") as f:
        pickle.dump(result, f)
    print(f"  saved -> {path}")
    return path


def _save_c4a_tokens(result: dict) -> None:
    np.save(C4A_TOKENS_PATH, np.asarray(result["token_counts"], dtype=np.int64))
    print(f"  saved C4a token counts -> {C4A_TOKENS_PATH}")


def _load_c4a_tokens() -> np.ndarray | None:
    if C4A_TOKENS_PATH.exists():
        return np.load(C4A_TOKENS_PATH)
    return None


def _run_c6(framing: str, art: dict, model, tokenizer, clean_lda) -> None:
    """Run the full 5-point C6 gradient sweep for one framing.

    For each gradient point (k_wrong, k_correct) in C6_GRADIENT_POINTS:
      1. Sample per-question agent assignments (seeded, reproducible).
      2. Build a closure conforming to run_experiment's builder interface.
      3. Run the experiment and augment the result with C6 metadata.
      4. Save to results/c6_{framing_short}_{k_wrong}v{k_correct}.pkl.

    ``jury_wrong`` merges the existing jury_strong corpus (gemma/qwen/mistral)
    with the Phi wrong-arguing corpus from jury_responses_phi_wrong.json so
    that all four agents are present. ``jury_correct`` is loaded from
    jury_responses_correct.json.

    """
    from src.prompts import (
        assign_agents_per_question,
        build_prompt_c6_self_framing,
        build_prompt_c6_user_role,
    )

    framing_short = "u" if framing == "user" else "s"
    builder = build_prompt_c6_user_role if framing == "user" else build_prompt_c6_self_framing

    # Load C6 jury corpora.
    jury_wrong_path = RESULTS_DIR / "jury_responses_phi_wrong.json"
    jury_correct_path = RESULTS_DIR / "jury_responses_correct.json"
    if not jury_wrong_path.exists():
        raise SystemExit(
            f"missing {jury_wrong_path} — generate the Phi wrong-arguing corpus first "
            "(see CONDITION_6_SPEC.md §4)"
        )
    if not jury_correct_path.exists():
        raise SystemExit(
            f"missing {jury_correct_path} — generate the correct-arguing corpus first "
            "(see CONDITION_6_SPEC.md §4)"
        )

    with open(jury_wrong_path) as f:
        jury_phi_wrong = json.load(f)
    with open(jury_correct_path) as f:
        jury_correct = json.load(f)

    # Merge existing wrong-arguing corpus (gemma/qwen/mistral) with Phi.
    jury_wrong = {
        "gemma":   art["jury_strong"]["gemma"],
        "qwen":    art["jury_strong"]["qwen"],
        "mistral": art["jury_strong"]["mistral"],
        "phi":     jury_phi_wrong["phi"],
    }

    n_questions = len(art["jury_strong"]["gemma"])

    for k_wrong, k_correct in C6_GRADIENT_POINTS:
        label = f"c6_{framing_short}_{k_wrong}v{k_correct}"
        description = f"C6_{framing}_{k_wrong}v{k_correct}"
        assignments = assign_agents_per_question(k_wrong, n_questions, seed=42)

        # Build a closure with the standard run_experiment builder interface:
        #   (q_idx, item, wrong_idx, jury_data, tokenizer) -> str
        # The closure ignores jury_data (wrong_idx already extracted by
        # run_experiment from jury_strong) and uses the C6 corpora captured
        # in its outer scope.
        def _make_closure(asgn, jw, jc, bldr):
            def _closure(q_idx, item, wrong_idx, jury_data, tok):
                wrong_agents = asgn[q_idx]["wrong"]
                correct_agents = asgn[q_idx]["correct"]
                return bldr(q_idx, item, wrong_idx, wrong_agents, correct_agents, jw, jc, tokenizer=tok)
            return _closure

        build_fn = _make_closure(assignments, jury_wrong, jury_correct, builder)

        print("\n" + "=" * 60)
        print(f"CONDITION {label}  ({k_wrong} wrong / {k_correct} correct)")
        print("=" * 60)

        # Pass jury_strong as the jury argument so run_experiment can extract
        # the seeded wrong_idx per question (same target as C4a/C4c/C4d).
        result = run_experiment(
            build_fn, art["jury_strong"], model, tokenizer,
            description=description, clean_lda=clean_lda,
        )

        # Augment with C6-specific metadata (see CONDITION_6_SPEC.md §9).
        result["gradient_point"] = (k_wrong, k_correct)
        result["framing"] = framing
        result["agent_assignments"] = assignments

        _save_result(label, result)
        yield_rate = result.get("yield_rate", float("nan"))
        print(f"  gradient_point={k_wrong}v{k_correct}  yield={yield_rate:.3f}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--conditions",
        required=True,
        help=(
            "comma-separated condition names from CONDITION_REGISTRY, "
            "or the special token 'c6' to run the disagreement-gradient sweep"
        ),
    )
    p.add_argument(
        "--framing",
        default=None,
        choices=["user", "self"],
        help="C6 only: framing to use for the gradient sweep (user-role or self-framing).",
    )
    p.add_argument(
        "--device",
        default=None,
        help="(advisory) CUDA device index. CUDA_VISIBLE_DEVICES is authoritative.",
    )
    p.add_argument(
        "--suffix",
        default="",
        help="Optional suffix appended to saved pickle filenames "
        "(e.g. '_pinned' writes results/c4a_pinned.pkl). Not applied to C6 outputs.",
    )
    args = p.parse_args()

    names = [n.strip() for n in args.conditions.split(",") if n.strip()]
    run_c6 = "c6" in names
    registry_names = [n for n in names if n != "c6"]

    unknown = [n for n in registry_names if n not in CONDITION_REGISTRY]
    if unknown:
        raise SystemExit(f"unknown conditions: {unknown}")

    if run_c6 and args.framing is None:
        raise SystemExit("--framing user|self is required when running c6")

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")

    art = load_artifacts()
    clean_lda = CleanLDA.fit_default()
    model, tokenizer = get_model_and_tokenizer()

    # ── Standard registry conditions ────────────────────────────────────────
    if registry_names:
        # C3 must see C4a token counts. If C4a isn't in this batch, load from disk.
        if "c3" in registry_names and "c4a" not in registry_names:
            toks = _load_c4a_tokens()
            if toks is None:
                raise SystemExit(
                    "c3 was requested but c4a has not been run. Run c4a first "
                    "(it saves token counts to results/c4a_token_counts.npy)."
                )
            set_c4a_token_counts(toks.tolist())
            print(f"loaded C4a token counts from {C4A_TOKENS_PATH}")

        ordered = _condition_order(registry_names)
        print(f"running conditions in order: {ordered}")

        for name in ordered:
            build_fn, jury_name = CONDITION_REGISTRY[name]
            jury = art["jury_strong"] if jury_name == "strong" else art["jury_weak"]
            print("\n" + "=" * 60)
            print(f"CONDITION {name}")
            print("=" * 60)
            result = run_experiment(
                build_fn, jury, model, tokenizer,
                description=name, clean_lda=clean_lda,
            )
            _save_result(name, result, suffix=args.suffix)
            if name == "c4a" and not args.suffix:
                # Only update the shared c4a_token_counts.npy from a default run;
                # suffixed runs (e.g. pinned-env triangulation) must not clobber it.
                set_c4a_token_counts(result["token_counts"])
                _save_c4a_tokens(result)

    # ── C6 disagreement-gradient sweep ──────────────────────────────────────
    if run_c6:
        print(f"\nrunning C6 gradient sweep  framing={args.framing}")
        _run_c6(args.framing, art, model, tokenizer, clean_lda)

    print("\nAll requested conditions finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
