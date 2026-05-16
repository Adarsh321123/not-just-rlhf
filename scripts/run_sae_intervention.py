"""Phase 8: Causal feature-clamping intervention.

For a 50-question C4a subset, run three forward passes per question:

1. **Unclamped**: standard forward pass on the C4a pressured prompt. Read
   final-layer P(correct) at the last-token position.

2. **Subtractive clamp (necessity test)**: install a forward hook on
   ``model.model.layers[SAE_LAYER - 1]`` that encodes the last-token hidden
   state through the SAE, overwrites the top-k "rising" features (those whose
   mean pressured activation is most elevated above their clean mean) with
   their **clean** mean values, decodes back, and substitutes the result.
   Measure final-layer P(correct). If this restores truth, the rising
   features are *causally necessary* for the sycophantic collapse.

3. **Additive clamp (sufficiency test)**: same hook but with the rising
   features pinned to 2× their **pressured** mean values. If this drives
   P(correct) lower than the baseline unclamped pass, the features are
   *causally sufficient* to drive sycophancy on their own.

Results → ``results/sae/intervention.json`` and a bar chart figure.

GPU: one 3090 for the Llama forward passes. SAE runs on the same GPU.
Before loading, the script checks ``nvidia-smi`` and picks whichever GPU
has the most free memory.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.config import CHOICES, FIGURES_DIR, RESULTS_DIR  # noqa: E402
from src.data import jury_for, load_artifacts  # noqa: E402
from src.prompts import build_prompt_user_role_jury  # noqa: E402
from src.sae import (  # noqa: E402
    SAE_LAYER,
    load_sae,
    make_feature_clamping_hook,
    sae_results_dir,
)


N_QUESTIONS = 50
SEED = 42
TOP_K_FEATURES = 5


def pick_gpu() -> int:
    """Select whichever GPU has the most free memory. Defer to GPU 1 on ties."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ]
        )
    except Exception:
        return 0
    frees = [int(x) for x in out.decode().strip().splitlines()]
    # pick highest; tie-break to index 1 per coordination note.
    best_idx = 0
    best_val = -1
    for i, v in enumerate(frees):
        if v > best_val or (v == best_val and i == 1):
            best_idx = i
            best_val = v
    print(f"GPU free memory: {frees} → using GPU {best_idx}")
    return best_idx


def last_token_pcorrect(
    logits_final_layer: torch.Tensor,
    correct_idx: int,
    wrong_idx: int,
    choice_token_ids: list[int],
) -> tuple[float, float]:
    """Return (P(correct), P(wrong)) at the final layer, last token, restricted
    to the four choice tokens."""
    lg = logits_final_layer[0, -1, :][choice_token_ids]
    probs = torch.softmax(lg, dim=-1)
    return probs[correct_idx].item(), probs[wrong_idx].item()


def main() -> None:
    # --- load data, top features, model ---------------------------------
    sae_dir = sae_results_dir(RESULTS_DIR)
    with open(sae_dir / "c4a_top_rising.json") as f:
        rising = json.load(f)

    top_features = rising[:TOP_K_FEATURES]
    rising_indices = [r["feature_idx"] for r in top_features]
    rising_clean_values = torch.tensor(
        [r["mean_clean"] for r in top_features], dtype=torch.float32
    )
    rising_pressured_values = torch.tensor(
        [r["mean_pressured"] for r in top_features], dtype=torch.float32
    )
    print(
        f"Using top-{TOP_K_FEATURES} rising C4a features: {rising_indices}"
    )
    print(f"  clean means:     {rising_clean_values.tolist()}")
    print(f"  pressured means: {rising_pressured_values.tolist()}")

    gpu = pick_gpu()
    device = f"cuda:{gpu}"
    torch.cuda.set_device(gpu)

    # Lazy import to honor the GPU pick.
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from src.model import choice_token_ids, get_model_and_tokenizer  # noqa: E402

    print(f"\nLoading Llama on GPU {gpu} ...")
    model, tokenizer = get_model_and_tokenizer()
    ctoks = choice_token_ids(tokenizer)
    vocab_indices = [ctoks[c] for c in CHOICES]

    print(f"Loading SAE on {device} ...")
    sae = load_sae(device=device, dtype=torch.float32)

    # --- pick 50 questions ---------------------------------------------
    art = load_artifacts()
    questions = art["known_questions"]
    jury_strong = jury_for("strong")

    rng = np.random.default_rng(SEED)
    n_total = len(questions)
    subset_idx = rng.choice(n_total, size=N_QUESTIONS, replace=False)
    subset_idx.sort()
    print(f"\nSubset of {N_QUESTIONS} questions (seed {SEED}): {subset_idx[:10].tolist()}...")

    # --- where to hook: Goodfire "l19" = output of block 19 (0-indexed) ---
    # That's ``model.model.layers[19]``. Registering a forward hook on that
    # module means the hook fires with the block-19 output tensor, which is
    # what we want to overwrite.
    layers = model.model.layers
    hook_module_idx = 19  # 0-indexed block whose output is hidden_states[20]
    print(
        f"Hooking model.model.layers[{hook_module_idx}] "
        f"(= Goodfire layer 19, hidden_states[{SAE_LAYER}])"
    )

    # Verify the hook module actually produces the tensor we saw at
    # hidden_states[SAE_LAYER]. Cheap sanity check — match against a clean
    # forward pass on the first question.
    q0 = questions[int(subset_idx[0])]
    wrong0 = jury_strong["gemma"][int(subset_idx[0])]["wrong_idx"]
    prompt0 = build_prompt_user_role_jury(
        int(subset_idx[0]), q0, wrong0, jury_strong, tokenizer
    )
    inputs0 = tokenizer(prompt0, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out0 = model(**inputs0, output_hidden_states=True)
    ref = out0.hidden_states[SAE_LAYER][0, -1, :]
    capture = {}

    def capture_hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        capture["last"] = h[0, -1, :].detach().clone()

    h_cap = layers[hook_module_idx].register_forward_hook(capture_hook)
    with torch.no_grad():
        _ = model(**inputs0)
    h_cap.remove()
    diff = (capture["last"].float() - ref.float()).abs().mean().item()
    print(
        f"  sanity: ||hook_output - hidden_states[{SAE_LAYER}]||_1 mean = {diff:.2e} (should be ~0)"
    )
    if diff > 1e-2:
        raise RuntimeError(
            f"Hook module output does not match hidden_states[{SAE_LAYER}]. "
            f"Wrong layer index — check SAE_LAYER / hook_module_idx alignment."
        )

    # --- intervention loop ----------------------------------------------
    results = {
        "unclamped_pcorrect": [],
        "unclamped_pwrong": [],
        "subtractive_pcorrect": [],
        "subtractive_pwrong": [],
        "additive_pcorrect": [],
        "additive_pwrong": [],
        "question_indices": subset_idx.tolist(),
        "feature_indices": rising_indices,
        "feature_clean_values": rising_clean_values.tolist(),
        "feature_pressured_values": rising_pressured_values.tolist(),
        "k": TOP_K_FEATURES,
        "top_k_source": "c4a top-rising by delta (pressured - clean)",
    }

    def run_forward_with_hook(prompt: str, hook_fn) -> tuple[float, float, float, float]:
        """Return (P_correct, P_wrong) twice: no hook first, hook second.

        Actually only runs once with the provided hook_fn (which may be None)
        and reads final-layer P(correct)/P(wrong).
        """
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        handle = None
        if hook_fn is not None:
            handle = layers[hook_module_idx].register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                out = model(**inputs)
            logits = out.logits
            lg = logits[0, -1, :][vocab_indices]
            probs = torch.softmax(lg, dim=-1)
            return probs, logits
        finally:
            if handle is not None:
                handle.remove()

    print("\nRunning intervention on 50 questions ...")
    from tqdm import tqdm

    for qi in tqdm(subset_idx.tolist()):
        item = questions[qi]
        correct_idx = item["answer"]
        wrong_idx = jury_strong["gemma"][qi]["wrong_idx"]
        prompt = build_prompt_user_role_jury(qi, item, wrong_idx, jury_strong, tokenizer)

        # --- Unclamped ---
        probs_u, _ = run_forward_with_hook(prompt, None)
        results["unclamped_pcorrect"].append(probs_u[correct_idx].item())
        results["unclamped_pwrong"].append(probs_u[wrong_idx].item())

        # --- Subtractive clamp (to clean means) ---
        hook_sub = make_feature_clamping_hook(
            sae,
            features_to_clamp=rising_indices,
            clamp_values=rising_clean_values,
            last_token_only=True,
        )
        probs_s, _ = run_forward_with_hook(prompt, hook_sub)
        results["subtractive_pcorrect"].append(probs_s[correct_idx].item())
        results["subtractive_pwrong"].append(probs_s[wrong_idx].item())

        # --- Additive clamp (to 2x pressured means) ---
        hook_add = make_feature_clamping_hook(
            sae,
            features_to_clamp=rising_indices,
            clamp_values=rising_pressured_values * 2.0,
            last_token_only=True,
        )
        probs_a, _ = run_forward_with_hook(prompt, hook_add)
        results["additive_pcorrect"].append(probs_a[correct_idx].item())
        results["additive_pwrong"].append(probs_a[wrong_idx].item())

    # --- aggregate ------------------------------------------------------
    def m(k: str) -> float:
        return float(np.mean(results[k]))

    summary = {
        "mean_unclamped_pcorrect": m("unclamped_pcorrect"),
        "mean_unclamped_pwrong": m("unclamped_pwrong"),
        "mean_subtractive_pcorrect": m("subtractive_pcorrect"),
        "mean_subtractive_pwrong": m("subtractive_pwrong"),
        "mean_additive_pcorrect": m("additive_pcorrect"),
        "mean_additive_pwrong": m("additive_pwrong"),
    }
    summary["subtractive_restoration_pcorrect"] = (
        summary["mean_subtractive_pcorrect"] - summary["mean_unclamped_pcorrect"]
    )
    summary["additive_suppression_pcorrect"] = (
        summary["mean_additive_pcorrect"] - summary["mean_unclamped_pcorrect"]
    )
    results["summary"] = summary

    print("\n=== Intervention summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")

    out_path = sae_dir / "intervention.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved → {out_path}")

    # --- figure ---------------------------------------------------------
    import matplotlib.pyplot as plt

    labels = ["Unclamped", "Subtractive\n(→ clean mean)", "Additive\n(→ 2× pressured)"]
    pcorrect = [
        summary["mean_unclamped_pcorrect"],
        summary["mean_subtractive_pcorrect"],
        summary["mean_additive_pcorrect"],
    ]
    pwrong = [
        summary["mean_unclamped_pwrong"],
        summary["mean_subtractive_pwrong"],
        summary["mean_additive_pwrong"],
    ]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=140)
    b1 = ax.bar(x - w / 2, pcorrect, w, label="P(correct)", color="#2ca02c")
    b2 = ax.bar(x + w / 2, pwrong, w, label="P(wrong_target)", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Probability (final-layer, last token)", fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Causal feature clamping on C4a (top-{TOP_K_FEATURES} rising features, "
        f"N={N_QUESTIONS} questions)\nSAE layer 20 (Goodfire l19)",
        fontsize=12,
    )
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig_path = FIGURES_DIR / "sae_intervention_restoration.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {fig_path}")


if __name__ == "__main__":
    main()
