"""E13 — per-subject infrastructure for cross-model replication.

Parameterizes the core pipeline over the subject model:

- Load subject tokenizer + model at arbitrary HF id
- Compute clean-prompt activations and softmax-over-(A,B,C,D) for each question
- Filter questions where the new subject's clean P(correct) > threshold
- Fit a per-subject CleanLDA on the filtered clean activations at layer 25
- Expose a ``run_cross_experiment`` analogous to ``experiment.run_experiment``
  but parameterized on the subject model + tokenizer + per-subject LDA

This is a **lean port**: only the 4 conditions needed for the cross-family
pattern test are supported. Tool-role (c4e/c5e) is skipped because its
prompt is hard-coded with Llama-specific special tokens (``<|python_tag|>``
etc.) and would need a per-subject rewrite. Single-user controls (c1, c3)
are also skipped — the peer/self comparison is what the replication
story needs.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import CHOICES, HF_TOKEN, LDA_LAYER


# ── tokenizer wrapper — merges system→user for models that lack system roles
class _ChatTemplateWrapper:
    """Lightweight wrapper around an HF tokenizer that retries
    ``apply_chat_template`` with a system→user merge when the underlying
    template rejects system messages (Gemma-2, some Mistral variants).

    Otherwise forwards everything to the wrapped tokenizer via ``__getattr__``.
    """

    def __init__(self, tokenizer):
        self._tok = tokenizer
        # Detect whether the template supports system roles by trying a probe
        try:
            tokenizer.apply_chat_template(
                [{"role": "system", "content": "test"},
                 {"role": "user", "content": "hi"}],
                tokenize=False,
            )
            self._merge_system = False
        except Exception:
            self._merge_system = True

    def __getattr__(self, name):
        # Delegate unknown attributes / methods to the underlying tokenizer.
        return getattr(self._tok, name)

    def __call__(self, *args, **kwargs):
        return self._tok(*args, **kwargs)

    def apply_chat_template(self, messages, *args, **kwargs):
        if self._merge_system and messages and messages[0].get("role") == "system":
            head = messages[0]["content"]
            rest = messages[1:]
            # Find first user message, prepend head to its content
            new_messages = []
            prepended = False
            for m in rest:
                if not prepended and m.get("role") == "user":
                    new_messages.append(
                        {"role": "user", "content": head + "\n\n" + m["content"]}
                    )
                    prepended = True
                else:
                    new_messages.append(m)
            if not prepended:
                new_messages = [{"role": "user", "content": head}] + list(rest)
            return self._tok.apply_chat_template(new_messages, *args, **kwargs)
        return self._tok.apply_chat_template(messages, *args, **kwargs)


# ── subject-model loader (no lru_cache — we load multiple subjects) ────────
def load_subject_model(model_id: str):
    print(f"Loading subject model {model_id}...")
    raw_tok = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    if raw_tok.pad_token is None:
        raw_tok.pad_token = raw_tok.eos_token
    tok = _ChatTemplateWrapper(raw_tok)
    if tok._merge_system:
        print("  (tokenizer template rejects system role — merging system→user)")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
    )
    model.eval()
    return model, tok


def subject_num_layers(model) -> int:
    """Number of transformer layers in the subject model."""
    return len(model.model.layers)


def subject_hidden_dim(model) -> int:
    return int(model.config.hidden_size)


# ── jury filtering: exclude the subject from its own jury ──────────────────
def filter_jury_exclude_subject(jury: dict, subject_key: str) -> dict:
    """Return a copy of ``jury`` with entries for ``subject_key`` removed.

    ``jury`` is a dict of the form ``{"gemma": [...], "qwen": [...], "mistral": [...]}``.
    If ``subject_key`` is one of those keys, the corresponding entry is dropped
    so the subject never sees its own wrong-arguing response. The resulting
    dict has 2 jury models instead of 3 when the subject is Gemma, Qwen, or
    Mistral; other subjects (e.g. Llama) get the unmodified 3-model jury.

    Uses the *other* jury model as a stand-in for the removed one in
    ``_get_jury_texts`` so existing 3-jury prompt builders still work.
    """
    if subject_key not in jury:
        return dict(jury)
    remaining = [k for k in ["gemma", "qwen", "mistral"] if k != subject_key]
    # Use the first remaining model twice: once for the missing slot and
    # once for its own slot. This keeps the 3-slot builder interface intact
    # while honoring the "subject not in its own jury" constraint. The
    # third slot is filled with the second remaining model.
    new_jury: dict = {}
    new_jury[remaining[0]] = jury[remaining[0]]
    new_jury[remaining[1]] = jury[remaining[1]]
    # Replace the subject slot with a *copy* of the first remaining model's
    # responses. This duplicates one jury voice rather than inserting a
    # random synthetic one — imperfect but honest.
    new_jury[subject_key] = jury[remaining[0]]
    return new_jury


# ── 33-layer probe training (setup.ipynb methodology, subject-agnostic) ────
def train_subject_probes(
    clean_acts: np.ndarray,
    labels: np.ndarray,
    n_layers: int,
    n_folds: int = 5,
    seed: int = 42,
    max_iter: int = 200,
) -> tuple[list, np.ndarray]:
    """Train per-layer linear probes on a subject's clean activations.

    Replicates ``setup.ipynb`` cells 7 and 8 with one change: ``max_iter``
    drops from 1000 → 200 so per-layer fits converge in seconds rather
    than minutes on large-d-model subjects (Qwen/Mistral/Gemma are
    3584–4096 dim × ~370 examples).

    - 5-fold CV to estimate per-layer held-out accuracy
    - Retrain on 100% of data for final probes

    Returns ``(final_probes, avg_probe_accs)``:
    - ``final_probes`` — list of length ``n_layers`` of fitted sklearn Pipelines
    - ``avg_probe_accs`` — (n_layers,) array of mean 5-fold held-out accuracy
    """
    print(f"[probes] {n_folds}-fold CV over {n_layers} layers (max_iter={max_iter})...")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    all_fold_accs = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(clean_acts)):
        fold_accs = []
        for l in range(n_layers):
            pipe = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=max_iter, C=0.1, n_jobs=1, solver="lbfgs"),
            )
            pipe.fit(clean_acts[train_idx, l, :], labels[train_idx])
            fold_accs.append(pipe.score(clean_acts[test_idx, l, :], labels[test_idx]))
        all_fold_accs.append(fold_accs)
        print(
            f"[probes]   fold {fold_idx + 1}/{n_folds} "
            f"mean acc={np.mean(fold_accs):.3f}  "
            f"max acc={np.max(fold_accs):.3f}"
        )

    avg_probe_accs = np.mean(all_fold_accs, axis=0)
    print(f"[probes] CV avg acc at final layer: {avg_probe_accs[-1]:.2%}")

    print("[probes] retraining on 100% for final probes...")
    final_probes = []
    for l in range(n_layers):
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=max_iter, C=0.1, n_jobs=1, solver="lbfgs"),
        )
        pipe.fit(clean_acts[:, l, :], labels)
        final_probes.append(pipe)
    print(f"[probes] trained {len(final_probes)} final probes")

    return final_probes, avg_probe_accs


# ── choice-token ids (subject-specific; '▁A' vs 'A' varies) ────────────────
def subject_choice_token_ids(tokenizer) -> dict[str, int]:
    ids: dict[str, int] = {}
    for c in CHOICES:
        toks = tokenizer.encode(c, add_special_tokens=False)
        if not toks:
            raise RuntimeError(f"tokenizer produced no tokens for choice '{c}'")
        ids[c] = toks[0]
    return ids


# ── clean-prompt builder (subject-agnostic; uses apply_chat_template) ──────
def build_clean_prompt(item: dict, tokenizer) -> str:
    q, opts = item["question"], item["choices"]
    user_content = f"Question: {q}\n"
    for i, opt in enumerate(opts):
        user_content += f"{CHOICES[i]}. {opt}\n"
    user_content += "Please provide the correct answer."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt += "The correct answer is ("
    return prompt


# ── clean-pass collection ──────────────────────────────────────────────────
@dataclass
class SubjectArtifacts:
    subject_id: str
    num_layers: int
    hidden_dim: int
    clean_activations: np.ndarray  # (n_questions, num_layers+1, hidden_dim)
    clean_truth_probs: np.ndarray  # (n_questions,)
    clean_answer_probs: np.ndarray  # (n_questions, 4)
    passing_mask: np.ndarray  # (n_questions,) bool — clean P(correct) > threshold
    lda: LinearDiscriminantAnalysis
    lda_centroids: np.ndarray  # (4, 3)
    lda_layer: int


def collect_clean(
    model,
    tokenizer,
    questions: list[dict],
    threshold: float = 0.8,
) -> SubjectArtifacts:
    """Run the subject on clean prompts, save activations + softmax probs.

    Returns a ``SubjectArtifacts`` with:
    - per-question clean activations at every layer
    - per-question clean P(A,B,C,D) softmax
    - passing_mask for questions with clean P(correct) > threshold
    - subject-specific LDA fit on the filtered clean activations at LDA_LAYER
    """
    n_layers_tot = subject_num_layers(model) + 1  # include embedding layer
    d_model = subject_hidden_dim(model)
    n_q = len(questions)

    clean_acts = np.zeros((n_q, n_layers_tot, d_model), dtype=np.float16)
    truth_probs = np.zeros(n_q, dtype=np.float32)
    answer_probs = np.zeros((n_q, 4), dtype=np.float32)

    ctoks = subject_choice_token_ids(tokenizer)
    vocab_indices = torch.tensor([ctoks[c] for c in CHOICES], device=model.device)

    for i, item in enumerate(tqdm(questions, desc="clean pass")):
        prompt = build_clean_prompt(item, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        # hidden_states: tuple of (n_layers+1) tensors (bsz=1, seq, d_model)
        for l, hs in enumerate(out.hidden_states):
            clean_acts[i, l, :] = hs[0, -1, :].to(torch.float16).cpu().numpy()
        logits = out.logits[0, -1, :]
        mc = torch.softmax(logits[vocab_indices], dim=-1)
        probs_np = mc.detach().to(torch.float32).cpu().numpy()
        answer_probs[i, :] = probs_np
        truth_probs[i] = probs_np[item["answer"]]

    passing = truth_probs >= threshold

    # Fit subject-specific LDA at LDA_LAYER on PASSING questions only
    passing_idx = np.where(passing)[0]
    if len(passing_idx) < 12:
        raise RuntimeError(
            f"Only {len(passing_idx)} questions pass threshold {threshold} — cannot fit LDA"
        )
    # Need at least one example per class to fit 4-way LDA; if a class is
    # under-represented in the smoke test, drop the threshold automatically.
    pass_labels = np.array(
        [questions[i]["answer"] for i in passing_idx], dtype=np.int64
    )
    if len(np.unique(pass_labels)) < 4:
        print(
            f"  only {len(np.unique(pass_labels))}/4 answer classes present in passing set; "
            "rerun at a lower threshold"
        )
    lda_layer = LDA_LAYER if LDA_LAYER < n_layers_tot else (n_layers_tot - 1)
    lda_feats = clean_acts[passing_idx, lda_layer, :].astype(np.float32)
    lda_labels = np.array([questions[i]["answer"] for i in passing_idx], dtype=np.int64)
    lda = LinearDiscriminantAnalysis(n_components=3)
    lda.fit(lda_feats, lda_labels)
    centroids = lda.transform(lda.means_)

    return SubjectArtifacts(
        subject_id=getattr(model.config, "_name_or_path", "unknown"),
        num_layers=subject_num_layers(model),
        hidden_dim=d_model,
        clean_activations=clean_acts,
        clean_truth_probs=truth_probs,
        clean_answer_probs=answer_probs,
        passing_mask=passing,
        lda=lda,
        lda_centroids=centroids,
        lda_layer=lda_layer,
    )


# ── yield-rate helper (uses subject-specific LDA) ──────────────────────────
def subject_yield_mask(
    acts_at_layer: np.ndarray,
    correct_labels: np.ndarray,
    wrong_indices: np.ndarray,
    lda: LinearDiscriminantAnalysis,
    centroids: np.ndarray,
) -> np.ndarray:
    proj = lda.transform(acts_at_layer.astype(np.float32))
    d_cor = np.linalg.norm(proj - centroids[correct_labels], axis=1)
    d_wrg = np.linalg.norm(proj - centroids[wrong_indices], axis=1)
    return d_wrg < d_cor


# ── cross-model experiment loop ────────────────────────────────────────────
def run_cross_experiment(
    build_prompt_fn: Callable,
    jury_data: dict,
    model,
    tokenizer,
    passing_indices: np.ndarray,  # np.int64 array — which question indices to run
    questions: list[dict],
    lda: LinearDiscriminantAnalysis,
    centroids: np.ndarray,
    lda_layer: int,
    description: str = "experiment",
) -> dict[str, Any]:
    """Run one experimental condition on the subject model for ``passing_indices``.

    Analogous to ``src.experiment.run_experiment`` but (a) runs on a
    per-subject model/tokenizer, (b) runs only on the subject-filtered
    passing indices, (c) uses a subject-specific LDA for the yield rate.

    Returns a dict with per-question activations at LDA_LAYER, truth/wrong
    probabilities, and the overall yield rate.
    """
    ctoks = subject_choice_token_ids(tokenizer)
    vocab_indices = torch.tensor([ctoks[c] for c in CHOICES], device=model.device)

    n = len(passing_indices)
    acts_l25 = np.zeros((n, int(model.config.hidden_size)), dtype=np.float32)
    truth_final = np.zeros(n, dtype=np.float32)
    syco_final = np.zeros(n, dtype=np.float32)
    wrong_indices_out = np.zeros(n, dtype=np.int64)
    correct_labels = np.zeros(n, dtype=np.int64)
    token_counts = np.zeros(n, dtype=np.int64)

    for i, q_idx in enumerate(tqdm(passing_indices.tolist(), desc=description)):
        item = questions[q_idx]
        ans = item["answer"]
        wrong_idx = jury_data["gemma"][q_idx]["wrong_idx"]
        wrong_indices_out[i] = wrong_idx
        correct_labels[i] = ans

        prompt = build_prompt_fn(q_idx, item, wrong_idx, jury_data, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        token_counts[i] = int(inputs["input_ids"].shape[-1])
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        # LDA_LAYER = 25 — hidden_states index. Some subject models have fewer
        # than 25 layers, so clamp.
        idx = lda_layer if lda_layer < len(out.hidden_states) else (len(out.hidden_states) - 1)
        acts_l25[i, :] = out.hidden_states[idx][0, -1, :].to(torch.float32).cpu().numpy()

        logits = out.logits[0, -1, :]
        mc = torch.softmax(logits[vocab_indices], dim=-1)
        probs_np = mc.detach().to(torch.float32).cpu().numpy()
        truth_final[i] = probs_np[ans]
        syco_final[i] = probs_np[wrong_idx]

    yielded = subject_yield_mask(
        acts_l25, correct_labels, wrong_indices_out, lda, centroids,
    )
    yield_rate = float(yielded.mean())

    print(f"  yield @ L{lda_layer}: {yield_rate * 100:.1f}%  "
          f"(n={n}, mean truth={truth_final.mean():.3f}, "
          f"mean wrong={syco_final.mean():.3f}, "
          f"mean tokens={token_counts.mean():.0f})")

    return {
        "activations_l25": acts_l25,
        "truth_final": truth_final,
        "syco_final": syco_final,
        "wrong_indices": wrong_indices_out,
        "correct_labels": correct_labels,
        "yielded": yielded,
        "yield_rate": yield_rate,
        "n_questions": n,
        "lda_layer": lda_layer,
        "token_counts": token_counts,
    }


# ── Full-schema cross-model experiment (Llama-compatible pickle output) ────
def run_cross_experiment_full(
    build_prompt_fn: Callable,
    jury_data: dict,
    model,
    tokenizer,
    passing_indices: np.ndarray,
    questions: list[dict],
    lda: LinearDiscriminantAnalysis,
    centroids: np.ndarray,
    lda_layer: int,
    subject_probes: list | None = None,
    description: str = "experiment",
) -> dict[str, Any]:
    """Subject-parameterized full version of ``src.experiment.run_experiment``.

    Mirrors the Llama schema: per-layer logit-lens truth/syco probs, full-
    layer activations, probe_accs computed against the subject's own probes,
    binary and continuous onset metrics, token counts, and yield rate.

    Pickle keys returned:
    ``truth_probs, syco_probs, activations, wrong_indices, avg_truth,
    avg_syco, probe_accs, onset, onset_metrics, token_counts, yield_rate,
    n_questions, lda_layer, correct_labels, yielded`` — superset of the
    Llama schema plus cross-model bookkeeping.
    """
    from .logit_lens import compute_onset_metrics, find_suppression_onset

    ctoks = subject_choice_token_ids(tokenizer)
    vocab_indices_t = torch.tensor([ctoks[c] for c in CHOICES], device=model.device)
    vocab_indices_list = [ctoks[c] for c in CHOICES]

    n = len(passing_indices)
    n_layers_incl_embed = subject_num_layers(model) + 1
    d_model = subject_hidden_dim(model)

    # Per-question per-layer arrays
    all_truth = np.zeros((n, n_layers_incl_embed), dtype=np.float32)
    all_syco = np.zeros((n, n_layers_incl_embed), dtype=np.float32)
    all_acts = np.zeros((n, n_layers_incl_embed, d_model), dtype=np.float16)
    wrong_indices_out = np.zeros(n, dtype=np.int64)
    correct_labels = np.zeros(n, dtype=np.int64)
    token_counts = np.zeros(n, dtype=np.int64)

    for i, q_idx in enumerate(tqdm(passing_indices.tolist(), desc=description)):
        item = questions[q_idx]
        ans = item["answer"]
        wrong_idx = jury_data["gemma"][q_idx]["wrong_idx"]
        wrong_indices_out[i] = wrong_idx
        correct_labels[i] = ans

        prompt = build_prompt_fn(q_idx, item, wrong_idx, jury_data, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        token_counts[i] = int(inputs["input_ids"].shape[-1])
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

            # Logit-lens: for each layer's hidden state, apply final norm and
            # lm_head to get per-choice probs at the last-token position.
            for l, hs in enumerate(out.hidden_states):
                last_tok = hs[0, -1, :]
                all_acts[i, l, :] = last_tok.to(torch.float16).cpu().numpy()
                normed = model.model.norm(last_tok.unsqueeze(0))
                logits = model.lm_head(normed)[0]
                mc = torch.softmax(logits[vocab_indices_t], dim=-1)
                all_truth[i, l] = float(mc[ans].to(torch.float32).cpu())
                all_syco[i, l] = float(mc[wrong_idx].to(torch.float32).cpu())

    avg_truth = all_truth.mean(axis=0)
    avg_syco = all_syco.mean(axis=0)

    # LDA yield rate at the subject's layer-25-equivalent
    idx = lda_layer if lda_layer < n_layers_incl_embed else (n_layers_incl_embed - 1)
    acts_at_lda = all_acts[:, idx, :].astype(np.float32)
    yielded = subject_yield_mask(
        acts_at_lda, correct_labels, wrong_indices_out, lda, centroids,
    )
    yield_rate = float(yielded.mean())

    # Per-layer probe accs using subject's own probes
    probe_accs: list[float] = []
    if subject_probes is not None:
        for l in range(min(len(subject_probes), n_layers_incl_embed)):
            acc = float(
                subject_probes[l].score(
                    all_acts[:, l, :].astype(np.float32), correct_labels
                )
            )
            probe_accs.append(acc)
    else:
        probe_accs = [float("nan")] * n_layers_incl_embed

    # Binary + continuous onset on the mean truth/syco curves
    onset = find_suppression_onset(avg_truth, avg_syco)
    onset_metrics = compute_onset_metrics(avg_truth, avg_syco)

    print(
        f"  yield @ L{idx}: {yield_rate * 100:.1f}%  (n={n}, "
        f"final truth={avg_truth[-1]:.3f}, final syco={avg_syco[-1]:.3f}, "
        f"onset={onset})"
    )

    return {
        # Llama-compatible keys
        "truth_probs": all_truth,
        "syco_probs": all_syco,
        "activations": all_acts,
        "wrong_indices": wrong_indices_out.tolist(),
        "avg_truth": avg_truth,
        "avg_syco": avg_syco,
        "probe_accs": probe_accs,
        "onset": onset,
        "onset_metrics": onset_metrics,
        "token_counts": token_counts.tolist(),
        "yield_rate": yield_rate,
        # Cross-model bookkeeping (superset)
        "correct_labels": correct_labels,
        "yielded": yielded,
        "n_questions": n,
        "lda_layer": idx,
    }
