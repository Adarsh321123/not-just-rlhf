"""Sparse Autoencoder (SAE) loader, encoder/decoder, feature-delta analysis,
and activation-level feature clamping hook for causal interventions.

Target SAE: ``Goodfire/Llama-3.1-8B-Instruct-SAE-l19``

Architecture (verified empirically against the released ``.pth``):
    * Linear encoder: (d_sae, d_in) + bias — ``encoder_linear.{weight,bias}``
    * Linear decoder: (d_in, d_sae) + bias — ``decoder_linear.{weight,bias}``
    * Nonlinearity: **Top-K with k = 91**. Raw ReLU over the pre-activations
      yields ~22k active latents per token against the stated L0 of 91, and
      only top-91-selection reproduces the advertised sparsity exactly.
    * d_in = 4096, d_sae = 65,536 (16× expansion)
    * Training data: LMSYS-Chat-1M activations from Llama-3.1-8B-Instruct.
      Goodfire report the hook point as "layer 19"; our mid-layer suppression
      onset is L17, so we are sampling the SAE two layers post-onset — the
      closest release available (no L16/L17/L18 SAE exists for this model).

This file is owned by the SAE track (second agent). It must not be edited by
the first agent's refactor work.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from .config import HF_TOKEN

# --- Release metadata ------------------------------------------------------

GOODFIRE_SAE_REPO = "Goodfire/Llama-3.1-8B-Instruct-SAE-l19"
GOODFIRE_SAE_WEIGHTS = "Llama-3.1-8B-Instruct-SAE-l19.pth"

# Goodfire's release name "l19" means the output of transformer block 19
# (0-indexed) = ``model.model.layers[19]``'s output. In HuggingFace's
# ``output_hidden_states`` tuple convention (length 33 for Llama-3.1-8B,
# with index 0 = embedding output and index i>=1 = output of block i-1),
# block 19's output lives at tuple index **20**. Our per-experiment
# ``activations`` pickles are saved with the same hidden_states indexing,
# so ``activations[:, 20, :]`` is the correct slice to feed to the SAE.
#
# Empirically confirmed by scanning hidden_states layers 15..25 through the
# SAE encoder/decoder and picking the layer with the highest reconstruction
# cosine against known clean MMLU activations (peak 0.87 at index 20; index
# 19 gives 0.80, index 21 gives 0.81). See SAE_REPORT.md §2 for the full
# scan table.
#
# Distance from suppression onset (L17 in the same indexing): the SAE
# samples three layers post-onset, which is the closest Goodfire release
# available for this model (no L16/L17/L18/L19 SAE exists on HF as of
# April 2026). The interpretation caveat is noted in SAE_REPORT.md.
SAE_LAYER = 20
SAE_GOODFIRE_LAYER_NAME = 19  # Goodfire's own "l19" label; here for documentation
SAE_D_IN = 4096          # Llama-3.1-8B residual stream dim
SAE_D_SAE = 65_536       # 16x expansion
SAE_TOPK = 91            # k in the Top-K SAE, matches Goodfire's advertised L0

# --- Core module ----------------------------------------------------------


@dataclass
class SAEConfig:
    d_in: int
    d_sae: int
    k: int
    layer: int
    release: str


class TopKSAE(nn.Module):
    """Top-K Sparse Autoencoder matching Goodfire's Llama-3.1-8B layer 19 release.

    forward(x) -> (features, recon):
        pre      = x @ W_enc.T + b_enc
        features = top_k_values on the top-k indices (ReLU applied), zeros else
        recon    = features @ W_dec.T + b_dec
    """

    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder_linear = nn.Linear(cfg.d_in, cfg.d_sae, bias=True)
        self.decoder_linear = nn.Linear(cfg.d_sae, cfg.d_in, bias=True)

    # --- encode / decode -------------------------------------------------

    def pre_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Return the raw pre-activation vector (no sparsification)."""
        return self.encoder_linear(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Top-K sparse feature activations. Preserves leading batch dims.

        Input:  (..., d_in)
        Output: (..., d_sae), sparse — exactly k entries non-zero per row.
        """
        pre = self.pre_activations(x)
        vals, idx = torch.topk(pre, self.cfg.k, dim=-1)
        vals = F.relu(vals)
        features = torch.zeros_like(pre)
        features.scatter_(-1, idx, vals)
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder_linear(features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.encode(x)
        recon = self.decode(feats)
        return feats, recon


# --- Loader ---------------------------------------------------------------


def load_sae(
    release: str = GOODFIRE_SAE_REPO,
    weights_filename: str = GOODFIRE_SAE_WEIGHTS,
    d_in: int = SAE_D_IN,
    d_sae: int = SAE_D_SAE,
    k: int = SAE_TOPK,
    layer: int = SAE_LAYER,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> TopKSAE:
    """Download and instantiate the Goodfire SAE, mapping its .pth state dict
    onto our ``TopKSAE`` module.

    Parameters default to the Goodfire Llama-3.1-8B-Instruct-SAE-l19 release.
    """
    path = hf_hub_download(release, weights_filename, token=HF_TOKEN)
    state = torch.load(path, map_location="cpu", weights_only=False)

    expected = {
        "encoder_linear.weight",
        "encoder_linear.bias",
        "decoder_linear.weight",
        "decoder_linear.bias",
    }
    missing = expected - set(state.keys())
    if missing:
        raise RuntimeError(
            f"SAE state dict missing keys {missing}; found {list(state.keys())}"
        )

    enc_w = state["encoder_linear.weight"]
    if enc_w.shape != (d_sae, d_in):
        raise RuntimeError(
            f"encoder_linear.weight shape {tuple(enc_w.shape)} does not match "
            f"expected ({d_sae}, {d_in}); check the release metadata."
        )

    cfg = SAEConfig(d_in=d_in, d_sae=d_sae, k=k, layer=layer, release=release)
    sae = TopKSAE(cfg)
    sae.load_state_dict(state, strict=True)
    sae = sae.to(device=device, dtype=dtype)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    return sae


# --- Encoding ------------------------------------------------------------


@torch.no_grad()
def encode_activations_to_features(
    hidden_states: torch.Tensor,
    sae: TopKSAE,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Encode per-question last-token hidden states to sparse features.

    Input: ``(N, d_in)``  (typically ``(400, 4096)``)
    Output: ``(N, d_sae)`` sparse feature activations in the SAE's dtype.

    Runs in batches for memory safety. The SAE is assumed to live on the same
    device (CPU or GPU) as the caller wants; inputs are moved to match.
    """
    if hidden_states.ndim != 2:
        raise ValueError(
            f"expected (N, d_in) hidden_states, got shape {tuple(hidden_states.shape)}"
        )
    dev = next(sae.parameters()).device
    dt = next(sae.parameters()).dtype
    n = hidden_states.shape[0]
    out = []
    for start in range(0, n, batch_size):
        chunk = hidden_states[start : start + batch_size].to(device=dev, dtype=dt)
        out.append(sae.encode(chunk).cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def decode_features(features: torch.Tensor, sae: TopKSAE) -> torch.Tensor:
    dev = next(sae.parameters()).device
    dt = next(sae.parameters()).dtype
    return sae.decode(features.to(device=dev, dtype=dt)).cpu()


# --- Feature deltas ------------------------------------------------------


def compute_feature_deltas(
    clean_features: torch.Tensor,
    pressured_features: torch.Tensor,
    top_k: int = 30,
) -> dict:
    """Per-feature mean activation delta + top-``top_k`` ranking.

    Inputs: ``(N, d_sae)`` each. N can differ across conditions in principle,
    but the Llama pipeline always yields 400 questions per condition so they
    should match for our use case.

    Returns a dict:
        mean_clean      : (d_sae,)  float tensor, mean clean activation per feature
        mean_pressured  : (d_sae,)  float tensor, mean pressured activation per feature
        deltas          : (d_sae,)  pressured - clean
        abs_deltas      : (d_sae,)  |deltas|
        top_indices     : (top_k,)  int, sorted descending by |delta|
        top_values      : (top_k,)  float, the |delta| values at those indices
        top_signed      : (top_k,)  float, the signed delta values
        top_clean       : (top_k,)  float, clean means at those indices
        top_pressured   : (top_k,)  float, pressured means at those indices
    """
    if clean_features.ndim != 2 or pressured_features.ndim != 2:
        raise ValueError("features must be 2D (N, d_sae)")
    if clean_features.shape[-1] != pressured_features.shape[-1]:
        raise ValueError(
            f"feature dims differ: clean {clean_features.shape[-1]} vs "
            f"pressured {pressured_features.shape[-1]}"
        )

    clean_features = clean_features.float()
    pressured_features = pressured_features.float()

    mean_clean = clean_features.mean(dim=0)
    mean_pressured = pressured_features.mean(dim=0)
    deltas = mean_pressured - mean_clean
    abs_deltas = deltas.abs()

    k = min(top_k, abs_deltas.numel())
    top_values, top_indices = torch.topk(abs_deltas, k)

    return {
        "mean_clean": mean_clean,
        "mean_pressured": mean_pressured,
        "deltas": deltas,
        "abs_deltas": abs_deltas,
        "top_indices": top_indices,
        "top_values": top_values,
        "top_signed": deltas[top_indices],
        "top_clean": mean_clean[top_indices],
        "top_pressured": mean_pressured[top_indices],
    }


def jaccard_overlap(indices_a: Iterable[int], indices_b: Iterable[int]) -> float:
    """Jaccard overlap between two sets of feature indices."""
    a = set(int(i) for i in indices_a)
    b = set(int(i) for i in indices_b)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# --- Feature clamping hook (for causal intervention) --------------------


def make_feature_clamping_hook(
    sae: TopKSAE,
    features_to_clamp: list[int],
    clamp_values: torch.Tensor,
    last_token_only: bool = True,
    verbose: bool = False,
) -> Callable:
    """Build a forward hook that substitutes the SAE-decoded activations at
    the specified features with fixed ``clamp_values``, for use on a Llama
    decoder layer output.

    How it works
    ------------
    Llama's ``model.model.layers[i]`` outputs a tuple whose first element is the
    residual-stream hidden state ``(batch, seq, d_in)``. We intercept that
    output, slice the last-token position (by default), encode it through the
    SAE, overwrite the selected features with ``clamp_values`` (while preserving
    the other features' sparse activations), decode back to hidden space, and
    write it into the tuple in-place.

    This is a **substitution hook**, not a residual add — it replaces the
    last-token hidden state with the decoded-from-clamped-features version.
    Everything downstream of the hooked layer then sees the clamped state.

    Parameters
    ----------
    sae:
        The loaded SAE. Must be on the same device as the model.
    features_to_clamp:
        List of integer feature indices to overwrite.
    clamp_values:
        1-D tensor of the same length as ``features_to_clamp``, containing the
        value each feature should be clamped to.
    last_token_only:
        If True, only the final token's hidden state is modified (standard for
        our next-token-prediction setup). If False, every token is modified —
        more invasive, not used in the paper's intervention.
    verbose:
        If True, print a debug line on the first call. Useful for sanity
        checks during development.

    Returns
    -------
    A PyTorch forward hook callable suitable for
    ``model.model.layers[SAE_LAYER - 1].register_forward_hook(hook)``.

    Note on layer indexing: Goodfire's "layer 19" corresponds to the hidden
    state *after* the 19th transformer block, i.e. ``hidden_states[19]`` in
    the ``output_hidden_states`` tuple (where index 0 is the embedding). To
    hook that, register on ``model.model.layers[18]`` (zero-indexed) so the
    hook fires when the 19th block's output is produced. Callers are
    responsible for getting this right; this function does not do any module
    lookup on its own.
    """
    if len(features_to_clamp) != len(clamp_values):
        raise ValueError(
            f"features_to_clamp ({len(features_to_clamp)}) vs clamp_values "
            f"({len(clamp_values)}) length mismatch"
        )
    idx_tensor = torch.tensor(features_to_clamp, dtype=torch.long)
    val_tensor = clamp_values.detach().clone().float()
    state = {"called": 0}

    def hook(module, inputs, output):
        # Llama decoder block output is either a Tensor or a tuple whose first
        # element is the hidden state. Handle both.
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
            is_tuple = True
        else:
            hidden = output
            rest = None
            is_tuple = False

        # hidden shape: (batch, seq, d_in)
        sae_device = next(sae.parameters()).device
        sae_dtype = next(sae.parameters()).dtype
        if last_token_only:
            target = hidden[:, -1:, :].to(device=sae_device, dtype=sae_dtype)  # (b, 1, d)
        else:
            target = hidden.to(device=sae_device, dtype=sae_dtype)

        b, s, d = target.shape
        flat = target.reshape(b * s, d)

        feats = sae.encode(flat)  # (bs, d_sae)

        idx = idx_tensor.to(feats.device)
        vals = val_tensor.to(device=feats.device, dtype=feats.dtype)
        feats[:, idx] = vals  # overwrite the clamped features

        new_flat = sae.decode(feats)  # (bs, d)
        new_target = new_flat.reshape(b, s, d).to(dtype=hidden.dtype, device=hidden.device)

        if last_token_only:
            hidden = hidden.clone()
            hidden[:, -1:, :] = new_target
        else:
            hidden = new_target

        if verbose and state["called"] == 0:
            print(
                f"[feature-clamp hook] first fire: "
                f"batch={b} seq={s} d={d} feats_clamped={len(features_to_clamp)}"
            )
        state["called"] += 1

        if is_tuple:
            return (hidden,) + tuple(rest)
        return hidden

    hook.state = state  # expose call count for tests / sanity
    return hook


# --- Convenience: paths for cached feature tensors ----------------------

SAE_RESULTS_SUBDIR = "sae"


def sae_results_dir(results_dir: Path | str) -> Path:
    p = Path(results_dir) / SAE_RESULTS_SUBDIR
    p.mkdir(parents=True, exist_ok=True)
    return p
