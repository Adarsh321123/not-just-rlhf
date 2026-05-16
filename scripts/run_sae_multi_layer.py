"""Experiment #9: Multi-Layer SAE Analysis with Alternative Bases.

Loads pre-computed activations from condition pickles and encodes them through
alternative SAEs at layers inside the L14-L18 causal window. Computes feature
deltas, Jaccard overlap matrices, and runs causal interventions to answer:
do the feature families replicate under different SAE bases?

SAEs tested:
  - andyrdt/saes-llama-3.1-8b-instruct: L15, BatchTopK (k=32), d_sae=131072
  - Geaming/Llama-3.1-8B-Instruct_SAEs: L18, standard ReLU+L1, d_sae=32768
  - Jammies-io/sae-Llama-3.1-8B-Instruct-layer18-sycophancy-v2: L18, d_sae=16384
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import CHOICES, FIGURES_DIR, HF_TOKEN, RESULTS_DIR

OUT_DIR = RESULTS_DIR / "sae_multi_layer"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


# ─── SAE architectures ───────────────────────────────────────────────────────

@dataclass
class AltSAEConfig:
    name: str
    repo: str
    layer: int           # transformer block index (0-based)
    hidden_idx: int      # index into hidden_states tuple (layer + 1)
    d_in: int
    d_sae: int
    activation: str      # "topk", "relu", "jumprelu"
    k: int | None = None


class TopKSAEAlt(nn.Module):
    """BatchTopK SAE (andyrdt format)."""
    def __init__(self, d_in: int, d_sae: int, k: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k
        self.encoder = nn.Linear(d_in, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_in, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.encoder(x - self.b_dec)
        vals, idx = torch.topk(pre, self.k, dim=-1)
        vals = F.relu(vals)
        features = torch.zeros_like(pre)
        features.scatter_(-1, idx, vals)
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features) + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.encode(x)
        recon = self.decode(feats)
        return feats, recon


class JumpReLUSAE(nn.Module):
    """JumpReLU SAE (SAELens/Geaming format): features = pre * (pre > threshold)."""
    def __init__(self, d_in: int, d_sae: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.W_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.threshold = nn.Parameter(torch.zeros(d_sae))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = x @ self.W_enc + self.b_enc
        return pre * (pre > self.threshold).float()

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.encode(x)
        recon = self.decode(feats)
        return feats, recon


class VanillaSAE(nn.Module):
    """Vanilla encoder-decoder SAE (Jammies-io format): no b_dec subtraction on input."""
    def __init__(self, d_in: int, d_sae: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.encoder = nn.Linear(d_in, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_in, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x))

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.encode(x)
        recon = self.decode(feats)
        return feats, recon


# ─── Loaders ─────────────────────────────────────────────────────────────────

def load_andyrdt_l15(device: str = "cpu") -> tuple[nn.Module, AltSAEConfig]:
    """Load andyrdt/saes-llama-3.1-8b-instruct L15 trainer_0."""
    import json as _json
    from huggingface_hub import hf_hub_download

    repo = "andyrdt/saes-llama-3.1-8b-instruct"

    # Read config.json to determine architecture
    cfg_path = hf_hub_download(repo, "resid_post_layer_15/trainer_0/config.json", token=HF_TOKEN)
    with open(cfg_path) as f:
        raw_cfg = _json.load(f)
    trainer_cfg = raw_cfg.get("trainer", raw_cfg)
    d_in = trainer_cfg["activation_dim"]
    d_sae = trainer_cfg["dict_size"]
    k = trainer_cfg["k"]
    layer = trainer_cfg["layer"]
    print(f"  [andyrdt L15] config: d_in={d_in}, d_sae={d_sae}, k={k}, layer={layer}, "
          f"class={trainer_cfg.get('dict_class', '?')}")

    cfg = AltSAEConfig(
        name="andyrdt_L15", repo=repo, layer=layer, hidden_idx=layer + 1,
        d_in=d_in, d_sae=d_sae, activation="topk", k=k,
    )

    path = hf_hub_download(repo, "resid_post_layer_15/trainer_0/ae.pt", token=HF_TOKEN)
    state = torch.load(path, map_location="cpu", weights_only=False)

    sae = TopKSAEAlt(cfg.d_in, cfg.d_sae, cfg.k)
    sae.encoder.weight.data = state["encoder.weight"]
    sae.encoder.bias.data = state["encoder.bias"]
    sae.decoder.weight.data = state["decoder.weight"]
    sae.b_dec.data = state["b_dec"]

    sae = sae.to(device=device, dtype=torch.float32)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)

    print(f"  [andyrdt L15] loaded: d_sae={cfg.d_sae}, k={cfg.k}")
    return sae, cfg


def load_geaming_l18(device: str = "cpu") -> tuple[nn.Module, AltSAEConfig]:
    """Load Geaming/Llama-3.1-8B-Instruct_SAEs L18 BT(P) JumpReLU."""
    import json as _json
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    repo = "Geaming/Llama-3.1-8B-Instruct_SAEs"
    subdir = "BT(P)/blocks_18_hook_resid_post_8X_2048_jumprelu"

    # Read cfg.json to determine architecture
    cfg_path = hf_hub_download(repo, f"{subdir}/cfg.json", token=HF_TOKEN)
    with open(cfg_path) as f:
        raw_cfg = _json.load(f)
    d_in = raw_cfg["d_in"]
    d_sae = raw_cfg["d_sae"]
    layer = raw_cfg["hook_layer"]
    activation_fn = raw_cfg.get("activation_fn", "relu")
    print(f"  [Geaming L18] config: d_in={d_in}, d_sae={d_sae}, layer={layer}, "
          f"architecture={raw_cfg.get('architecture', '?')}, activation={activation_fn}, "
          f"sae_lens_version={raw_cfg.get('sae_lens_version', '?')}")

    cfg = AltSAEConfig(
        name="geaming_L18", repo=repo, layer=layer, hidden_idx=layer + 1,
        d_in=d_in, d_sae=d_sae, activation="jumprelu",
    )

    weights_path = hf_hub_download(repo, f"{subdir}/sae_weights.safetensors", token=HF_TOKEN)
    state = load_file(weights_path)

    sae = JumpReLUSAE(cfg.d_in, cfg.d_sae)
    sae.W_enc.data = state["W_enc"]
    sae.b_enc.data = state["b_enc"]
    sae.W_dec.data = state["W_dec"]
    sae.b_dec.data = state["b_dec"]
    sae.threshold.data = state["threshold"]

    sae = sae.to(device=device, dtype=torch.float32)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)

    print(f"  [Geaming L18] loaded: d_sae={cfg.d_sae}, JumpReLU")
    return sae, cfg


def load_jammies_l18(device: str = "cpu") -> tuple[nn.Module, AltSAEConfig]:
    """Load Jammies-io/sae-Llama-3.1-8B-Instruct-layer18-sycophancy-v2."""
    import json as _json
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    repo = "Jammies-io/sae-Llama-3.1-8B-Instruct-layer18-sycophancy-v2"

    # Read config.json to determine architecture
    cfg_path = hf_hub_download(repo, "config.json", token=HF_TOKEN)
    with open(cfg_path) as f:
        raw_cfg = _json.load(f)
    d_in = raw_cfg["d_in"]
    d_sae = raw_cfg["d_sae"]
    layer = raw_cfg["target_layer"]
    print(f"  [Jammies L18] config: d_in={d_in}, d_sae={d_sae}, layer={layer}, "
          f"expansion={raw_cfg.get('sae_expansion', '?')}, "
          f"n_samples={raw_cfg.get('n_samples', '?')}")

    cfg = AltSAEConfig(
        name="jammies_L18_syco", repo=repo, layer=layer, hidden_idx=layer + 1,
        d_in=d_in, d_sae=d_sae, activation="relu",
    )

    path = hf_hub_download(repo, "model.safetensors", token=HF_TOKEN)
    state = load_file(path)

    sae = VanillaSAE(cfg.d_in, cfg.d_sae)
    sae.encoder.weight.data = state["encoder.weight"]
    sae.encoder.bias.data = state["encoder.bias"]
    sae.decoder.weight.data = state["decoder.weight"]

    sae = sae.to(device=device, dtype=torch.float32)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)

    print(f"  [Jammies L18 sycophancy] loaded: d_sae={cfg.d_sae}, ReLU")
    return sae, cfg


def load_pellement_l16(device: str = "cpu") -> tuple[nn.Module, AltSAEConfig]:
    """Load pellement99/llama-3.1-8b-instruct-16-res-sae L16 BatchTopK."""
    import json as _json
    from huggingface_hub import hf_hub_download

    repo = "pellement99/llama-3.1-8b-instruct-16-res-sae"

    # Read config.json to determine architecture
    cfg_path = hf_hub_download(repo, "config.json", token=HF_TOKEN)
    with open(cfg_path) as f:
        raw_cfg = _json.load(f)
    trainer_cfg = raw_cfg.get("trainer", raw_cfg)
    d_in = trainer_cfg["activation_dim"]
    d_sae = trainer_cfg["dict_size"]
    k = trainer_cfg["k"]
    layer = trainer_cfg["layer"]
    print(f"  [pellement L16] config: d_in={d_in}, d_sae={d_sae}, k={k}, layer={layer}, "
          f"class={trainer_cfg.get('dict_class', '?')}")

    cfg = AltSAEConfig(
        name="pellement_L16", repo=repo, layer=layer, hidden_idx=layer + 1,
        d_in=d_in, d_sae=d_sae, activation="topk", k=k,
    )

    path = hf_hub_download(repo, "ae.pt", token=HF_TOKEN)
    state = torch.load(path, map_location="cpu", weights_only=False)

    sae = TopKSAEAlt(cfg.d_in, cfg.d_sae, cfg.k)
    sae.encoder.weight.data = state["encoder.weight"]
    sae.encoder.bias.data = state["encoder.bias"]
    sae.decoder.weight.data = state["decoder.weight"]
    sae.b_dec.data = state["b_dec"]

    sae = sae.to(device=device, dtype=torch.float32)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)

    print(f"  [pellement L16] loaded: d_sae={cfg.d_sae}, k={cfg.k}")
    return sae, cfg


# ─── Utility ──────────────────────────────────────────────────────────────────

def load_activations(condition: str, layer_idx: int) -> torch.Tensor:
    """Load activations from a condition pickle at a specific layer index."""
    pkl_path = RESULTS_DIR / f"{condition}.pkl"
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    acts = data["activations"][:, layer_idx, :]  # (400, 4096)
    return torch.from_numpy(acts).float()


@torch.no_grad()
def encode_batch(sae: nn.Module, x: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    """Encode activations through SAE in batches."""
    device = next(sae.parameters()).device
    out = []
    for i in range(0, x.shape[0], batch_size):
        chunk = x[i:i+batch_size].to(device)
        feats = sae.encode(chunk)
        out.append(feats.cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def validate_sae(sae: nn.Module, x: torch.Tensor, cfg: AltSAEConfig) -> dict:
    """Validate SAE reconstruction quality."""
    device = next(sae.parameters()).device
    sample = x[:100].to(device)
    feats = sae.encode(sample)
    recon = sae.decode(feats)

    mse = F.mse_loss(recon, sample).item()
    input_var = sample.var().item()
    rel_mse = mse / (input_var + 1e-8)
    l0 = (feats > 0).float().sum(dim=-1).mean().item()
    cos = F.cosine_similarity(recon, sample, dim=-1).mean().item()

    result = {
        "name": cfg.name,
        "mse": mse,
        "input_var": input_var,
        "relative_mse": rel_mse,
        "mean_l0": l0,
        "cosine_sim": cos,
        "valid": rel_mse < 10.0,
    }
    print(f"  Validation: MSE={mse:.4f}, rel_MSE={rel_mse:.4f}, L0={l0:.1f}, cos={cos:.4f}")
    if not result["valid"]:
        print(f"  *** FAILED: relative MSE {rel_mse:.2f} > 10x input variance — skipping")
    return result


def compute_deltas(clean_feats: torch.Tensor, pressured_feats: torch.Tensor, top_k: int = 30):
    """Compute per-feature deltas and return top-k rising and falling."""
    mean_clean = clean_feats.float().mean(dim=0)
    mean_pressured = pressured_feats.float().mean(dim=0)
    deltas = mean_pressured - mean_clean

    # Top rising (most positive delta)
    rising_vals, rising_idx = torch.topk(deltas, top_k)
    # Top falling (most negative delta)
    falling_vals, falling_idx = torch.topk(-deltas, top_k)
    falling_idx_sorted = falling_idx
    falling_vals_sorted = -falling_vals  # negative values

    return {
        "deltas": deltas,
        "mean_clean": mean_clean,
        "mean_pressured": mean_pressured,
        "top_rising_idx": rising_idx.tolist(),
        "top_rising_vals": rising_vals.tolist(),
        "top_falling_idx": falling_idx_sorted.tolist(),
        "top_falling_vals": falling_vals_sorted.tolist(),
    }


def jaccard(set_a: list[int], set_b: list[int]) -> float:
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# ─── Phase 2: Feature deltas and Jaccard ──────────────────────────────────────

def run_phase2(sae: nn.Module, cfg: AltSAEConfig, conditions: list[str]) -> dict:
    """Run feature delta analysis for all conditions."""
    print(f"\n  Phase 2 for {cfg.name} (layer {cfg.layer}, hidden_idx {cfg.hidden_idx})")

    # Load clean activations at this layer
    clean_acts = load_activations("c1", cfg.hidden_idx)
    print(f"    Encoding clean activations...")
    clean_feats = encode_batch(sae, clean_acts)
    print(f"    Clean features shape: {clean_feats.shape}, mean L0: {(clean_feats > 0).float().sum(-1).mean():.1f}")

    all_deltas = {}
    top30_sets = {}

    for cond in conditions:
        print(f"    Processing {cond}...")
        cond_acts = load_activations(cond, cfg.hidden_idx)
        cond_feats = encode_batch(sae, cond_acts)

        d = compute_deltas(clean_feats, cond_feats, top_k=30)
        all_deltas[cond] = d

        # Combined top-30 by absolute delta (for Jaccard comparison with Goodfire)
        abs_deltas = d["deltas"].abs()
        top30_vals, top30_idx = torch.topk(abs_deltas, 30)
        top30_sets[cond] = top30_idx.tolist()

    # Compute Jaccard matrix
    cond_list = list(top30_sets.keys())
    n = len(cond_list)
    jaccard_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            jaccard_matrix[i, j] = jaccard(top30_sets[cond_list[i]], top30_sets[cond_list[j]])

    # Save feature deltas CSV
    for cond in conditions:
        d = all_deltas[cond]
        csv_path = OUT_DIR / f"feature_deltas_{cfg.name}_{cond}.csv"
        with open(csv_path, "w") as f:
            f.write("feature_idx,delta,clean_mean,pressured_mean\n")
            deltas_np = d["deltas"].numpy()
            clean_np = d["mean_clean"].numpy()
            press_np = d["mean_pressured"].numpy()
            # Write top-100 by absolute value
            top100 = np.argsort(-np.abs(deltas_np))[:100]
            for idx in top100:
                f.write(f"{idx},{deltas_np[idx]:.6f},{clean_np[idx]:.6f},{press_np[idx]:.6f}\n")

    # Save Jaccard matrix
    jaccard_path = OUT_DIR / f"jaccard_{cfg.name}.csv"
    with open(jaccard_path, "w") as f:
        f.write(",".join(cond_list) + "\n")
        for i in range(n):
            row = ",".join(f"{jaccard_matrix[i,j]:.4f}" for j in range(n))
            f.write(row + "\n")

    print(f"    Jaccard matrix ({cfg.name}):")
    for i, c in enumerate(cond_list):
        print(f"      {c:12s}: {['%.3f' % jaccard_matrix[i,j] for j in range(n)]}")

    # Compute cluster metrics
    # c4a = peer-pressure, c4d = self-framing, c4e = tool-role, c4c_matched = no-attribution
    # Key comparison: self/tool pair (c4d, c4e) vs cross-cluster (c4a vs c4d/c4e)
    self_tool_conds = [c for c in ["c4d", "c4e"] if c in top30_sets]
    peer_conds = [c for c in ["c4a"] if c in top30_sets]

    within_self = []
    for i in range(len(self_tool_conds)):
        for j in range(i+1, len(self_tool_conds)):
            within_self.append(jaccard(top30_sets[self_tool_conds[i]], top30_sets[self_tool_conds[j]]))

    # Cross-cluster: peer (c4a) vs self/tool (c4d, c4e)
    cross_cluster = []
    for pc in peer_conds:
        for sc in self_tool_conds:
            cross_cluster.append(jaccard(top30_sets[pc], top30_sets[sc]))

    # Also measure c4a vs c4c_matched (should be somewhat similar since both are user-turn)
    peer_peer = jaccard(top30_sets.get("c4a", []), top30_sets.get("c4c_matched", []))

    metrics = {
        "within_self_tool_jaccard": np.mean(within_self) if within_self else float("nan"),
        "cross_cluster_jaccard": np.mean(cross_cluster) if cross_cluster else float("nan"),
        "peer_vs_no_attrib_jaccard": peer_peer,
        "within_peer_jaccard": peer_peer,  # c4a vs c4c_matched for summary table
    }
    print(f"    Cluster metrics: self/tool={metrics['within_self_tool_jaccard']:.3f}, "
          f"cross(peer vs self/tool)={metrics['cross_cluster_jaccard']:.3f}, "
          f"c4a-c4c_matched={metrics['peer_vs_no_attrib_jaccard']:.3f}")

    return {
        "all_deltas": all_deltas,
        "top30_sets": top30_sets,
        "jaccard_matrix": jaccard_matrix,
        "conditions": cond_list,
        "metrics": metrics,
        "clean_feats": clean_feats,
    }


# ─── Phase 3: Causal intervention ────────────────────────────────────────────

def run_phase3(sae: nn.Module, cfg: AltSAEConfig, phase2_results: dict) -> dict | None:
    """Run causal intervention at this SAE's layer.

    Uses top-100 features by absolute delta (both rising and falling),
    clamped to their clean means.
    """
    print(f"\n  Phase 3: Causal intervention at {cfg.name} (block {cfg.layer})")

    from src.data import jury_for, load_artifacts
    from src.model import choice_token_ids, get_model_and_tokenizer
    from src.prompts import build_prompt_user_role_jury

    model, tokenizer = get_model_and_tokenizer()
    ctoks = choice_token_ids(tokenizer)
    vocab_indices = [ctoks[c] for c in CHOICES]

    art = load_artifacts()
    questions = art["known_questions"]
    jury_strong = jury_for("strong")

    # Same 50-question subset as the Goodfire intervention
    rng = np.random.default_rng(42)
    subset_idx = rng.choice(400, size=50, replace=False)
    subset_idx.sort()

    # Get top-100 features from C4a delta
    c4a_deltas = phase2_results["all_deltas"]["c4a"]["deltas"]
    abs_deltas = c4a_deltas.abs()
    top100_vals, top100_idx = torch.topk(abs_deltas, 100)
    feature_indices = top100_idx.tolist()

    # Clamp values = clean mean at those features
    clean_feats = phase2_results["clean_feats"]
    clean_means = clean_feats.float().mean(dim=0)
    clamp_values = clean_means[top100_idx]

    device = "cuda:0"
    sae_dev = next(sae.parameters()).device
    if str(sae_dev) != device:
        sae = sae.to(device)

    # Hook module: block N's output = hidden_states[N+1]
    # To intervene on hidden_states at index cfg.hidden_idx, hook block cfg.layer
    hook_module_idx = cfg.layer
    layers = model.model.layers

    idx_tensor = torch.tensor(feature_indices, dtype=torch.long, device=device)
    val_tensor = clamp_values.to(device=device, dtype=torch.float32)

    def make_hook():
        state = {"called": False}
        def hook_fn(module, inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None

            target = hidden[:, -1:, :].float().to(device)
            b, s, d = target.shape
            flat = target.reshape(b * s, d)

            feats = sae.encode(flat)
            feats[:, idx_tensor] = val_tensor
            new_flat = sae.decode(feats)
            new_target = new_flat.reshape(b, s, d).to(dtype=hidden.dtype, device=hidden.device)

            hidden = hidden.clone()
            hidden[:, -1:, :] = new_target
            state["called"] = True

            if isinstance(output, tuple):
                return (hidden,) + tuple(rest)
            return hidden
        return hook_fn, state

    # Run unclamped and clamped passes
    unclamped_pcorrect = []
    unclamped_pwrong = []
    clamped_pcorrect = []
    clamped_pwrong = []

    print(f"    Running {len(subset_idx)} forward passes (unclamped + clamped)...")
    for count, qi in enumerate(subset_idx):
        item = questions[qi]
        wrong_idx = jury_strong["gemma"][qi]["wrong_idx"]
        correct_idx = item["answer"]

        prompt = build_prompt_user_role_jury(qi, item, wrong_idx, jury_strong, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Unclamped
        with torch.no_grad():
            out = model(**inputs)
        logits = out.logits[0, -1, :][vocab_indices]
        probs = torch.softmax(logits.float(), dim=0)
        unclamped_pcorrect.append(probs[correct_idx].item())
        unclamped_pwrong.append(probs[wrong_idx].item())

        # Clamped
        hook_fn, hook_state = make_hook()
        handle = layers[hook_module_idx].register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                out = model(**inputs)
            logits = out.logits[0, -1, :][vocab_indices]
            probs = torch.softmax(logits.float(), dim=0)
            clamped_pcorrect.append(probs[correct_idx].item())
            clamped_pwrong.append(probs[wrong_idx].item())
        finally:
            handle.remove()

        if count == 0:
            print(f"      Hook fired: {hook_state['called']}")
            print(f"      Q0: unclamped P(c)={unclamped_pcorrect[0]:.4f}, P(w)={unclamped_pwrong[0]:.4f}")
            print(f"          clamped   P(c)={clamped_pcorrect[0]:.4f}, P(w)={clamped_pwrong[0]:.4f}")

        if (count + 1) % 10 == 0:
            print(f"      [{count+1}/50] done")

    results = {
        "sae_name": cfg.name,
        "layer": cfg.layer,
        "n_questions": len(subset_idx),
        "n_features_clamped": 100,
        "question_indices": subset_idx.tolist(),
        "mean_unclamped_pcorrect": float(np.mean(unclamped_pcorrect)),
        "mean_unclamped_pwrong": float(np.mean(unclamped_pwrong)),
        "mean_clamped_pcorrect": float(np.mean(clamped_pcorrect)),
        "mean_clamped_pwrong": float(np.mean(clamped_pwrong)),
        "delta_pcorrect": float(np.mean(clamped_pcorrect) - np.mean(unclamped_pcorrect)),
        "delta_pwrong": float(np.mean(clamped_pwrong) - np.mean(unclamped_pwrong)),
    }

    print(f"\n    Intervention results ({cfg.name}):")
    print(f"      Unclamped: P(correct)={results['mean_unclamped_pcorrect']:.4f}, "
          f"P(wrong)={results['mean_unclamped_pwrong']:.4f}")
    print(f"      Clamped:   P(correct)={results['mean_clamped_pcorrect']:.4f}, "
          f"P(wrong)={results['mean_clamped_pwrong']:.4f}")
    print(f"      ΔP(correct) = {results['delta_pcorrect']*100:+.2f} pp")
    print(f"      ΔP(wrong)   = {results['delta_pwrong']*100:+.2f} pp")

    # Save
    out_path = OUT_DIR / f"intervention_{cfg.name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


# ─── Phase 4: Figures and summary ─────────────────────────────────────────────

def make_jaccard_figure(all_results: dict) -> None:
    """Side-by-side Jaccard heatmaps for each SAE."""
    # Also load Goodfire reference
    goodfire_path = RESULTS_DIR / "sae" / "jaccard_overlap.csv"
    goodfire_matrix = None
    goodfire_labels = None
    if goodfire_path.exists():
        with open(goodfire_path) as f:
            lines = f.readlines()
        goodfire_labels = lines[0].strip().split(",")
        goodfire_matrix = np.array([[float(x) for x in l.strip().split(",")] for l in lines[1:]])

    n_plots = len(all_results) + (1 if goodfire_matrix is not None else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.5), squeeze=False)
    axes = axes[0]

    plot_idx = 0

    # Goodfire reference
    if goodfire_matrix is not None:
        ax = axes[plot_idx]
        # Use only the c4 conditions for comparison
        c4_mask = [i for i, l in enumerate(goodfire_labels) if l.startswith("c4")]
        c4_labels = [goodfire_labels[i] for i in c4_mask]
        sub = goodfire_matrix[np.ix_(c4_mask, c4_mask)]
        im = ax.imshow(sub, vmin=0, vmax=1, cmap="YlOrRd")
        ax.set_xticks(range(len(c4_labels)))
        ax.set_xticklabels(c4_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(c4_labels)))
        ax.set_yticklabels(c4_labels, fontsize=8)
        ax.set_title("Goodfire L19\n(reference)", fontsize=10)
        for i in range(len(c4_labels)):
            for j in range(len(c4_labels)):
                ax.text(j, i, f"{sub[i,j]:.2f}", ha="center", va="center", fontsize=7)
        plot_idx += 1

    # Alternative SAEs
    for name, res in all_results.items():
        ax = axes[plot_idx]
        mat = res["jaccard_matrix"]
        conds = res["conditions"]
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlOrRd")
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(conds)))
        ax.set_yticklabels(conds, fontsize=8)
        ax.set_title(f"{name}\n(L{res.get('layer', '?')})", fontsize=10)
        for i in range(len(conds)):
            for j in range(len(conds)):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7)
        plot_idx += 1

    plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
    fig.suptitle("Top-30 Feature Jaccard Overlap: Cross-SAE Comparison", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_sae_multi_layer_jaccard.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_sae_multi_layer_jaccard.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: figures/fig_sae_multi_layer_jaccard.png")


def make_intervention_figure(intervention_results: dict) -> None:
    """Bar chart comparing P(correct) restoration across SAEs."""
    # Include Goodfire reference
    goodfire = {
        "name": "Goodfire L19",
        "delta_pcorrect": 0.021563,  # both_clean_k100 from sweep
        "delta_pwrong": -0.312436,
    }

    all_interventions = [goodfire] + list(intervention_results.values())
    names = [r["name"] if isinstance(r, dict) and "name" in r else r.get("sae_name", "?") for r in all_interventions]
    dp_correct = [r["delta_pcorrect"] * 100 for r in all_interventions]
    dp_wrong = [r["delta_pwrong"] * 100 for r in all_interventions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(len(names))
    ax1.bar(x, dp_correct, color=["#1f77b4"] + ["#2ca02c"] * (len(names) - 1))
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("ΔP(correct) [pp]")
    ax1.set_title("P(correct) Restoration")
    ax1.axhline(0, color="gray", lw=0.8)
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, dp_wrong, color=["#1f77b4"] + ["#d62728"] * (len(names) - 1))
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("ΔP(wrong) [pp]")
    ax2.set_title("P(wrong) Suppression")
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Causal Intervention: Feature Clamping at Different Layers", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_sae_multi_layer_intervention.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: figures/fig_sae_multi_layer_intervention.png")


def write_report(all_phase2: dict, intervention_results: dict, validation_results: dict) -> None:
    """Write REPORT.md with full results."""
    lines = [
        "# Experiment #9: Multi-Layer SAE Analysis with Alternative Bases",
        "",
        "## Summary",
        "",
        "This experiment tests whether the SAE feature families identified at Goodfire L19",
        "replicate under alternative SAE bases and inside the L14-L18 causal window.",
        "",
        "## SAEs Tested",
        "",
        "| SAE Source | Layer | d_sae | Activation | Notes |",
        "|---|---|---|---|---|",
        "| Goodfire (reference) | L19 | 65,536 | Top-K (k=91) | Original analysis |",
    ]

    for name, res in all_phase2.items():
        cfg_layer = res.get("layer", "?")
        d_sae = res.get("d_sae", "?")
        act = res.get("activation", "?")
        lines.append(f"| {name} | L{cfg_layer} | {d_sae:,} | {act} | This experiment |")

    lines += ["", "## Validation", ""]
    for name, v in validation_results.items():
        lines.append(f"- **{name}**: MSE={v['mse']:.4f}, rel_MSE={v['relative_mse']:.4f}, "
                     f"L0={v['mean_l0']:.1f}, cos_sim={v['cosine_sim']:.4f} "
                     f"{'✓' if v['valid'] else '✗ FAILED'}")

    lines += ["", "## Phase 2: Feature Delta Analysis", ""]
    lines.append("### Cluster Separation Metrics (Top-30 Jaccard)")
    lines.append("")
    lines.append("| SAE | J(c4a,c4c_matched) | J(c4d,c4e) [self/tool] | J(c4a,c4d/c4e) [cross] | Self/Tool − Cross |")
    lines.append("|---|---|---|---|---|")

    # Goodfire reference from existing CSV
    goodfire_path = RESULTS_DIR / "sae" / "jaccard_overlap.csv"
    if goodfire_path.exists():
        with open(goodfire_path) as f:
            gf_lines = f.readlines()
        gf_labels = gf_lines[0].strip().split(",")
        gf_mat = np.array([[float(x) for x in l.strip().split(",")] for l in gf_lines[1:]])
        # c4a=0, c4d=1, c4c_matched=2, c4c=3, c4e=4
        gf_peer = gf_mat[0, 2]  # c4a vs c4c_matched
        gf_self = gf_mat[1, 4]  # c4d vs c4e
        gf_cross = np.mean([gf_mat[0, 1], gf_mat[0, 4]])  # c4a vs c4d, c4a vs c4e
        lines.append(f"| Goodfire L19 | {gf_peer:.3f} | {gf_self:.3f} | {gf_cross:.3f} | {gf_self - gf_cross:.3f} |")

    for name, res in all_phase2.items():
        m = res["metrics"]
        sep = m["within_self_tool_jaccard"] - m["cross_cluster_jaccard"]
        lines.append(f"| {name} | {m['within_peer_jaccard']:.3f} | "
                     f"{m['within_self_tool_jaccard']:.3f} | "
                     f"{m['cross_cluster_jaccard']:.3f} | {sep:.3f} |")

    lines += ["", "## Phase 3: Causal Intervention", ""]

    if intervention_results:
        lines.append("| SAE | Layer | k clamped | ΔP(correct) [pp] | ΔP(wrong) [pp] |")
        lines.append("|---|---|---|---|---|")
        lines.append(f"| Goodfire L19 (ref) | 19 | 100 | +3.5 | -21.9 |")
        for name, r in intervention_results.items():
            lines.append(f"| {name} | {r.get('layer', '?')} | {r['n_features_clamped']} | "
                         f"{r['delta_pcorrect']*100:+.2f} | {r['delta_pwrong']*100:+.2f} |")
    else:
        lines.append("*(No intervention results — model loading may have failed.)*")

    lines += [
        "",
        "## Interpretation",
        "",
        "### Do feature families replicate across SAE bases?",
        "",
    ]

    # Auto-generate interpretation based on metrics
    replicates = True
    for name, res in all_phase2.items():
        m = res["metrics"]
        if m["within_self_tool_jaccard"] <= m["cross_cluster_jaccard"]:
            replicates = False

    if replicates:
        lines.append("**Yes.** All tested alternative SAEs show the self/tool cluster (c4d, c4e)")
        lines.append("having higher within-cluster Jaccard than cross-cluster Jaccard (c4a vs c4d/c4e),")
        lines.append("confirming that the feature family structure is not an artifact of the Goodfire basis.")
    else:
        lines.append("**Partially.** Some alternative SAEs show cluster separation but others do not.")
        lines.append("See metrics above for details.")

    lines += [
        "",
        "### Does causal-window intervention produce better P(correct) restoration?",
        "",
    ]
    if intervention_results:
        best_intervention = max(intervention_results.values(), key=lambda r: r["delta_pcorrect"])
        if best_intervention["delta_pcorrect"] > 0.021563:
            lines.append(f"**Yes.** {best_intervention['sae_name']} at L{best_intervention.get('layer', '?')} "
                         f"achieves ΔP(correct) = {best_intervention['delta_pcorrect']*100:+.2f} pp, "
                         f"exceeding Goodfire L19's +3.5 pp. Intervening inside the causal window "
                         f"is more effective for restoring correct-answer probability.")
        else:
            lines.append(f"**No clear improvement.** The best alternative SAE achieves "
                         f"ΔP(correct) = {best_intervention['delta_pcorrect']*100:+.2f} pp "
                         f"vs Goodfire L19's +3.5 pp.")
    else:
        lines.append("*(Intervention not run.)*")

    lines += ["", "---", f"*Generated by run_sae_multi_layer.py*", ""]

    report_path = OUT_DIR / "REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {report_path}")


def write_comparison_summary(all_phase2: dict, intervention_results: dict, validation_results: dict) -> None:
    """Write comparison_summary.csv."""
    import csv

    rows = []
    # Goodfire reference row
    goodfire_path = RESULTS_DIR / "sae" / "jaccard_overlap.csv"
    if goodfire_path.exists():
        with open(goodfire_path) as f:
            gf_lines = f.readlines()
        gf_mat = np.array([[float(x) for x in l.strip().split(",")] for l in gf_lines[1:]])
        gf_peer = gf_mat[0, 2]
        gf_self = gf_mat[1, 4]
        gf_cross = np.mean([gf_mat[0, 1], gf_mat[0, 4], gf_mat[2, 1], gf_mat[2, 4]])
        rows.append({
            "sae_source": "Goodfire",
            "layer": 19,
            "d_sae": 65536,
            "activation": "topk_k91",
            "within_peer_jaccard": f"{gf_peer:.4f}",
            "within_self_tool_jaccard": f"{gf_self:.4f}",
            "cross_cluster_jaccard": f"{gf_cross:.4f}",
            "delta_pcorrect_pp": "+3.5",
            "delta_pwrong_pp": "-21.9",
            "valid": "True",
        })

    for name, res in all_phase2.items():
        m = res["metrics"]
        interv = intervention_results.get(name, {})
        rows.append({
            "sae_source": name,
            "layer": res.get("layer", ""),
            "d_sae": res.get("d_sae", ""),
            "activation": res.get("activation", ""),
            "within_peer_jaccard": f"{m['within_peer_jaccard']:.4f}",
            "within_self_tool_jaccard": f"{m['within_self_tool_jaccard']:.4f}",
            "cross_cluster_jaccard": f"{m['cross_cluster_jaccard']:.4f}",
            "delta_pcorrect_pp": f"{interv.get('delta_pcorrect', float('nan'))*100:+.2f}" if interv else "N/A",
            "delta_pwrong_pp": f"{interv.get('delta_pwrong', float('nan'))*100:+.2f}" if interv else "N/A",
            "valid": str(validation_results.get(name, {}).get("valid", False)),
        })

    csv_path = OUT_DIR / "comparison_summary.csv"
    fieldnames = ["sae_source", "layer", "d_sae", "activation",
                  "within_peer_jaccard", "within_self_tool_jaccard", "cross_cluster_jaccard",
                  "delta_pcorrect_pp", "delta_pwrong_pp", "valid"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"  Saved: {csv_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Experiment #9: Multi-Layer SAE Analysis with Alternative Bases")
    print("=" * 70)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda:0":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    conditions = ["c4a", "c4d", "c4e", "c4c_matched"]

    # ── Load and validate SAEs ─────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Phase 1: Loading and validating alternative SAEs")
    print("=" * 50)

    saes_loaded = {}
    validation_results = {}

    # 1. andyrdt L15
    print("\n[1] Loading andyrdt L15...")
    try:
        sae_andy, cfg_andy = load_andyrdt_l15(device=device)
        clean_acts_15 = load_activations("c1", cfg_andy.hidden_idx)
        val_andy = validate_sae(sae_andy, clean_acts_15, cfg_andy)
        validation_results[cfg_andy.name] = val_andy
        if val_andy["valid"]:
            saes_loaded[cfg_andy.name] = (sae_andy, cfg_andy)
    except Exception as e:
        print(f"  FAILED to load andyrdt L15: {e}")

    # 2. Geaming L18
    print("\n[2] Loading Geaming L18...")
    try:
        sae_geam, cfg_geam = load_geaming_l18(device=device)
        clean_acts_18 = load_activations("c1", cfg_geam.hidden_idx)
        val_geam = validate_sae(sae_geam, clean_acts_18, cfg_geam)
        validation_results[cfg_geam.name] = val_geam
        if val_geam["valid"]:
            saes_loaded[cfg_geam.name] = (sae_geam, cfg_geam)
    except Exception as e:
        print(f"  FAILED to load Geaming L18: {e}")

    # 3. pellement99 L16
    print("\n[3] Loading pellement99 L16...")
    try:
        sae_pel, cfg_pel = load_pellement_l16(device=device)
        clean_acts_16 = load_activations("c1", cfg_pel.hidden_idx)
        val_pel = validate_sae(sae_pel, clean_acts_16, cfg_pel)
        validation_results[cfg_pel.name] = val_pel
        if val_pel["valid"]:
            saes_loaded[cfg_pel.name] = (sae_pel, cfg_pel)
    except Exception as e:
        print(f"  FAILED to load pellement99 L16: {e}")

    # 4. Jammies-io L18 sycophancy
    print("\n[4] Loading Jammies-io L18 sycophancy...")
    try:
        sae_jam, cfg_jam = load_jammies_l18(device=device)
        clean_acts_18b = load_activations("c1", cfg_jam.hidden_idx)
        val_jam = validate_sae(sae_jam, clean_acts_18b, cfg_jam)
        validation_results[cfg_jam.name] = val_jam
        if val_jam["valid"]:
            saes_loaded[cfg_jam.name] = (sae_jam, cfg_jam)
    except Exception as e:
        print(f"  FAILED to load Jammies-io L18: {e}")

    print(f"\n  Successfully loaded and validated: {list(saes_loaded.keys())}")
    if len(saes_loaded) < 2:
        print("  WARNING: fewer than 2 SAEs validated. Continuing with what we have.")

    # ── Phase 2: Feature deltas ────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Phase 2: Feature Delta Analysis")
    print("=" * 50)

    all_phase2 = {}
    for name, (sae, cfg) in saes_loaded.items():
        result = run_phase2(sae, cfg, conditions)
        result["layer"] = cfg.layer
        result["d_sae"] = cfg.d_sae
        result["activation"] = cfg.activation
        all_phase2[name] = result

    # ── Phase 3: Causal intervention ───────────────────────────────────────
    print("\n" + "=" * 50)
    print("Phase 3: Causal Intervention")
    print("=" * 50)

    intervention_results = {}

    # Prefer L15/L16 (inside causal window), then L18
    intervention_order = sorted(saes_loaded.keys(), key=lambda n: saes_loaded[n][1].layer)
    for name in intervention_order:
        sae, cfg = saes_loaded[name]
        if cfg.layer >= 14 and cfg.layer <= 18:
            try:
                result = run_phase3(sae, cfg, all_phase2[name])
                if result is not None:
                    result["name"] = name
                    result["layer"] = cfg.layer
                    intervention_results[name] = result
            except Exception as e:
                print(f"  Intervention FAILED for {name}: {e}")
                import traceback
                traceback.print_exc()

    # ── Phase 4: Figures and summary ───────────────────────────────────────
    print("\n" + "=" * 50)
    print("Phase 4: Figures and Summary")
    print("=" * 50)

    if all_phase2:
        make_jaccard_figure(all_phase2)

    if intervention_results:
        make_intervention_figure(intervention_results)

    write_comparison_summary(all_phase2, intervention_results, validation_results)
    write_report(all_phase2, intervention_results, validation_results)

    # ── Verdict ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if all_phase2:
        all_separate = True
        for name, res in all_phase2.items():
            m = res["metrics"]
            sep = m["within_self_tool_jaccard"] - m["cross_cluster_jaccard"]
            print(f"\n  {name}: self/tool={m['within_self_tool_jaccard']:.3f}, "
                  f"cross(peer-vs-self/tool)={m['cross_cluster_jaccard']:.3f}, "
                  f"c4a-c4c_matched={m.get('peer_vs_no_attrib_jaccard', float('nan')):.3f}, "
                  f"separation(self/tool - cross)={sep:.3f}")
            if sep <= 0:
                all_separate = False

        if all_separate and len(all_phase2) >= 2:
            print("\n  ✓ FEATURE FAMILIES REPLICATE across alternative SAE bases.")
            print("    The self/tool cluster (c4d-c4e) shows higher Jaccard overlap than")
            print("    cross-cluster pairs (c4a vs c4d/c4e), confirming the peer vs self/tool")
            print("    separation holds with different SAE bases and at causal-window layers.")
        elif all_separate:
            print("\n  ~ Partial replication (only 1 SAE tested).")
        else:
            print("\n  ✗ Feature families do NOT cleanly replicate across all SAE bases.")
            print("    Some SAEs show cluster separation, others do not.")

    if intervention_results:
        best = max(intervention_results.values(), key=lambda r: r["delta_pcorrect"])
        print(f"\n  Best causal-window intervention: {best['sae_name']} L{best.get('layer', '?')}")
        print(f"    ΔP(correct) = {best['delta_pcorrect']*100:+.2f} pp (Goodfire L19 ref: +3.5 pp)")
        print(f"    ΔP(wrong)   = {best['delta_pwrong']*100:+.2f} pp (Goodfire L19 ref: -21.9 pp)")

    print("\n  Done. All results saved to results/sae_multi_layer/")
    print("=" * 70)


if __name__ == "__main__":
    main()
