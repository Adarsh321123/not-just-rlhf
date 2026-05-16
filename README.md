# Not Just RLHF: Why Alignment Alone Won't Fix Multi-Agent Sycophancy

**[Paper (arXiv)](https://arxiv.org/abs/2605.12991)**

Multi-agent LLM pipelines flip from correct to incorrect answers under simulated peer disagreement at rates we term *yield*. We show this vulnerability is pretrained (not RLHF-induced), localize it to an attention-dominant mid-layer circuit (L14--L18), decompose the attack surface into channel framing x consensus strength, and demonstrate that a single correctly-arguing dissenter generalizes across framings where prompt-level defenses do not.

## Citation

```bibtex
@article{kumarappan2026notjust,
  title={Not Just RLHF: Why Alignment Alone Won't Fix Multi-Agent Sycophancy},
  author={Kumarappan, Adarsh and Mujoo, Ananya},
  journal={arXiv preprint arXiv:2605.12991},
  year={2026}
}
```

## Requirements

- Python 3.10
- NVIDIA GPU with >= 24 GB VRAM (tested on RTX 3090)
- HuggingFace account with access to gated models (Llama-3.1-8B-Instruct, Gemma-2-9B-it)

## Setup

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_huggingface_token>
```

You must accept the Llama-3.1-8B-Instruct license on HuggingFace before running any experiment.

## Repository Structure

```
src/           Core library (model loading, prompt construction, patching, probes, SAE)
scripts/       All experiment entry points (one script per experiment)
data/          Pre-generated artifacts (questions, probes, jury response corpora)
```

## Pre-generated Data

The `data/` directory contains everything needed to run experiments without regeneration:

- `questions.json` — 400 filtered MMLU questions (humanities, 4 categories)
- `final_probes.joblib` — 33 per-layer linear probes trained on clean activations
- `avg_probe_accs.joblib` — probe accuracy curve (5-fold CV)
- `jury_responses_*.json` — jury corpora from multiple models (Gemma, Qwen, Mistral, Phi, Llama-3.2, Yi-1.5)

To regenerate the wrong-agent count sweep jury corpora (requires GPU):

```bash
python scripts/generate_c6_jury.py
python scripts/generate_c6_jury_extra.py
```

### Optional: Regenerate data from scratch

Replicates the full data pipeline (MMLU filtering, probe training, jury generation):

```bash
python scripts/generate_data.py
```

## Reproducing Results

A quick sanity check (5 questions, ~2 min):

```bash
python scripts/smoke_test.py
```

### Core behavioral sweep (Tables 1-2)

Runs all 16 conditions from Tables 1-2. Condition codes map to paper names as follows:

| Code | Paper name |
|---|---|
| `c1` | Direct user assertion |
| `c3` | User assertion (peer-jury length) |
| `c4a` / `c5a` | Named peer jury (strong / weak) |
| `c4c` / `c5c` | Anonymous perspectives (strong / weak) |
| `c4c_matched` / `c5c_matched` | Anonymous jury (strong / weak) |
| `c4d` / `c5d` | Assistant-role jury (strong / weak) |
| `c4e` / `c5e` | Tool-role jury (strong / weak) |
| `c4d_unmatched` / `c5d_unmatched` | Assist.-role, no consensus (strong / weak) |
| `c4e_unmatched` / `c5e_unmatched` | Tool-role, no consensus (strong / weak) |

```bash
# Strong conditions (Table 1)
python scripts/run_all_conditions.py --conditions c1,c3,c4a,c4c,c4c_matched,c4d,c4e,c4d_unmatched,c4e_unmatched
# Weak conditions
python scripts/run_all_conditions.py --conditions c5a,c5c,c5c_matched,c5d,c5e,c5d_unmatched,c5e_unmatched
# Bootstrap CIs
python scripts/compute_bootstrap.py
```

### Mechanistic analyses (Figures 3, 7)

```bash
python scripts/run_patching.py
python scripts/run_component_patching.py
python scripts/run_dissenter_patching.py
```

### Attack surface: wrong-agent count sweep, defenses, dissenter rescue (Figure 2, Tables 3-8)

```bash
python scripts/run_adaptive_attacker.py
python scripts/analyze_adaptive_attacker.py
python scripts/run_defense_matrix.py
python scripts/run_minimal_dissenter.py
python scripts/run_c6_conditional_patching.py
python scripts/run_c6_tool_role.py
```

### Cross-family evidence: base vs. Instruct (Figure 4, Table 12)

```bash
python scripts/run_cross_model.py
python scripts/run_cross_model_patching.py
python scripts/run_base_model.py
python scripts/run_base_vs_instruct_cross_family.py --family mistral
python scripts/run_base_vs_instruct_cross_family.py --family gemma
python scripts/run_base_vs_instruct_cross_family.py --family qwen
python scripts/analyze_base_vs_instruct.py
```

### Feature suppression: SAE & difference-in-means (Tables 9-11, Figure 8)

```bash
python scripts/run_sae_analysis.py
python scripts/run_sae_intervention.py
python scripts/run_sae_multi_layer.py
python scripts/run_dim_analysis.py
```

### Cross-domain and cross-benchmark generalization (Tables 13-14, Figure 10)

```bash
python scripts/run_stem_extension.py
python scripts/run_c1_stem_addendum.py
python scripts/run_c1_stem_investigation.py
python scripts/run_cross_benchmark_transfer.py
python scripts/run_cross_benchmark_transfer_v2.py
```

### Robustness: wrong-agent count sweep scaling, calibration, jury audit (Tables 15-18)

```bash
python scripts/run_disagreement_gradient.py
python scripts/run_c6_user_sweep.py     # user-role framing sweep
python scripts/run_c6_self_sweep.py     # assistant-role framing sweep
python scripts/run_c6_scaling.py        # N=5 and N=6 jury sizes
python scripts/run_attention_ablation.py
python scripts/jury_audit_llm.py
```

### Generate figures

```bash
python scripts/generate_figures.py           # all figures
python scripts/generate_figures.py --fig 3   # specific figure only
```

All results are saved to `results/` and figures to `figures/`.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
