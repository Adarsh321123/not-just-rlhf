"""Shared configuration: model IDs, paths, seeds."""
from __future__ import annotations

import os
from pathlib import Path

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_REVISION = os.environ.get("MODEL_REVISION") or "0e9e39f249a16976918f6564b8830bc894c89659"

HF_TOKEN = os.environ.get("HF_TOKEN", "")

NUM_LAYERS = 33
CHOICES = ["A", "B", "C", "D"]

WRONG_IDX_SEED = 42

LDA_LAYER = 25

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
