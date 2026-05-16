"""Load MMLU artifacts and jury corpora from the bundled data/ directory."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import joblib
import numpy as np

from .config import DATA_DIR


@lru_cache(maxsize=1)
def load_artifacts() -> dict[str, Any]:
    """Load all MMLU/probe/jury artifacts from the local data/ directory."""
    probes = joblib.load(DATA_DIR / "final_probes.joblib")
    acc = joblib.load(DATA_DIR / "avg_probe_accs.joblib")

    with open(DATA_DIR / "questions.json") as f:
        questions = json.load(f)
    with open(DATA_DIR / "jury_responses_4a.json") as f:
        jury_strong = json.load(f)
    with open(DATA_DIR / "jury_responses_4b.json") as f:
        jury_weak = json.load(f)

    result = {
        "final_probes": probes,
        "avg_probe_accs": acc,
        "known_questions": questions,
        "jury_strong": jury_strong,
        "jury_weak": jury_weak,
    }

    npz_path = DATA_DIR / "dataset.npz"
    if npz_path.exists():
        loaded = np.load(npz_path)
        result["known_acts"] = loaded["acts"]
        result["known_labels"] = loaded["labels"]

    return result


def jury_for(name: str) -> dict:
    """Return the jury dict for 'strong' or 'weak'."""
    art = load_artifacts()
    if name == "strong":
        return art["jury_strong"]
    if name == "weak":
        return art["jury_weak"]
    raise ValueError(f"unknown jury name: {name!r}")
