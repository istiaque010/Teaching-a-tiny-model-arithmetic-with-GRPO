"""Project 8: GRPO post-training of a small autoregressive language model (Project 1's 13.9M GPT).

Code package `grpo`. Importing it makes Project 1's `src` package importable (model, tokenizer); Project 1 is only
read, never changed, and bytecode writing is off so nothing is written into it.
"""
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True

PROJECT8 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(PROJECT8)
PROJECT1 = os.path.join(ROOT, "project1")
CONFIGS = os.path.join(PROJECT8, "configs")
DATA = os.path.join(PROJECT8, "data")
CHECKPOINTS = os.path.join(PROJECT8, "checkpoints")
RESULTS = os.path.join(PROJECT8, "results")
METRICS = os.path.join(RESULTS, "metrics")
GENERATIONS = os.path.join(RESULTS, "generations")
FIGURES = os.path.join(RESULTS, "figures")
LABELS = os.path.join(RESULTS, "error_labels")

BASE_CKPT = os.path.join(PROJECT1, "checkpoints", "small", "model_best.pt")
TOKENIZER = os.path.join(PROJECT1, "data", "tokenizer.json")

if PROJECT1 not in sys.path:
    sys.path.insert(0, PROJECT1)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(obj, *path):
    p = os.path.join(*path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return p


def load_json(*path):
    with open(os.path.join(*path), encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(rows, *path):
    p = os.path.join(*path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def read_jsonl(*path):
    with open(os.path.join(*path), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_config(name: str) -> dict:
    import yaml

    with open(os.path.join(CONFIGS, name), encoding="utf-8") as f:
        return yaml.safe_load(f)


def ckpt_path(name: str) -> str:
    """'base' -> Project 1 checkpoint; 'sft' -> checkpoints/sft/final.pt; 'sft/step_01000' -> that file."""
    if name == "base":
        return BASE_CKPT
    if "/" in name:
        return os.path.join(CHECKPOINTS, name + ".pt")
    return os.path.join(CHECKPOINTS, name, "final.pt")
