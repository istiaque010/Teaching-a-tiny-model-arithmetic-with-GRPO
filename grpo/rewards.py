"""Verifiable rewards (Phase 6).

    R = w_correct * correctness + w_format * format
    correctness = 1 if the parsed final answer equals the true answer, else 0
    format      = 1 if the response has exactly one parseable final answer (or agreeing duplicates), else 0

Length is never rewarded: a long explanation earns nothing by itself, and a correct final answer does not prove that
the steps before it are right (Phase 15 checks that separately). Components are always returned separately so the
format bonus can be ablated (Phase 14).
"""
from __future__ import annotations

from .answer_parser import grade


def reward(text: str, answer: int, w_correct: float = 1.0, w_format: float = 0.1) -> dict:
    g = grade(text, answer)
    correctness = 1.0 if g["correct"] else 0.0
    fmt = 1.0 if g["valid"] else 0.0
    return {"reward": w_correct * correctness + w_format * fmt, "correctness": correctness, "format": fmt,
            "status": g["status"], "pred": g["pred"]}
