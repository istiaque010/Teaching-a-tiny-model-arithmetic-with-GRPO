"""One fixed prompt / answer format for the whole experiment (Phase 3).

    <BOS>Question: Mia has 3 bags with 4 apples in each. How many apples does Mia have?
    Solution: 3 * 4 = 12.
    Final answer: <answer>12</answer><EOS>

The prompt ends after "Solution:"; everything after it (the worked steps, the final-answer line and <EOS>) is the
*response* the model must write. Losses are computed only on response tokens.
"""
from __future__ import annotations

from typing import List

PROMPT = "Question: {question}\nSolution:"
TARGET = " {solution}\nFinal answer: <answer>{answer}</answer>"


def prompt_text(question: str) -> str:
    return PROMPT.format(question=question)


def target_text(solution: str, answer) -> str:
    return TARGET.format(solution=solution, answer=answer)


def encode_prompt(tok, question: str) -> List[int]:
    return tok.encode(prompt_text(question), add_bos=True)


def encode_target(tok, solution: str, answer) -> List[int]:
    return tok.encode(target_text(solution, answer)) + [tok.eos_id]
