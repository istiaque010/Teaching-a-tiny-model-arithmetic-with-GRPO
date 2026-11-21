"""Deterministic final-answer parser and grader (Phase 3).

Rules (strict on purpose, so the reward cannot be gamed by odd outputs):
  * the answer must be inside <answer>...</answer>; no complete tag -> "no_tag" (an opened but unclosed tag
    -> "unclosed");
  * the tag content, after trimming spaces, must be a plain whole number: optional sign, digits, optional
    thousands commas ("1,234") and an optional zero decimal part ("12.0"). Anything else ("twelve", "1e3", "12.5",
    "12 apples") -> "not_number";
  * several tags are allowed only if they all hold the same number ("duplicate", still valid); different numbers
    -> "conflict" (invalid);
  * at most 9 digits ("too_long" otherwise).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

TAG = re.compile(r"<answer>(.*?)</answer>", re.S)
NUMBER = re.compile(r"^[+-]?(\d{1,3}(,\d{3})+|\d+)(\.0+)?$")


@dataclass
class Parsed:
    valid: bool
    value: Optional[int]
    status: str          # ok | duplicate | no_tag | unclosed | not_number | conflict | too_long | empty
    n_tags: int


def normalize(s: str) -> Optional[int]:
    s = s.strip()
    if not NUMBER.match(s):
        return None
    s = s.replace(",", "").split(".")[0]
    if len(s.lstrip("+-")) > 9:
        raise OverflowError
    return int(s)


def parse(text: str) -> Parsed:
    if not text or not text.strip():
        return Parsed(False, None, "empty", 0)
    tags = TAG.findall(text)
    if not tags:
        return Parsed(False, None, "unclosed" if "<answer>" in text else "no_tag", 0)
    values = []
    for t in tags:
        try:
            v = normalize(t)
        except OverflowError:
            return Parsed(False, None, "too_long", len(tags))
        if v is None:
            return Parsed(False, None, "not_number", len(tags))
        values.append(v)
    if len(set(values)) > 1:
        return Parsed(False, None, "conflict", len(tags))
    return Parsed(True, values[0], "ok" if len(tags) == 1 else "duplicate", len(tags))


def grade(text: str, answer: int) -> dict:
    p = parse(text)
    return {"valid": p.valid, "pred": p.value, "status": p.status, "correct": bool(p.valid and p.value == answer)}
