"""Answer parser, grader and reward: valid, invalid and adversarial outputs (Phases 3 and 6)."""
import pytest

import grpo  # noqa: F401
from grpo.answer_parser import parse, grade
from grpo.rewards import reward


@pytest.mark.parametrize("text,value,status", [
    (" 3 * 4 = 12.\nFinal answer: <answer>12</answer>", 12, "ok"),
    ("<answer> 12 </answer>", 12, "ok"),
    ("<answer>+12</answer>", 12, "ok"),
    ("<answer>12.0</answer>", 12, "ok"),
    ("<answer>1,234</answer>", 1234, "ok"),
    ("<answer>0</answer>", 0, "ok"),
    ("<answer>-7</answer>", -7, "ok"),
    ("<answer>12</answer> again <answer>12</answer>", 12, "duplicate"),
])
def test_valid(text, value, status):
    p = parse(text)
    assert p.valid and p.value == value and p.status == status


@pytest.mark.parametrize("text,status", [
    ("", "empty"),
    ("   \n", "empty"),
    ("3 * 4 = 12.", "no_tag"),
    ("Final answer: 12", "no_tag"),
    ("Final answer: <answer>12", "unclosed"),
    ("<answer>twelve</answer>", "not_number"),
    ("<answer>12 apples</answer>", "not_number"),
    ("<answer>12.5</answer>", "not_number"),
    ("<answer>1e3</answer>", "not_number"),
    ("<answer></answer>", "not_number"),
    ("<answer>12,34</answer>", "not_number"),
    ("<answer><answer>12</answer></answer>", "not_number"),
    ("<answer>12</answer><answer>13</answer>", "conflict"),
    ("<answer>1234567890</answer>", "too_long"),
])
def test_invalid(text, status):
    p = parse(text)
    assert not p.valid and p.value is None and p.status == status


def test_grade_and_reward():
    good = " 3 * 4 = 12.\nFinal answer: <answer>12</answer>"
    assert grade(good, 12)["correct"]
    assert not grade(good, 13)["correct"]
    r = reward(good, 12)
    assert r["reward"] == pytest.approx(1.1) and r["correctness"] == 1 and r["format"] == 1
    r = reward(good, 13)  # well formed but wrong: only the format bonus
    assert r["reward"] == pytest.approx(0.1) and r["correctness"] == 0
    for bad in ["", "12", "<answer>12</answer><answer>11</answer>"]:
        r = reward(bad, 12)
        assert r["reward"] == 0 and r["correctness"] == 0 and r["format"] == 0
    assert reward(good, 12, w_format=0.0)["reward"] == 1.0


def test_length_is_not_rewarded():
    short = "<answer>12</answer>"
    long = "Let me think. " * 20 + "<answer>12</answer>"
    assert reward(short, 12)["reward"] == reward(long, 12)["reward"]
