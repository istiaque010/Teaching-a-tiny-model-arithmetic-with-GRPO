"""Synthetic tasks (Phase 2): answers are right, splits are deterministic and never share a computation."""
import random
import re

import grpo  # noqa: F401
from grpo import tasks
from grpo.answer_parser import grade
from grpo.prompts import target_text


def _check_solution(p):
    last = None
    for a, op, b, r in re.findall(r"(\d+) ([+\-*]) (\d+) = (\d+)\.", p.solution):
        a, b, r = int(a), int(b), int(r)
        assert r == {"+": a + b, "-": a - b, "*": a * b}[op]
        last = r
    assert last == p.answer and p.answer >= 0
    assert len(re.findall(r" = ", p.solution)) == p.steps


def test_every_family_answers_are_correct():
    rng = random.Random(0)
    for fam in tasks.FAMILIES:
        for _ in range(300):
            p = tasks.sample(rng, fam)
            _check_solution(p)
            assert p.difficulty == ("easy" if p.steps == 1 else "multi")


def test_gold_target_grades_correct():
    rng = random.Random(1)
    for _ in range(100):
        p = tasks.sample(rng)
        assert grade(target_text(p.solution, p.answer), p.answer)["correct"]


def test_split_is_a_function_of_chain_and_commutative():
    assert tasks.split_of("3*4=12") == tasks.split_of("3*4=12")
    p1 = tasks.Problem("mul", 1, "easy", "7 * 16 = ?", "7 * 16 = 112.", 112, tasks._canon(7, "*", 16))
    p2 = tasks.Problem("mul", 1, "easy", "16 * 7 = ?", "16 * 7 = 112.", 112, tasks._canon(16, "*", 7))
    assert p1.chain == p2.chain
    assert tasks._canon(9, "-", 4) != tasks._canon(4, "-", 9)


def test_build_split_deterministic_unique_and_disjoint():
    a = tasks.build_split("val", 200, seed=5)
    b = tasks.build_split("val", 200, seed=5)
    assert [p.question for p in a] == [p.question for p in b]
    assert len({p.question for p in a}) == 200
    t = tasks.build_split("test", 200, seed=6)
    assert not {p.chain for p in a} & {p.chain for p in t}
    assert all(p.split == "val" for p in a) and all(p.split == "test" for p in t)
