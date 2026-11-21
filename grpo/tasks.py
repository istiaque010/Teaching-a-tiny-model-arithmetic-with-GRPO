"""Synthetic, verifiable arithmetic tasks (Phase 2).

Every problem is generated from integers, so its answer is known exactly. Five families, each with a difficulty tag:

    family       steps  difficulty  example
    add          1      easy        "47 + 38 = ?"            (addition or subtraction, numbers 1-99)
    mul          1      easy        "7 * 16 = ?"             (one factor 2-12, the other 2-99)
    multi        2      multi       "(12 + 8) * 3 = ?"       (two operations, brackets when needed)
    word1        1      easy        "Mia has 3 bags with 4 apples in each bag. How many apples does Mia have?"
    word2        2      multi       "... She eats 5 of them. How many apples are left?"

Splits never share a computation: the split is decided by a hash of the *chain* (e.g. "3*4=12;12-5=7"), so the same
arithmetic cannot appear in train and test even with different wording. All answers are whole numbers >= 0.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass

FAMILIES = ["add", "mul", "multi", "word1", "word2"]
WEIGHTS = [0.20, 0.15, 0.30, 0.15, 0.20]
SPLIT_RANGES = {"train": range(0, 88), "val": range(88, 92), "test": range(92, 100)}  # hash bucket 0-99

NAMES = [("Mia", "She"), ("Tom", "He"), ("Lily", "She"), ("Ben", "He"), ("Sam", "He"), ("Anna", "She"),
         ("Max", "He"), ("Lucy", "She"), ("Jack", "He"), ("Zoe", "She"), ("Tim", "He"), ("Sue", "She")]
ITEMS = ["apples", "balls", "cookies", "toys", "flowers", "stickers", "books", "cars", "shells", "stars",
         "rocks", "candies"]
BOXES = ["bags", "boxes", "baskets", "jars", "cups", "bowls"]


@dataclass
class Problem:
    family: str
    steps: int
    difficulty: str
    question: str
    solution: str      # the worked steps, e.g. "3 * 4 = 12. 12 - 5 = 7."
    answer: int
    chain: str         # canonical computation, decides the split
    split: str = ""
    id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _op(a: int, op: str, b: int) -> int:
    return a + b if op == "+" else a - b if op == "-" else a * b


def _eq(a: int, op: str, b: int) -> str:
    return f"{a} {op} {b} = {_op(a, op, b)}."


def _canon(a: int, op: str, b: int) -> str:
    """Canonical form of one step; + and * are commutative, so 7*16 and 16*7 are the same computation."""
    if op in "+*" and a > b:
        a, b = b, a
    return f"{a}{op}{b}={_op(a, op, b)}"


def _addsub(rng: random.Random):
    a, b = rng.randint(1, 99), rng.randint(1, 99)
    if rng.random() < 0.5:
        return a, "+", b
    a, b = max(a, b), min(a, b)
    return a, "-", b


def _mul(rng: random.Random):
    a, b = rng.randint(2, 12), rng.randint(2, 99)
    return (a, "*", b) if rng.random() < 0.5 else (b, "*", a)


def gen_add(rng):
    a, op, b = _addsub(rng)
    r = _op(a, op, b)
    return Problem("add", 1, "easy", f"{a} {op} {b} = ?", _eq(a, op, b), r, _canon(a, op, b))


def gen_mul(rng):
    a, op, b = _mul(rng)
    r = _op(a, op, b)
    return Problem("mul", 1, "easy", f"{a} {op} {b} = ?", _eq(a, op, b), r, _canon(a, op, b))


def gen_multi(rng):
    """Two operations. Shapes: (a o b) o c  or  a o (b o c) written with brackets, plus a*b+c style."""
    while True:
        shape = rng.randrange(3)
        if shape == 0:      # (a +- b) * c
            a, op1, b = _addsub(rng)
            a, b = (a % 30) + 1, (b % 30) + 1
            if op1 == "-" and a < b:
                a, b = b, a
            c = rng.randint(2, 9)
            r1 = _op(a, op1, b)
            q = f"({a} {op1} {b}) * {c} = ?"
            steps = [(a, op1, b), (r1, "*", c)]
        elif shape == 1:    # a * b +- c
            a, _, b = _mul(rng)
            c = rng.randint(1, 99)
            op2 = rng.choice("+-")
            r1 = a * b
            if op2 == "-" and c > r1:
                continue
            q = f"{a} * {b} {op2} {c} = ?"
            steps = [(a, "*", b), (r1, op2, c)]
        else:               # a +- b +- c
            a, op1, b = _addsub(rng)
            r1 = _op(a, op1, b)
            c = rng.randint(1, 99)
            op2 = rng.choice("+-")
            if op2 == "-" and c > r1:
                continue
            q = f"{a} {op1} {b} {op2} {c} = ?"
            steps = [(a, op1, b), (r1, op2, c)]
        r = _op(*steps[1])
        sol = " ".join(_eq(*s) for s in steps)
        chain = ";".join(_canon(*st) for st in steps)
        return Problem("multi", 2, "multi", q, sol, r, chain)


def _who(rng):
    (name, pron), (friend, _) = rng.sample(NAMES, 2)
    return name, pron, friend, rng.choice(ITEMS)


def gen_word1(rng):
    name, pron, friend, item = _who(rng)
    kind = rng.randrange(3)
    if kind == 0:
        a, b = rng.randint(1, 60), rng.randint(1, 40)
        q = f"{name} has {a} {item}. {pron} gets {b} more {item}. How many {item} does {name} have now?"
        s = (a, "+", b)
    elif kind == 1:
        a = rng.randint(2, 99)
        b = rng.randint(1, a)
        q = f"{name} has {a} {item}. {pron} gives {b} {item} to {friend}. How many {item} does {name} have now?"
        s = (a, "-", b)
    else:
        a, b = rng.randint(2, 9), rng.randint(2, 12)
        box = rng.choice(BOXES)
        q = f"{name} has {a} {box} with {b} {item} in each. How many {item} does {name} have?"
        s = (a, "*", b)
    r = _op(*s)
    return Problem("word1", 1, "easy", q, _eq(*s), r, _canon(*s))


def gen_word2(rng):
    while True:
        name, pron, friend, item = _who(rng)
        kind = rng.randrange(3)
        if kind == 0:       # boxes, then eats / gives some
            a, b = rng.randint(2, 9), rng.randint(2, 12)
            c = rng.randint(1, 20)
            box = rng.choice(BOXES)
            if c > a * b:
                continue
            q = (f"{name} has {a} {box} with {b} {item} in each. {pron} gives {c} {item} to {friend}. "
                 f"How many {item} does {name} have now?")
            steps = [(a, "*", b), (a * b, "-", c)]
        elif kind == 1:     # has some, gets more, gives some
            a, b = rng.randint(1, 60), rng.randint(1, 40)
            c = rng.randint(1, 60)
            if c > a + b:
                continue
            q = (f"{name} has {a} {item}. {pron} finds {b} more. Then {pron.lower()} gives {c} to {friend}. "
                 f"How many {item} does {name} have now?")
            steps = [(a, "+", b), (a + b, "-", c)]
        else:               # two friends combine, then a multiple of the total
            a, b = rng.randint(1, 40), rng.randint(1, 40)
            c = rng.randint(2, 5)
            q = (f"{name} has {a} {item} and {friend} has {b} {item}. Their mom gives them {c} times as many "
                 f"{item} as they have together. How many {item} does their mom give them?")
            steps = [(a, "+", b), (a + b, "*", c)]
        r = _op(*steps[1])
        sol = " ".join(_eq(*s) for s in steps)
        chain = ";".join(_canon(*st) for st in steps)
        return Problem("word2", 2, "multi", q, sol, r, chain)


GENERATORS = {"add": gen_add, "mul": gen_mul, "multi": gen_multi, "word1": gen_word1, "word2": gen_word2}


def bucket(chain: str) -> int:
    return int(hashlib.sha256(chain.encode()).hexdigest(), 16) % 100


def split_of(chain: str) -> str:
    b = bucket(chain)
    return next(s for s, r in SPLIT_RANGES.items() if b in r)


def sample(rng: random.Random, family: str | None = None) -> Problem:
    fam = family or rng.choices(FAMILIES, WEIGHTS)[0]
    p = GENERATORS[fam](rng)
    p.split = split_of(p.chain)
    return p


def build_split(split: str, n: int, seed: int, balanced: bool = False) -> list[Problem]:
    """n unique problems (by question text) of one split. balanced=True gives equal counts per family."""
    rng = random.Random(seed)
    seen, out = set(), []
    quota = {f: n // len(FAMILIES) for f in FAMILIES} if balanced else None
    while len(out) < n:
        fam = None
        if quota:
            left = [f for f in FAMILIES if quota[f] > 0]
            if not left:
                break
            fam = rng.choice(left)
        p = sample(rng, fam)
        if p.split != split or p.question in seen:
            continue
        seen.add(p.question)
        p.id = f"{split}-{len(out):06d}"
        out.append(p)
        if quota:
            quota[fam] -= 1
    return out
