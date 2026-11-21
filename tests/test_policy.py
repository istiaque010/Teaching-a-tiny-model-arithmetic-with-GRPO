"""The padded and cached forward passes give exactly Project 1's model outputs (Phases 1 and 7)."""
import random

import pytest
import torch

import grpo  # noqa: F401
from grpo import policy as P
from grpo import tasks
from grpo.prompts import encode_prompt, encode_target, prompt_text, target_text


@pytest.fixture(scope="module")
def mt():
    torch.manual_seed(0)
    from src.model import GPT, GPTConfig

    m = P.no_dropout(GPT(GPTConfig(vocab_size=8192, n_layer=2, n_head=2, n_embd=64, block_size=256))).eval()
    return m, P.load_tokenizer()


def _batch(tok, n=6):
    qs = [tasks.sample(random.Random(i)).question for i in range(n)]
    ps = [encode_prompt(tok, q) for q in qs]
    rs = [encode_target(tok, "12 + 30 = 42.", 42)[:k] for k in [3, 9, 1, 12, 5, 7][:n]]
    return ps, rs


def test_prompt_target_tokenize_like_one_string(mt):
    _, tok = mt
    q = "Mia has 3 bags with 4 apples in each. How many apples does Mia have?"
    joint = tok.encode(prompt_text(q) + target_text("3 * 4 = 12.", 12), add_bos=True) + [tok.eos_id]
    assert encode_prompt(tok, q) + encode_target(tok, "3 * 4 = 12.", 12) == joint


def test_padded_forward_matches_project1(mt):
    m, tok = mt
    ps, rs = _batch(tok)
    ids, attn, _ = P.pack(ps, rs, tok.pad_id)
    lg = P.forward_padded(m, ids, attn)
    Pm = max(len(p) for p in ps)
    for i, (p, r) in enumerate(zip(ps, rs)):
        ref, _ = m(torch.tensor([p + r]))
        st = Pm - len(p)
        assert torch.allclose(lg[i, st:st + len(p) + len(r)], ref[0], atol=1e-4)


def test_cached_greedy_matches_naive(mt):
    m, tok = mt
    from src.generate import generate

    ps, _ = _batch(tok)
    R, _ = P.sample(m, ps, 20, tok.eos_id, tok.pad_id, greedy=True)
    for p, r in zip(ps, R):
        out, _ = generate(m, torch.tensor([p]), len(r), method="greedy")
        assert out[0, len(p):].tolist() == r


def test_rollout_logprobs_match_recompute(mt):
    m, tok = mt
    ps, _ = _batch(tok)
    g = torch.Generator().manual_seed(1)
    for temp in (1.0, 0.7):
        R, L = P.sample(m, ps, 20, tok.eos_id, tok.pad_id, temperature=temp, generator=g)
        ids, attn, resp = P.pack(ps, R, tok.pad_id)
        lp, _, rm = P.token_logprobs(m, ids, attn, resp, temperature=temp)
        flat = torch.cat([torch.tensor(x) for x in L])
        assert torch.allclose(lp[rm], flat, atol=1e-4)


def test_response_mask_covers_only_response(mt):
    _, tok = mt
    ps, rs = _batch(tok)
    ids, attn, resp = P.pack(ps, rs, tok.pad_id)
    assert resp.sum().item() == sum(len(r) for r in rs)
    for i, r in enumerate(rs):
        tgt = ids[i, 1:][resp[i, :-1]].tolist()
        assert tgt == r


def test_eos_stops_a_row(mt):
    m, tok = mt
    ps, _ = _batch(tok, 2)
    R, L = P.sample(m, ps, 30, tok.eos_id, tok.pad_id, generator=torch.Generator().manual_seed(0))
    for r, l in zip(R, L):
        assert len(r) == len(l)
        assert tok.eos_id not in r[:-1]
