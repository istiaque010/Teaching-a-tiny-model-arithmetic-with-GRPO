"""The policy: Project 1's GPT with padding-aware and key/value-cached forward passes (Phases 1 and 7).

Project 1's own `GPT.forward` takes one unpadded batch and recomputes the whole sequence for every new token. GRPO
needs (a) many responses of different lengths in one batch and (b) fast sampling, so this module re-uses the *same
weights* through two extra forward functions:

  forward_padded(model, ids, attn)      full sequences, any mix of left/right padding (training, scoring)
  sample(model, prompts, ...)           batched sampling with a key/value cache (rollouts, evaluation)

Both are tested to give the same logits as `GPT.forward` on unpadded input (tests/test_policy.py).
Positions count real tokens only, so a left-padded prompt sees exactly the positions it would see unpadded.
Dropout is switched off for all post-training: the sampling policy and the policy being optimised are then the same
function, so the probability ratio is exactly 1 before the first update.
"""
from __future__ import annotations

import math
import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import BASE_CKPT, TOKENIZER, sha256

BASE_SHA256 = None  # filled at first load and stored in results/metrics/env.json


def no_dropout(model: nn.Module) -> nn.Module:
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    model.cfg.dropout = 0.0
    return model


def load_tokenizer():
    from src.tokenizer import StoryTokenizer

    return StoryTokenizer(TOKENIZER)


def load_policy(path: str = BASE_CKPT, device: str = "cpu"):
    """Project 1 checkpoint or a Project 8 checkpoint (same format). Returns (model, ckpt dict)."""
    from src.model import load_checkpoint

    model, ckpt = load_checkpoint(path, device)
    return no_dropout(model).eval(), ckpt


def save_policy(model, path: str, **meta) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(), "model_config": model.cfg.to_dict(), "tokenizer": TOKENIZER,
                "meta": meta}, path)
    return path


def frozen_copy(model):
    import copy

    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def _attend(attn_mod, x, mask, cache=None, start=None):
    """Causal self-attention of Project 1's CausalSelfAttention weights. mask: (B,1,Tq,Tk) bool, True = may attend."""
    B, T, C = x.shape
    h, hd = attn_mod.n_head, attn_mod.head_dim
    q, k, v = attn_mod.qkv(x).split(C, dim=2)
    q = q.view(B, T, h, hd).transpose(1, 2)
    k = k.view(B, T, h, hd).transpose(1, 2)
    v = v.view(B, T, h, hd).transpose(1, 2)
    if cache is not None:
        K, V = cache
        K[:, :, start:start + T] = k
        V[:, :, start:start + T] = v
        k, v = K[:, :, :start + T], V[:, :, :start + T]
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return attn_mod.proj(y.transpose(1, 2).contiguous().view(B, T, C))


def _blocks(model, x, mask, caches=None, start=None):
    for i, blk in enumerate(model.blocks):
        x = x + _attend(blk.attn, blk.ln1(x), mask, None if caches is None else caches[i], start)
        x = x + blk.mlp(blk.ln2(x))
    return model.head(model.ln_f(x))


def forward_padded(model, ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    """ids, attn: (B, T); attn True on real tokens. Returns logits (B, T, V). Pad positions give junk (ignored)."""
    B, T = ids.shape
    pos = (attn.long().cumsum(1) - 1).clamp(min=0)
    x = model.tok_emb(ids) + model.pos_emb(pos)
    causal = torch.ones(T, T, dtype=torch.bool, device=ids.device).tril()
    mask = causal[None] & attn[:, None, :]
    mask = mask | torch.eye(T, dtype=torch.bool, device=ids.device)[None]  # pad rows must see something
    return _blocks(model, x, mask[:, None])


def pack(prompts: List[List[int]], responses: List[List[int]], pad_id: int, device="cpu"):
    """[pad.. prompt response ..pad] for each row. Returns ids, attn, resp_mask (True where the *target* token at
    t+1 is a response token, aligned with logits[:, t])."""
    P = max(len(p) for p in prompts)
    R = max(max((len(r) for r in responses), default=1), 1)
    B = len(prompts)
    ids = torch.full((B, P + R), pad_id, dtype=torch.long)
    attn = torch.zeros(B, P + R, dtype=torch.bool)
    resp = torch.zeros(B, P + R, dtype=torch.bool)
    for i, (p, r) in enumerate(zip(prompts, responses)):
        s = P - len(p)
        ids[i, s:P] = torch.tensor(p)
        attn[i, s:P] = True
        if r:
            ids[i, P:P + len(r)] = torch.tensor(r)
            attn[i, P:P + len(r)] = True
            resp[i, P - 1:P - 1 + len(r)] = True
    return ids.to(device), attn.to(device), resp.to(device)


def token_logprobs(model, ids, attn, resp, temperature: float = 1.0, with_entropy: bool = False):
    """Log-probability of each next token under logits / temperature. Returns (logp (B, T-1... aligned to t),
    entropy or None); only positions with resp=True are meaningful."""
    logits = forward_padded(model, ids, attn)[:, :-1] / temperature
    logp_all = F.log_softmax(logits.float(), dim=-1)
    tgt = ids[:, 1:]
    logp = logp_all.gather(-1, tgt[..., None]).squeeze(-1)
    ent = None
    if with_entropy:
        ent = -(logp_all.exp() * logp_all).sum(-1)
    return logp, ent, resp[:, :-1]


@torch.no_grad()
def sample(model, prompts: List[List[int]], max_new: int, eos_id: int, pad_id: int, temperature: float = 1.0,
           greedy: bool = False, generator: Optional[torch.Generator] = None):
    """Batched sampling with a key/value cache. Returns (responses, logprobs): per row the generated token ids
    (ending with <EOS> if one was produced) and the log-probability of each under logits / temperature."""
    model.eval()
    dev = model.tok_emb.weight.device
    B = len(prompts)
    P = max(len(p) for p in prompts)
    L = P + max_new
    assert L <= model.cfg.block_size, f"prompt {P} + {max_new} new tokens > block_size"
    h, hd = model.blocks[0].attn.n_head, model.blocks[0].attn.head_dim
    caches = [(torch.zeros(B, h, L, hd, device=dev), torch.zeros(B, h, L, hd, device=dev)) for _ in model.blocks]
    ids = torch.full((B, P), pad_id, dtype=torch.long, device=dev)
    keymask = torch.zeros(B, L, dtype=torch.bool, device=dev)
    lens = torch.tensor([len(p) for p in prompts], device=dev)
    for i, p in enumerate(prompts):
        ids[i, P - len(p):] = torch.tensor(p, device=dev)
        keymask[i, P - len(p):P] = True
    pos = (keymask[:, :P].long().cumsum(1) - 1).clamp(min=0)
    causal = torch.ones(P, P, dtype=torch.bool, device=dev).tril()
    mask = (causal[None] & keymask[:, None, :P]) | torch.eye(P, dtype=torch.bool, device=dev)[None]
    logits = _blocks(model, model.tok_emb(ids) + model.pos_emb(pos), mask[:, None], caches, 0)[:, -1]
    out = torch.full((B, max_new), pad_id, dtype=torch.long, device=dev)
    lps = torch.zeros(B, max_new, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    n = torch.zeros(B, dtype=torch.long, device=dev)  # tokens generated per row (incl. EOS)
    for t in range(max_new):
        logp = F.log_softmax(logits.float() / temperature, dim=-1)
        if greedy:
            nxt = logp.argmax(-1)
        else:
            nxt = torch.multinomial(logp.exp(), 1, generator=generator).squeeze(-1)
        nxt = torch.where(done, torch.full_like(nxt, pad_id), nxt)
        out[:, t] = nxt
        lps[:, t] = logp.gather(-1, nxt[:, None]).squeeze(-1)
        n = n + (~done).long()
        done = done | (nxt == eos_id)
        if done.all() or t == max_new - 1:
            break
        keymask[:, P + t] = True  # the new token is a key for later steps (finished rows are ignored anyway)
        pos_t = (lens + t)[:, None]
        mask_t = keymask[:, None, None, :P + t + 1]
        x = model.tok_emb(nxt[:, None]) + model.pos_emb(pos_t)
        logits = _blocks(model, x, mask_t, caches, P + t)[:, -1]
    resp, lp = [], []
    for i in range(B):
        k = int(n[i])
        resp.append(out[i, :k].tolist())
        lp.append(lps[i, :k].tolist())
    return resp, lp


def decode_response(tok, resp: List[int]) -> str:
    if resp and resp[-1] == tok.eos_id:
        resp = resp[:-1]
    return tok.decode(resp)
