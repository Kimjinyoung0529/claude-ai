"""작은 GPT (처음부터). 글자 단위. KV 캐시 생성 지원."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, past=None):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        sp = lambda t: t.view(B, T, self.h, D // self.h).transpose(1, 2)
        q, k, v = sp(q), sp(k), sp(v)
        if past is not None:
            k = torch.cat([past[0], k], 2); v = torch.cat([past[1], v], 2)
        causal = past is None
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        x = x + self.drop(self.proj(y.transpose(1, 2).reshape(B, T, D)))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x, (k, v)


class GPT(nn.Module):
    def __init__(self, vocab, d=320, layers=6, heads=4, ctx=640, drop=0.05):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads, drop) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, ids, past=None, start=0):
        B, T = ids.shape
        x = self.tok(ids) + self.pos(torch.arange(start, start + T, device=ids.device))
        new = []
        for i, b in enumerate(self.blocks):
            x, kv = b(x, None if past is None else past[i])
            new.append(kv)
        return self.head(self.ln(x)), new


def n_params(m):
    return sum(p.numel() for p in m.parameters())
