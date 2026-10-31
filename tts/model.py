"""소형 비자기회귀 TTS: 텍스트 인코더 + MAS 정렬 + 길이 조절기 + 디코더.

정렬(음소별 길이)은 외부 도구 없이 학습 중에 Monotonic Alignment Search(MAS)로 찾는다.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tts.audio import N_MELS


class PosEnc(nn.Module):
    def __init__(self, d, maxlen=4000):
        super().__init__()
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(maxlen).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class FFTBlock(nn.Module):
    def __init__(self, d, heads, ff, k=9, drop=0.1):
        super().__init__()
        self.att = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.conv = nn.Sequential(
            nn.Conv1d(d, ff, k, padding=k // 2), nn.ReLU(), nn.Dropout(drop),
            nn.Conv1d(ff, d, 1),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x, pad):
        a, _ = self.att(x, x, x, key_padding_mask=pad, need_weights=False)
        x = self.n1(x + self.drop(a))
        c = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.n2(x + self.drop(c))


class Stack(nn.Module):
    def __init__(self, d, layers, heads, ff):
        super().__init__()
        self.pos = PosEnc(d)
        self.blocks = nn.ModuleList(FFTBlock(d, heads, ff) for _ in range(layers))

    def forward(self, x, pad):
        x = self.pos(x)
        for b in self.blocks:
            x = b(x, pad)
        return x.masked_fill(pad.unsqueeze(-1), 0.0)


@torch.no_grad()
def mas(logp: torch.Tensor, tmask: torch.Tensor, fmask: torch.Tensor) -> torch.Tensor:
    """Monotonic alignment search. logp: (B, T, F). 반환: 정수 길이 (B, T)."""
    B, T, Fm = logp.shape
    out = torch.zeros(B, T)
    lp = logp.cpu().numpy()
    for b in range(B):
        t = int(tmask[b].sum())
        f = int(fmask[b].sum())
        v = lp[b, :t, :f]
        Q = np.full((t, f), -1e9, np.float32)
        Q[0, 0] = v[0, 0]
        for j in range(1, f):
            stay = Q[:, j - 1]
            move = np.concatenate([[-1e9], Q[:-1, j - 1]])
            Q[:, j] = np.maximum(stay, move) + v[:, j]
        # backtrack
        i = t - 1
        dur = np.zeros(t, np.int64)
        for j in range(f - 1, -1, -1):
            dur[i] += 1
            if i > 0 and j > 0 and Q[i - 1, j - 1] >= Q[i, j - 1]:
                i -= 1
        out[b, :t] = torch.from_numpy(dur)
    return out


def length_regulate(h: torch.Tensor, dur: torch.Tensor, max_len: int):
    """h: (B,T,D), dur: (B,T) 정수 -> (B,max_len,D)."""
    B, T, D = h.shape
    out = h.new_zeros(B, max_len, D)
    for b in range(B):
        e = torch.repeat_interleave(h[b], dur[b].long().clamp(min=0), dim=0)[:max_len]
        out[b, : e.size(0)] = e
    return out


class TTS(nn.Module):
    def __init__(self, n_vocab, d=192, enc_layers=4, dec_layers=4, heads=2, ff=768):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, d, padding_idx=0)
        self.enc = Stack(d, enc_layers, heads, ff)
        self.prior = nn.Linear(d, N_MELS)             # MAS 용 프라이어 평균
        self.dur = nn.Sequential(
            nn.Conv1d(d, 128, 3, padding=1), nn.ReLU(), nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(), nn.Conv1d(128, 1, 1),
        )
        self.dec = Stack(d, dec_layers, heads, ff)
        self.out = nn.Linear(d, N_MELS)

    def encode(self, ids):
        pad = ids.eq(0)
        h = self.enc(self.emb(ids) * math.sqrt(self.emb.embedding_dim), pad)
        logd = self.dur(h.transpose(1, 2)).squeeze(1)
        return h, pad, logd

    def forward(self, ids, mel, mel_len):
        h, pad, logd = self.encode(ids)
        B, T = ids.shape
        Fm = mel.size(1)
        fpad = torch.arange(Fm)[None] >= mel_len[:, None]
        # MAS: 프라이어 평균과 멜의 가우시안 로그우도
        mu = self.prior(h)
        with torch.no_grad():
            m = mel.clone()
            d2 = (mu.detach().pow(2).sum(-1, keepdim=True)
                  - 2 * mu.detach() @ m.transpose(1, 2) + m.pow(2).sum(-1).unsqueeze(1))
            dur = mas(-0.5 * d2, ~pad, ~fpad).to(ids.device)
        # 프라이어 손실 (정렬된 평균이 멜에 가까워지도록)
        mu_up = length_regulate(mu, dur, Fm)
        prior_loss = ((mu_up - mel).pow(2).mean(-1).masked_fill(fpad, 0).sum() / (~fpad).sum())
        # 길이 손실
        dloss = ((logd - torch.log(dur + 1)).pow(2).masked_fill(pad, 0).sum() / (~pad).sum())
        x = length_regulate(h, dur, Fm)
        y = self.out(self.dec(x, fpad))
        mel_loss = ((y - mel).abs().mean(-1).masked_fill(fpad, 0).sum() / (~fpad).sum())
        return mel_loss, prior_loss, dloss

    @torch.no_grad()
    def infer(self, ids, speed=1.0):
        h, pad, logd = self.encode(ids)
        dur = (torch.exp(logd) - 1).clamp(min=1).mul(1.0 / speed).round().long()
        dur = dur.masked_fill(pad, 0)
        L = int(dur.sum(1).max())
        x = length_regulate(h, dur, L)
        fpad = torch.arange(L)[None] >= dur.sum(1)[:, None]
        return self.out(self.dec(x, fpad))
