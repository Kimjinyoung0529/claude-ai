"""소형 비자기회귀 TTS (FastPitch 스타일 학습형 정렬기).

- Aligner: 텍스트/멜 임베딩 코사인 유사도 + 대각선 유도 사전분포(beta-binomial)
  + forward-sum(CTC) 손실로 정렬을 스스로 학습, MAS로 하드 정렬(음소별 길이) 추출
- 인코더 -> 길이 조절기 -> 디코더 -> 멜
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import betabinom

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
    B, T, _ = logp.shape
    out = torch.zeros(B, T)
    lp = logp.cpu().numpy()
    for b in range(B):
        t, f = int(tmask[b].sum()), int(fmask[b].sum())
        v = lp[b, :t, :f]
        Q = np.full((t, f), -1e9, np.float32)
        Q[0, 0] = v[0, 0]
        for j in range(1, f):
            move = np.concatenate([[-1e9], Q[:-1, j - 1]])
            Q[:, j] = np.maximum(Q[:, j - 1], move) + v[:, j]
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


def beta_prior(tlen, flen, Tmax, Fmax, scale=1.0):
    """대각선 근처에 확률을 몰아주는 log-prior (B, F, T)."""
    out = torch.full((len(tlen), Fmax, Tmax), -1e4)
    for b, (t, f) in enumerate(zip(tlen.tolist(), flen.tolist())):
        i = np.arange(1, f + 1)[:, None]
        x = np.arange(t)[None, :]
        lp = betabinom.logpmf(x, t - 1, scale * i, scale * (f + 1 - i))
        out[b, :f, :t] = torch.from_numpy(np.clip(lp, -1e4, None)).float()
    return out


def hard_attn(dur, Fm):
    cum = dur.cumsum(1)
    start = cum - dur
    f = torch.arange(Fm, device=dur.device)[None, :, None]
    return ((f >= start[:, None, :]) & (f < cum[:, None, :])).float()


class Aligner(nn.Module):
    def __init__(self, n_vocab, c=128, temp=15.0):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, c, padding_idx=0)
        self.key = nn.Sequential(nn.Conv1d(c, 2 * c, 3, padding=1), nn.ReLU(), nn.Conv1d(2 * c, c, 1))
        self.query = nn.Sequential(
            nn.Conv1d(N_MELS, 2 * N_MELS, 3, padding=1), nn.ReLU(),
            nn.Conv1d(2 * N_MELS, N_MELS, 1), nn.ReLU(), nn.Conv1d(N_MELS, c, 1),
        )
        self.temp = temp

    def forward(self, ids, mel, tpad, log_prior):
        k = F.normalize(self.key(self.emb(ids).transpose(1, 2)), dim=1)
        q = F.normalize(self.query(mel.transpose(1, 2)), dim=1)
        logits = self.temp * torch.einsum("bcf,bct->bft", q, k)
        logits = logits.masked_fill(tpad[:, None, :], -1e4)
        return F.log_softmax(logits, dim=-1) + log_prior  # (B, F, T)


def forward_sum(logp, tlen, flen, blank=-1.0):
    B, Fm, T = logp.shape
    lp = F.pad(logp, (1, 0), value=blank)
    lp = F.log_softmax(lp, dim=-1).transpose(0, 1)  # (F, B, T+1)
    tgt = torch.arange(1, T + 1)[None].expand(B, -1)
    loss = F.ctc_loss(lp, tgt, flen, tlen, blank=0, reduction="sum", zero_infinity=True)
    return loss / tlen.sum()


class TTS(nn.Module):
    def __init__(self, n_vocab, d=192, enc_layers=4, dec_layers=4, heads=2, ff=768, n_spk=1):
        super().__init__()
        self.spk = nn.Embedding(n_spk, d)
        self.emb = nn.Embedding(n_vocab, d, padding_idx=0)
        self.enc = Stack(d, enc_layers, heads, ff)
        self.aligner = Aligner(n_vocab)
        self.dur = nn.Sequential(
            nn.Conv1d(d, 128, 3, padding=1), nn.ReLU(), nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(), nn.Conv1d(128, 1, 1),
        )
        self.dec = Stack(d, dec_layers, heads, ff)
        self.out = nn.Linear(d, N_MELS)

    def encode(self, ids, spk=None):
        pad = ids.eq(0)
        h = self.enc(self.emb(ids) * math.sqrt(self.emb.embedding_dim), pad)
        if spk is not None:
            h = (h + self.spk(spk)[:, None, :]).masked_fill(pad.unsqueeze(-1), 0.0)
        logd = self.dur(h.transpose(1, 2)).squeeze(1)
        return h, pad, logd

    def forward(self, ids, mel, mel_len, spk=None):
        """반환: dict(mel, dur, fs, bin, spread). spread=가장 긴 음소가 차지하는 프레임 비율(붕괴 감시)."""
        h, pad, logd = self.encode(ids, spk)
        B, T = ids.shape
        Fm = mel.size(1)
        fpad = torch.arange(Fm)[None] >= mel_len[:, None]
        tlen = (~pad).sum(1)

        logp = self.aligner(ids, mel, pad, beta_prior(tlen, mel_len, T, Fm))
        fs = forward_sum(logp, tlen, mel_len)
        dur = mas(logp.detach().transpose(1, 2), ~pad, ~fpad).to(ids.device)
        hard = hard_attn(dur, Fm)
        soft = torch.softmax(logp, dim=-1)
        bin_loss = -(torch.log(soft.clamp(min=1e-8)) * hard).sum() / hard.sum()

        dloss = ((logd - torch.log(dur + 1)).pow(2).masked_fill(pad, 0).sum() / (~pad).sum())
        x = length_regulate(h, dur, Fm)
        y = self.out(self.dec(x, fpad))
        mel_loss = ((y - mel).abs().mean(-1).masked_fill(fpad, 0).sum() / (~fpad).sum())
        spread = (dur.max(1).values / mel_len).mean()
        return {"mel": mel_loss, "dur": dloss, "fs": fs, "bin": bin_loss, "spread": spread}

    @torch.no_grad()
    def infer(self, ids, spk=None, speed=1.0):
        h, pad, logd = self.encode(ids, spk)
        dur = (torch.exp(logd) - 1).clamp(min=1).mul(1.0 / speed).round().long()
        dur = dur.masked_fill(pad, 0)
        L = int(dur.sum(1).max())
        x = length_regulate(h, dur, L)
        fpad = torch.arange(L)[None] >= dur.sum(1)[:, None]
        return self.out(self.dec(x, fpad))
