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
        # --- v2: 토큰별 피치/에너지 예측기 (FastPitch 방식). 임베딩은 0 초기화 -> 처음엔 v1 과 동일하게 동작
        def _pred():
            return nn.Sequential(nn.Conv1d(d, 128, 3, padding=1), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Conv1d(128, 1, 1))
        self.pitch_pred, self.energy_pred = _pred(), _pred()
        self.pitch_emb, self.energy_emb = nn.Conv1d(1, d, 3, padding=1), nn.Conv1d(1, d, 3, padding=1)
        for m in (self.pitch_emb, self.energy_emb):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        # 정규화 통계: [평균, 표준편차]  (log f0 / 프레임 에너지 / 멜)
        self.register_buffer("lf0_ms", torch.tensor([5.4, 0.3]))
        self.register_buffer("e_ms", torch.tensor([0.0, 1.0]))
        self.register_buffer("mel_ms", torch.tensor([0.0, 1.0]))

    def set_stats(self, lf0_mean, lf0_std, e_mean, e_std, mel_mean, mel_std):
        self.lf0_ms.copy_(torch.tensor([lf0_mean, lf0_std]))
        self.e_ms.copy_(torch.tensor([e_mean, e_std]))
        self.mel_ms.copy_(torch.tensor([mel_mean, mel_std]))

    def predict_var(self, h):
        return self.pitch_pred(h.transpose(1, 2)).squeeze(1), self.energy_pred(h.transpose(1, 2)).squeeze(1)

    def add_var(self, h, pad, p, e):
        h = h + self.pitch_emb(p.unsqueeze(1)).transpose(1, 2) + self.energy_emb(e.unsqueeze(1)).transpose(1, 2)
        return h.masked_fill(pad.unsqueeze(-1), 0.0)

    def encode(self, ids, spk=None):
        pad = ids.eq(0)
        h = self.enc(self.emb(ids) * math.sqrt(self.emb.embedding_dim), pad)
        if spk is not None:
            h = (h + self.spk(spk)[:, None, :]).masked_fill(pad.unsqueeze(-1), 0.0)
        logd = self.dur(h.transpose(1, 2)).squeeze(1)
        return h, pad, logd

    def forward(self, ids, mel, mel_len, spk=None, f0=None, extra=False):
        """반환: dict(mel, dur, fs, bin, spread[, pitch, energy]). spread=가장 긴 음소가 차지하는 프레임 비율.
        f0(B,F; Hz, 0=무성)를 주면 토큰별 피치/에너지 목표로 교사강요 + 예측기 손실을 계산한다."""
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
        out = {}
        if f0 is not None:
            vm = (f0 > 0).float().masked_fill(fpad, 0.0)
            lf0 = (torch.log(f0.clamp(min=1.0)) - self.lf0_ms[0]) / self.lf0_ms[1]
            en = (torch.logsumexp(mel * self.mel_ms[1] + self.mel_ms[0], dim=-1) - self.e_ms[0]) / self.e_ms[1]
            hT = hard.transpose(1, 2)                                   # (B, T, F)
            cnt = hT.sum(-1).clamp(min=1)
            v_cnt = (hT @ vm.unsqueeze(-1)).squeeze(-1)
            p_tok = (hT @ (lf0 * vm).unsqueeze(-1)).squeeze(-1) / v_cnt.clamp(min=1) * (v_cnt > 0).float()
            e_tok = (hT @ en.masked_fill(fpad, 0.0).unsqueeze(-1)).squeeze(-1) / cnt
            p_hat, e_hat = self.predict_var(h)
            ntok = (~pad).sum()
            out["pitch"] = ((p_hat - p_tok).pow(2).masked_fill(pad, 0).sum() / ntok)
            out["energy"] = ((e_hat - e_tok).pow(2).masked_fill(pad, 0).sum() / ntok)
            if extra:
                out["p_hat"], out["p_tok"], out["pad"] = p_hat.detach(), p_tok.detach(), pad
            h = self.add_var(h, pad, p_tok, e_tok)
        x = length_regulate(h, dur, Fm)
        y = self.out(self.dec(x, fpad))
        mel_loss = ((y - mel).abs().mean(-1).masked_fill(fpad, 0).sum() / (~fpad).sum())
        spread = (dur.max(1).values / mel_len).mean()
        out.update({"mel": mel_loss, "dur": dloss, "fs": fs, "bin": bin_loss, "spread": spread})
        return out

    @torch.no_grad()
    def infer(self, ids, spk=None, speed=1.0, dur_jitter=0.0, var_scale=1.0):
        """dur_jitter: 길이에 곱하는 로그정규 잡음의 표준편차(박자 변화). var_scale: 예측 피치/에너지 편차 배율(억양 강조)."""
        h, pad, logd = self.encode(ids, spk)
        p_hat, e_hat = self.predict_var(h)
        h = self.add_var(h, pad, p_hat * var_scale, e_hat * var_scale)
        dur = (torch.exp(logd) - 1).clamp(min=1)
        if dur_jitter > 0:
            dur = dur * torch.exp(torch.randn_like(dur) * dur_jitter)
        dur = dur.mul(1.0 / speed).round().long().clamp(min=1)
        dur = dur.masked_fill(pad, 0)
        L = int(dur.sum(1).max())
        x = length_regulate(h, dur, L)
        fpad = torch.arange(L)[None] >= dur.sum(1)[:, None]
        return self.out(self.dec(x, fpad))
