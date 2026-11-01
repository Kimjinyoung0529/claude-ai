"""호출어 감지기: 로그멜 특징 + 작은 시간 합성곱 네트워크(TC-ResNet 계열). 실시간 CPU용.

클래스: 0 = 부정, 1 = hey("헤이 하나"), 2 = ya("하나야"), 3 = ssi("하나씨")
입력: 1.5초 파형 -> 10ms 간격 40채널 로그멜 150프레임.
"""
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kws import config as C

N_FFT, HOP, N_MELS = 512, 160, 40
CLASSES = ["neg", "hey", "ya", "ssi"]
_FB = None
_WIN = None


def mel_fb():
    global _FB, _WIN
    if _FB is None:
        _FB = torch.from_numpy(librosa.filters.mel(sr=C.SR, n_fft=N_FFT, n_mels=N_MELS, fmin=60, fmax=7600)).float()
        _WIN = torch.hann_window(N_FFT)
    return _FB, _WIN


def features(wav):
    """wav: (B, T) float [-1,1] -> (B, N_MELS, frames) 로그멜 (대략 평균 0 표준편차 1)"""
    fb, win = mel_fb()
    s = torch.stft(wav, N_FFT, HOP, N_FFT, win, center=True, return_complex=True).abs() ** 2
    m = torch.log(torch.clamp(fb @ s, min=1e-6))
    return (m + 6.0) / 3.0


class Block(nn.Module):
    def __init__(self, cin, cout, stride, dil):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 9, stride, padding=4 * dil, dilation=dil, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 9, 1, padding=4, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.sk = nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout)) \
            if (cin != cout or stride != 1) else nn.Identity()

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + self.sk(x))


class KWS(nn.Module):
    def __init__(self, width=1.0, n_cls=len(CLASSES)):
        super().__init__()
        w = lambda c: int(c * width)
        self.stem = nn.Sequential(nn.Conv1d(N_MELS, w(64), 3, padding=1, bias=False), nn.BatchNorm1d(w(64)), nn.ReLU())
        self.blocks = nn.Sequential(Block(w(64), w(64), 1, 1), Block(w(64), w(96), 2, 1),
                                    Block(w(96), w(96), 1, 2), Block(w(96), w(128), 2, 1),
                                    Block(w(128), w(128), 1, 2))
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(w(128) * 2, n_cls)

    def forward(self, feat):                      # (B, N_MELS, T)
        h = self.blocks(self.stem(feat))
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=1)
        return self.head(self.drop(h))

    def from_wav(self, wav):
        return self(features(wav))


def n_params(m):
    return sum(p.numel() for p in m.parameters())
