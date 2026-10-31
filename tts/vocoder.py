"""경량 HiFi-GAN (CPU 학습용). 입력: 정규화된 로그멜 (B, 80, T) -> 파형 (B, 1, T*256).

생성기는 pre-emphasis 된 파형을 만든다(추론 시 deemphasis).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from tts.audio import N_MELS

UP_RATES = (8, 8, 2, 2)  # 곱 = 256 = HOP
UP_KERNELS = (16, 16, 4, 4)
RB_KERNELS = (3, 7, 11)
RB_DILATIONS = (1, 3, 5)


class ResBlock(nn.Module):
    def __init__(self, ch, k):
        super().__init__()
        self.c1 = nn.ModuleList(
            weight_norm(nn.Conv1d(ch, ch, k, 1, dilation=d, padding=(k - 1) * d // 2)) for d in RB_DILATIONS)
        self.c2 = nn.ModuleList(
            weight_norm(nn.Conv1d(ch, ch, k, 1, padding=(k - 1) // 2)) for _ in RB_DILATIONS)

    def forward(self, x):
        for a, b in zip(self.c1, self.c2):
            x = x + b(F.leaky_relu(a(F.leaky_relu(x, 0.1)), 0.1))
        return x


class Generator(nn.Module):
    def __init__(self, ch=256):
        super().__init__()
        self.pre = weight_norm(nn.Conv1d(N_MELS, ch, 7, 1, padding=3))
        self.ups, self.rbs = nn.ModuleList(), nn.ModuleList()
        for i, (u, k) in enumerate(zip(UP_RATES, UP_KERNELS)):
            cin, cout = ch // 2 ** i, ch // 2 ** (i + 1)
            self.ups.append(weight_norm(nn.ConvTranspose1d(cin, cout, k, u, padding=(k - u) // 2)))
            self.rbs.append(nn.ModuleList(ResBlock(cout, rk) for rk in RB_KERNELS))
        self.post = weight_norm(nn.Conv1d(ch // 2 ** len(UP_RATES), 1, 7, 1, padding=3))

    def forward(self, mel):
        x = self.pre(mel)
        for up, rbs in zip(self.ups, self.rbs):
            x = up(F.leaky_relu(x, 0.1))
            x = sum(rb(x) for rb in rbs) / len(rbs)
        return torch.tanh(self.post(F.leaky_relu(x)))


class PeriodD(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.p = period
        chs = (1, 16, 32, 64, 128)
        self.convs = nn.ModuleList(
            weight_norm(nn.Conv2d(chs[i], chs[i + 1], (5, 1), (3, 1), padding=(2, 0))) for i in range(4))
        self.convs.append(weight_norm(nn.Conv2d(128, 128, (5, 1), 1, padding=(2, 0))))
        self.post = weight_norm(nn.Conv2d(128, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        B, C, T = x.shape
        if T % self.p:
            x = F.pad(x, (0, self.p - T % self.p), mode="reflect")
        x = x.view(B, C, -1, self.p)
        fm = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            fm.append(x)
        x = self.post(x)
        fm.append(x)
        return x.flatten(1), fm


class ScaleD(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = [(1, 16, 15, 1, 7, 1), (16, 32, 41, 4, 20, 4), (32, 64, 41, 4, 20, 16),
               (64, 128, 41, 4, 20, 16), (128, 128, 5, 1, 2, 1)]
        self.convs = nn.ModuleList(
            weight_norm(nn.Conv1d(i, o, k, s, padding=p, groups=g)) for i, o, k, s, p, g in cfg)
        self.post = weight_norm(nn.Conv1d(128, 1, 3, 1, padding=1))

    def forward(self, x):
        fm = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            fm.append(x)
        x = self.post(x)
        fm.append(x)
        return x.flatten(1), fm


class Discriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.mpd = nn.ModuleList(PeriodD(p) for p in periods)
        self.msd = nn.ModuleList([ScaleD(), ScaleD()])

    def forward(self, x):
        outs, fms = [], []
        for d in self.mpd:
            o, f = d(x)
            outs.append(o); fms.append(f)
        for i, d in enumerate(self.msd):
            o, f = d(x if i == 0 else F.avg_pool1d(x, 4, 2, padding=2))
            outs.append(o); fms.append(f)
        return outs, fms
