"""HiFi-GAN 보코더 학습 (KSS 단일 화자, CPU).

사용: python -m tts.train_vocoder --minutes 300 --threads 2
- 멜: data.pkl 의 정규화 멜 (TTS 출력과 같은 공간), 파형: data/proc/kss_wav.pkl
- 손실: LSGAN + feature matching(x2) + 멜 L1(x45)
"""
import argparse
import os
import pickle
import random
import time

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from tts import audio
from tts.vocoder import Discriminator, Generator

SEG = 32  # 프레임 -> 8192 샘플
N_HOLD = 40


def make_mel_fn():
    fb = torch.from_numpy(librosa.filters.mel(sr=audio.SR, n_fft=audio.N_FFT, n_mels=audio.N_MELS,
                                              fmin=audio.FMIN, fmax=audio.FMAX)).float()
    win = torch.hann_window(audio.WIN)

    def fn(y):  # y: (B, 1, T), pre-emphasis 된 파형
        s = torch.stft(y.squeeze(1), audio.N_FFT, audio.HOP, audio.WIN, win, center=True,
                       pad_mode="reflect", return_complex=True).abs()
        return torch.log(torch.clamp(fb @ s, min=1e-5))
    return fn


def preemph(y):
    return np.append(y[0], y[1:] - 0.97 * y[:-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=120)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    d = pickle.load(open("data/proc/data.pkl", "rb"))
    mean, std = d["mean"], d["std"]
    mels = [x["mel"] for x in d["items"]]
    wavs = pickle.load(open("data/proc/kss_wav.pkl", "rb"))
    keep = [i for i in range(len(wavs) - N_HOLD) if len(mels[i]) > SEG + 2]
    print("utts", len(keep), flush=True)

    G, D = Generator(), Discriminator()
    print("G params(M)", sum(p.numel() for p in G.parameters()) / 1e6,
          "D params(M)", sum(p.numel() for p in D.parameters()) / 1e6, flush=True)
    oG = torch.optim.AdamW(G.parameters(), a.lr, betas=(0.8, 0.99))
    oD = torch.optim.AdamW(D.parameters(), a.lr, betas=(0.8, 0.99))
    mel_fn = make_mel_fn()
    os.makedirs("ckpt", exist_ok=True)
    step = 0
    if os.path.exists("ckpt/voc.pt"):
        s = torch.load("ckpt/voc.pt")
        G.load_state_dict(s["G"]); D.load_state_dict(s["D"])
        oG.load_state_dict(s["oG"]); oD.load_state_dict(s["oD"]); step = s["step"]
        print("resumed at", step, flush=True)

    def batch():
        mm, yy = [], []
        for i in random.sample(keep, a.batch):
            n = len(mels[i]) - SEG - 1
            f0 = random.randint(0, n)
            mm.append(mels[i][f0:f0 + SEG].astype(np.float32).T)
            seg = wavs[i][max(0, f0 * audio.HOP - 1): (f0 + SEG) * audio.HOP].astype(np.float32) / 32767
            seg = preemph(seg)[-SEG * audio.HOP:] if f0 > 0 else preemph(seg)
            yy.append(seg[: SEG * audio.HOP])
        return torch.from_numpy(np.stack(mm)), torch.from_numpy(np.stack(yy))[:, None]

    def export():
        os.makedirs("release", exist_ok=True)
        torch.save({"G": {k: v.half() for k, v in G.state_dict().items()}, "mean": mean, "std": std},
                   "release/vocoder.pt")

    t0 = last_save = time.time()
    deadline = t0 + a.minutes * 60
    G.train(); D.train()
    while time.time() < deadline:
        m, y = batch()
        yg = G(m)
        # --- D
        ro, _ = D(y)
        fo, _ = D(yg.detach())
        lD = sum(((r - 1) ** 2).mean() + (f ** 2).mean() for r, f in zip(ro, fo))
        oD.zero_grad(); lD.backward(); oD.step()
        # --- G
        fo, ffm = D(yg)
        with torch.no_grad():
            _, rfm = D(y)
        adv = sum(((f - 1) ** 2).mean() for f in fo)
        fm = sum(F.l1_loss(a_, b_) for fa, fb in zip(ffm, rfm) for a_, b_ in zip(fa, fb))
        lm = F.l1_loss(mel_fn(yg), mel_fn(y))
        lG = adv + 2 * fm + 45 * lm
        oG.zero_grad(); lG.backward(); oG.step()
        step += 1
        if step % 20 == 0:
            print(f"step {step} mel_l1 {lm.item():.3f} adv {adv.item():.3f} fm {fm.item():.3f} "
                  f"D {lD.item():.3f} elapsed {(time.time() - t0) / 60:.1f}m", flush=True)
        if time.time() - last_save > 120:
            torch.save({"G": G.state_dict(), "D": D.state_dict(), "oG": oG.state_dict(),
                        "oD": oD.state_dict(), "step": step}, "ckpt/voc.pt")
            last_save = time.time()
    torch.save({"G": G.state_dict(), "D": D.state_dict(), "oG": oG.state_dict(),
                "oD": oD.state_dict(), "step": step}, "ckpt/voc.pt")
    export()
    print("done, step", step, flush=True)


if __name__ == "__main__":
    main()
