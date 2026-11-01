"""HiFi-GAN 보코더 v2 (CPU): 배속 증강 3벌 + 멀티해상도 STFT 손실 + 학습률 감쇠 + (선택) 변형별 GTA 멜.

사용: python -m tts.train_vocoder_v2 --tag v2 --init ckpt/voc.pt --minutes 110 --threads 2
GTA 마무리: ... --tag v2gta --init ckpt/v2.pt --gta_dir data/proc/gta_v2 --lr 1e-4
이어서 학습: 같은 명령 재실행 (ckpt/<tag>.pt 에서 자동 재개). 결과: release/vocoder_<tag>.pt (fp16 생성기)
"""
import argparse
import json
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from tts import audio
from tts.train_vocoder import SEG, make_mel_fn, preemph
from tts.vocoder import Discriminator, Generator

N_ORIG = 12854
N_HOLD = 40
TAGS = ["s100", "s095", "s105"]
WAV_FILES = {"s100": "data/proc/kss_wav.pkl", "s095": "data/proc/kss_wav_s095.pkl", "s105": "data/proc/kss_wav_s105.pkl"}
STFT_CFG = ((256, 64, 256), (512, 128, 512), (1024, 256, 1024))
_WIN = {}


def mr_stft_loss(yg, y):
    tot = 0.0
    for n_fft, hop, win in STFT_CFG:
        w = _WIN.setdefault(win, torch.hann_window(win))
        a = torch.stft(yg.squeeze(1), n_fft, hop, win, w, return_complex=True).abs().clamp(min=1e-7)
        b = torch.stft(y.squeeze(1), n_fft, hop, win, w, return_complex=True).abs().clamp(min=1e-7)
        sc = torch.norm(b - a, p="fro") / torch.norm(b, p="fro")
        tot = tot + sc + F.l1_loss(torch.log(a), torch.log(b))
    return tot / len(STFT_CFG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--minutes", type=float, default=110)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_half_steps", type=int, default=30000, help="이 스텝마다 학습률 절반")
    ap.add_argument("--init", default="ckpt/voc.pt")
    ap.add_argument("--gta_dir", default="", help="gta_s100.pkl, gta_s095.pkl, gta_s105.pkl 이 있는 폴더")
    ap.add_argument("--gta_ratio", type=float, default=0.8)
    ap.add_argument("--stft_w", type=float, default=5.0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    CK, EXPORT = f"ckpt/{a.tag}.pt", f"release/vocoder_{a.tag}.pt"
    stats = json.load(open("data/proc/v2_stats.json"))

    V = {}
    for t in TAGS:
        items = pickle.load(open(f"data/proc/v2_{t}.pkl", "rb"))
        mels = [x["mel"] for x in items]
        wavs = pickle.load(open(WAV_FILES[t], "rb"))
        gta = pickle.load(open(f"{a.gta_dir}/gta_{t}.pkl", "rb")) if a.gta_dir else None
        keep = [i for i in range(N_ORIG - N_HOLD) if len(mels[i]) > SEG + 2 and abs(len(wavs[i]) // audio.HOP + 1 - len(mels[i])) <= 1]
        V[t] = dict(mels=mels, wavs=wavs, gta=gta, keep=keep)
        print(t, "발화", len(keep), "GTA" if gta else "", flush=True)
    weights = [0.5, 0.25, 0.25]

    G, D = Generator(), Discriminator()
    oG = torch.optim.AdamW(G.parameters(), a.lr, betas=(0.8, 0.99))
    oD = torch.optim.AdamW(D.parameters(), a.lr, betas=(0.8, 0.99))
    mel_fn = make_mel_fn()
    os.makedirs("ckpt", exist_ok=True)
    step = 0
    if os.path.exists(CK):
        s = torch.load(CK)
        G.load_state_dict(s["G"]); D.load_state_dict(s["D"])
        oG.load_state_dict(s["oG"]); oD.load_state_dict(s["oD"]); step = s["step"]
        print("resumed at", step, flush=True)
    else:
        s = torch.load(a.init)
        G.load_state_dict(s["G"]); D.load_state_dict(s["D"])
        print("initialized from", a.init, "(step", s.get("step"), ")", flush=True)

    def batch():
        mm, yy = [], []
        for _ in range(a.batch):
            t = random.choices(TAGS, weights)[0]
            v = V[t]
            i = random.choice(v["keep"])
            n = len(v["mels"][i]) - SEG - 1
            f0 = random.randint(0, n)
            use_gta = v["gta"] is not None and random.random() < a.gta_ratio
            src = v["gta"][i] if use_gta else v["mels"][i]
            mm.append(src[f0:f0 + SEG].astype(np.float32).T)
            seg = v["wavs"][i][max(0, f0 * audio.HOP - 1): (f0 + SEG) * audio.HOP].astype(np.float32) / 32767
            seg = preemph(seg)[-SEG * audio.HOP:] if f0 > 0 else preemph(seg)
            seg = seg[: SEG * audio.HOP]
            if len(seg) < SEG * audio.HOP:
                seg = np.pad(seg, (0, SEG * audio.HOP - len(seg)))
            yy.append(seg)
        return torch.from_numpy(np.stack(mm)), torch.from_numpy(np.stack(yy))[:, None]

    def save(p):
        torch.save({"G": G.state_dict(), "D": D.state_dict(), "oG": oG.state_dict(),
                    "oD": oD.state_dict(), "step": step}, p)

    def export():
        os.makedirs("release", exist_ok=True)
        torch.save({"G": {k: x.half() for k, x in G.state_dict().items()},
                    "mean": stats["mel_mean"], "std": stats["mel_std"]}, EXPORT)

    t0 = last_save = time.time()
    deadline = t0 + a.minutes * 60
    G.train(); D.train()
    while time.time() < deadline:
        lr = a.lr * max(0.1, 0.5 ** (step / a.lr_half_steps))
        for o in (oG, oD):
            for g in o.param_groups:
                g["lr"] = lr
        m, y = batch()
        yg = G(m)
        ro, _ = D(y)
        fo, _ = D(yg.detach())
        lD = sum(((r - 1) ** 2).mean() + (f ** 2).mean() for r, f in zip(ro, fo))
        oD.zero_grad(); lD.backward(); oD.step()
        fo, ffm = D(yg)
        with torch.no_grad():
            _, rfm = D(y)
        adv = sum(((f - 1) ** 2).mean() for f in fo)
        fm = sum(F.l1_loss(a_, b_) for fa, fb in zip(ffm, rfm) for a_, b_ in zip(fa, fb))
        lm = F.l1_loss(mel_fn(yg), mel_fn(y))
        ls = mr_stft_loss(yg, y)
        lG = adv + 2 * fm + 45 * lm + a.stft_w * ls
        oG.zero_grad(); lG.backward(); oG.step()
        step += 1
        if step % 20 == 0:
            print(f"step {step} mel_l1 {lm.item():.3f} stft {ls.item():.3f} adv {adv.item():.3f} fm {fm.item():.3f} "
                  f"D {lD.item():.3f} lr {lr:.2e} elapsed {(time.time() - t0) / 60:.1f}m", flush=True)
        if time.time() - last_save > 120:
            save(CK); last_save = time.time()
    save(CK)
    export()
    print("done, step", step, flush=True)


if __name__ == "__main__":
    main()
