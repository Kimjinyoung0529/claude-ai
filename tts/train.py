"""CPU 학습. 사용: python -m tts.train --minutes 120

체크포인트는 ckpt/last.pt 에 주기적으로 저장되며, 재실행하면 이어서 학습한다.
"""
import argparse
import os
import pickle
import random
import time

import numpy as np
import torch

from tts.model import TTS
from tts.text import Vocab

torch.set_num_threads(4)


def batches(items, max_frames, shuffle=True):
    idx = sorted(range(len(items)), key=lambda i: len(items[i]["mel"]))
    chunks, cur, mx = [], [], 0
    for i in idx:
        n = len(items[i]["mel"])
        if cur and max(mx, n) * (len(cur) + 1) > max_frames:
            chunks.append(cur)
            cur, mx = [], 0
        cur.append(i)
        mx = max(mx, n)
    if cur:
        chunks.append(cur)
    if shuffle:
        random.shuffle(chunks)
    return chunks


def collate(items, ids):
    b = [items[i] for i in ids]
    T = max(len(x["ids"]) for x in b)
    F = max(len(x["mel"]) for x in b)
    tx = torch.zeros(len(b), T, dtype=torch.long)
    mel = torch.zeros(len(b), F, 80)
    ml = torch.zeros(len(b), dtype=torch.long)
    for k, x in enumerate(b):
        tx[k, : len(x["ids"])] = torch.from_numpy(x["ids"].astype(np.int64))
        mel[k, : len(x["mel"])] = torch.from_numpy(x["mel"].astype(np.float32))
        ml[k] = len(x["mel"])
    return tx, mel, ml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--max_frames", type=int, default=3500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_len", type=int, default=450, help="너무 긴 발화 제외(프레임)")
    a = ap.parse_args()

    data = pickle.load(open("data/proc/data.pkl", "rb"))
    items = [x for x in data["items"] if len(x["mel"]) <= a.max_len and len(x["ids"]) > 3]
    vocab = Vocab.load("data/proc/vocab.json")
    print(f"utts={len(items)} vocab={len(vocab)}", flush=True)

    model = TTS(len(vocab))
    print("params(M):", sum(p.numel() for p in model.parameters()) / 1e6, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=1e-6)
    os.makedirs("ckpt", exist_ok=True)
    step = 0
    if os.path.exists("ckpt/last.pt"):
        s = torch.load("ckpt/last.pt")
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); step = s["step"]
        print("resumed at", step, flush=True)

    t0, last_save, deadline = time.time(), time.time(), time.time() + a.minutes * 60
    model.train()
    while time.time() < deadline:
        for ids in batches(items, a.max_frames):
            tx, mel, ml = collate(items, ids)
            warm = min(1.0, (step + 1) / 400)
            for g in opt.param_groups:
                g["lr"] = a.lr * warm
            mel_l, pri_l, dur_l = model(tx, mel, ml)
            loss = mel_l + 0.5 * pri_l + dur_l
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 20 == 0:
                print(f"step {step} mel {mel_l.item():.3f} prior {pri_l.item():.3f} "
                      f"dur {dur_l.item():.3f} elapsed {(time.time() - t0) / 60:.1f}m", flush=True)
            if time.time() - last_save > 300 or time.time() > deadline:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                            "mean": data["mean"], "std": data["std"]}, "ckpt/last.pt")
                last_save = time.time()
            if time.time() > deadline:
                break
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                "mean": data["mean"], "std": data["std"]}, "ckpt/last.pt")
    print("done, step", step, flush=True)


if __name__ == "__main__":
    main()
