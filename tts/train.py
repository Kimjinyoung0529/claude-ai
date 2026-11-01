"""CPU 학습 (다화자 사전학습 / 단일 화자 미세조정 공용).

예)  python -m tts.train --stage pre --data "data/proc/data.pkl,data/proc_zeroth/shard*.pkl" --minutes 180
     python -m tts.train --stage ft  --data data/proc/data.pkl --init ckpt/pre.pt --lr 3e-4 --minutes 90

- 평가용으로 KSS 마지막 40개 발화를 항상 떼어 둔다(학습에 쓰지 않음).
- 주기적으로 '예측 총길이 / 실제 길이' 를 재서 정렬/길이 붕괴를 감지하고,
  붕괴하면 마지막 정상 체크포인트로 되돌리고 학습률을 절반으로 줄인다.
- 평가 때마다 samples/train/ 에 샘플 wav 를 남긴다.
"""
import argparse
import glob
import json
import os
import pickle
import random
import time

import numpy as np
import soundfile as sf
import torch

from tts import audio
from tts.model import TTS
from tts.text import Vocab

SENTENCE = "안녕하세요. 오늘 날씨가 정말 좋네요."
N_HOLD = 40


def load_items(spec):
    kss = pickle.load(open("data/proc/data.pkl", "rb"))
    mean, std = kss["mean"], kss["std"]
    for x in kss["items"]:
        x["spk"] = 0
    held = kss["items"][-N_HOLD:]
    items = []
    for part in spec.split(","):
        for f in sorted(glob.glob(part)):
            if f.endswith("data/proc/data.pkl") or f == "data/proc/data.pkl":
                items += kss["items"][:-N_HOLD]
            else:
                items += pickle.load(open(f, "rb"))
    return items, held, mean, std


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
    spk = torch.zeros(len(b), dtype=torch.long)
    for k, x in enumerate(b):
        tx[k, : len(x["ids"])] = torch.from_numpy(x["ids"].astype(np.int64))
        mel[k, : len(x["mel"])] = torch.from_numpy(x["mel"].astype(np.float32))
        ml[k] = len(x["mel"])
        spk[k] = x["spk"]
    return tx, mel, ml, spk


@torch.no_grad()
def evaluate(model, held):
    """예측 총길이/실제 길이 비율(평균, 표준편차). 1.0 에 가까울수록 정상."""
    model.eval()
    r = []
    for it in held:
        ids = torch.tensor(it["ids"].astype(np.int64))[None]
        _, _, logd = model.encode(ids, torch.tensor([0]))
        r.append((torch.exp(logd) - 1).clamp(min=1).sum().item() / len(it["mel"]))
    model.train()
    return float(np.mean(r)), float(np.std(r))


@torch.no_grad()
def dump_sample(model, vocab, mean, std, path):
    model.eval()
    ids = torch.tensor([vocab.encode(SENTENCE)])
    mel = model.infer(ids, torch.tensor([0]))[0].numpy() * std + mean
    model.train()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, audio.mel_to_wav(mel, n_iter=32), audio.SR)
    return len(mel) * audio.HOP / audio.SR


def export_fp16(model, n_spk, mean, std, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": {k: v.half() for k, v in model.state_dict().items()},
                "n_spk": n_spk, "mean": mean, "std": std}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="pre")
    ap.add_argument("--data", default="data/proc/data.pkl")
    ap.add_argument("--init", default="")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--max_frames", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_len", type=int, default=800, help="너무 긴 발화 제외(프레임)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--bin_start", type=int, default=3000)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    items, held, mean, std = load_items(a.data)
    items = [x for x in items if len(x["mel"]) <= a.max_len and len(x["ids"]) > 3]
    vocab = Vocab.load("data/proc/vocab.json")
    n_spk = 1
    if os.path.exists("data/proc_zeroth/spk_map.json"):
        n_spk = 1 + len(json.load(open("data/proc_zeroth/spk_map.json")))
    frames = sum(len(x["mel"]) for x in items)
    print(f"[{a.stage}] utts={len(items)} hours={frames * audio.HOP / audio.SR / 3600:.1f} "
          f"vocab={len(vocab)} n_spk={n_spk}", flush=True)

    model = TTS(len(vocab), n_spk=n_spk)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=1e-6)
    os.makedirs("ckpt", exist_ok=True)
    path, best_path = f"ckpt/{a.stage}.pt", f"ckpt/{a.stage}_best.pt"
    step, lr_scale, warm_from = 0, 1.0, int(a.stage != "pre")
    if os.path.exists(path):
        s = torch.load(path)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        step, lr_scale = s["step"], s.get("lr_scale", 1.0)
        print("resumed at", step, flush=True)
    elif a.init:
        model.load_state_dict(torch.load(a.init)["model"])
        warm_from = 1  # 미세조정은 정렬이 이미 잡혀 있으므로 이진화 손실을 바로 쓴다
        print("initialized from", a.init, flush=True)

    def save(p):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                    "lr_scale": lr_scale, "mean": mean, "std": std, "n_spk": n_spk}, p)

    t0 = last_save = time.time()
    deadline = t0 + a.minutes * 60
    recent_spread = []
    model.train()
    while time.time() < deadline:
        for ids in batches(items, a.max_frames):
            tx, mel, ml, spk = collate(items, ids)
            warm = min(1.0, (step + 1) / 400)
            for g in opt.param_groups:
                g["lr"] = a.lr * warm * lr_scale
            r = model(tx, mel, ml, spk)
            w_bin = 1.0 if warm_from else min(1.0, max(0.0, (step - a.bin_start) / 4000))
            loss = r["mel"] + r["dur"] + r["fs"] + w_bin * r["bin"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            step += 1
            recent_spread = (recent_spread + [r["spread"].item()])[-100:]
            if step % 20 == 0:
                print(f"step {step} mel {r['mel'].item():.3f} dur {r['dur'].item():.3f} "
                      f"fs {r['fs'].item():.3f} bin {r['bin'].item():.3f} "
                      f"spread {r['spread'].item():.2f} elapsed {(time.time() - t0) / 60:.1f}m", flush=True)

            if step % a.eval_every == 0:
                ratio, sd = evaluate(model, held)
                spread = float(np.mean(recent_spread))
                secs = dump_sample(model, vocab, mean, std, f"samples/train/{a.stage}_{step}.wav")
                ok = 0.7 <= ratio <= 1.4 and spread < 0.35
                print(f"EVAL step {step}: len_ratio {ratio:.2f}±{sd:.2f} spread {spread:.2f} "
                      f"sample {secs:.1f}s -> {'OK' if ok else 'COLLAPSE?'}", flush=True)
                if ok:
                    save(best_path)
                elif step >= 2000 and os.path.exists(best_path):
                    s = torch.load(best_path)
                    model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
                    lr_scale *= 0.5
                    print(f"!! 붕괴 감지 -> 마지막 정상 체크포인트(step {s['step']})로 복구, lr_scale={lr_scale}",
                          flush=True)

            if time.time() - last_save > 120 or time.time() > deadline:
                save(path)
                last_save = time.time()
            if time.time() > deadline:
                break
    save(path)
    export_fp16(model, n_spk, mean, std, f"release/tts_{a.stage}.pt")
    print("done, step", step, flush=True)


if __name__ == "__main__":
    main()
