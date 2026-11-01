"""TTS v2 학습: 피치/에너지 예측기 + 발음 변환 입력 + 배속 증강 + 긴 문장 혼합.

예) python -m tts.train_v2 --stage v2 --init release/tts_long.pt --minutes 110 --threads 2
이어서 학습: 같은 명령 재실행 (ckpt/<stage>.pt 에서 자동 재개)
- 평가용: KSS 마지막 40개 발화(idx >= 12814)는 모든 변형에서 제외
- 평가: 길이 비율(붕괴 감시), 학습 안 쓴 발화의 멜 L1(교사강요), 피치 표준편차 비율(억양 밋밋함 지표)
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
N_ORIG = 12854


def load_items(spec):
    items, held = [], []
    for part in spec.split(","):
        for f in sorted(glob.glob(part)):
            for x in pickle.load(open(f, "rb")):
                if x["idx"] >= N_ORIG - N_HOLD:
                    if f.endswith("v2_s100.pkl"):
                        held.append(x)
                    continue
                items.append(x)
    return items, held


def batches(items, max_frames):
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
    random.shuffle(chunks)
    return chunks


def collate(items, ids):
    b = [items[i] for i in ids]
    T, F = max(len(x["ids"]) for x in b), max(len(x["mel"]) for x in b)
    tx = torch.zeros(len(b), T, dtype=torch.long)
    mel = torch.zeros(len(b), F, 80)
    f0 = torch.zeros(len(b), F)
    ml = torch.zeros(len(b), dtype=torch.long)
    spk = torch.zeros(len(b), dtype=torch.long)
    for k, x in enumerate(b):
        tx[k, :len(x["ids"])] = torch.from_numpy(x["ids"].astype(np.int64))
        mel[k, :len(x["mel"])] = torch.from_numpy(x["mel"].astype(np.float32))
        f0[k, :len(x["f0"])] = torch.from_numpy(x["f0"].astype(np.float32))
        ml[k] = len(x["mel"])
        spk[k] = x["spk"]
    return tx, mel, ml, spk, f0


@torch.no_grad()
def evaluate(model, held):
    model.eval()
    ratios, l1s, sd_pred, sd_true = [], [], [], []
    for it in held:
        tx, mel, ml, spk, f0 = collate([it], [0])
        h, pad, logd = model.encode(tx, spk)
        ratios.append((torch.exp(logd) - 1).clamp(min=1).sum().item() / len(it["mel"]))
        r = model(tx, mel, ml, spk, f0, extra=True)
        l1s.append(r["mel"].item())
        ok = ~r["pad"][0]
        sd_pred.append(r["p_hat"][0][ok].std().item())
        sd_true.append(r["p_tok"][0][ok].std().item())
    model.train()
    return (float(np.mean(ratios)), float(np.mean(l1s)),
            float(np.mean(sd_pred) / max(np.mean(sd_true), 1e-6)))


@torch.no_grad()
def dump_sample(model, vocab, mean, std, path):
    from tts.g2p_ko import g2p
    model.eval()
    ids = torch.tensor([vocab.encode(g2p(SENTENCE))])
    mel = model.infer(ids, torch.tensor([0]), 0.92)[0].numpy() * std + mean
    model.train()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, audio.mel_to_wav(mel, n_iter=32), audio.SR)
    return len(mel) * audio.HOP / audio.SR


def export_fp16(model, n_spk, mean, std, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": {k: (v.half() if v.is_floating_point() and v.numel() > 8 else v)
                          for k, v in model.state_dict().items()},
                "n_spk": n_spk, "mean": mean, "std": std, "g2p": True}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="v2")
    ap.add_argument("--data", default="data/proc/v2_s100.pkl,data/proc/v2_s095.pkl,data/proc/v2_s105.pkl,data/proc/long_v2.pkl")
    ap.add_argument("--init", default="release/tts_long.pt")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--max_frames", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max_len", type=int, default=800)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--var_w", type=float, default=0.3)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    items, held = load_items(a.data)
    items = [x for x in items if len(x["mel"]) <= a.max_len and len(x["ids"]) > 3 and len(x["mel"]) > len(x["ids"])]
    vocab = Vocab.load("data/proc/vocab.json")
    stats = json.load(open("data/proc/v2_stats.json"))
    mean, std = stats["mel_mean"], stats["mel_std"]
    n_spk = 1 + len(json.load(open("data/proc_zeroth/spk_map.json")))
    print(f"[{a.stage}] 학습 항목 {len(items)}  평가 {len(held)}  stats {stats}", flush=True)

    model = TTS(len(vocab), n_spk=n_spk)
    model.set_stats(stats["lf0_mean"], stats["lf0_std"], stats["e_mean"], stats["e_std"], mean, std)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=1e-6)
    os.makedirs("ckpt", exist_ok=True)
    path, best_path = f"ckpt/{a.stage}.pt", f"ckpt/{a.stage}_best.pt"
    step, lr_scale = 0, 1.0
    if os.path.exists(path):
        s = torch.load(path)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        step, lr_scale = s["step"], s.get("lr_scale", 1.0)
        print("resumed at", step, flush=True)
    else:
        s = torch.load(a.init)
        miss = model.load_state_dict({k: v.float() for k, v in s["model"].items()}, strict=False)
        model.set_stats(stats["lf0_mean"], stats["lf0_std"], stats["e_mean"], stats["e_std"], mean, std)
        print("initialized from", a.init, "| 새로 학습할 모듈:", sorted({k.split('.')[0] for k in miss.missing_keys}), flush=True)

    def save(p):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                    "lr_scale": lr_scale}, p)

    t0 = last_save = time.time()
    deadline = t0 + a.minutes * 60
    recent = []
    model.train()
    while time.time() < deadline:
        for ids in batches(items, a.max_frames):
            tx, mel, ml, spk, f0 = collate(items, ids)
            warm = min(1.0, (step + 1) / 300)
            for g in opt.param_groups:
                g["lr"] = a.lr * warm * lr_scale
            r = model(tx, mel, ml, spk, f0)
            loss = r["mel"] + r["dur"] + r["fs"] + r["bin"] + a.var_w * (r["pitch"] + r["energy"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            step += 1
            recent = (recent + [r["spread"].item()])[-100:]
            if step % 20 == 0:
                print(f"step {step} mel {r['mel'].item():.3f} dur {r['dur'].item():.3f} pitch {r['pitch'].item():.3f} "
                      f"energy {r['energy'].item():.3f} fs {r['fs'].item():.3f} bin {r['bin'].item():.3f} "
                      f"spread {r['spread'].item():.2f} elapsed {(time.time() - t0) / 60:.1f}m", flush=True)
            if step % a.eval_every == 0:
                ratio, l1, psd = evaluate(model, held)
                spread = float(np.mean(recent))
                secs = dump_sample(model, vocab, mean, std, f"samples/train/{a.stage}_{step}.wav")
                ok = 0.7 <= ratio <= 1.4 and spread < 0.35
                print(f"EVAL step {step}: len_ratio {ratio:.2f} heldout_mel_L1 {l1:.4f} pitch_std_ratio {psd:.2f} "
                      f"spread {spread:.2f} sample {secs:.1f}s -> {'OK' if ok else 'COLLAPSE?'}", flush=True)
                if ok:
                    save(best_path)
                elif step >= 1500 and os.path.exists(best_path):
                    s2 = torch.load(best_path)
                    model.load_state_dict(s2["model"]); opt.load_state_dict(s2["opt"])
                    lr_scale *= 0.5
                    print(f"!! 붕괴 감지 -> 복구(step {s2['step']}), lr_scale={lr_scale}", flush=True)
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
