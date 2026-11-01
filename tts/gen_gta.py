"""GTA(ground-truth-aligned) 멜 생성: 학습 발화에 대해 TTS 가 '실제 길이/정렬로' 예측한 멜.

보코더를 이 멜로 추가 학습하면, 합성 때 들어오는 (실제보다 흐린) TTS 멜에 맞춰진다.
사용: python -m tts.gen_gta --ckpt release/tts_long.pt --out data/proc/gta.pkl --threads 4
"""
import argparse
import pickle
import time

import numpy as np
import torch

from tts.infer import load_tts
from tts.model import beta_prior, hard_attn, length_regulate, mas
from tts.text import Vocab


@torch.no_grad()
def gta_batch(model, ids, mel, mel_len, spk):
    h, pad, _ = model.encode(ids, spk)
    B, T = ids.shape
    Fm = mel.size(1)
    fpad = torch.arange(Fm)[None] >= mel_len[:, None]
    tlen = (~pad).sum(1)
    logp = model.aligner(ids, mel, pad, beta_prior(tlen, mel_len, T, Fm))
    dur = mas(logp.transpose(1, 2), ~pad, ~fpad)
    x = length_regulate(h, dur, Fm)
    return model.out(model.dec(x, fpad))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="release/tts_long.pt")
    ap.add_argument("--out", default="data/proc/gta.pkl")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max_frames", type=int, default=4000)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    vocab = Vocab.load("data/proc/vocab.json")
    model, mean, std = load_tts(a.ckpt, vocab)
    items = pickle.load(open("data/proc/data.pkl", "rb"))["items"]
    order = sorted(range(len(items)), key=lambda i: len(items[i]["mel"]))
    out = [None] * len(items)
    t0, i, batches = time.time(), 0, 0
    while i < len(order):
        grp, mx = [], 0
        while i < len(order):
            n = len(items[order[i]]["mel"])
            if grp and max(mx, n) * (len(grp) + 1) > a.max_frames:
                break
            grp.append(order[i]); mx = max(mx, n); i += 1
        b = [items[j] for j in grp]
        T, Fm = max(len(x["ids"]) for x in b), max(len(x["mel"]) for x in b)
        ids = torch.zeros(len(b), T, dtype=torch.long)
        mel = torch.zeros(len(b), Fm, 80)
        ml = torch.zeros(len(b), dtype=torch.long)
        for k, x in enumerate(b):
            ids[k, :len(x["ids"])] = torch.from_numpy(x["ids"].astype(np.int64))
            mel[k, :len(x["mel"])] = torch.from_numpy(x["mel"].astype(np.float32))
            ml[k] = len(x["mel"])
        if min(len(x["ids"]) for x in b) < 1 or (ml < ids.ne(0).sum(1)).any():
            for k, j in zip(range(len(b)), grp):  # 정렬 불가(프레임 < 글자) 발화는 실제 멜을 그대로 둔다
                out[j] = items[j]["mel"]
            continue
        y = gta_batch(model, ids, mel, ml, torch.zeros(len(b), dtype=torch.long)).numpy()
        for k, j in enumerate(grp):
            out[j] = y[k, :ml[k]].astype(np.float16)
        batches += 1
        if batches % 50 == 0:
            print(f"{i}/{len(order)} utts, {(time.time() - t0) / 60:.1f} min", flush=True)
    err = np.mean([np.abs(o.astype(np.float32) - x["mel"].astype(np.float32)).mean()
                   for o, x in zip(out[:300], items[:300])])
    pickle.dump(out, open(a.out, "wb"))
    print(f"saved {a.out}: {len(out)} utts, GTA-vs-real mel L1 (first 300) = {err:.3f}, "
          f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
