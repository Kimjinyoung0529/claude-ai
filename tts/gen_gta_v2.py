"""GTA 멜 생성 v2: TTS v2 모델로 3개 변형(s100/s095/s105)의 학습 발화에 대한 GTA 멜을 만든다.

사용: python -m tts.gen_gta_v2 --ckpt release/tts_v2.pt --out data/proc/gta_v2 --threads 4
출력: <out>/gta_s100.pkl, gta_s095.pkl, gta_s105.pkl  (각각 v2_<변형>.pkl 항목과 같은 순서)
"""
import argparse
import os
import pickle
import time

import numpy as np
import torch

from tts.infer import load_tts
from tts.model import beta_prior, hard_attn, length_regulate, mas
from tts.text import Vocab


@torch.no_grad()
def gta_batch(model, ids, mel, mel_len, spk, f0):
    h, pad, _ = model.encode(ids, spk)
    B, T = ids.shape
    Fm = mel.size(1)
    fpad = torch.arange(Fm)[None] >= mel_len[:, None]
    tlen = (~pad).sum(1)
    logp = model.aligner(ids, mel, pad, beta_prior(tlen, mel_len, T, Fm))
    dur = mas(logp.transpose(1, 2), ~pad, ~fpad)
    hard = hard_attn(dur, Fm)
    vm = (f0 > 0).float().masked_fill(fpad, 0.0)
    lf0 = (torch.log(f0.clamp(min=1.0)) - model.lf0_ms[0]) / model.lf0_ms[1]
    en = (torch.logsumexp(mel * model.mel_ms[1] + model.mel_ms[0], dim=-1) - model.e_ms[0]) / model.e_ms[1]
    hT = hard.transpose(1, 2)
    cnt = hT.sum(-1).clamp(min=1)
    v_cnt = (hT @ vm.unsqueeze(-1)).squeeze(-1)
    p_tok = (hT @ (lf0 * vm).unsqueeze(-1)).squeeze(-1) / v_cnt.clamp(min=1) * (v_cnt > 0).float()
    e_tok = (hT @ en.masked_fill(fpad, 0.0).unsqueeze(-1)).squeeze(-1) / cnt
    h = model.add_var(h, pad, p_tok, e_tok)
    x = length_regulate(h, dur, Fm)
    return model.out(model.dec(x, fpad))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="release/tts_v2.pt")
    ap.add_argument("--out", default="data/proc/gta_v2")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max_frames", type=int, default=4000)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    vocab = Vocab.load("data/proc/vocab.json")
    model, mean, std = load_tts(a.ckpt, vocab)
    t0 = time.time()
    for tag in ("s100", "s095", "s105"):
        items = pickle.load(open(f"data/proc/v2_{tag}.pkl", "rb"))
        order = sorted(range(len(items)), key=lambda i: len(items[i]["mel"]))
        res = [None] * len(items)
        i = 0
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
            f0 = torch.zeros(len(b), Fm)
            ml = torch.zeros(len(b), dtype=torch.long)
            for k, x in enumerate(b):
                ids[k, :len(x["ids"])] = torch.from_numpy(x["ids"].astype(np.int64))
                mel[k, :len(x["mel"])] = torch.from_numpy(x["mel"].astype(np.float32))
                f0[k, :len(x["f0"])] = torch.from_numpy(x["f0"].astype(np.float32))
                ml[k] = len(x["mel"])
            if (ml < ids.ne(0).sum(1)).any():
                for j in grp:
                    res[j] = items[j]["mel"]
                continue
            y = gta_batch(model, ids, mel, ml, torch.zeros(len(b), dtype=torch.long), f0).numpy()
            for k, j in enumerate(grp):
                res[j] = y[k, :ml[k]].astype(np.float16)
        pickle.dump(res, open(f"{a.out}/gta_{tag}.pkl", "wb"))
        print(f"{tag} 완료 ({(time.time() - t0) / 60:.1f}분)", flush=True)


if __name__ == "__main__":
    main()
