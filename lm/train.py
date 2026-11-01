"""한국어 소형 GPT 학습 (CPU). 처음부터.

사용: python -m lm.train --minutes 120 --threads 4
이어서: 같은 명령 재실행 (ckpt/lm.pt 재개). 결과: release/lm.pt
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from agent.model import GPT, n_params
from lm.bpe import BPE


def get_batch(data, bs, ctx, rng):
    ix = rng.integers(0, len(data) - ctx - 1, bs)
    x = np.stack([data[i:i + ctx] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + ctx + 1] for i in ix]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def val_loss(model, val, ctx, n=40, bs=16):
    model.eval()
    rng = np.random.default_rng(0)
    tot = 0.0
    for _ in range(n):
        x, y = get_batch(val, bs, ctx, rng)
        lo, _ = model(x)
        tot += F.cross_entropy(lo.reshape(-1, lo.size(-1)), y.reshape(-1)).item()
    model.train()
    return tot / n


@torch.no_grad()
def sample(model, bpe, prompt, n=80, temp=0.8, top_k=40, seed=0):
    torch.manual_seed(seed)
    model.eval()
    ids = bpe.encode(prompt)
    x = torch.tensor([ids])
    logits, past = model(x, None, 0)
    out, pos = list(ids), len(ids)
    for _ in range(n):
        lo = logits[0, -1] / temp
        v, _ = lo.topk(top_k)
        lo[lo < v[-1]] = -1e9
        nxt = torch.multinomial(F.softmax(lo, -1), 1).item()
        if nxt == 3:
            break
        out.append(nxt)
        if pos + 1 >= model.ctx:
            break
        logits, past = model(torch.tensor([[nxt]]), past, pos)
        pos += 1
    model.train()
    return bpe.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=120)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--eval_every", type=int, default=300)
    ap.add_argument("--ckpt", default="ckpt/lm.pt")
    ap.add_argument("--out", default="release/lm.pt")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    bpe = BPE.load("data/lm/bpe.json")
    train = np.memmap("data/lm/train.bin", dtype=np.uint16, mode="r")
    val = np.memmap("data/lm/val.bin", dtype=np.uint16, mode="r")
    print(f"학습 토큰 {len(train) / 1e6:.1f}M, 검증 토큰 {len(val) / 1e6:.2f}M, 어휘 {len(bpe)}", flush=True)
    model = GPT(len(bpe), d=a.d, layers=a.layers, heads=a.heads, ctx=a.ctx, drop=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    step, seen = 0, 0
    os.makedirs("ckpt", exist_ok=True); os.makedirs("release", exist_ok=True)
    elapsed0 = 0.0
    if os.path.exists(a.ckpt):
        s = torch.load(a.ckpt)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        step, seen, elapsed0 = s["step"], s["seen"], s.get("elapsed", 0.0)
        print("resumed at", step, flush=True)
    print(f"파라미터 {n_params(model) / 1e6:.1f}M (임베딩 포함)", flush=True)
    rng = np.random.default_rng(step)
    t0, total = time.time(), a.minutes * 60
    last_save = t0
    model.train()
    PROMPTS = ["대한민국은", "수학은", "서울특별시는", "김민수는"]
    while time.time() - t0 < total:
        el = elapsed0 + time.time() - t0
        frac = min(1.0, (time.time() - t0) / total)
        lr = a.lr * min(1.0, (step + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = lr
        loss_acc = 0.0
        for _ in range(a.accum):
            x, y = get_batch(train, a.bs, a.ctx, rng)
            lo, _ = model(x)
            loss = F.cross_entropy(lo.reshape(-1, lo.size(-1)), y.reshape(-1)) / a.accum
            loss.backward(); loss_acc += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        step += 1; seen += a.bs * a.ctx * a.accum
        if step % 25 == 0:
            tps = seen / max(1.0, el)
            print(f"step {step} loss {loss_acc:.3f} lr {lr:.2e} 토큰 {seen / 1e6:.1f}M ({tps:.0f}/초) {(time.time() - t0) / 60:.1f}분", flush=True)
        if step % a.eval_every == 0:
            vl = val_loss(model, val, a.ctx)
            print(f"VAL step {step}: loss {vl:.3f} 퍼플렉시티 {math.exp(vl):.1f}", flush=True)
            for p in PROMPTS[:2]:
                print("  >", sample(model, bpe, p, 50).replace("\n", " / "), flush=True)
        if time.time() - last_save > 180:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "seen": seen,
                        "elapsed": el}, a.ckpt)
            last_save = time.time()
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "seen": seen,
                "elapsed": elapsed0 + time.time() - t0}, a.ckpt)
    torch.save({"model": {k: v.half() for k, v in model.state_dict().items()},
                "cfg": dict(vocab=len(bpe), d=a.d, layers=a.layers, heads=a.heads, ctx=a.ctx)}, a.out)
    vl = val_loss(model, val, a.ctx, n=100)
    print(f"최종 VAL loss {vl:.3f} 퍼플렉시티 {math.exp(vl):.1f}", flush=True)
    for p in PROMPTS:
        print("  >", sample(model, bpe, p, 80).replace("\n", " / "), flush=True)
    print("done, step", step, flush=True)


if __name__ == "__main__":
    main()
