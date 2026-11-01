"""호출어 감지기 학습 (CPU). 어려운 부정 예시 자동 선별(OHEM) 포함.

사용: python -m kws.train --minutes 40 --threads 2 --workers 2
이어서 학습: 같은 명령 재실행 (ckpt/kws.pt 에서 재개). 결과: release/kws.pt (fp16)
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from kws import config as C
from kws.data import BatchStream, load_all, make_val
from kws.model import CLASSES, KWS, features, n_params


@torch.no_grad()
def validate(model, D, vals, hneg_va):
    model.eval()
    out = {}
    for name, (X, y) in vals.items():
        p = torch.cat([F.softmax(model.from_wav(X[i:i + 256]), -1) for i in range(0, len(X), 256)])
        pred = p.argmax(1)
        out[f"recall_{name}"] = float((pred == y).float().mean())
    # 어려운 부정(검증 화자) 오작동률: 양성 클래스로 판정된 비율
    rng = np.random.default_rng(7)
    from kws import augment as A
    ws = [hneg_va[i] for i in rng.choice(len(hneg_va), min(1500, len(hneg_va)), replace=False)]
    X = torch.from_numpy(np.stack([A.place(w.astype(np.float32) / 32768 / (np.abs(w).max() / 32768 + 1e-6) * 0.6, C.CLIP, int(0.15 * C.SR), rng).astype(np.float32) for w in ws]))
    p = torch.cat([F.softmax(model.from_wav(X[i:i + 256]), -1) for i in range(0, len(X), 256)])
    out["hardneg_FA"] = float((p.argmax(1) != 0).float().mean())
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--ohem", type=float, default=1.5, help="후보 배치 배수 (그 중 어려운 부정만 학습)")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=float, default=0.7)
    ap.add_argument("--eval_every", type=int, default=400)
    ap.add_argument("--ckpt", default="ckpt/kws.pt")
    ap.add_argument("--out", default="release/kws.pt")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    D, _ = load_all()
    print(f"데이터: 양성 합성 {sum(len(v) for v in D['pos_tr'].values())}개, 어려운 부정 합성 {len(D['hneg_tr'])}, "
          f"Zeroth 부정 창 {len(D['neg'])}, 실제 '하나' 부정 {len(D['hana'])}, 소음 {len(D['noise_tr'])}종 클립", flush=True)
    vals = {"clean": make_val(D), "snr10": make_val(D, snr=10), "snr0": make_val(D, snr=0)}
    model = KWS(width=a.width)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    step = 0
    os.makedirs("ckpt", exist_ok=True); os.makedirs("release", exist_ok=True)
    if os.path.exists(a.ckpt):
        s = torch.load(a.ckpt)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); step = s["step"]
        print("resumed at", step, flush=True)
    print(f"모델 파라미터 {n_params(model) / 1e3:.0f}천 개", flush=True)

    loader = torch.utils.data.DataLoader(BatchStream(D, int(a.batch * a.ohem), seed=step), batch_size=None,
                                         num_workers=a.workers, prefetch_factor=4)
    t0 = time.time(); total = a.minutes * 60; last_save = t0
    model.train()
    for X, y in loader:
        el = time.time() - t0
        if el > total:
            break
        lr = a.lr * (min(1.0, (step + 1) / 200)) * 0.5 * (1 + math.cos(math.pi * min(1.0, el / total)))
        for g in opt.param_groups:
            g["lr"] = lr
        logits = model.from_wav(X)
        loss_each = F.cross_entropy(logits, y, reduction="none", label_smoothing=0.05)
        # OHEM: 양성은 전부, 부정은 손실이 큰 순서로 60%
        neg = (y == 0).nonzero().squeeze(1); pos = (y != 0).nonzero().squeeze(1)
        k = max(1, int(0.6 * len(neg)))
        hard = neg[loss_each[neg].topk(k).indices] if len(neg) else neg
        sel = torch.cat([pos, hard])
        loss = loss_each[sel].mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); step += 1
        if step % 50 == 0:
            acc = (logits.argmax(1) == y).float().mean().item()
            print(f"step {step} loss {loss.item():.3f} 배치정확도 {acc:.3f} lr {lr:.2e} {el / 60:.1f}분", flush=True)
        if step % a.eval_every == 0:
            v = validate(model, D, vals, D["hneg_va"])
            print("VAL step %d: " % step + " ".join(f"{k} {x:.3f}" for k, x in v.items()), flush=True)
        if time.time() - last_save > 120:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, a.ckpt)
            last_save = time.time()
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, a.ckpt)
    torch.save({"model": {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()},
                "width": a.width, "classes": CLASSES}, a.out)
    v = validate(model, D, vals, D["hneg_va"])
    print("최종 VAL: " + " ".join(f"{k} {x:.3f}" for k, x in v.items()), flush=True)
    print("done, step", step, flush=True)


if __name__ == "__main__":
    main()
