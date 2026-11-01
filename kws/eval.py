"""호출어 평가: 시간당 오작동(FA/h), 소음별 인식률, 지연/용량. 학습에 안 쓴 화자의 연속 음성 사용.

사용: python -m kws.eval --model release/kws.pt --target_fa 1.0
- 평가 화자를 반으로 나눠 한쪽(calib)으로 임계값을 정하고, 다른 쪽(test)으로 보고한다 (임계값이 평가에 새지 않게).
- 연속 음성에 1.5초 창을 hop 간격으로 밀며 확률을 구하고, 최근 2개 평균 + 불응기 1.5초로 트리거를 센다.
"""
import argparse
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F

from kws import config as C
from kws.model import CLASSES, KWS, n_params


def load_model(path):
    s = torch.load(path)
    m = KWS(width=s["width"])
    m.load_state_dict({k: v.float() for k, v in s["model"].items()})
    return m.eval()


@torch.no_grad()
def stream_probs(model, wav, hop=0.15, batch=256):
    """연속 음성 -> (시각 배열, 확률 [N, 4])"""
    step = int(hop * C.SR)
    starts = np.arange(0, max(1, len(wav) - C.CLIP), step)
    P = []
    for i in range(0, len(starts), batch):
        X = np.stack([wav[s:s + C.CLIP] for s in starts[i:i + batch]]).astype(np.float32) / 32768
        P.append(F.softmax(model.from_wav(torch.from_numpy(X)), -1).numpy())
    return (starts + C.CLIP) / C.SR, np.concatenate(P)


def triggers(times, P, thr, smooth=2, refractory=1.5):
    """클래스별 임계값 thr(길이 3)로 트리거 시각과 클래스를 돌려준다."""
    Ps = np.copy(P)
    for k in range(1, smooth):
        Ps[k:] = Ps[k:] + P[:-k]
    Ps /= smooth
    out, last = [], -9
    for t, p in zip(times, Ps):
        c = int(np.argmax(p[1:])) + 1
        if p[c] >= thr[c - 1] and t - last >= refractory:
            out.append((t, c)); last = t
    return out


def build_streams(held, spks):
    wav, texts, off = [], [], 0
    gap = np.zeros(int(0.3 * C.SR), np.int16)
    for s in spks:
        for w, text in held[s]:
            texts.append((off / C.SR, (off + len(w)) / C.SR, text)); off += len(w) + len(gap)
            wav += [w, gap]
    return np.concatenate(wav), texts


def fa_per_hour(model, held, spks, thr, hop):
    wav, texts = build_streams(held, spks)
    times, P = stream_probs(model, wav, hop)
    tr = triggers(times, P, thr)
    hours = len(wav) / C.SR / 3600
    by = {CLASSES[c]: sum(1 for _, k in tr if k == c) / hours for c in range(1, 4)}
    ctx = []
    for t, c in tr:
        for a, b, text in texts:
            if a <= t <= b + 0.2:
                ctx.append((CLASSES[c], text)); break
    return hours, by, ctx, (times, P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="release/kws.pt")
    ap.add_argument("--target_fa", type=float, default=1.0, help="클래스별 목표 시간당 오작동")
    ap.add_argument("--hop", type=float, default=0.15)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model = load_model(a.model)
    held = pickle.load(open("data/kws/heldout.pkl", "rb"))
    spks = sorted(held)
    calib, test = spks[0::2], spks[1::2]
    print(f"모델 파라미터 {n_params(model) / 1e3:.0f}천 개 | 평가 화자 {len(spks)}명: 보정 {len(calib)}명, 시험 {len(test)}명")

    # 1) 보정 화자로 클래스별 임계값 결정: 목표 FA/h 를 만족하는 가장 낮은 임계값
    wav, _ = build_streams(held, calib)
    times, P = stream_probs(model, wav, a.hop)
    hours_c = len(wav) / C.SR / 3600
    thr = []
    for c in range(1, 4):
        pick = 0.99
        for th in np.arange(0.30, 0.995, 0.005):
            t = [th if k == c - 1 else 2.0 for k in range(3)]   # 다른 클래스는 끔
            n = len(triggers(times, P, t))
            if n / hours_c <= a.target_fa:
                pick = float(th); break
        thr.append(pick)
    print("보정(시간 %.2f): 임계값 hey %.3f / ya %.3f / ssi %.3f (목표 %.1f회/시간)" % ((hours_c,) + tuple(thr) + (a.target_fa,)))

    # 2) 시험 화자로 시간당 오작동 보고
    t0 = time.time()
    hours, by, ctx, _ = fa_per_hour(model, held, test, thr, a.hop)
    print(f"\n[시험 화자 {hours:.2f}시간] 시간당 오작동: " + ", ".join(f"{k} {v:.2f}" for k, v in by.items()) +
          f" | 합계 {sum(by.values()):.2f}  (분석 {time.time() - t0:.0f}초)")
    if ctx:
        print("  오작동 시 문장 예:")
        for k, tx in ctx[:8]:
            print(f"   - [{k}] {tx[:60]}")
    pickle.dump({"thr": thr}, open("data/kws/thresholds.pkl", "wb"))

    # 3) 검증 양성 인식률 (합성 검증 화자 x 소음 fold 5), 보정한 임계값 적용
    from kws.data import load_all, make_val
    D, _ = load_all()
    print("\n[합성 검증 화자 인식률] (보정 임계값 적용)")
    for name, snr in [("깨끗", None), ("SNR 20dB", 20), ("SNR 10dB", 10), ("SNR 5dB", 5), ("SNR 0dB", 0)]:
        X, y = make_val(D, 200, snr=snr)
        with torch.no_grad():
            p = torch.cat([F.softmax(model.from_wav(X[i:i + 256]), -1) for i in range(0, len(X), 256)])
        hit = []
        for c in range(1, 4):
            m = (y == c)
            hit.append(float((p[m][:, c] >= thr[c - 1]).float().mean()))
        print(f"  {name:9s} hey {hit[0]:.1%} / ya {hit[1]:.1%} / ssi {hit[2]:.1%}")
    # 4) 지연
    x = torch.randn(1, C.CLIP) * 0.05
    t = time.time()
    with torch.no_grad():
        for _ in range(30):
            model.from_wav(x)
    print(f"\n창 1개 추론 {(time.time() - t) / 30 * 1000:.1f} ms (1.5초 창, {a.threads}스레드)")


if __name__ == "__main__":
    main()
