"""실시간 마이크 호출어 감지 (사용자 PC 용). 사용: python -m kws.live --model release/kws.pt

필요 패키지: pip install sounddevice numpy torch librosa
임계값은 kws.eval 이 저장한 data/kws/thresholds.pkl 을 쓰거나 --thr 로 직접 지정.
"""
import argparse
import collections
import pickle
import queue
import time

import numpy as np
import torch
import torch.nn.functional as F

from kws import config as C
from kws.eval import load_model
from kws.model import CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="release/kws.pt")
    ap.add_argument("--thr", type=float, nargs=3, default=None, help="hey ya ssi 임계값")
    ap.add_argument("--hop", type=float, default=0.1)
    ap.add_argument("--smooth", type=int, default=2)
    a = ap.parse_args()
    import sounddevice as sd
    model = load_model(a.model)
    thr = a.thr
    if thr is None:
        try:
            thr = pickle.load(open("data/kws/thresholds.pkl", "rb"))["thr"]
        except Exception:
            thr = [0.9, 0.95, 0.95]
    print("임계값(hey/ya/ssi):", [round(t, 3) for t in thr], "| 호출어: '헤이 하나' / '하나야' / '하나씨' | Ctrl+C 로 종료")
    q = queue.Queue()
    sd.InputStream(samplerate=C.SR, channels=1, dtype="int16", blocksize=int(a.hop * C.SR),
                   callback=lambda x, f, t, s: q.put(x[:, 0].copy())).start()
    buf = np.zeros(C.CLIP, np.int16)
    recent = collections.deque(maxlen=a.smooth)
    last = 0.0
    while True:
        buf = np.concatenate([buf, q.get()])[-C.CLIP:]
        with torch.no_grad():
            p = F.softmax(model.from_wav(torch.from_numpy(buf[None].astype(np.float32) / 32768)), -1)[0].numpy()
        recent.append(p)
        ps = np.mean(recent, 0)
        c = int(np.argmax(ps[1:])) + 1
        if ps[c] >= thr[c - 1] and time.time() - last > 1.5:
            last = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] 호출어 감지! ({CLASSES[c]}, 확률 {ps[c]:.2f})", flush=True)


if __name__ == "__main__":
    main()
