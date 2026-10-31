"""KSS 의 (트림된) 파형을 int16 으로 캐시 -> data/proc/kss_wav.pkl (보코더 학습용).

prep.py 와 같은 순서/같은 트림을 써서 data.pkl 의 멜과 프레임이 1:1 로 맞는다.
"""
import glob
import io
import pickle
from multiprocessing import Pool

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from tts import audio


def work(b):
    y, sr = sf.read(io.BytesIO(b), dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    if sr != audio.SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=audio.SR)
    y, _ = librosa.effects.trim(y, top_db=35)
    return (np.clip(y, -1, 1) * 32767).astype(np.int16)


def main():
    blobs = []
    for f in sorted(glob.glob("data/raw/*.parquet")):
        d = pd.read_parquet(f, columns=["audio"])
        blobs += [r["audio"]["bytes"] for _, r in d.iterrows()]
    with Pool(4) as p:
        wavs = p.map(work, blobs, chunksize=32)
    ref = pickle.load(open("data/proc/data.pkl", "rb"))["items"]
    assert len(ref) == len(wavs), (len(ref), len(wavs))
    bad = sum(len(w) // audio.HOP + 1 != len(it["mel"]) for w, it in zip(wavs, ref))
    print("utts", len(wavs), "frame-mismatch", bad)
    pickle.dump(wavs, open("data/proc/kss_wav.pkl", "wb"))


if __name__ == "__main__":
    main()
