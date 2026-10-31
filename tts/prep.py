"""KSS parquet -> 멜 스펙트로그램 + 텍스트 id 캐시 (data/proc/).

사용: python -m tts.prep [--limit N]
"""
import argparse
import glob
import io
import os
import pickle
from multiprocessing import Pool

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from tts import audio
from tts.text import Vocab, normalize

RAW = "data/raw"
OUT = "data/proc"


def work(args):
    wav_bytes, text = args
    y, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    if sr != audio.SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=audio.SR)
    y, _ = librosa.effects.trim(y, top_db=35)
    return audio.wav_to_mel(y), text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    jobs = []
    for f in sorted(glob.glob(f"{RAW}/*.parquet")):
        d = pd.read_parquet(f, columns=["audio", "expanded_script"])
        jobs += [(r["audio"]["bytes"], normalize(r["expanded_script"])) for _, r in d.iterrows()]
        if a.limit and len(jobs) >= a.limit:
            jobs = jobs[: a.limit]
            break
    print("utterances:", len(jobs))

    with Pool(4) as p:
        res = p.map(work, jobs, chunksize=32)

    vocab = Vocab.build([t for _, t in res])
    vocab.save(f"{OUT}/vocab.json")
    mels = [m for m, _ in res]
    allm = np.concatenate(mels)
    mean, std = float(allm.mean()), float(allm.std())
    items = []
    for m, t in res:
        ids = vocab.encode(t)
        items.append({"text": t, "ids": np.array(ids, np.int16),
                      "mel": ((m - mean) / std).astype(np.float16)})
    pickle.dump({"items": items, "mean": mean, "std": std}, open(f"{OUT}/data.pkl", "wb"))
    frames = sum(len(m) for m in mels)
    print(f"vocab={len(vocab)} mean={mean:.3f} std={std:.3f} hours={frames * audio.HOP / audio.SR / 3600:.2f}")


if __name__ == "__main__":
    main()
