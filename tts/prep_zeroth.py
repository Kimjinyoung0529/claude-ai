"""Zeroth-Korean(다화자, 약 51h) -> 사전학습용 캐시 (data/proc_zeroth/shardN.pkl).

KSS 의 어휘/정규화 통계를 그대로 써서 두 데이터가 같은 공간에 놓이게 한다.
사용: python -m tts.prep_zeroth
"""
import glob
import io
import json
import os
import pickle
from multiprocessing import Pool

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from tts import audio
from tts.text import Vocab, normalize, to_jamo

RAW = "data/zeroth"
OUT = "data/proc_zeroth"
MAX_FRAMES = 800


def work(b):
    y, sr = sf.read(io.BytesIO(b), dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    if sr != audio.SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=audio.SR)
    y, _ = librosa.effects.trim(y, top_db=35)
    if len(y) > MAX_FRAMES * audio.HOP:
        return None
    return audio.wav_to_mel(y)


def main():
    os.makedirs(OUT, exist_ok=True)
    kss = pickle.load(open("data/proc/data.pkl", "rb"))
    mean, std = kss["mean"], kss["std"]
    del kss
    vocab = Vocab.load("data/proc/vocab.json")
    allowed = set(vocab.symbols[1:])
    spk_map = {}
    kept = dropped = 0
    with Pool(4) as pool:
        for si, f in enumerate(sorted(glob.glob(f"{RAW}/train-*.parquet"))):
            d = pd.read_parquet(f, columns=["speaker_id", "audio", "text"])
            rows = []
            for _, r in d.iterrows():
                t = normalize(r["text"])
                if not t or any(c not in allowed for c in to_jamo(t)):
                    dropped += 1
                    continue
                rows.append((int(r["speaker_id"]), t, r["audio"]["bytes"]))
            mels = pool.map(work, [r[2] for r in rows], chunksize=16)
            items = []
            for (spk, t, _), m in zip(rows, mels):
                ids = vocab.encode(t)
                if m is None or len(ids) < 4 or len(m) < len(ids):
                    dropped += 1
                    continue
                spk_map.setdefault(spk, 1 + len(spk_map))  # 0 은 KSS
                items.append({"text": t, "ids": np.array(ids, np.int16),
                              "mel": ((m - mean) / std).astype(np.float16), "spk": spk_map[spk]})
            kept += len(items)
            pickle.dump(items, open(f"{OUT}/shard{si}.pkl", "wb"))
            print(f"shard {si}: kept {len(items)} (total kept {kept}, dropped {dropped}, speakers {len(spk_map)})", flush=True)
    json.dump({str(k): v for k, v in spk_map.items()}, open(f"{OUT}/spk_map.json", "w"))
    print("done. n_spk (incl. KSS) =", 1 + len(spk_map))


if __name__ == "__main__":
    main()
