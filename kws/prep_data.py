"""호출어 학습용 부정/소음 데이터 준비.

사용: python -m kws.prep_data noise     # ESC-50 -> 16kHz int16
      python -m kws.prep_data neg       # Zeroth -> 부정 창, 평가용 연속 음성(화자 분리), 실제 '하나' 어려운 부정
출력(data/kws/): noise.pkl, neg_train.pkl, neg_hana_train.pkl, heldout.pkl
- 화자를 학습/평가로 나눠서 평가 오작동률이 학습 화자와 무관하게 측정되도록 한다.
- '하나' 위치는 TTS 정렬기(tts_pre)로 찾는다(음소 단위 정렬).
"""
import glob
import io
import pickle
import random
import sys
import time
from multiprocessing import Pool

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from kws import config as C

OUT = "data/kws"
HELD_EVERY = 10          # 화자 10명 중 1명을 평가용으로 분리
WIN = C.CLIP
WORDS = ["하나"]


def _decode(b):
    y, sr = sf.read(io.BytesIO(b), dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    if sr != C.SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=C.SR)
    return y


def to_i16(y):
    return (np.clip(y, -1, 1) * 32767).astype(np.int16)


# --------------------------------------------------------------------- 소음
def noise():
    clips = []
    for f in sorted(glob.glob(f"{OUT}/noise/*.parquet")):
        d = pd.read_parquet(f)
        for _, r in d.iterrows():
            clips.append({"cat": r["category"], "fold": int(r["fold"]), "wav": to_i16(_decode(r["audio"]["bytes"]))})
    pickle.dump(clips, open(f"{OUT}/noise.pkl", "wb"))
    print(f"소음 {len(clips)}클립, {len(set(c['cat'] for c in clips))}종류 저장")


# --------------------------------------------------------------------- 부정
_ST = {}


def _init_align():
    import torch
    torch.set_num_threads(1)
    from tts.infer import load_tts
    from tts.text import Vocab
    vocab = Vocab.load("data/proc/vocab.json")
    m, mean, std = load_tts("release/tts_pre.pt", vocab)
    _ST.update(model=m, mean=mean, std=std, vocab=vocab)


def _locate(y, text):
    """'하나' 가 말해진 구간(초) 목록을 정렬기로 찾는다. 실패하면 []."""
    import torch
    from tts import audio
    from tts.model import beta_prior, mas
    from tts.text import to_jamo
    m, vocab = _ST["model"], _ST["vocab"]
    chars = list(text)
    ids = vocab.encode(text)
    if len(ids) != sum(len(to_jamo(c)) for c in chars):   # 어휘 밖 글자가 있으면 건너뜀
        return []
    mel = (audio.wav_to_mel(y) - _ST["mean"]) / _ST["std"]
    if len(mel) < len(ids) + 2:
        return []
    with torch.no_grad():
        idt = torch.tensor([ids]); mt = torch.tensor(mel[None].astype(np.float32))
        h, pad, _ = m.encode(idt)
        T, Fm = idt.shape[1], mt.shape[1]
        lp = m.aligner(idt, mt, pad, beta_prior(torch.tensor([T]), torch.tensor([Fm]), T, Fm))
        dur = mas(lp.transpose(1, 2), torch.ones(1, T, dtype=torch.bool), torch.ones(1, Fm, dtype=torch.bool))[0].numpy()
    cum = np.concatenate([[0], np.cumsum(dur)])
    ntok = np.cumsum([0] + [len(to_jamo(c)) for c in chars])
    spans = []
    for w in WORDS:
        for mm in text_finditer(text, w):
            s, e = ntok[mm[0]], ntok[mm[1]]
            spans.append((cum[s] * audio.HOP / C.SR, cum[e] * audio.HOP / C.SR))
    return spans


def text_finditer(text, w):
    i = text.find(w)
    while i >= 0:
        yield (i, i + len(w))
        i = text.find(w, i + 1)


def _shard(args):
    f, held_spk, seed = args
    rng = random.Random(seed)
    d = pd.read_parquet(f, columns=["speaker_id", "audio", "text"])
    train_win, hana, held = [], [], {}
    from tts.text import normalize
    allowed = set(_ST["vocab"].symbols[1:])
    from tts.text import to_jamo
    for _, r in d.iterrows():
        spk = int(r["speaker_id"])
        y = _decode(r["audio"]["bytes"])
        text = normalize(r["text"])
        if spk in held_spk:
            held.setdefault(spk, []).append((to_i16(y), text))
            continue
        # 무작위 부정 창
        for _ in range(2):
            if len(y) > WIN:
                s = rng.randrange(0, len(y) - WIN)
                train_win.append(to_i16(y[s:s + WIN]))
        # 실제 '하나' 어려운 부정
        if any(w in text for w in WORDS) and all(c in allowed for c in to_jamo(text)):
            for (t0, t1) in _locate(y, text):
                a, b = int(t0 * C.SR), int(t1 * C.SR)
                if b - a > WIN - 4000:
                    continue
                for _ in range(8):
                    lo = max(0, b - WIN + 2000); hi = min(len(y) - WIN, a - 2000)
                    if hi <= lo:
                        continue
                    s = rng.randrange(lo, hi)
                    hana.append(to_i16(y[s:s + WIN]))
    return train_win, hana, held


def neg():
    files = sorted(glob.glob("data/zeroth/train-*.parquet"))
    spks = set()
    for f in files:
        spks |= set(pd.read_parquet(f, columns=["speaker_id"])["speaker_id"].astype(int))
    spks = sorted(spks)
    held_spk = set(spks[::HELD_EVERY])
    print(f"화자 {len(spks)}명 중 평가용 {len(held_spk)}명 분리", flush=True)
    t0 = time.time()
    tw, hn, held = [], [], {}
    with Pool(min(4, len(files)), initializer=_init_align) as pool:
        for i, (a, b, c) in enumerate(pool.imap_unordered(_shard, [(f, held_spk, i) for i, f in enumerate(files)])):
            tw += a; hn += b
            for k, v in c.items():
                held.setdefault(k, []).extend(v)
            print(f"  조각 {i + 1}/{len(files)} 완료 ({(time.time() - t0) / 60:.1f}분) 창 {len(tw)} 하나 {len(hn)}", flush=True)
    pickle.dump(np.stack(tw), open(f"{OUT}/neg_train.pkl", "wb"))
    pickle.dump(np.stack(hn) if hn else np.zeros((0, WIN), np.int16), open(f"{OUT}/neg_hana_train.pkl", "wb"))
    secs = sum(len(w) for v in held.values() for w, _ in v) / C.SR
    pickle.dump(held, open(f"{OUT}/heldout.pkl", "wb"))
    print(f"학습 부정 창 {len(tw)}개, 실제 '하나' 부정 {len(hn)}개, 평가용 연속 음성 {secs / 3600:.2f}시간 ({len(held)}명) 저장", flush=True)


if __name__ == "__main__":
    {"noise": noise, "neg": neg}[sys.argv[1]]()
