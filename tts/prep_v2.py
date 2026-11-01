"""데이터 v2: 프레임별 피치(pyin) + 배속 증강(0.95/1.05) + 발음 변환(g2p) 적용 텍스트.

출력 (data/proc/):
  v2_s100.pkl, v2_s095.pkl, v2_s105.pkl : [{text, ids(g2p), mel(f16, 정규화), f0(f16, Hz, 0=무성), spk, idx}]
  kss_wav_s095.pkl, kss_wav_s105.pkl      : 증강 파형(int16) - 보코더 학습용 (원본은 kss_wav.pkl)
  v2_stats.json                            : log-f0 평균/표준편차(유성 프레임), 에너지 평균/표준편차
배속 s: 파형을 s배 빠르게 재생한 것과 같다(길이 1/s, 피치 x s). 피치는 원본에서 시간축 재표본 후 x s.
사용: python -m tts.prep_v2
"""
import json
import pickle
import time
from multiprocessing import Pool

import librosa
import numpy as np

from tts import audio
from tts.g2p_ko import g2p
from tts.text import Vocab

SPEEDS = [1.0, 0.95, 1.05]
OUT = "data/proc"


def work(args):
    w, mean, std = args
    y = w.astype(np.float32) / 32767
    f0, _, _ = librosa.pyin(y, fmin=70, fmax=450, sr=audio.SR, frame_length=1024,
                            hop_length=audio.HOP, resolution=0.2, n_thresholds=50, fill_na=0.0)
    res = {}
    for s in SPEEDS:
        if s == 1.0:
            ya, fa = y, f0
        else:
            ya = librosa.resample(y, orig_sr=int(audio.SR * s), target_sr=audio.SR)
            n_new = len(ya) // audio.HOP + 1
            idx = np.clip(np.round(np.linspace(0, len(f0) - 1, n_new)).astype(int), 0, len(f0) - 1)
            fa = f0[idx] * s
        mel = audio.wav_to_mel(ya)
        n = min(len(mel), len(fa))
        res[s] = (((mel[:n] - mean) / std).astype(np.float16), fa[:n].astype(np.float16),
                  (np.clip(ya, -1, 1) * 32767).astype(np.int16))
    return res


def main():
    d = pickle.load(open(f"{OUT}/data.pkl", "rb"))
    mean, std, items = d["mean"], d["std"], d["items"]
    wavs = pickle.load(open(f"{OUT}/kss_wav.pkl", "rb"))
    vocab = Vocab.load(f"{OUT}/vocab.json")
    print(f"utts {len(items)}  pyin 시작", flush=True)

    t0 = time.time()
    out = {s: [] for s in SPEEDS}
    wav_out = {s: [] for s in SPEEDS if s != 1.0}
    with Pool(4) as pool:
        for i, res in enumerate(pool.imap(work, ((w, mean, std) for w in wavs), chunksize=8)):
            it = items[i]
            ptext = g2p(it["text"])
            ids = np.array(vocab.encode(ptext), np.int16)
            for s in SPEEDS:
                mel, f0, wv = res[s]
                out[s].append({"text": it["text"], "ptext": ptext, "ids": ids, "mel": mel, "f0": f0,
                               "spk": 0, "idx": i})
                if s != 1.0:
                    wav_out[s].append(wv)
            if (i + 1) % 500 == 0:
                el = (time.time() - t0) / 60
                print(f"{i + 1}/{len(items)}  {el:.1f}분 경과, 남은 약 {el / (i + 1) * (len(items) - i - 1):.0f}분", flush=True)

    # 통계 (원본 속도, 유성 프레임)
    lf0 = np.concatenate([np.log(x["f0"][x["f0"] > 0].astype(np.float32)) for x in out[1.0]])
    en = np.concatenate([np.log(np.exp(x["mel"].astype(np.float32) * std + mean).sum(-1) + 1e-5) for x in out[1.0]])
    stats = {"lf0_mean": float(lf0.mean()), "lf0_std": float(lf0.std()),
             "e_mean": float(en.mean()), "e_std": float(en.std()),
             "mel_mean": float(mean), "mel_std": float(std)}
    json.dump(stats, open(f"{OUT}/v2_stats.json", "w"))
    print("stats", stats, flush=True)
    for s in SPEEDS:
        tag = f"s{int(round(s * 100)):03d}"
        pickle.dump(out[s], open(f"{OUT}/v2_{tag}.pkl", "wb"))
        if s != 1.0:
            pickle.dump(wav_out[s], open(f"{OUT}/kss_wav_{tag}.pkl", "wb"))
    voiced = np.mean([(x["f0"] > 0).mean() for x in out[1.0]])
    print(f"완료. 유성 프레임 비율 {voiced:.2f}, 총 {(time.time() - t0) / 60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
