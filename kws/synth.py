"""호출어 학습용 음성 합성: 다화자 TTS 로 양성/어려운 부정 문구를 만든다.

사용: python -m kws.synth --quick          (20개 시험)
      python -m kws.synth --procs 4        (전체)
화자 0 = tts_v2b(KSS) + vocoder_v2gta, 화자 1..105 = tts_pre(다화자) + vocoder.pt.
변형: TTS 속도(0.8~1.15), 재생 속도/피치(리샘플 0.9~1.1), 무음 패딩은 학습에서 처리.
출력: data/kws/synth.pkl  [{"text","cls","phrase","spk","wav"(int16)}]
"""
import argparse
import os
import pickle
import random
import time
from multiprocessing import Pool

import librosa
import numpy as np
import torch

from kws import config as C

_STATE = {}


def _init():
    torch.set_num_threads(1)
    from tts.infer import load_tts, load_vocoder
    from tts.text import Vocab
    vocab = Vocab.load("data/proc/vocab.json")
    _STATE["vocab"] = vocab
    _STATE["pre"] = load_tts("release/tts_pre.pt", vocab)          # (model, mean, std), g2p 없음
    _STATE["v2"] = load_tts("release/tts_v2b.pt", vocab)           # KSS 화자 0, g2p 있음
    _STATE["voc_pre"] = load_vocoder("release/vocoder.pt")
    _STATE["voc_v2"] = load_vocoder("release/vocoder_v2gta.pt")


def _job(args):
    from tts.infer import synthesize
    text, cls, phrase, spk, speed, rs, seed = args
    try:
        random.seed(seed); torch.manual_seed(seed)
        if spk == 0:
            m, mean, std = _STATE["v2"]; voc = _STATE["voc_v2"]
        else:
            m, mean, std = _STATE["pre"]; voc = _STATE["voc_pre"]
        y = synthesize(text, m, mean, std, voc, spk, speed, _STATE["vocab"])
        if abs(rs - 1.0) > 1e-3:   # 리샘플 = 속도·피치 동시 변화 (화자 다양성 확대)
            y = librosa.resample(y, orig_sr=int(C.SR * rs), target_sr=C.SR)
        y = np.clip(y, -1, 1)
        return {"text": text, "cls": cls, "phrase": phrase, "spk": spk, "wav": (y * 32767).astype(np.int16)}
    except Exception as e:  # 한 건 실패가 전체를 막지 않게
        return {"error": f"{text}/{spk}: {e}"}


def build_jobs(quick, seed=0):
    rng = random.Random(seed)
    jobs = []
    spks = list(range(106))
    # 양성: 변형 id 별 모든 문구 x 모든 화자 x 4가지 속도/피치 조합
    for var, phrases in C.POS.items():
        for text in phrases:
            for spk in spks:
                for _ in range(4):
                    jobs.append((text, 1, var, spk, rng.uniform(0.8, 1.15), rng.uniform(0.9, 1.1), rng.randrange(10**9)))
    # 어려운 부정: 문구마다 무작위 화자 40명 x 3가지
    for text in C.HARD_NEG_SYNTH:
        for spk in rng.sample(spks, 40):
            for _ in range(3):
                jobs.append((text, 0, "neg", spk, rng.uniform(0.8, 1.15), rng.uniform(0.9, 1.1), rng.randrange(10**9)))
    rng.shuffle(jobs)
    return jobs[:20] if quick else jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", default="data/kws/synth.pkl")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    jobs = build_jobs(a.quick)
    npos = sum(j[1] for j in jobs)
    print(f"합성 작업 {len(jobs)}개 (양성 {npos}, 부정 {len(jobs) - npos}), 프로세스 {a.procs}", flush=True)
    t0 = time.time(); out, errs = [], []
    with Pool(a.procs, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs, chunksize=8)):
            (errs if "error" in r else out).append(r if "error" not in r else r["error"])
            if (i + 1) % 500 == 0:
                el = (time.time() - t0) / 60
                print(f"{i + 1}/{len(jobs)}  {el:.1f}분 경과, 남은 약 {el / (i + 1) * (len(jobs) - i - 1):.0f}분", flush=True)
    pickle.dump(out, open(a.out, "wb"))
    dur = np.mean([len(x["wav"]) / C.SR for x in out])
    print(f"완료: {len(out)}개 저장, 실패 {len(errs)}, 평균 길이 {dur:.2f}초, {(time.time() - t0) / 60:.1f}분", flush=True)
    for e in errs[:5]: print("  실패 예:", e)


if __name__ == "__main__":
    main()
