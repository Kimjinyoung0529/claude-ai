"""호출어 학습 데이터 로더: 합성 양성/부정 + Zeroth 부정 + 소음을 섞어 배치를 만든다.

분리 규칙(평가가 학습과 섞이지 않도록):
- 합성 화자: spk % 10 == 5 는 검증 전용 (학습 제외)
- 소음: ESC-50 fold 5 는 검증/시험 전용
- Zeroth 화자: prep_data 에서 이미 학습/평가용 분리
"""
import pickle

import numpy as np
import torch

from kws import config as C
from kws import augment as A
from kws.model import CLASSES

VAL_SPK = lambda s: s % 10 == 5
VAL_FOLD = 5


def load_all(out="data/kws"):
    synth = pickle.load(open(f"{out}/synth.pkl", "rb"))
    noise = pickle.load(open(f"{out}/noise.pkl", "rb"))
    neg = pickle.load(open(f"{out}/neg_train.pkl", "rb"))
    hana = pickle.load(open(f"{out}/neg_hana_train.pkl", "rb"))
    D = {"neg": neg, "hana": hana,
         "noise_tr": [n["wav"] for n in noise if n["fold"] != VAL_FOLD],
         "noise_va": [n["wav"] for n in noise if n["fold"] == VAL_FOLD]}
    cls_id = {c: i for i, c in enumerate(CLASSES)}
    D["pos_tr"] = {c: [x["wav"] for x in synth if x["cls"] == 1 and x["phrase"] == c and not VAL_SPK(x["spk"])] for c in CLASSES[1:]}
    D["pos_va"] = {c: [x["wav"] for x in synth if x["cls"] == 1 and x["phrase"] == c and VAL_SPK(x["spk"])] for c in CLASSES[1:]}
    D["hneg_tr"] = [x["wav"] for x in synth if x["cls"] == 0 and not VAL_SPK(x["spk"])]
    D["hneg_va"] = [x["wav"] for x in synth if x["cls"] == 0 and VAL_SPK(x["spk"])]
    return D, cls_id


class BatchStream(torch.utils.data.IterableDataset):
    """무한 배치 생성기. 배치 구성 비율은 mix 에 지정."""

    def __init__(self, D, batch=128, mix=None, seed=0):
        self.D, self.batch, self.seed = D, batch, seed
        self.mix = mix or {"pos": 0.25, "hneg": 0.20, "neg": 0.30, "hana": 0.07, "bg": 0.18}

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (info.id if info else 0) * 9973)
        D, B = self.D, self.batch
        kinds = list(self.mix)
        p = np.array([self.mix[k] for k in kinds]); p = p / p.sum()
        while True:
            X = np.zeros((B, C.CLIP), np.float32); y = np.zeros(B, np.int64)
            for i, k in enumerate(rng.choice(kinds, B, p=p)):
                if k == "pos":
                    c = int(rng.integers(1, len(CLASSES)))
                    pool = D["pos_tr"][CLASSES[c]]
                    X[i] = A.make_positive(pool[rng.integers(len(pool))], D["neg"], D["noise_tr"], rng); y[i] = c
                elif k == "hneg":
                    w = D["hneg_tr"][rng.integers(len(D["hneg_tr"]))]
                    X[i] = A.make_positive(w, D["neg"], D["noise_tr"], rng)   # 어려운 부정도 같은 위치/환경으로
                elif k == "neg":
                    X[i] = A.make_negative(D["neg"][rng.integers(len(D["neg"]))], D["noise_tr"], rng)
                elif k == "hana" and len(D["hana"]):
                    X[i] = A.make_negative(D["hana"][rng.integers(len(D["hana"]))], D["noise_tr"], rng)
                else:
                    X[i] = A.make_background(D["noise_tr"], D["neg"], rng)
            yield torch.from_numpy(X), torch.from_numpy(y)


def make_val(D, n_per_class=300, snr=None, seed=123):
    """검증용 양성(합성 검증 화자 + 시험 소음) 고정 세트. snr=None 이면 깨끗한 환경."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(1, len(CLASSES)):
        pool = D["pos_va"][CLASSES[c]]
        for _ in range(n_per_class):
            w = pool[rng.integers(len(pool))].astype(np.float32) / 32768
            w = w / (np.max(np.abs(w)) + 1e-6) * 0.6
            x = A.place(w, C.CLIP, int(0.15 * C.SR), rng)
            if snr is not None:
                n = D["noise_va"][rng.integers(len(D["noise_va"]))].astype(np.float32) / 32768
                x = A.mix_snr(x, n, snr, rng)
            X.append(x.astype(np.float32)); y.append(c)
    return torch.from_numpy(np.stack(X)), torch.tensor(y)
