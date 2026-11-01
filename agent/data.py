"""에이전트 학습 데이터: 교사 궤적 생성, 글자 토크나이저, 행동 토큰 마스크."""
import json
import os
import pickle
import random
import re

from agent.world import run_oracle

KIND_W = {"calc": 0.10, "lookup": 0.15, "lookup_calc": 0.15, "compare": 0.12, "fs_total": 0.16, "abstain": 0.12, "memory": 0.10}
SPECIAL = ["<pad>", "<unk>", "<task>", "<a>", "<o>", "<end>"]
_SPLIT_RE = re.compile(r"(<task>|<a>|<o>|<end>)")


class Tok:
    def __init__(self, chars):
        self.itos = SPECIAL + sorted(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def build(cls, texts):
        chars = set()
        for t in texts:
            for part in _SPLIT_RE.split(t):
                if part not in SPECIAL:
                    chars |= set(part)
        return cls(chars)

    def encode(self, text):
        ids = []
        for part in _SPLIT_RE.split(text):
            if not part:
                continue
            if part in self.stoi and part in SPECIAL:
                ids.append(self.stoi[part])
            else:
                ids += [self.stoi.get(c, 1) for c in part]
        return ids

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def save(self, path):
        json.dump(self.itos, open(path, "w"), ensure_ascii=False)

    @classmethod
    def load(cls, path):
        t = cls([])
        t.itos = json.load(open(path)); t.stoi = {c: i for i, c in enumerate(t.itos)}
        return t

    def __len__(self):
        return len(self.itos)


def action_mask(tok, ids):
    """행동 줄(<a> 마커 + 호출 + 줄바꿈)과 <end> 만 학습 대상(1). 질문/관찰은 0."""
    A, O, T, E = (tok.stoi[x] for x in ("<a>", "<o>", "<task>", "<end>"))
    mask, on = [], False
    for i in ids:
        if i == A:
            on = True
        elif i in (O, T):
            on = False
        mask.append(1 if (on or i == E) else 0)
    # <a> 줄의 줄바꿈 이후 <o> 가 오기 전까지가 on 이므로, 관찰 줄바꿈은 <o> 로 꺼진다
    return mask


def make_dataset(n, split="train", seed0=0, p_typo=0.25, fail_rate=0.08):
    rng = random.Random(seed0)
    kinds, w = list(KIND_W), list(KIND_W.values())
    out, s = [], seed0 * 1_000_003
    while len(out) < n:
        s += 1
        kind = rng.choices(kinds, w)[0]
        txt, ok, _ = run_oracle(s, split=split, kind=kind, p_typo=p_typo, fail_rate=fail_rate)
        if ok:
            out.append(txt)
    return out


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
    os.makedirs("data/agent", exist_ok=True)
    train = make_dataset(n, "train", seed0=1)
    val = make_dataset(1000, "test", seed0=2)          # 학습에 없던 이름
    tok = Tok.build(train + val + make_dataset(2000, "test", seed0=3))
    tok.save("data/agent/vocab.json")
    pickle.dump({"train": train, "val": val}, open("data/agent/data.pkl", "wb"))
    L = [len(tok.encode(t)) for t in train]
    print(f"학습 궤적 {len(train)}개, 평가 궤적 {len(val)}개, 어휘 {len(tok)}개, 길이 평균 {sum(L) / len(L):.0f} 최대 {max(L)} 토큰")
