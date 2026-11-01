"""긴 문장 학습용 데이터: KSS 연속 발화 2~3개를 이어 붙인다 -> data/proc/long.pkl

KSS 는 지문 순서대로 정렬돼 있어서 연속 발화를 이어 붙이면 자연스러운 긴 글이 된다.
발화 사이에는 짧은 무음 프레임(데이터에서 뽑은 저에너지 멜)을 넣는다.
"""
import pickle
import random

import numpy as np

from tts.text import Vocab

N_HOLD = 40
N_ITEMS = 6000
MAX_FRAMES = 780


def main():
    random.seed(0)
    d = pickle.load(open("data/proc/data.pkl", "rb"))
    items = d["items"][:-N_HOLD]
    vocab = Vocab.load("data/proc/vocab.json")
    space = vocab.idx[" "]

    sample = np.concatenate([items[i]["mel"].astype(np.float32)
                             for i in random.sample(range(len(items)), 300)])
    silence = np.percentile(sample, 2, axis=0).astype(np.float32)  # (80,)

    out, tries = [], 0
    while len(out) < N_ITEMS and tries < N_ITEMS * 20:
        tries += 1
        n = random.choice([2, 3])
        s = random.randrange(0, len(items) - n)
        grp = items[s:s + n]
        gap = random.randint(6, 16)
        mel = grp[0]["mel"].astype(np.float32)
        ids = list(grp[0]["ids"])
        text = grp[0]["text"]
        for g in grp[1:]:
            mel = np.concatenate([mel, np.tile(silence, (gap, 1)), g["mel"].astype(np.float32)])
            ids += [space] + list(g["ids"])
            text += " " + g["text"]
        if len(mel) > MAX_FRAMES:
            continue
        out.append({"text": text, "ids": np.array(ids, np.int16), "mel": mel.astype(np.float16), "spk": 0})
    pickle.dump(out, open("data/proc/long.pkl", "wb"))
    fr = [len(x["mel"]) for x in out]
    print(f"long items: {len(out)}  frames mean {np.mean(fr):.0f} max {max(fr)}  (tries {tries})")


if __name__ == "__main__":
    main()
