"""긴 문장용 데이터 v2: v2 항목(발음 변환 ids + f0)의 연속 발화 2~3개를 이어 붙인다 -> data/proc/long_v2.pkl"""
import pickle
import random

import numpy as np

from tts.text import Vocab

N_HOLD = 40
N_ITEMS = 8000
MAX_FRAMES = 780


def main():
    random.seed(1)
    vocab = Vocab.load("data/proc/vocab.json")
    space = vocab.idx[" "]
    out = []
    for tag in ("s100", "s095", "s105"):
        items = [x for x in pickle.load(open(f"data/proc/v2_{tag}.pkl", "rb")) if x["idx"] < 12854 - N_HOLD]
        sample = np.concatenate([items[i]["mel"].astype(np.float32) for i in random.sample(range(len(items)), 300)])
        silence = np.percentile(sample, 2, axis=0).astype(np.float32)
        target = N_ITEMS if tag == "s100" else N_ITEMS // 4
        got, tries = 0, 0
        while got < target and tries < target * 20:
            tries += 1
            n = random.choice([2, 3])
            s = random.randrange(0, len(items) - n)
            grp = items[s:s + n]
            gap = random.randint(6, 16)
            mel = grp[0]["mel"].astype(np.float32)
            f0 = grp[0]["f0"].astype(np.float32)
            ids = list(grp[0]["ids"])
            text = grp[0]["text"]
            for g in grp[1:]:
                mel = np.concatenate([mel, np.tile(silence, (gap, 1)), g["mel"].astype(np.float32)])
                f0 = np.concatenate([f0, np.zeros(gap, np.float32), g["f0"].astype(np.float32)])
                ids += [space] + list(g["ids"])
                text += " " + g["text"]
            if len(mel) > MAX_FRAMES:
                continue
            out.append({"text": text, "ids": np.array(ids, np.int16), "mel": mel.astype(np.float16),
                        "f0": f0.astype(np.float16), "spk": 0, "idx": -1})
            got += 1
        print(tag, "긴 항목", got)
    pickle.dump(out, open("data/proc/long_v2.pkl", "wb"))
    print("long_v2 총", len(out), "평균 프레임", int(np.mean([len(x["mel"]) for x in out])))


if __name__ == "__main__":
    main()
