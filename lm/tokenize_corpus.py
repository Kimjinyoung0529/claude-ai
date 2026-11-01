"""말뭉치를 토큰 id 로 바꿔서 uint16 배열로 저장. 문서 사이에는 <eos>.

사용: python -m lm.tokenize_corpus
출력: data/lm/train.bin, data/lm/val.bin
"""
import json
import time

import numpy as np

from lm.bpe import BPE


def run(src, dst, bpe, limit=None):
    out, n, t0 = [], 0, time.time()
    for line in open(src, encoding="utf-8"):
        d = json.loads(line)
        ids = bpe.encode(d["t"] + "\n" + d["x"] + "\n")
        out.append(np.array(ids + [3], dtype=np.uint16))   # 3 = <eos>
        n += 1
        if n % 50000 == 0:
            print(f"  {n}문서, {time.time() - t0:.0f}초", flush=True)
        if limit and n >= limit:
            break
    arr = np.concatenate(out)
    arr.tofile(dst)
    return n, len(arr)


if __name__ == "__main__":
    bpe = BPE.load("data/lm/bpe.json")
    n, t = run("data/lm/val.jsonl", "data/lm/val.bin", bpe)
    print(f"검증 {n}문서 {t / 1e6:.2f}M 토큰")
    n, t = run("data/lm/train.jsonl", "data/lm/train.bin", bpe)
    print(f"학습 {n}문서 {t / 1e6:.1f}M 토큰")
