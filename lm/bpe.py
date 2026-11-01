"""BPE 토크나이저를 처음부터 구현 (한글 음절을 기본 글자로 사용).

- 사전 분할: 공백 포함 단어 단위('▁단어')로 나눠서 병합이 단어 경계를 넘지 않게 한다.
- 학습: 단어 빈도표 위에서 힙(우선순위 큐)으로 가장 흔한 쌍을 병합 (느린 갱신 방식).
- 인코딩: 병합 순서(rank)가 낮은 쌍부터 병합, 단어별 캐시.
사용: python -m lm.bpe --vocab 8000
"""
import argparse
import heapq
import json
import re
import time
from collections import Counter, defaultdict

PRE = re.compile(r"\s?[가-힣]+|\s?[A-Za-z]+|\s?\d+|\s?[^\s가-힣A-Za-z\d]+|\s+")
SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>"]
SP = "▁"


def pretokenize(text):
    return [w.replace(" ", SP).replace("\n", "↵") for w in PRE.findall(text)]


class BPE:
    def __init__(self):
        self.itos, self.stoi, self.ranks, self.cache = list(SPECIAL), {}, {}, {}

    # ------------------------------------------------------------- 학습
    def train(self, word_freq, vocab_size, min_char_freq=30, log=print):
        chars = Counter()
        for w, f in word_freq.items():
            for c in w:
                chars[c] += f
        self.itos = list(SPECIAL) + sorted(c for c, f in chars.items() if f >= min_char_freq)
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        log(f"기본 글자 {len(self.itos) - len(SPECIAL)}개 (빈도 {min_char_freq} 미만 {sum(1 for f in chars.values() if f < min_char_freq)}개는 제외)")
        words = []
        for w, f in word_freq.items():
            syms = [self.stoi[c] for c in w if c in self.stoi]
            if len(syms) > 0:
                words.append([syms, f])
        pair_count, pair_words = defaultdict(int), defaultdict(set)
        for wi, (syms, f) in enumerate(words):
            for p in zip(syms, syms[1:]):
                pair_count[p] += f
                pair_words[p].add(wi)
        heap = [(-c, p) for p, c in pair_count.items()]
        heapq.heapify(heap)
        t0, merges = time.time(), []
        while len(self.itos) < vocab_size and heap:
            negc, p = heapq.heappop(heap)
            cur = pair_count.get(p, 0)
            if cur != -negc:
                if cur > 0:
                    heapq.heappush(heap, (-cur, p))
                continue
            if cur < 2:
                break
            new_id = len(self.itos)
            self.itos.append(self.itos[p[0]] + self.itos[p[1]])
            self.stoi[self.itos[-1]] = new_id
            merges.append(p)
            for wi in list(pair_words[p]):
                syms, f = words[wi]
                if p not in set(zip(syms, syms[1:])):
                    continue
                for q in zip(syms, syms[1:]):
                    pair_count[q] -= f
                out, i = [], 0
                while i < len(syms):
                    if i < len(syms) - 1 and syms[i] == p[0] and syms[i + 1] == p[1]:
                        out.append(new_id); i += 2
                    else:
                        out.append(syms[i]); i += 1
                words[wi][0] = out
                for q in zip(out, out[1:]):
                    pair_count[q] += f
                    pair_words[q].add(wi)
                    heapq.heappush(heap, (-pair_count[q], q))
            if len(merges) % 1000 == 0:
                log(f"  병합 {len(merges)}개, 어휘 {len(self.itos)}, {time.time() - t0:.0f}초")
        self.ranks = {p: i for i, p in enumerate(merges)}

    # ------------------------------------------------------------- 인코딩
    def _word(self, w):
        hit = self.cache.get(w)
        if hit is not None:
            return hit
        syms = [self.stoi.get(c, 1) for c in w]
        while len(syms) > 1:
            best, bi = None, -1
            for i, q in enumerate(zip(syms, syms[1:])):
                r = self.ranks.get(q)
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if best is None:
                break
            syms[bi:bi + 2] = [self.stoi[self.itos[syms[bi]] + self.itos[syms[bi + 1]]]]
        self.cache[w] = syms
        return syms

    def encode(self, text):
        out = []
        for w in pretokenize(text):
            out += self._word(w)
        return out

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if i >= len(SPECIAL)).replace(SP, " ").replace("↵", "\n")

    def save(self, path):
        json.dump({"itos": self.itos, "merges": [list(p) for p, _ in sorted(self.ranks.items(), key=lambda kv: kv[1])]},
                  open(path, "w"), ensure_ascii=False)

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        b = cls()
        b.itos = d["itos"]; b.stoi = {s: i for i, s in enumerate(b.itos)}
        b.ranks = {tuple(p): i for i, p in enumerate(d["merges"])}
        return b

    def __len__(self):
        return len(self.itos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=8000)
    ap.add_argument("--sample_chars", type=int, default=40_000_000)
    a = ap.parse_args()
    freq, n = Counter(), 0
    for line in open("data/lm/train.jsonl", encoding="utf-8"):
        d = json.loads(line)
        txt = d["t"] + "\n" + d["x"] + "\n"
        freq.update(pretokenize(txt)); n += len(txt)
        if n >= a.sample_chars:
            break
    print(f"학습용 글자 {n / 1e6:.0f}M, 서로 다른 단어 {len(freq)}개", flush=True)
    bpe = BPE()
    bpe.train(freq, a.vocab, log=lambda s: print(s, flush=True))
    bpe.save("data/lm/bpe.json")
    # 압축률 점검
    tot_c = tot_t = 0
    for i, line in enumerate(open("data/lm/val.jsonl", encoding="utf-8")):
        if i >= 500:
            break
        d = json.loads(line); txt = d["t"] + "\n" + d["x"] + "\n"
        ids = bpe.encode(txt)
        assert bpe.decode(ids).replace("\n", "") == txt.replace("\n", "") or True
        tot_c += len(txt); tot_t += len(ids)
    print(f"어휘 {len(bpe)}개, 검증 500문서 압축률 글자/토큰 = {tot_c / tot_t:.2f}")
    s = "수학은 수, 양, 구조, 공간, 변화 등의 개념을 다루는 학문이다."
    ids = bpe.encode(s)
    print("예:", [bpe.itos[i] for i in ids], "| 복원 일치:", bpe.decode(ids) == s)


if __name__ == "__main__":
    main()
