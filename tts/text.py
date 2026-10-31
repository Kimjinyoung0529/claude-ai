"""한글 텍스트 -> 자모 시퀀스 -> id."""
import json
import re
from jamo import h2j

PAD, SPACE = 0, 1


def normalize(s: str) -> str:
    s = re.sub(r"[\"'“”‘’()\[\]]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def to_jamo(s: str) -> str:
    # 초성/중성/종성 conjoining jamo 로 분해 (공백, 문장부호는 유지)
    return "".join(h2j(c) if "가" <= c <= "힣" else c for c in normalize(s))


class Vocab:
    def __init__(self, symbols):
        self.symbols = ["<pad>"] + list(symbols)
        self.idx = {s: i for i, s in enumerate(self.symbols)}

    @classmethod
    def build(cls, texts):
        chars = sorted({c for t in texts for c in to_jamo(t)})
        return cls(chars)

    def encode(self, s: str):
        return [self.idx[c] for c in to_jamo(s) if c in self.idx]

    def save(self, path):
        json.dump(self.symbols[1:], open(path, "w"), ensure_ascii=False)

    @classmethod
    def load(cls, path):
        return cls(json.load(open(path)))

    def __len__(self):
        return len(self.symbols)
