"""한국어 위키 -> 학습 텍스트. 각 문서의 첫 두 문단(최대 MAXC 글자)만 쓴다.

사용: python -m lm.prep
출력: data/lm/train.jsonl, data/lm/val.jsonl  (한 줄에 {"t": 제목, "x": 본문})
"""
import glob
import hashlib
import json
import re

import pyarrow.parquet as pq

MAXC = 600
MINC = 80


def clean(text):
    paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    paras = [p for p in paras if not p.startswith("=") and len(p) > 20]
    body = " ".join(paras[:2])
    body = re.sub(r"\s+", " ", body)
    body = re.sub(r"\s*\([^()]{0,60}\)", "", body) if False else body       # 괄호는 유지 (원어 표기도 배운다)
    if len(body) > MAXC:
        cut = max(body.rfind("다.", 0, MAXC), body.rfind(". ", 0, MAXC))
        body = body[:cut + 2] if cut > MINC else body[:MAXC]
    return body.strip()


def main():
    tr = open("data/lm/train.jsonl", "w", encoding="utf-8")
    va = open("data/lm/val.jsonl", "w", encoding="utf-8")
    n_tr = n_va = chars = 0
    for f in sorted(glob.glob("data/lm/wiki-*.parquet")):
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=2000, columns=["title", "text"]):
            for r in b.to_pylist():
                body = clean(r["text"])
                if len(body) < MINC:
                    continue
                line = json.dumps({"t": r["title"].strip(), "x": body}, ensure_ascii=False) + "\n"
                if int(hashlib.md5(r["title"].encode()).hexdigest(), 16) % 200 == 0:
                    va.write(line); n_va += 1
                else:
                    tr.write(line); n_tr += 1; chars += len(body) + len(r["title"]) + 1
        print(f"{f} 처리 완료 (학습 {n_tr}, 검증 {n_va})", flush=True)
    print(f"학습 문서 {n_tr}개, 검증 문서 {n_va}개, 학습 글자 {chars / 1e6:.0f}M")


if __name__ == "__main__":
    main()
