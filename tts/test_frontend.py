"""텍스트 프런트엔드 테스트. 사용: python -m tts.test_frontend"""
from tts.frontend import normalize_text, split_text, english_word

NUM_CASES = [
    ("2026년 10월 31일", "이천이십육 년 시월 삼십일 일"),
    ("오후 3시 30분에 만나요", "오후 세 시 삼십 분에 만나요"),
    ("3시에 출발합니다", "세 시에 출발합니다"),
    ("12시", "열두 시"),
    ("사과 3개", "사과 세 개"),
    ("20명이 왔어요", "스무 명이 왔어요"),
    ("21살", "스물한 살"),
    ("100원", "백 원"),
    ("1,500원", "천오백 원"),
    ("300만 원", "삼백만 원"),
    ("10000", "만"),
    ("114에 전화", "백십사에 전화"),
    ("3.5%", "삼 점 오 퍼센트"),
    ("010-1234-5678", "공일공 일이삼사 오육칠팔"),
    ("6월 6일", "유월 육 일"),
    ("5km를 달렸다", "오 킬로미터를 달렸다"),
    ("1번째", "첫 번째"),
    ("7년간", "칠 년간"),
    ("소주 5병", "소주 다섯 병"),
    ("3~5개", "삼에서 다섯 개"),
]

ENG_CASES = [
    ("GPT", "지피티"), ("API", "에이피아이"), ("IMF", "아이엠에프"), ("GPU", "지피유"),
    ("transformer", "트랜스포머"), ("attention", "어텐션"), ("benchmark", "벤치마크"),
    ("network", "네트워크"), ("speech", "스피치"), ("cloud", "클라우드"), ("server", "서버"),
    ("code", "코드"), ("data", "데이터"),
]


def main():
    fails = 0
    for src, exp in NUM_CASES:
        got, _ = normalize_text(src)
        ok = got == exp
        fails += not ok
        print(("OK  " if ok else "FAIL"), src, "->", got, "" if ok else f"(기대: {exp})")
    print()
    for src, exp in ENG_CASES:
        got = english_word(src) if src.upper() != src else english_word(src)
        ok = got == exp
        fails += not ok
        print(("OK  " if ok else "FAIL"), src, "->", got, "" if ok else f"(기대: {exp})")
    t, w = normalize_text("GPT-4와 BERT는 대규모 언어 모델이며, 2026년 10월 31일 오후 3시 30분에 API로 호출할 수 있습니다.")
    print("\n혼합:", t, "| 경고:", w)
    long = ("이 모델은 텍스트를 입력받아 음소 단위로 길이를 예측한 뒤 멜 스펙트로그램을 생성하고 마지막으로 보코더가 "
            "이를 파형으로 변환하며 그 결과를 이어 붙여서 하나의 긴 문장으로 들려줍니다.")
    for c, p in split_text(normalize_text(long)[0]):
        print(f"  [{len(c):2d}자, 쉼 {p}s] {c}")
    print("\n실패:", fails)
    return fails


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
