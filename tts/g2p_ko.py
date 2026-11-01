"""한국어 표기 -> 표준 발음 변환 (규칙 기반, 단어(공백 단위) 안에서만 적용).

적용 규칙: 겹받침 단순화, 연음(구개음화 포함), ㅎ 규칙(격음화, ㅎ 탈락),
대표음 교체, 경음화(받침 ㄱㄷㅂ + 평음), 비음화, 유음화.
미적용(예외가 많음): ㄴ 첨가, 사이시옷, 용언 어간 경음화, 겹받침 ㄼ 예외 등.
"""
import re

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ",
        "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

DOUBLE = {"ㄳ": ("ㄱ", "ㅅ"), "ㄵ": ("ㄴ", "ㅈ"), "ㄺ": ("ㄹ", "ㄱ"), "ㄻ": ("ㄹ", "ㅁ"),
          "ㄼ": ("ㄹ", "ㅂ"), "ㄽ": ("ㄹ", "ㅅ"), "ㄾ": ("ㄹ", "ㅌ"), "ㄿ": ("ㄹ", "ㅍ"),
          "ㅄ": ("ㅂ", "ㅅ")}
REP = {"ㄲ": "ㄱ", "ㅋ": "ㄱ", "ㅅ": "ㄷ", "ㅆ": "ㄷ", "ㅈ": "ㄷ", "ㅊ": "ㄷ", "ㅌ": "ㄷ", "ㅎ": "ㄷ",
       "ㅍ": "ㅂ", "ㄳ": "ㄱ", "ㄺ": "ㄱ", "ㄵ": "ㄴ", "ㄶ": "ㄴ", "ㄻ": "ㅁ", "ㄼ": "ㄹ", "ㄽ": "ㄹ",
       "ㄾ": "ㄹ", "ㅀ": "ㄹ", "ㅄ": "ㅂ", "ㄿ": "ㅂ"}
TENSE = {"ㄱ": "ㄲ", "ㄷ": "ㄸ", "ㅂ": "ㅃ", "ㅅ": "ㅆ", "ㅈ": "ㅉ"}
ASPIR = {"ㄱ": "ㅋ", "ㄷ": "ㅌ", "ㅈ": "ㅊ", "ㅂ": "ㅍ"}
NASAL = {"ㄱ": "ㅇ", "ㄷ": "ㄴ", "ㅂ": "ㅁ"}


def _split(ch):
    o = ord(ch) - 0xAC00
    return [CHO[o // 588], JUNG[(o % 588) // 28], JONG[o % 28]]


def _join(s):
    return chr(0xAC00 + (CHO.index(s[0]) * 21 + JUNG.index(s[1])) * 28 + JONG.index(s[2]))


def _word(w):
    syl = [_split(c) for c in w]
    n = len(syl)
    for i in range(n - 1):
        f, nc, nv = syl[i][2], syl[i + 1][0], syl[i + 1][1]
        if not f:
            continue
        # --- ㅎ 관련
        if f in ("ㅎ", "ㄶ", "ㅀ"):
            rest = {"ㅎ": "", "ㄶ": "ㄴ", "ㅀ": "ㄹ"}[f]
            if nc == "ㅇ":                       # 좋아 -> 조아, 많아 -> 마나
                syl[i][2] = ""
                if rest:
                    syl[i + 1][0] = rest
                continue
            if nc in ("ㄱ", "ㄷ", "ㅈ"):          # 좋고 -> 조코, 않다 -> 안타
                syl[i][2] = rest
                syl[i + 1][0] = ASPIR[nc]
                continue
            if nc == "ㄴ":                       # 놓는 -> 논는
                syl[i][2] = "ㄴ" if f != "ㅀ" else "ㄹ"
                if f == "ㅀ":
                    syl[i + 1][0] = "ㄹ"
                continue
            syl[i][2] = rest or "ㄷ"
            f = syl[i][2]
        # --- 뒤 음절이 ㅎ (축하 -> 추카, 입학 -> 이팍)
        if nc == "ㅎ" and f in ("ㄱ", "ㄷ", "ㅂ", "ㅈ", "ㅋ", "ㄲ", "ㅊ", "ㅌ", "ㅍ", "ㄺ"):
            base = {"ㄲ": "ㄱ", "ㅋ": "ㄱ", "ㅊ": "ㅈ", "ㅌ": "ㄷ", "ㅍ": "ㅂ"}.get(f, f)
            if f == "ㄺ":
                syl[i][2] = "ㄹ"
                syl[i + 1][0] = "ㅋ"
            else:
                syl[i][2] = ""
                syl[i + 1][0] = ASPIR[base]
            continue
        # --- 연음
        if nc == "ㅇ" and f != "ㅇ":
            if f in DOUBLE:
                keep, move = DOUBLE[f]
                syl[i][2] = keep
                syl[i + 1][0] = TENSE.get(move, move) if keep in ("ㄱ", "ㅂ") and move == "ㅅ" else move
            else:
                move = f
                syl[i][2] = ""
                if move in ("ㄷ", "ㅌ") and nv == "ㅣ":   # 구개음화: 같이 -> 가치, 굳이 -> 구지
                    move = "ㅈ" if move == "ㄷ" else "ㅊ"
                syl[i + 1][0] = move
            continue
        # --- 용언 ㄺ + ㄱ: 읽고 -> 일꼬, 맑게 -> 말께
        if f == "ㄺ" and nc == "ㄱ":
            syl[i][2] = "ㄹ"
            syl[i + 1][0] = "ㄲ"
            continue
        # --- 대표음
        f = REP.get(f, f)
        syl[i][2] = f
        # --- 경음화
        if f in ("ㄱ", "ㄷ", "ㅂ") and nc in TENSE:
            syl[i + 1][0] = TENSE[nc]
        # --- 비음화
        if f in NASAL and nc in ("ㄴ", "ㅁ"):
            syl[i][2] = NASAL[f]
        elif f in ("ㅁ", "ㅇ") and nc == "ㄹ":        # 심리 -> 심니
            syl[i + 1][0] = "ㄴ"
        elif f in ("ㄱ", "ㅂ") and nc == "ㄹ":        # 독립 -> 동닙
            syl[i][2] = NASAL[f]
            syl[i + 1][0] = "ㄴ"
        # --- 유음화
        elif f == "ㄴ" and nc == "ㄹ":                # 신라 -> 실라
            syl[i][2] = "ㄹ"
        elif f == "ㄹ" and nc == "ㄴ":                # 칼날 -> 칼랄
            syl[i + 1][0] = "ㄹ"
    # 어말 대표음
    if syl[-1][2]:
        syl[-1][2] = REP.get(syl[-1][2], syl[-1][2])
    return "".join(_join(s) for s in syl)


def g2p(text):
    """한글 음절만 변환하고 나머지(공백/문장부호)는 그대로 둔다."""
    return re.sub(r"[가-힣]+", lambda m: _word(m[0]), text)


if __name__ == "__main__":
    cases = {
        "같이": "가치", "굳이": "구지", "국물": "궁물", "신라": "실라", "닭이": "달기", "값이": "갑씨",
        "읽다": "익따", "좋아": "조아", "놓는": "논는", "합니다": "함니다", "먹는": "멍는", "맑게": "말께",
        "좋고": "조코", "않다": "안타", "축하": "추카", "입학": "이팍", "독립": "동닙", "심리": "심니",
        "칼날": "칼랄", "옷이": "오시", "맛있다": "마싣따", "꽃": "꼳", "앞": "압", "많아": "마나",
        "읽고": "일꼬", "학교": "학꾜", "국어": "구거", "믿음": "미듬", "밥물": "밤물",
    }
    bad = 0
    for k, v in cases.items():
        got = g2p(k)
        ok = got == v
        bad += not ok
        print(("OK  " if ok else "FAIL"), k, "->", got, "" if ok else f"(기대 {v})")
    print("실패:", bad)
