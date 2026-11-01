"""텍스트 프런트엔드: 숫자/영어/단위 -> 한글 발음, 긴 글 분할.

학습 데이터(KSS)의 표기 관례를 따른다: "칠 년", "세 시", "다섯 병", "삼백만 원",
약어는 붙여서 "아이엠에프", 전화번호/번호는 숫자별 "일일사".
영어 (2수준): 단어는 CMUdict 발음을 한글 소리로 근사, 약어/미등록어는 알파벳 이름으로 읽는다.
"""
import json
import os
import re
import unicodedata

# ----------------------------------------------------------------- 한글 조합
CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ",
        "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]


def compose(c, v, f=""):
    return chr(0xAC00 + (CHO.index(c) * 21 + JUNG.index(v)) * 28 + JONG.index(f))


# ----------------------------------------------------------------- 숫자
SINO_DIG = "영일이삼사오육칠팔구"
DIGIT_READ = "공일이삼사오육칠팔구"
NATIVE_ONES = ["", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉"]
NATIVE_TENS = ["", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔"]
NATIVE_ORD = {1: "첫", 2: "두", 3: "세", 4: "네"}


def _sino4(n):
    th, hu, te, on = n // 1000, n // 100 % 10, n // 10 % 10, n % 10
    s = ""
    if th:
        s += ("" if th == 1 else SINO_DIG[th]) + "천"
    if hu:
        s += ("" if hu == 1 else SINO_DIG[hu]) + "백"
    if te:
        s += ("" if te == 1 else SINO_DIG[te]) + "십"
    if on:
        s += SINO_DIG[on]
    return s


def sino(n):
    if n == 0:
        return "영"
    units = ["", "만", "억", "조", "경"]
    parts, i = [], 0
    while n > 0 and i < len(units):
        chunk = n % 10000
        if chunk:
            s = _sino4(chunk)
            if i == 1 and chunk == 1:
                s = ""  # 일만 -> 만
            parts.append(s + units[i])
        n //= 10000
        i += 1
    return " ".join(reversed(parts))


def native(n):
    """고유어 수 (단위 앞 형태). 100 이상이면 한자어로."""
    if n <= 0 or n >= 100:
        return sino(n)
    t, o = divmod(n, 10)
    if t == 2 and o == 0:
        return "스무"
    return NATIVE_TENS[t] + NATIVE_ONES[o]


NUM = r"\d{1,3}(?:,\d{3})+|\d+"
DEC = rf"(?:{NUM})(?:\.\d+)?"


def _int(s):
    return int(s.replace(",", ""))


def read_number(s, use_native=False):
    if "." in s:
        a, b = s.split(".", 1)
        return f"{sino(_int(a))} 점 " + "".join(SINO_DIG[int(c)] for c in b)
    if len(s) > 1 and s.startswith("0"):
        return "".join(DIGIT_READ[int(c)] for c in s)
    n = _int(s)
    return native(n) if use_native else sino(n)


def read_digits(s):
    return "".join(DIGIT_READ[int(c)] for c in s if c.isdigit())


NATIVE_C = ["마리", "시간", "번째", "켤레", "방울", "조각", "군데", "가지", "그루", "송이",
            "살", "시", "개", "명", "권", "장", "대", "병", "잔", "곳", "채", "벌", "척", "동",
            "통", "알", "끼", "판", "번", "달"]
SINO_C = ["개월", "주년", "학년", "호선", "번지", "페이지", "퍼센트", "년", "월", "일", "분", "초",
          "원", "도", "층", "호", "주", "세", "위", "등", "반", "쪽", "점", "배", "차", "회",
          "기", "부", "편", "화", "절"]
_CNT = "|".join(sorted(NATIVE_C + SINO_C, key=len, reverse=True))
RE_COUNTER = re.compile(rf"(?P<n>{DEC})\s*(?P<c>{_CNT})")

UNITS = {
    "km": "킬로미터", "kg": "킬로그램", "cm": "센티미터", "mm": "밀리미터", "ml": "밀리리터",
    "mg": "밀리그램", "GB": "기가바이트", "MB": "메가바이트", "KB": "킬로바이트", "TB": "테라바이트",
    "kHz": "킬로헤르츠", "MHz": "메가헤르츠", "GHz": "기가헤르츠", "Hz": "헤르츠", "ms": "밀리초",
    "fps": "에프피에스", "px": "픽셀", "m": "미터", "g": "그램", "l": "리터", "L": "리터", "s": "초",
    "h": "시간", "°C": "도", "℃": "도", "°F": "도",
}
_UNIT = "|".join(sorted((re.escape(u) for u in UNITS), key=len, reverse=True))
RE_UNIT = re.compile(rf"(?P<n>{DEC})\s*(?P<u>{_UNIT})(?![A-Za-z])")


def _month(n):
    return {6: "유월", 10: "시월"}.get(n, sino(n) + "월")


def expand_numbers(t):
    # 범위 3~5개 -> 3에서 5개 (단위 처리보다 먼저)
    t = re.sub(r"(?<=\d)\s*~\s*(?=\d)", "에서 ", t)
    # 날짜 2026-10-31 / 2026.10.31 / 2026/10/31
    t = re.sub(r"\b(\d{4})[-./](\d{1,2})[-./](\d{1,2})\b",
               lambda m: f"{sino(int(m[1]))} 년 {_month(int(m[2]))} {sino(int(m[3]))} 일", t)
    # 시각 3:30
    def _time(m):
        h, mi = int(m[1]), int(m[2])
        s = f"{native(h)} 시"
        return s + (f" {sino(mi)} 분" if mi else "")
    t = re.sub(r"\b(\d{1,2}):(\d{2})\b", _time, t)
    # 전화번호
    t = re.sub(r"\b(0\d{1,2})-(\d{3,4})-(\d{4})\b",
               lambda m: " ".join(read_digits(g) for g in m.groups()), t)
    # 퍼센트 / 통화
    t = re.sub(rf"(?P<n>{DEC})\s*%", lambda m: read_number(m["n"]) + " 퍼센트", t)
    t = re.sub(rf"\$\s*(?P<n>{DEC})", lambda m: read_number(m["n"]) + " 달러", t)
    t = re.sub(rf"₩\s*(?P<n>{DEC})", lambda m: read_number(m["n"]) + " 원", t)
    # 라틴 단위
    t = RE_UNIT.sub(lambda m: f"{read_number(m['n'])} {UNITS[m['u']]}", t)

    # 한글 단위
    def _counter(m):
        n_s, c = m["n"], m["c"]
        if c == "번째":
            n = _int(n_s) if "." not in n_s else 0
            return (NATIVE_ORD.get(n) or native(n)) + " 번째"
        if c == "월" and "." not in n_s:
            return _month(_int(n_s))
        return f"{read_number(n_s, use_native=c in NATIVE_C)} {c}"
    t = RE_COUNTER.sub(_counter, t)
    # 음수
    t = re.sub(r"(?<![\w])-(?=\d)", "마이너스 ", t)
    # 남은 숫자
    t = re.sub(DEC, lambda m: read_number(m[0]), t)
    return t


# ----------------------------------------------------------------- 영어
LETTER = {"A": "에이", "B": "비", "C": "씨", "D": "디", "E": "이", "F": "에프", "G": "지", "H": "에이치",
          "I": "아이", "J": "제이", "K": "케이", "L": "엘", "M": "엠", "N": "엔", "O": "오", "P": "피",
          "Q": "큐", "R": "알", "S": "에스", "T": "티", "U": "유", "V": "브이", "W": "더블유",
          "X": "엑스", "Y": "와이", "Z": "제트"}

TERMS = {
    "gpt-4": "지피티 포", "gpt-3": "지피티 쓰리", "gpt": "지피티", "chatgpt": "챗지피티",
    "openai": "오픈에이아이", "anthropic": "앤트로픽", "claude": "클로드", "gemini": "제미나이",
    "llama": "라마", "bert": "버트", "lora": "로라", "rag": "래그", "nasa": "나사", "nvidia": "엔비디아",
    "ai": "에이아이", "api": "에이피아이", "gpu": "지피유", "cpu": "씨피유", "llm": "엘엘엠",
    "python": "파이썬", "github": "깃허브", "linux": "리눅스", "google": "구글", "youtube": "유튜브",
    "iphone": "아이폰", "android": "안드로이드", "windows": "윈도우", "chrome": "크롬",
    "transformer": "트랜스포머", "token": "토큰", "tokens": "토큰", "model": "모델", "models": "모델",
    "attention": "어텐션", "embedding": "임베딩", "prompt": "프롬프트", "dataset": "데이터셋",
    "data": "데이터", "benchmark": "벤치마크", "fine-tuning": "파인튜닝", "finetuning": "파인튜닝",
    "fine": "파인", "tuning": "튜닝", "training": "트레이닝", "inference": "인퍼런스",
    "deep": "딥", "learning": "러닝", "machine": "머신", "neural": "뉴럴", "network": "네트워크",
    "language": "랭귀지", "agent": "에이전트", "agents": "에이전트", "context": "컨텍스트",
    "window": "윈도우", "server": "서버", "cloud": "클라우드", "code": "코드", "open": "오픈",
    "source": "소스", "software": "소프트웨어", "hardware": "하드웨어", "internet": "인터넷",
    "computer": "컴퓨터", "program": "프로그램", "system": "시스템", "service": "서비스",
    "text": "텍스트", "speech": "스피치", "voice": "보이스", "vocoder": "보코더",
    "algorithm": "알고리즘", "database": "데이터베이스", "web": "웹", "app": "앱", "online": "온라인",
    "ok": "오케이", "hello": "헬로", "hi": "하이",
}

_cmu = None


def _cmudict():
    global _cmu
    if _cmu is None:
        try:
            import cmudict
            _cmu = cmudict.dict()
        except Exception:  # 패키지가 없으면 알파벳 낭독으로 대체
            _cmu = {}
    return _cmu


CONS = {"B": "ㅂ", "CH": "ㅊ", "D": "ㄷ", "DH": "ㄷ", "F": "ㅍ", "G": "ㄱ", "HH": "ㅎ", "JH": "ㅈ",
        "K": "ㅋ", "L": "ㄹ", "M": "ㅁ", "N": "ㄴ", "NG": "ㅇ", "P": "ㅍ", "R": "ㄹ", "S": "ㅅ",
        "SH": "ㅅ", "T": "ㅌ", "TH": "ㅅ", "V": "ㅂ", "Z": "ㅈ", "ZH": "ㅈ"}
VOW = {"AA": ["ㅏ"], "AE": ["ㅐ"], "AH": ["ㅓ"], "AO": ["ㅗ"], "AW": ["ㅏ", "ㅜ"], "AY": ["ㅏ", "ㅣ"],
       "EH": ["ㅔ"], "ER": ["ㅓ"], "EY": ["ㅔ", "ㅣ"], "IH": ["ㅣ"], "IY": ["ㅣ"], "OW": ["ㅗ"],
       "OY": ["ㅗ", "ㅣ"], "UH": ["ㅜ"], "UW": ["ㅜ"]}
SHORT = {"AE", "AH", "IH", "UH", "EH", "AA", "AO"}
CODA_OK = {"N": "ㄴ", "M": "ㅁ", "NG": "ㅇ", "L": "ㄹ"}
CODA_STOP = {"K": "ㄱ", "T": "ㅅ", "P": "ㅂ"}
YMAP = {"ㅏ": "ㅑ", "ㅓ": "ㅕ", "ㅗ": "ㅛ", "ㅜ": "ㅠ", "ㅔ": "ㅖ", "ㅐ": "ㅒ", "ㅣ": "ㅣ"}
WMAP = {"ㅏ": "ㅘ", "ㅓ": "ㅝ", "ㅔ": "ㅞ", "ㅐ": "ㅙ", "ㅣ": "ㅟ", "ㅜ": "ㅜ", "ㅗ": "ㅗ"}


def arpa_to_hangul(phones):
    syl = []  # [초성, 중성, 종성, 짧은모음?]
    pend = []

    def epen(c):
        syl.append([CONS[c], "ㅣ" if c in ("SH", "CH", "JH", "ZH") else "ㅡ", "", False])

    def pre_cluster(cons, allow_stop_coda=True):
        long_r = bool(cons) and cons[0] == "R"
        first = True
        for c in cons:
            if c == "R":
                continue
            if first and syl and syl[-1][2] == "":
                if c in CODA_OK:
                    syl[-1][2] = CODA_OK[c]; first = False; continue
                if allow_stop_coda and c in CODA_STOP and syl[-1][3] and not long_r:
                    syl[-1][2] = CODA_STOP[c]; first = False; continue
            first = False
            epen(c)

    for p in phones:
        ph = re.sub(r"\d", "", p)
        if ph in VOW:
            vs, short = VOW[ph], ph in SHORT
            cons, pend = pend, []
            glide = cons.pop() if cons and cons[-1] in ("Y", "W") else None
            onset = cons.pop() if cons else None
            stop_ok = True
            if glide == "W" and onset not in (None, "K", "G", "HH"):
                cons.append(onset); onset = None; stop_ok = False  # tw, dw, sw: W 는 독립 음절
            direct = (not cons) and onset == "L" and bool(syl)
            pre_cluster(cons, stop_ok)
            v0 = vs[0]
            if glide == "Y":
                v0 = YMAP.get(v0, v0)
            elif glide == "W":
                v0 = WMAP.get(v0, v0)
            elif onset == "SH":
                v0 = YMAP.get(v0, v0)
            if onset == "NG":
                if syl and syl[-1][2] == "":
                    syl[-1][2] = "ㅇ"
                cho = "ㅇ"
            elif onset is None:
                cho = "ㅇ"
            else:
                cho = CONS[onset]
            if direct and syl[-1][2] == "":
                syl[-1][2] = "ㄹ"  # 모음 사이 L: 컬러, 올라
            syl.append([cho, v0, "", short])
            for ev in vs[1:]:
                syl.append(["ㅇ", ev, "", False])
        elif ph in CONS or ph in ("Y", "W"):
            pend.append(ph)
    if pend:
        long_r = pend[0] == "R"
        cons = [c for c in pend if c != "R"]
        if syl and cons:
            f, rest = cons[0], cons[1:]
            if f in CODA_OK and syl[-1][2] == "":
                syl[-1][2] = CODA_OK[f]
            elif f in CODA_STOP and syl[-1][3] and syl[-1][2] == "" and not long_r:
                syl[-1][2] = CODA_STOP[f]
            else:
                epen(f)
            for c in rest:
                epen(c)
        else:
            for c in cons:
                epen(c)
    return "".join(compose(c, v, f) for c, v, f, _ in syl)


def spell_letters(word):
    return "".join(LETTER.get(ch.upper(), "") for ch in word if ch.isalpha())


def english_word(word):
    low = word.lower()
    if low in TERMS:
        return TERMS[low]
    if word.isupper() and 2 <= len(word) <= 8 and low not in _cmudict():
        return spell_letters(word)
    if word.isupper() and len(word) <= 5 and len(word) >= 2:
        return spell_letters(word)  # GPU, API 처럼 사전에 있어도 약어는 알파벳으로
    prons = _cmudict().get(low.replace("'", "").replace("’", ""))
    if prons:
        return arpa_to_hangul(prons[0])
    return spell_letters(word)


RE_ENG = re.compile(r"[A-Za-z][A-Za-z0-9'’]*(?:-[A-Za-z0-9]+)*")


def expand_english(t):
    def _tok(m):
        w = m[0]
        before = m.string[:m.start()].rstrip()
        if before[-1:].isdigit() and w in UNITS:
            return w  # 숫자 뒤의 단위(km, GB 등)는 숫자 처리에서 읽는다
        if w.lower() in TERMS:
            return TERMS[w.lower()]
        parts = w.split("-")
        out = []
        for p in parts:
            runs = re.findall(r"[A-Za-z]+|\d+", p)  # H2O -> H, 2, O (숫자는 뒤의 숫자 처리에서 읽음)
            out.append(" ".join(english_word(r) if r.isalpha() else r for r in runs))
        return " ".join(out)
    return RE_ENG.sub(_tok, t)


# ----------------------------------------------------------------- 정리
_ALLOWED = None


def allowed_jamo(vocab_path="data/proc/vocab.json"):
    global _ALLOWED
    if _ALLOWED is None:
        _ALLOWED = set(json.load(open(vocab_path))) if os.path.exists(vocab_path) else None
    return _ALLOWED


def normalize_text(text, vocab_path="data/proc/vocab.json"):
    """원문 -> 모델이 읽을 수 있는 한글/문장부호 텍스트. 반환: (텍스트, 경고 목록)."""
    t = unicodedata.normalize("NFC", text)
    t = re.sub(r"[*`#_>]+", " ", t)
    t = t.replace("…", ".").replace("—", ",").replace("–", ",").replace(":", ",").replace(";", ",")
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", ",", t)  # 천 단위 쉼표는 숫자 처리에서 소거
    t = expand_english(t)   # 먼저: GPT-4 같은 영문+숫자 토큰을 통째로 처리
    t = expand_numbers(t)   # 그 뒤: 날짜/시각/단위/숫자 (영어 뒤에 남은 숫자 포함)
    t = re.sub(r"[()\[\]{}]", ", ", t)
    t = re.sub(r"[\"'“”‘’]", "", t)
    warnings = []
    keep = []
    for ch in t:
        if "가" <= ch <= "힣" or ch in " .,?!\n":
            keep.append(ch)
        elif ch.isspace():
            keep.append(" ")
        else:
            warnings.append(ch)
    t = "".join(keep)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s*,\s*(,\s*)+", ", ", t)
    t = re.sub(r"\s+([.,?!])", r"\1", t)
    t = t.strip()
    allowed = allowed_jamo(vocab_path)
    if allowed:
        from jamo import h2j
        bad = sorted({j for ch in t if "가" <= ch <= "힣" for j in h2j(ch) if j not in allowed})
        if bad:
            warnings.append(f"모델 어휘에 없는 자모: {bad}")
    return t, sorted(set(warnings), key=str)


# ----------------------------------------------------------------- 분할
CONNECT = re.compile(r"(?<=[고며서])\s|(?<=면서)\s|(?<=는데)\s|(?<=은데)\s|(?<=지만)\s|(?<=으나)\s|(?<=니까)\s|(?<=다가)\s")


def _force_split(s, max_chars):
    if len(s) <= max_chars:
        return [s]
    mid = len(s) // 2
    cands = [m.end() for m in CONNECT.finditer(s) if 12 <= m.end() <= len(s) - 12]
    if not cands:
        cands = [m.end() for m in re.finditer(r"\s", s) if 12 <= m.end() <= len(s) - 12]
    if not cands:
        return [s]
    cut = min(cands, key=lambda x: abs(x - mid))
    a, b = s[:cut].strip(), s[cut:].strip()
    if not a.endswith((",", ".", "?", "!")):
        a += ","
    return _force_split(a, max_chars) + _force_split(b, max_chars)


def split_text(text, max_chars=64):
    """정규화된 텍스트 -> [(조각, 뒤에 둘 쉼 초)]. 문장 끝은 길게, 쉼표는 짧게."""
    out = []
    paras = [p for p in re.split(r"\n+", text) if p.strip()]
    for pi, para in enumerate(paras):
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para.strip()) if s.strip()]
        for si, s in enumerate(sents):
            pieces = []
            for clause in re.split(r"(?<=,)\s*", s):
                if clause.strip():
                    pieces += _force_split(clause.strip(), max_chars)
            groups, cur = [], ""
            for p in pieces:
                if cur and len(cur) + 1 + len(p) > max_chars:
                    groups.append(cur); cur = p
                else:
                    cur = (cur + " " + p).strip()
            if cur:
                groups.append(cur)
            for gi, g in enumerate(groups):
                last = gi == len(groups) - 1
                if not g.endswith((".", "?", "!", ",")):
                    g += "." if last else ","
                if last:
                    pause = 0.5 if si == len(sents) - 1 and pi < len(paras) - 1 else 0.38
                else:
                    pause = 0.16
                out.append((g, pause))
    if out:
        out[-1] = (out[-1][0], 0.0)
    return out
