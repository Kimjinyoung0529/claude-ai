"""미니 에이전트의 세계: 도구, 지식, 가상 파일, 한국어 과제, 교사(oracle) 정책.

도구 호출 문법 (모델이 한 줄에 하나씩 출력):
  calc(식)  search(검색어)  ls(경로)  read(경로)  note(키=값)  recall(키)  answer(답)
궤적 텍스트:
  <task>질문
  <a>search(서울 인구)
  <o>서울 | 인구 | 9500000
  <a>answer(9500000)
  <end>
세계는 시드로 생성되며 값이 매번 달라지므로, 모델은 값을 외울 수 없고 도구로 알아내야 한다.
"""
import ast
import hashlib
import operator
import random
import re

# ----------------------------------------------------------------- 이름/값 풀
SYL = list("가나다라마바사아자차카타파하고노도로모보소오조초코토포호구누두루무부수우주추쿠투푸후기니디리미비시이지치키티피히")
CITY_SUF = ["시", "군", "읍"]
FAMILY = list("김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남심노하곽성차주우구민나진지엄채원천방공")
GIVEN = ["민수", "지영", "서연", "도윤", "하준", "수빈", "예린", "준호", "서준", "지우", "현우", "유진", "채원", "태민", "나은", "시우",
         "다은", "건우", "소율", "재민", "승현", "아린", "은서", "정우", "주원", "하린", "동현", "미래", "세진", "가온"]
PRODUCTS = ["사과", "배", "귤", "포도", "딸기", "키보드", "마우스", "모니터", "노트북", "의자", "책상", "연필", "공책", "가방", "신발",
            "우산", "텀블러", "이어폰", "충전기", "달력", "수건", "비누", "샴푸", "양말"]
ATTR = {"도시": [("인구", "명", 1000, 9_800_000), ("면적", "제곱킬로미터", 5, 900)],
        "사람": [("나이", "세", 18, 85), ("월급", "만원", 150, 900)],
        "제품": [("가격", "원", 500, 900_000), ("재고", "개", 0, 999)]}
TEST_MOD = 5          # 이름 해시 % 5 == 0 이면 시험 전용 이름 (학습에 없는 이름)


def is_test_name(name):
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % TEST_MOD == 0


def make_name(kind, rng, split):
    for _ in range(200):
        if kind == "도시":
            n = rng.choice(SYL) + rng.choice(SYL) + rng.choice(CITY_SUF)
        elif kind == "사람":
            n = rng.choice(FAMILY) + rng.choice(GIVEN)
        else:
            n = rng.choice(["", "프리미엄 ", "미니 ", "대형 ", "신형 "]) + rng.choice(PRODUCTS)
        if split == "any" or (split == "test") == is_test_name(n):
            return n
    return n


# ----------------------------------------------------------------- 안전한 계산기
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg, ast.Pow: operator.pow}


def _ev(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        a, b = _ev(node.left), _ev(node.right)
        if isinstance(node.op, ast.Pow) and (abs(b) > 6 or abs(a) > 1e6):
            raise ValueError("거듭제곱 범위 초과")
        return _OPS[type(node.op)](a, b)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_ev(node.operand))
    raise ValueError("허용되지 않는 식")


def fmt_num(x):
    if isinstance(x, float):
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return ("%.4f" % x).rstrip("0").rstrip(".")
    return str(x)


class World:
    """한 에피소드의 세계. 지식(facts), 가상 파일(fs), 메모(notes)."""

    def __init__(self, seed, split="any", fail_rate=0.0):
        self.rng = random.Random(seed)
        self.split, self.fail_rate = split, fail_rate
        self.facts, self.names, self.fs, self.notes = {}, {"도시": [], "사람": [], "제품": []}, {}, {}
        for kind in ATTR:
            for _ in range(self.rng.randint(5, 8)):
                n = make_name(kind, self.rng, split)
                if n in self.names[kind]:
                    continue
                self.names[kind].append(n)
                for (attr, unit, lo, hi) in ATTR[kind]:
                    self.facts[(n, attr)] = self.rng.randint(lo, hi)
        # 판매 파일: /sales/2024-01.txt ...
        prods = list({make_name("제품", self.rng, split) for _ in range(5)})
        self.sales_products = prods
        for m in range(1, self.rng.randint(3, 5) + 1):
            rows = [f"{p},{self.rng.randint(1, 60)}" for p in prods if self.rng.random() < 0.8] or [f"{prods[0]},5"]
            self.fs[f"/sales/2024-{m:02d}.txt"] = "상품,수량\n" + "\n".join(rows)
        self.calls = 0

    # ---- 도구
    def execute(self, call):
        """'name(args)' 문자열을 실행하고 관찰 문자열을 돌려준다."""
        self.calls += 1
        m = re.fullmatch(r"\s*(\w+)\((.*)\)\s*", call, flags=re.S)
        if not m:
            return "오류: 형식이 잘못됨"
        name, arg = m.group(1), m.group(2).strip()
        if self.fail_rate and name not in ("answer",) and self.rng.random() < self.fail_rate:
            return "오류: 일시적 실패, 다시 시도하세요"
        if name == "calc":
            try:
                return fmt_num(_ev(ast.parse(arg.replace(",", ""), mode="eval").body))
            except ZeroDivisionError:
                return "오류: 0으로 나눔"
            except Exception:
                return "오류: 계산할 수 없는 식"
        if name == "search":
            toks = [t for t in re.split(r"\s+", arg) if t]
            scored = []
            for (ent, attr), v in self.facts.items():
                s = sum(1 for t in toks if t == ent or t == attr or (len(t) > 1 and t in ent))
                if s and (any(t in ent for t in toks)):
                    scored.append((s + (0.5 if any(t == attr for t in toks) else 0), ent, attr, v))
            scored.sort(key=lambda x: (-x[0], x[1], x[2]))
            return "\n".join(f"{e} | {a} | {v}" for _, e, a, v in scored[:3]) or "결과 없음"
        if name == "ls":
            d = arg.rstrip("/") + "/"
            files = sorted(p for p in self.fs if p.startswith(d))
            return "\n".join(files) if files else "오류: 경로 없음"
        if name == "read":
            return self.fs.get(arg, "오류: 파일 없음")
        if name == "note":
            if "=" not in arg:
                return "오류: 키=값 형식"
            k, v = arg.split("=", 1)
            self.notes[k.strip()] = v.strip()
            return "저장됨"
        if name == "recall":
            return self.notes.get(arg, "오류: 메모 없음")
        if name == "answer":
            return "완료"
        return "오류: 알 수 없는 도구"


# ----------------------------------------------------------------- 과제 + 교사 정책
def norm_ans(s):
    s = re.sub(r"[,\s]", "", str(s))
    try:
        return fmt_num(float(s)) if re.fullmatch(r"-?\d+(\.\d+)?", s) else s
    except Exception:
        return s


class Task:
    def __init__(self, kind, prompt, steps, expected):
        self.kind, self.prompt, self.steps, self.expected = kind, prompt, steps, expected


def _typo(name, rng):
    return name[:-1] if len(name) > 2 else name + "x"


def make_task(world, rng, kind=None, p_typo=0.0):
    """과제 하나와 교사 행동 목록을 만든다. steps: [(call, 'obs 로 부터 계산되는 인자 람다' ...)].
    교사는 관찰을 읽어야 다음 행동을 알 수 있으므로 generator 로 구현한다."""
    kinds = ["calc", "lookup", "lookup_calc", "compare", "fs_total", "abstain", "memory"]
    kind = kind or rng.choice(kinds)
    W = world

    def lookup_steps(ent, attr):
        """검색으로 값을 얻는 교사 행동 (오타 복구 포함). 값을 yield 로 돌려받는다."""
        q = ent
        if rng.random() < p_typo:
            obs = yield f"search({_typo(ent, rng)} {attr})"
            if obs == "결과 없음" or ent not in obs:
                obs = yield f"search({ent} {attr})"
        else:
            obs = yield f"search({ent} {attr})"
        for _ in range(6):
            for line in obs.split("\n"):
                p = [x.strip() for x in line.split("|")]
                if len(p) == 3 and p[0] == ent and p[1] == attr:
                    return p[2]
            if obs.startswith("오류: 일시"):
                obs = yield f"search({ent} {attr})"
            else:
                obs = yield f"search({ent})"
        return None

    if kind == "calc":
        a, b = rng.randint(2, 999), rng.randint(2, 99)
        op = rng.choice(["+", "-", "*"])
        word = {"+": "더하기", "-": "빼기", "*": "곱하기"}[op]
        prompt = rng.choice([f"{a} {word} {b}는 얼마야?", f"{a}{ '' }과 {b}를 {word} 하면?", f"{a} {word} {b} 계산해줘"])
        val = {"+": a + b, "-": a - b, "*": a * b}[op]

        def teacher():
            obs = yield f"calc({a}{op}{b})"
            while obs.startswith("오류: 일시"):
                obs = yield f"calc({a}{op}{b})"
            yield f"answer({obs})"
        return Task(kind, prompt, teacher, norm_ans(val))

    if kind == "lookup":
        k = rng.choice(list(ATTR)); ent = rng.choice(W.names[k]); attr, unit = rng.choice(ATTR[k])[:2]
        prompt = rng.choice([f"{ent}의 {attr}는?", f"{ent} {attr} 알려줘", f"{ent}의 {attr}가 얼마야?"])

        def teacher():
            v = yield from lookup_steps(ent, attr)
            yield f"answer({v})" if v is not None else "answer(알 수 없음)"
        return Task(kind, prompt, teacher, norm_ans(W.facts[(ent, attr)]))

    if kind in ("lookup_calc", "compare"):
        k = rng.choice(list(ATTR)); attr = rng.choice(ATTR[k])[0]
        a, b = rng.sample(W.names[k], 2)
        va, vb = W.facts[(a, attr)], W.facts[(b, attr)]
        if kind == "lookup_calc":
            op = rng.choice(["+", "-"])
            word = "합" if op == "+" else "차이"
            prompt = rng.choice([f"{a}와 {b}의 {attr} {word}은?", f"{a}, {b}의 {attr} {word} 구해줘"])
            val = va + vb if op == "+" else abs(va - vb)
        else:
            op, val = "/", None
            prompt = rng.choice([f"{a}의 {attr}는 {b}의 몇 배야?", f"{a}의 {attr}는 {b}의 {attr}의 몇 배인가요?"])
            val = va / vb if vb else None
            if val is None:
                return make_task(world, rng, "calc", p_typo)

        def teacher():
            x = yield from lookup_steps(a, attr)
            y = yield from lookup_steps(b, attr)
            if x is None or y is None:
                yield "answer(알 수 없음)"; return
            expr = f"{x}+{y}" if (kind == "lookup_calc" and op == "+") else (f"abs({x}-{y})" if False else
                    (f"{x}-{y}" if int(x) >= int(y) else f"{y}-{x}") if kind == "lookup_calc" else f"{x}/{y}")
            obs = yield f"calc({expr})"
            while obs.startswith("오류: 일시"):
                obs = yield f"calc({expr})"
            yield f"answer({obs})"
        return Task(kind, prompt, teacher, norm_ans(val))

    if kind == "fs_total":
        prod = rng.choice(W.sales_products)
        total = 0
        for p, txt in W.fs.items():
            for line in txt.split("\n")[1:]:
                n, q = line.split(",")
                if n == prod:
                    total += int(q)
        prompt = rng.choice([f"/sales 폴더 전체에서 {prod}의 총 판매 수량은?", f"/sales의 모든 파일을 보고 {prod}를 몇 개 팔았는지 합쳐줘"])

        def teacher():
            obs = yield "ls(/sales)"
            while obs.startswith("오류: 일시"):
                obs = yield "ls(/sales)"
            qs = []
            for path in [x for x in obs.split("\n") if x.startswith("/sales/")]:
                txt = yield f"read({path})"
                while txt.startswith("오류: 일시"):
                    txt = yield f"read({path})"
                for line in txt.split("\n")[1:]:
                    if "," in line and line.split(",")[0] == prod:
                        qs.append(line.split(",")[1])
            if not qs:
                yield "answer(0)"; return
            obs = yield "calc(" + "+".join(qs) + ")"
            while obs.startswith("오류: 일시"):
                obs = yield "calc(" + "+".join(qs) + ")"
            yield f"answer({obs})"
        return Task(kind, prompt, teacher, norm_ans(total))

    if kind == "abstain":
        k = rng.choice(list(ATTR)); attr = rng.choice(ATTR[k])[0]
        ent = make_name(k, rng, W.split)
        while ent in W.names[k]:
            ent = make_name(k, rng, W.split)
        prompt = rng.choice([f"{ent}의 {attr}는?", f"{ent} {attr} 알려줘"])

        def teacher():
            obs = yield f"search({ent} {attr})"
            while obs.startswith("오류: 일시"):
                obs = yield f"search({ent} {attr})"
            if ent in obs:      # 이름이 우연히 일치하면 그 값을 쓴다
                v = None
                for line in obs.split("\n"):
                    p = [x.strip() for x in line.split("|")]
                    if len(p) == 3 and p[0] == ent and p[1] == attr:
                        v = p[2]
                yield f"answer({v if v else '알 수 없음'})"
            else:
                obs = yield f"search({ent})"
                yield "answer(알 수 없음)"
        return Task(kind, prompt, teacher, "알수없음")

    if kind == "memory":
        k = rng.choice(list(ATTR)); attr = rng.choice(ATTR[k])[0]
        a, b = rng.sample(W.names[k], 2)
        va, vb = W.facts[(a, attr)], W.facts[(b, attr)]
        prompt = f"{a}의 {attr}를 메모해 두고, {b}의 {attr}와 합쳐서 알려줘"

        def retry(call):
            obs = yield call
            while obs.startswith("오류: 일시"):
                obs = yield call
            return obs

        def teacher():
            x = yield from lookup_steps(a, attr)
            if x is None:
                yield "answer(알 수 없음)"; return
            yield from retry(f"note({a}={x})")
            y = yield from lookup_steps(b, attr)
            if y is None:
                yield "answer(알 수 없음)"; return
            r = yield from retry(f"recall({a})")
            obs = yield from retry(f"calc({r}+{y})")
            yield f"answer({obs})"
        return Task(kind, prompt, teacher, norm_ans(va + vb))


def run_oracle(seed, split="any", kind=None, p_typo=0.0, fail_rate=0.0):
    """교사가 과제를 푸는 전체 궤적을 만든다. 반환: (텍스트, 정답여부, 종류)"""
    rng = random.Random(seed * 7919 + 13)
    W = World(seed, split, fail_rate)
    t = make_task(W, rng, kind, p_typo)
    lines = [f"<task>{t.prompt}"]
    gen = t.steps()
    final = None
    try:
        call = next(gen)
        for _ in range(40):
            lines.append(f"<a>{call}")
            obs = W.execute(call)
            m = re.fullmatch(r"answer\((.*)\)", call, flags=re.S)
            if m:
                final = m.group(1); break
            lines.append(f"<o>{obs}")
            call = gen.send(obs)
    except StopIteration:
        pass
    lines.append("<end>")
    return "\n".join(lines), norm_ans(final) == t.expected if final is not None else False, t.kind


if __name__ == "__main__":
    import collections
    ok = collections.Counter(); n = collections.Counter(); lens = []
    for s in range(3000):
        txt, good, kind = run_oracle(s, p_typo=0.25, fail_rate=0.08)
        n[kind] += 1; ok[kind] += good; lens.append(len(txt))
    print("교사 정답률(오류·오타 주입 상태):", {k: f"{ok[k]}/{n[k]}" for k in n})
    print("궤적 길이(글자) 평균 %.0f, 최대 %d" % (sum(lens) / len(lens), max(lens)))
    for s, kind in [(1, "lookup_calc"), (5, "fs_total"), (9, "abstain")]:
        print("\n--- 예시:", kind); print(run_oracle(s, kind=kind, p_typo=0.6)[0])
