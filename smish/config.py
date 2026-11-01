"""스미싱 점검기 설정: 브랜드/공식 도메인(근거 수준 포함), 위험 신호 규칙.

근거 수준(evidence):
  official  = 공식 안내/공식 사이트 내용으로 확인
  site      = 해당 주소에 접속해 그 회사의 사이트임을 확인
  dataset   = 데이터의 정상 문자에서 반복 관찰(보조 근거)
  secondary = 블로그/뉴스 등 2차 출처만
  unverified= 직접 확인 실패(단정 금지: '공식 주소인지 직접 확인하세요' 로만 안내)
브랜드를 추가할 때는 evidence 를 정직하게 적는다.
"""

BRANDS = [
    {"name": "정부24", "aliases": ["정부24", "정부 24", "민원24"], "domains": ["gov.kr"], "evidence": "official",
     "source": "https://www.gov.kr 공식 공지"},
    {"name": "경찰청 교통민원24(이파인)", "aliases": ["교통24", "경찰청교통24", "교통민원24", "경찰교통24", "경찰청민원24", "이파인",
                                              "경찰청교통민원", "경찰청"], "domains": ["efine.go.kr", "police.go.kr"],
     "evidence": "unverified", "source": "직접 접속 실패(리디렉트 과다), 블로그 2차 출처. 정부기관은 .go.kr 이어야 함"},
    {"name": "국민건강보험", "aliases": ["국민건강보험", "건강보험공단", "건보공단", "건강보험"], "domains": ["nhis.or.kr"],
     "evidence": "official", "source": "공단 안내문: 공식 문자에는 대표번호와 www.nhis.or.kr 외 URL 없음"},
    {"name": "금융감독원", "aliases": ["금융감독원", "금감원", "FSS"], "domains": ["fss.or.kr"], "evidence": "secondary",
     "source": "뉴스/소비자경보 인용 (fss.or.kr, phishing-keeper.fss.or.kr)"},
    {"name": "CJ대한통운", "aliases": ["CJ대한통운", "대한통운", "CJ택배", "CJ 대한통운"], "domains": ["cjlogistics.com"],
     "evidence": "site", "source": "접속 확인('CJ Logistics') + 정상 문자 56건에서 관찰"},
    {"name": "한진택배", "aliases": ["한진택배", "한진"], "domains": ["hanjin.com", "hanjin.co.kr"], "evidence": "site",
     "source": "hanjin.com 접속 확인((주)한진), hanjin.co.kr 은 301 로 hanjin.com 이동"},
    {"name": "로젠택배", "aliases": ["로젠택배", "로젠"], "domains": ["ilogen.com"], "evidence": "secondary",
     "source": "블로그 2차 출처, 접속 시 빈 페이지"},
    {"name": "케이뱅크", "aliases": ["케이뱅크"], "domains": ["kbanknow.com"], "evidence": "site",
     "source": "kbanknow.com 접속 확인(케이뱅크 사이트)"},
    {"name": "SKT", "aliases": ["SKT", "SK텔레콤"], "domains": ["tworldfriends.co.kr", "tworld.co.kr", "sktelecom.com", "t-mms.kr"],
     "evidence": "dataset", "source": "정상 문자에서 tworldfriends.co.kr(64), t-mms.kr(128) 반복 관찰"},
]

# 정부·공공기관을 사칭하는 말(링크가 .go.kr / gov.kr / or.kr 이 아니면 근거 있는 불일치)
GOV_CLAIMS = ["정부", "경찰", "검찰", "국세청", "홈택스", "건강보험", "국민연금", "금융감독원", "금감원", "법원", "대검찰청",
              "행정안전부", "민원", "범칙금", "과태료", "환급금", "지원금"]
GOV_OK_SUFFIX = (".go.kr", ".or.kr", ".re.kr")
GOV_OK_DOMAINS = {"gov.kr"}

SHORTENERS = {"bit.ly", "abit.ly", "t.do", "vo.la", "c11.kr", "buly.kr", "tuney.kr", "w0q.de", "me2.do", "han.gl", "tinyurl.com",
              "goo.gl", "t.co", "is.gd", "url.kr", "naver.me", "kakao.com", "ow.ly", "rb.gy", "cutt.ly", "shorturl.at", "bitly.kr",
              "zrr.kr", "m.site.naver.com", "forms.gle"}
# 한국 2단계 등록 접미사(등록 가능한 도메인을 3단계로 계산)
SLD_SUFFIX = {"co.kr", "or.kr", "go.kr", "ne.kr", "re.kr", "pe.kr", "ac.kr", "hs.kr", "ms.kr", "es.kr", "sc.kr", "kg.kr"}

# 위험 신호 규칙: id -> (설명, 정규식, 강도 weak/medium/strong)
RULES = {
    "ask_otp": ("인증번호·OTP·비밀번호를 알려 달라는 요구", r"(인증\s?번호|OTP|보안\s?카드|비밀\s?번호|비번|CVC|CVV|계좌\s?(비밀)?번호).{0,12}(알려|보내|입력|말씀|전송|공유|기재)", "strong"),
    "ask_account": ("계좌·카드 정보를 보내라는 요구", r"(계좌|카드)\s?(번호|정보).{0,10}(보내|알려|입력|전송)", "strong"),
    "install_app": ("앱 설치(apk 등)를 유도", r"(\.apk|앱\s?설치|어플\s?설치|애플리케이션\s?설치|설치\s?후|다운로드\s?(후|하여|하세요)|원격\s?(제어|지원)|보안\s?앱)", "strong"),
    "urgent": ("긴급·기한을 압박", r"(즉시|긴급|지금\s?바로|오늘\s?(안|까지)|금일\s?(까지|중)|24\s?시간\s?(이내|내)|당일\s?(까지|진행)|기한\s?내|미납|연체|정지|차단\s?예정)", "weak"),
    "family": ("가족·지인을 사칭하는 말투", r"^(엄마|아빠|엄니|아들|딸|여보|누나|형|삼촌|이모|고모|나\s?(폰|핸드폰|휴대폰)\s?(이|가)?\s?(고장|망가|떨어|액정|수리))", "strong"),
    "phone_broken": ("'폰이 고장났다'는 전형적 가족 사칭 문구", r"(폰|핸드폰|휴대폰).{0,8}(고장|망가|액정|수리|침수|분실)", "medium"),
    "money_ask": ("송금·입금을 요구", r"(입금|송금|이체|결제).{0,10}(해\s?줘|해주|부탁|요청|바랍|하세요|해\s?야)", "medium"),
    "refund_bait": ("환급금·지원금·당첨 미끼", r"(환급금|지원금|당첨|무료\s?(증정|지급)|보상금|상품권\s?(지급|증정)|캐시백)", "medium"),
    "fine_bait": ("범칙금·과태료·고지서 미끼", r"(범칙금|과태료|교통\s?위반|고지서|체납|세금\s?(미납|환급))", "medium"),
    "delivery_bait": ("택배 주소불일치·반송 미끼", r"(주소\s?(불일치|확인|오류|불명)|반송|배송\s?(불가|지연|실패)|미수령|부재중|배달\s?(불가|지연))", "medium"),
    "overseas_login": ("해외 로그인·IP 차단 미끼", r"(해외\s?(IP|로그인|접속)|비정상\s?(로그인|접속)|도용\s?(위험|우려)|계정\s?(정지|잠금))", "medium"),
    "invest_bait": ("투자·고수익·리딩방 권유", r"(고수익|수익\s?(보장|률)|리딩방|종목\s?추천|VIP\s?(방|리딩)|무료\s?(리딩|상담)|원금\s?보장|코인\s?(투자|추천))", "strong"),
    "job_bait": ("고액 알바·재택 부업 미끼", r"(재택\s?(근무|부업|알바)|부업|일당\s?\d|고액\s?알바|일\s?\d+\s?만\s?원)", "medium"),
    "loan_bait": ("대출 승인·저금리 미끼", r"(대출\s?(승인|가능|한도)|저금리\s?대환|무담보|신용\s?무관)", "medium"),
    "call_me": ("전화·메신저로 유도 (카톡/텔레그램 ID 등)", r"(텔레그램|카톡\s?ID|카카오톡\s?ID|라인\s?ID|오픈\s?채팅|문의\s?(카톡|텔레))", "medium"),
}
