# 검증 상태 (STATUS)

검증 수준을 4단계로 구분합니다. **iPhone 실기기 검증은 한 건도 없습니다.**

| 표기 | 의미 |
| --- | --- |
| 구현 | 코드가 있음 |
| 단위·로직 | Node 테스트로 확인 (모델 추론 없음) |
| 실행 검증 | 헤드리스 Chromium(데스크톱 VM, 소프트웨어 렌더링)에서 **실제 모델/실제 서버**로 확인 |
| 실기기 | iPhone Safari 실측 — **없음** |

측정 환경: 4코어 샌드박스 VM, 헤드리스 Chromium, WebGPU는 SwiftShader(소프트웨어)뿐. 속도 수치는 **이 VM의 CPU 값**이며 iPhone 성능을 뜻하지 않습니다.

## 기능

| 기능 | 상태 | 근거 |
| --- | --- | --- |
| 대화(스트리밍·중단·분기·재생성·기록) | 실행 검증 | Qwen3-0.6B q8 WASM, "서울" 정답, 중단 시 토큰 수 감소, IndexedDB 저장 (`reports/e2e-llm.json`) |
| 노트 CRUD·할 일·카드·백업/복원 | 단위·로직 + 실행 검증 | 새로고침 후 IndexedDB 유지, 백업 라운드트립 |
| 문장 편집 12종 | 실행 검증(UI) / 프롬프트 기반 | dialog 12개 가로 넘침 없음. 개별 품질은 모델에 의존하며 12종 모두를 실제 생성으로 평가하진 않음 |
| 의미 검색 (multilingual-e5-small q8) | 실행 검증 | "제빵 반죽 오븐" 질의 → "빵 만들기" 노트 1위 (점수 0.86), 3개 노트 소규모 |
| PDF 추출 / OCR | 구현만 | PDF.js·Tesseract 연결 코드와 번들만 있음. 실제 파일로 검증하지 않음. OCR은 언어 데이터를 CDN에서 받음 |
| 녹음·전사(Whisper tiny/base/small) | 실행 검증(영어) | JFK 11초 음성 정확 전사(tiny, 6.4초). 마이크 녹음 자체(MediaRecorder)는 실제 마이크로 검증하지 않음. **한국어 정확도 미측정** |
| 화자 구분 | 실행 검증(영어 합성) | 아래 평가 참고. 한국어·실제 대화·겹침·잡음 미평가 |
| 로컬 도구 29개 | 단위·로직 | 충돌 검사(rev)·승인 정책 포함 테스트 |
| 에이전트(커스텀 JSON 액션) | 실행 검증 | Qwen3-0.6B가 `note_list` 호출 → "노트는 3개입니다…" 완료. 초기엔 형식 오류·시간 초과가 있었고 프롬프트를 줄여 개선. **소형 모델이라 복잡한 작업은 실패 가능** |
| MCP(Streamable HTTP) | 실행 검증 | 로컬 SDK 서버에 연결·도구 목록·호출(40+2=42)·스키마 거부·오류·타임아웃·취소·401·해제. OAuth·stdio는 미지원 |
| PWA(manifest·서비스 워커) | 구현 / 일부 검증 | 설치 가능 여부는 실기기 미확인 |
| 반응형·접근성 | 실행 검증 | 320/393/430/768/1440px 가로 넘침 없음, 포커스 링, reduced-motion, safe-area CSS. VoiceOver 등은 미확인 |

## 가속 엔진

- 구현·단위 테스트: 백엔드 계약, 작업별 후보 선택, 직렬화, 취소, 출력 검증 후 대체 실행, cold/warm 분리 기록, device loss 쿨다운, `ModelResidency`(큰 모델 동시 적재 방지), 출력이 일치한 백엔드만 비교.
- 실제 커널 측정(이 VM, SwiftShader WebGPU, `reports/e2e-misc.json`): 벡터 덧셈 1M — js 4.58ms / **wasm-simd 3.57ms** / webgpu 15.2ms. 행렬곱 256 — js 35.8ms / webgpu 65.9ms. 소프트웨어 GPU라 WebGPU가 느린 것은 정상이며 **실제 GPU 성능이 아님**.
- 측정으로 얻은 정책: 브라우저 CPU(WASM)에서 **int4(MatMulNBits)는 q8보다 약 4배 느림**(같은 Qwen3-0.6B: q8 4.3 tok/s vs q4 0.98 tok/s) → CPU는 q8 우선으로 선택.
- 하지 않은 것: LLM 전체의 다중 백엔드 자동 보정, WebNN 어댑터, ANE 직접 제어, LLM 커널 튜닝. 대화 모델 GPU/CPU 비교 버튼은 있으나 GPU가 실제 가속기인 환경에서 돌려보지 못했습니다.
- "배속" 수치는 표시하지 않습니다(같은 입력 출력 일치가 확인된 측정만 표시).

## 화자 구분 평가 (`reports/diarization-eval-*.json`)

- 데이터: LibriSpeech dev-clean(영어) 화자별 발화를 이어 붙인 합성 대화 2종(61초 2명 / 78초 3명). 정답 화자 수와 일치.
- 화자 혼동 0%, 놓침 14~17%(정답 구간에 문장 내 쉼이 포함됨). INT8 세그멘테이션 DER 14.0~14.6%, FP32 15.2~17.0%.
- 이는 **한국어 평가도, 겹침·잡음·실제 대화 평가도 아닙니다.** 짧은 단독 발화는 미분류로 남깁니다.
- 브라우저 전체 파이프라인(Whisper + 화자)도 JFK(1명)·합성 2화자에서 실제로 돌려 화자 1·2명 표시를 확인했습니다.
- 반복·환각 탐지, 무음 구간 전사 생략(에너지 기반), 청크 이어서 전사, 화자 이름 변경·합치기·문장별 재지정이 구현되어 있습니다. 소형 VAD 모델은 추가하지 않았습니다.

## 모델

실제 상태는 `reports/model-status.json`(앱 설정 화면에도 표시). 원본 메타데이터는 `reports/model-audit.json`(조회 시점의 revision·라이선스·gated).

**브라우저에서 실제 실행 검증됨**
- `onnx-community/Qwen3-0.6B-ONNX` q8 — 로드 15초, 생성·중단·에이전트·UI 대화 확인.
- Whisper tiny(영어 전사), multilingual-e5-small(의미 검색), pyannote-segmentation INT8 + ReDimNet(화자).

**어댑터는 있으나 이 세션에서 실행 검증하지 않음**: Qwen3.5-0.8B/2B/4B, Qwen3-1.7B, DeepSeek-R1-Distill-1.5B (파일 크기·dtype은 HF에서 확인, 로딩은 미검증), Whisper base/small.

**직접 변환·검증**
- `K-intelligence/Midm-2.0-Mini-Instruct`(llama, MIT): onnxruntime-genai int4로 변환. optimum fp32 export는 RSS 13.9GB에서 **OOM**으로 실패해 빌더로 우회. ORT Web에 없는 `GatherBlockQuantized`는 원본 fp16 임베딩 `Gather`+`Cast`로 교체, KV 헤드 차원 고정, 브라우저용 external data 설정 추가.
  - 원본 PyTorch fp32 대비(6 프롬프트, greedy 24토큰): 첫 토큰 5/6 일치, 평균 공통 접두 44%. int4라 동일하지 않음(번역 프롬프트는 원본이 번역하지 않고 되풀이, 변환본은 번역). 정확도 벤치마크 아님.
  - 브라우저 WASM 로딩 26초, "대한민국의 수도는 서울입니다." 생성. **속도 0.13~0.19 tok/s로 CPU에서는 실용 불가**. GPU 경로는 미검증.
  - 산출물(1.93GB)은 저장소에 없음 → 재현: `scripts/convert-llm.py` 대신 아래 "변환 재현" 참고.

**기존 ONNX 재사용**: `yasserrmd/glm5.1-distill-onnx` — 결과는 `reports/e2e-reuse-glm5.1-distill-onnx.json`(있을 때) 참고.

**변환하지 못한 모델과 이유(모델별)**

| 모델 | 구분 | 이유 |
| --- | --- | --- |
| Noema-1.5-4B-Fast / Deep, clzoro Qwen3.5-4B Claude / KIMI | 자원 | 원본 ≈8.4GB. 메모리 한도 ≈14GB에서 fp32 export 불가. 구조(qwen3_5)는 런타임 지원 |
| kakaocorp/kanana-2-1.3b-instruct | 런타임 | `kanana2_tiny` 사용자 정의 구조, Transformers.js 4.3.2 미지원 |
| LGAI-EXAONE/EXAONE-4.0-1.2B | 런타임·exporter | `exaone4` 를 optimum·Transformers.js 모두 지원하지 않음 |
| naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B | 권한 | 원본 gated(auto): HF 로그인·승인 필요. llama 구조라 승인 후 Midm과 같은 절차로 가능 |
| kakaocorp/kanana-2-3b-instruct, ibm-granite/granite-4.2-3b | **시도 안 함** | 구조는 빌더·런타임 지원. 3B int4+임베딩이 2GB 단일 청크 한도를 넘고(청크 분할 필요), CPU int4가 매우 느림이 측정됨. GPU 검증 환경 필요 |
| LiquidAI/LFM2.5-2.6B, gemma-4-E2B-it, Ministral-3-3B | exporter | 런타임은 지원하나 이 환경에서 변환 레시피를 확인하지 못함(시도 안 함) |

"모든 모델 변환 완료"가 아닙니다. 14개 후보 중 직접 변환·검증은 1개(Midm), 기존 ONNX 재사용 1개(검증 결과 위 참고)입니다. 더 적합한 최신 모델로의 교체 제안은 하지 못했습니다(원본 최신성 조사는 HF API 메타데이터 수준).

## 변환 재현

```bash
python3 -m venv .cache/venv2 && .cache/venv2/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.cache/venv2/bin/pip install transformers onnxruntime-genai onnx onnxruntime onnx_ir onnxscript huggingface_hub safetensors accelerate
.cache/venv2/bin/python -m onnxruntime_genai.models.builder -m <HF_ID> -o .cache/genai/m -p int4 -e cpu -c .cache/genai/cache --extra_options hf_token=false
.cache/venv2/bin/python scripts/finalize-genai.py .cache/genai/m <HF_ID> .cache/converted/m
.cache/venv2/bin/python scripts/patch-embed.py .cache/converted/m <HF_ID>      # GatherBlockQuantized 교체
.cache/venv2/bin/python scripts/ref-generate.py <HF_ID> .cache/ref.json         # 원본 기준값
node scripts/verify-onnx.mjs .cache/converted/m .cache/ref.json reports/verify-m.json
```
결과물을 본인 HF 저장소에 올린 뒤 앱 설정 → "직접 변환한 모델 추가"로 등록합니다.

## 알려진 한계

- iPhone 17 Pro 실기기 성능·메모리·설치(PWA)·마이크 권한·키보드 동작 **전부 미검증**. iOS Safari의 탭 메모리 한도 때문에 1.5GB 이상 모델은 실패할 수 있음.
- Safari는 `navigator.deviceMemory`를 제공하지 않아 메모리 부족 사전 경고가 부정확함.
- 브라우저 Cache API 용량 부족 시 모델이 영구 캐시되지 않을 수 있음(헤드리스 환경에서 `QuotaExceededError` 확인).
- MCP: 브라우저는 서버의 CORS 허용이 필요. 토큰은 메모리에만 두며 새로고침 시 사라짐.
- 배포하지 않음. 기존 사이트와 그 접근 범위는 건드리지 않음.
