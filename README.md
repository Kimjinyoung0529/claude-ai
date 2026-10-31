# 내곁 · Night Signal

개인용 로컬 AI 작업실 (iPhone Safari PWA 대상). 대화, AI 메모, 문장 편집, 의미 검색, 녹음·전사·화자 구분, 실험적 에이전트, 외부 MCP 연결을 하나의 앱으로 묶었습니다. 개인 기록과 모델 추론은 기기(브라우저) 안에서 처리합니다.

> 이 저장소는 처음부터 새로 만든 구현입니다. 이전에 로컬에서 만들었다고 한 `naegyeot-gpu`, `sites/naegyeot-night-signal` 소스는 이 세션에 제공되지 않았습니다.
> 배포하지 않았습니다. 사이트(`naegyeot-night-signal.eqkg1v.chatgpt.site`)는 건드리지 않았고 접근 범위도 바꾸지 않았습니다.

## 실행

```bash
npm install
npm run build        # dist/ 생성 (esbuild)
npm run serve        # http://127.0.0.1:5173  (COOP/COEP 켜짐: WASM 스레드)
npm test             # Node 테스트
node scripts/e2e.mjs smoke|dialogs|crud|mcp|llm|speech|misc   # 실제 Chromium e2e
```

- iPhone 설치: HTTPS로 호스팅한 뒤 Safari → 공유 → 홈 화면에 추가. 마이크·서비스 워커는 HTTPS(또는 localhost)가 필요합니다.
- 교차 출처 격리 헤더(COOP/COEP)를 보내지 못하는 호스팅에서는 WASM 스레드가 꺼지고 단일 스레드로 동작합니다.
- 모델은 처음 사용할 때 Hugging Face에서 내려받아 브라우저 Cache Storage에 저장됩니다(설정에서 삭제 가능).
- 외부 MCP 서버를 연결하면 호출 내용이 그 서버로 전송됩니다. 승인 창과 화면 표시로 로컬 처리와 구분합니다.

## 구조

| 경로 | 역할 |
| --- | --- |
| `src/accel/` | 기능 감지, `HybridScheduler`(백엔드 계약·직렬화·취소·출력 검증·대체 실행·cold/warm 측정), `ModelResidency`, JS/WASM-SIMD/WebGPU 커널 백엔드 |
| `src/llm/catalog.mjs` | 실행 어댑터가 연결된 모델과 후보 모델 상태 |
| `src/web/` | PWA UI, LLM·음성·임베딩 Web Worker |
| `src/tools/` | 로컬 도구 29개(노트·할 일·앱 내부 파일·텍스트·계산·시간·해시), 안전한 계산기 |
| `src/agent/harness.mjs` | 커스텀 JSON 액션 에이전트 루프(승인·최대 단계·시간 제한·취소·형식 검증) |
| `src/mcp/connections.mjs` | 공식 SDK 기반 Streamable HTTP 클라이언트(HTTPS 또는 localhost, 스키마 검증, timeout/abort) |
| `src/speech/` | 화자 구분 로직(`diarize.mjs`)과 모델 비의존 파이프라인(`pipeline.mjs`) |
| `scripts/` | 빌드, 정적 서버, 모델 감사·변환, 화자 평가, e2e |
| `reports/` | 모델 감사, 화자 평가, e2e 결과 (아래 `STATUS.md` 참고) |

가속 엔진은 브라우저 GPU 드라이버가 아니라 WebGPU/WASM 위의 스케줄링·측정 계층입니다. 여러 가속기를 연결해도 성능이 합산되지 않으며, 작업마다 한 백엔드를 고릅니다. 비교 수치는 같은 입력에서 출력이 일치한 경우에만 표시합니다.

상세 검증 상태와 한계는 [`STATUS.md`](./STATUS.md)에 있습니다.
