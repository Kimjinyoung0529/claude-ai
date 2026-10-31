// reports/model-status.json 생성: 감사 보고서 + 변환 시도 기록 + 런타임 지원 확인 결과를 합친다. 근거 없는 'runtime' 표기는 하지 않는다.
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
const audit = JSON.parse(readFileSync('reports/model-audit.json')); const byId = Object.fromEntries(Object.values(audit.groups).flat().map(m => [m.id, m]));
const notes = JSON.parse(existsSync('reports/conversion-notes.json') ? readFileSync('reports/conversion-notes.json') : '{}');
const defaults = {
  'NoemaAI-labs/Noema-1.5-4B-Fast': ['blocked-resource', '원본 safetensors ≈8.4GB. 이 환경의 메모리 한도(≈14GB)에서 fp32 export 불가. qwen3_5 구조라 런타임은 지원.'],
  'NoemaAI-labs/Noema-1.5-4B-Deep': ['blocked-resource', '위와 동일(같은 구조·크기).'],
  'clzoro/Qwen3.5-4B-Claude-distill': ['blocked-resource', '4B(≈8.4GB). qwen3_5 구조: onnx-community/Qwen3.5-4B-ONNX 와 같은 구조라 같은 exporter 레시피가 필요.'],
  'clzoro/Qwen3.5-4B-KIMI-Distill': ['blocked-resource', '위와 동일.'],
  'yasserrmd/glm5.1-distill-onnx': ['onnx-reuse', '기존 ONNX(lfm2, q4 834MB) 존재. 런타임 lfm2 지원 확인됨. 브라우저 로딩·생성은 아직 검증하지 않음.'],
  'kakaocorp/kanana-2-1.3b-instruct': ['blocked-exporter', 'kanana2_tiny 는 사용자 정의 코드(trust_remote_code)이고 Transformers.js 4.3.2 가 지원하지 않는 구조.'],
  'kakaocorp/kanana-2-3b-instruct': ['needs-export', 'qwen3 구조라 런타임 지원. 3B fp32 export 는 메모리 한도 때문에 genai 빌더 경로로 시도 필요.'],
  'K-intelligence/Midm-2.0-Mini-Instruct': ['needs-export', 'llama 구조(MIT). optimum fp32 export 는 RSS 13.9GB 에서 OOM. genai 빌더(int4) 경로 시도.'],
  'LGAI-EXAONE/EXAONE-4.0-1.2B': ['blocked-exporter', 'exaone4 는 optimum exporter 도 Transformers.js 4.3.2 도 지원하지 않음(exaone 만 지원).'],
  'naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B': ['blocked-access', '원본이 gated(auto) — Hugging Face 로그인·접근 승인이 필요해 이 세션에서 받을 수 없음. llama 구조라 승인 후에는 Midm 과 같은 경로로 변환 가능.'],
  'ibm-granite/granite-4.2-3b': ['needs-export', 'granite 구조(Apache-2.0), optimum·런타임 모두 지원. 3B 라 메모리 한도 때문에 genai 빌더 경로로 시도 필요.'],
  'LiquidAI/LFM2.5-2.6B': ['blocked-exporter', 'lfm2 는 런타임 지원이나 optimum exporter 미지원. 직접 export 레시피 필요.'],
  'google/gemma-4-E2B-it': ['blocked-exporter', 'gemma4(멀티모달·PLE) 는 런타임이 지원하나 변환 레시피를 이 환경에서 확인하지 못함.'],
  'mistralai/Ministral-3-3B-Instruct-2512': ['blocked-exporter', 'mistral3(멀티모달). 런타임 지원이나 변환 레시피를 이 환경에서 확인하지 못함.'],
};
const models = Object.entries(defaults).map(([id, [status, detail]]) => { const a = byId[id] || {}; const n = notes[id] || {}; return { id, status: n.status || status, detail: n.detail || detail, license: a.license || null, revision: a.revision || null, gated: a.gated || false, ...(n.extra || {}) }; });
writeFileSync('reports/model-status.json', JSON.stringify({ generatedAt: new Date().toISOString(), note: '목록에 있다고 실행되는 것이 아님. status 는 실제 조사·변환 시도 결과.', models }, null, 2)); console.log(models.map(m => m.status + '\t' + m.id).join('\n'));
