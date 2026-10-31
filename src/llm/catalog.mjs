// 모델 카탈로그. runtime=브라우저에서 실행 어댑터가 연결된 모델. 나머지는 후보(status 로 실제 상태를 구분).
// sizes: dtype → 다운로드 MB (HF 파일 크기 합, reports/model-sizes.json 에서 확인한 값)
export const RUNTIME_MODELS = [
  { id: 'onnx-community/Qwen3-0.6B-ONNX', label: 'Qwen3 0.6B', family: 'Qwen3', lang: 'ko·en', license: 'Apache-2.0', thinking: true, sizes: { q4f16: 570, q4: 919, q8: 618 }, rec: 'light', note: '가장 가벼움. 간단한 요약·다듬기' },
  { id: 'onnx-community/Qwen3.5-0.8B-ONNX', label: 'Qwen3.5 0.8B', family: 'Qwen3.5', lang: 'ko·en', license: 'Apache-2.0', thinking: true, sizes: { q4f16: 438, q4: 486 }, rec: 'default', note: '기본 추천. 작고 한국어 대응' },
  { id: 'onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX', label: 'DeepSeek-R1 Distill 1.5B', family: 'Qwen2', lang: 'en·zh', license: '확인 필요', thinking: true, sizes: { q4f16: 1369, q4: 1966 }, rec: 'reasoning', note: '추론 과정을 길게 출력' },
  { id: 'onnx-community/Qwen3-1.7B-ONNX', label: 'Qwen3 1.7B', family: 'Qwen3', lang: 'ko·en', license: 'Apache-2.0', thinking: true, sizes: { q4f16: 1426, q4: 2147 }, rec: 'balanced', note: '품질 균형' },
  { id: 'onnx-community/Qwen3.5-2B-ONNX', label: 'Qwen3.5 2B', family: 'Qwen3.5', lang: 'ko·en', license: 'Apache-2.0', thinking: true, sizes: { q4f16: 1091, q4: 1210 }, rec: 'balanced', note: '2B급 중 가벼운 다운로드' },
  { id: 'onnx-community/Qwen3.5-4B-ONNX', label: 'Qwen3.5 4B', family: 'Qwen3.5', lang: 'ko·en', license: 'Apache-2.0', thinking: true, sizes: { q4f16: 2436, q4: 2702 }, rec: 'heavy', note: 'iPhone 메모리에 부담. 최신 기기에서만 시도', heavy: true },
];
export function pickDtype(model, caps) {
  const keys = Object.keys(model.sizes);
  if (caps.webgpu && caps.fp16 && keys.includes('q4f16')) return 'q4f16';
  const cpu = keys.filter(k => !/f16/.test(k)); // fp16 가중치는 WASM 에서 쓰지 않는다
  return cpu.sort((a, b) => model.sizes[a] - model.sizes[b])[0] || keys[0]; // 가장 작은 다운로드
}
export function pickDevice(caps) { return caps.webgpu ? 'webgpu' : 'wasm'; }
export function modelBytes(model, dtype) { return (model.sizes[dtype] || 0) * 1e6; }
// 메모리 예산 추정: 다운로드 크기 × 1.4 가 상주 메모리 하한이라는 보수적 가정(실측 아님).
export function memoryBudgetOk(model, dtype, caps) {
  const need = modelBytes(model, dtype) * 1.4; const dm = caps.deviceMemoryGB; // Safari 는 deviceMemory 미제공 → null
  return dm ? { ok: need < dm * 1e9 * 0.5, needMB: Math.round(need / 1e6) } : { ok: true, needMB: Math.round(need / 1e6), unknown: true };
}

export const CANDIDATE_GROUPS = [
  { key: 'experimental', title: '실험적·증류 모델', ids: ['NoemaAI-labs/Noema-1.5-4B-Fast', 'NoemaAI-labs/Noema-1.5-4B-Deep', 'clzoro/Qwen3.5-4B-Claude-distill', 'clzoro/Qwen3.5-4B-KIMI-Distill', 'yasserrmd/glm5.1-distill-onnx'] },
  { key: 'korean', title: '한국어 모델', ids: ['kakaocorp/kanana-2-1.3b-instruct', 'kakaocorp/kanana-2-3b-instruct', 'K-intelligence/Midm-2.0-Mini-Instruct', 'LGAI-EXAONE/EXAONE-4.0-1.2B', 'naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B'] },
  { key: 'base', title: '기본 모델', ids: ['ibm-granite/granite-4.2-3b', 'LiquidAI/LFM2.5-2.6B', 'google/gemma-4-E2B-it', 'mistralai/Ministral-3-3B-Instruct-2512'] },
];
// 실제 감사·변환 결과(reports/model-status.json)를 병합해 UI 에 보여준다.
export function mergeStatus(statusJson) {
  const m = new Map((statusJson?.models || []).map(x => [x.id, x]));
  return CANDIDATE_GROUPS.map(g => ({ ...g, items: g.ids.map(id => ({ id, ...(m.get(id) || { status: 'unverified', detail: '아직 조사되지 않음' }) })) }));
}
export const STATUS_LABEL = { runtime: '실행 가능(브라우저 검증됨)', 'onnx-reuse': '기존 ONNX 있음 · 연결 미검증', 'needs-export': '직접 변환 필요 · 미변환', 'blocked-resource': '자원 부족으로 미변환', 'blocked-access': '접근 권한 필요', 'blocked-exporter': '변환기 미지원', converted: '변환됨 · 미검증', unverified: '미확인' };
