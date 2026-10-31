// 후보 모델의 실제 Hugging Face 메타데이터(revision/라이선스/접근/ONNX 파일)를 조회해 reports/model-audit.json 생성.
import { writeFileSync, mkdirSync } from 'node:fs';
export const CANDIDATES = {
  runtime: ['onnx-community/Qwen3.5-0.8B-ONNX','onnx-community/Qwen3.5-2B-ONNX','onnx-community/Qwen3.5-4B-ONNX','onnx-community/Qwen3-0.6B-ONNX','onnx-community/Qwen3-1.7B-ONNX','onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX'],
  experimental: ['NoemaAI-labs/Noema-1.5-4B-Fast','NoemaAI-labs/Noema-1.5-4B-Deep','clzoro/Qwen3.5-4B-Claude-distill','clzoro/Qwen3.5-4B-KIMI-Distill','yasserrmd/glm5.1-distill-onnx'],
  korean: ['kakaocorp/kanana-2-1.3b-instruct','kakaocorp/kanana-2-3b-instruct','K-intelligence/Midm-2.0-Mini-Instruct','LGAI-EXAONE/EXAONE-4.0-1.2B','naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B'],
  base: ['ibm-granite/granite-4.2-3b','LiquidAI/LFM2.5-2.6B','google/gemma-4-E2B-it','mistralai/Ministral-3-3B-Instruct-2512'],
  speech: ['onnx-community/whisper-tiny','onnx-community/whisper-base','onnx-community/whisper-small','onnx-community/pyannote-segmentation-3.0','OpenVoiceOS/redimnet-b2-vox2-onnx','Xenova/multilingual-e5-small'],
};
async function audit(id) {
  const r = await fetch(`https://huggingface.co/api/models/${id}`);
  if (!r.ok) return { id, found: false, status: r.status };
  const m = await r.json();
  const files = (m.siblings || []).map(s => s.rfilename);
  const onnx = files.filter(f => f.endsWith('.onnx') || f.endsWith('.onnx_data'));
  const st = files.filter(f => f.endsWith('.safetensors'));
  const tag = (m.tags || []);
  return {
    id, found: true, revision: m.sha, lastModified: m.lastModified,
    gated: m.gated || false, private: m.private || false,
    license: m.cardData?.license || tag.find(t => t.startsWith('license:'))?.slice(8) || null,
    pipeline: m.pipeline_tag || null, library: m.library_name || null,
    arch: (m.config?.architectures || [])[0] || null, modelType: m.config?.model_type || null,
    hasChatTemplate: !!m.config?.tokenizer_config?.chat_template,
    hasOnnx: onnx.length > 0, onnxFiles: onnx.slice(0, 24), safetensorsCount: st.length,
    transformersJsTag: tag.includes('transformers.js'),
    downloads: m.downloads ?? null,
  };
}
if (import.meta.url === `file://${process.argv[1]}`) {
const out = { generatedAt: new Date().toISOString(), note: 'HF API 메타데이터 조회 결과. 변환/검증 상태는 reports/model-status.json 참조.', groups: {} };
for (const [g, ids] of Object.entries(CANDIDATES)) {
  out.groups[g] = [];
  for (const id of ids) { try { out.groups[g].push(await audit(id)); } catch (e) { out.groups[g].push({ id, error: String(e) }); } }
}
mkdirSync('reports', { recursive: true });
writeFileSync('reports/model-audit.json', JSON.stringify(out, null, 2));
for (const [g, l] of Object.entries(out.groups)) for (const m of l) console.log(g.padEnd(12), m.id.padEnd(55), m.found ? `${m.license||'?'} gated=${m.gated} onnx=${m.hasOnnx} st=${m.safetensorsCount} ${m.modelType}` : `NOT FOUND ${m.status||m.error}`);
}
