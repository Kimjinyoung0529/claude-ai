// 의미 검색용 임베딩(multilingual-e5-small). 실제 모델 추론.
import { pipeline, env } from '@huggingface/transformers';
import { ORT_BASE } from './ort-config.mjs';
env.allowLocalModels = false; env.backends.onnx.wasm.wasmPaths = ORT_BASE;
let fx = null; const files = new Map();
const post = m => self.postMessage(m);
self.onmessage = async ({ data: m }) => {
  try {
    if (m.type === 'load') { fx = await pipeline('feature-extraction', 'Xenova/multilingual-e5-small', { dtype: 'q8', device: 'wasm', progress_callback: p => { if (p.status === 'progress' && p.total) { files.set(p.file, p); let l = 0, t = 0; for (const f of files.values()) { l += f.loaded; t += f.total; } post({ type: 'progress', loaded: l, total: t }); } } }); post({ type: 'ready' }); }
    else if (m.type === 'embed') { const out = []; for (const t of m.texts) { const r = await fx((m.kind === 'query' ? 'query: ' : 'passage: ') + t.slice(0, 1500), { pooling: 'mean', normalize: true }); out.push(Array.from(r.data)); } post({ type: 'embedded', rid: m.rid, vectors: out }); }
  } catch (e) { post({ type: 'error', rid: m.rid, message: String(e?.message || e) }); }
};
