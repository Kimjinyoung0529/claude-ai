// 대화 모델 전용 Web Worker. Transformers.js 로 실제 ONNX 모델을 로드·생성한다. mock 없음.
import { pipeline, TextStreamer, InterruptableStoppingCriteria, env } from '@huggingface/transformers';
import { ORT_BASE } from './ort-config.mjs';
import { hfHost } from './hf-config.mjs';
const __h = await hfHost(); if (__h) env.remoteHost = __h;
env.allowLocalModels = false; env.backends.onnx.wasm.wasmPaths = ORT_BASE;
let gen = null, stopper = null, info = null, loadAbort = false, genCount = 0;
const post = (m) => self.postMessage(m);
const files = new Map();
function onProgress(p) {
  if (p.status === 'progress' && p.total) { files.set(p.file, { loaded: p.loaded, total: p.total }); let l = 0, t = 0; for (const f of files.values()) { l += f.loaded; t += f.total; } post({ type: 'progress', loaded: l, total: t, file: p.file }); }
  else if (p.status === 'initiate' || p.status === 'download') post({ type: 'status', text: '파일 확인 중…' });
}
async function tryLoad(id, device, dtype) {
  const t0 = performance.now(); const g = await pipeline('text-generation', id, { device, dtype, progress_callback: onProgress });
  return { g, ms: performance.now() - t0 };
}
async function load({ id, device, dtype, fallbackDtype }) {
  files.clear(); loadAbort = false; await unload();
  const plans = device === 'webgpu' ? [['webgpu', dtype], ['wasm', fallbackDtype]] : [['wasm', fallbackDtype || dtype]]; const errors = [];
  for (const [dev, dt] of plans) {
    try { post({ type: 'status', text: `${dev === 'webgpu' ? 'GPU' : 'CPU(WASM)'}로 불러오는 중…` }); const r = await tryLoad(id, dev, dt); gen = r.g; genCount = 0;
      info = { id, backend: dev, dtype: dt, loadMs: Math.round(r.ms), fellBack: dev !== plans[0][0], errors }; post({ type: 'ready', info }); return; }
    catch (e) { errors.push(`${dev}: ${e?.message || e}`); post({ type: 'status', text: `${dev} 실패 → 다른 방법 시도` }); }
  }
  post({ type: 'error', phase: 'load', message: errors.join(' / ') });
}
async function unload() { if (gen) { try { await gen.dispose?.(); } catch {} gen = null; info = null; post({ type: 'unloaded' }); } }
async function generate({ rid, messages, max_new_tokens = 512, temperature = 0.7, thinking = false }) {
  if (!gen) return post({ type: 'error', rid, phase: 'generate', message: '모델이 로드되지 않았습니다' });
  stopper = new InterruptableStoppingCriteria(); const t0 = performance.now(); let first = null, tokens = 0, text = '';
  const cold = genCount++ === 0;
  try {
    const prompt = gen.tokenizer.apply_chat_template(messages, { tokenize: false, add_generation_prompt: true, enable_thinking: thinking });
    const streamer = new TextStreamer(gen.tokenizer, { skip_prompt: true, skip_special_tokens: true, callback_function: (t) => { text += t; post({ type: 'token', rid, text: t }); }, token_callback_function: () => { tokens++; if (first == null) first = performance.now() - t0; } });
    await gen(prompt, { max_new_tokens, do_sample: temperature > 0, temperature: temperature || undefined, top_p: 0.9, streamer, stopping_criteria: stopper, return_full_text: false });
    const total = performance.now() - t0; post({ type: 'done', rid, text, stats: { tokens, ttftMs: Math.round(first ?? total), totalMs: Math.round(total), tokPerSec: tokens > 1 ? +((tokens - 1) / ((total - (first ?? 0)) / 1000)).toFixed(2) : null, cold, backend: info.backend, dtype: info.dtype, stopped: stopper.stop_requested } });
  } catch (e) { post({ type: 'error', rid, phase: 'generate', message: String(e?.message || e), deviceLost: /device.*lost|GPU/i.test(String(e)) }); }
}
self.onmessage = (e) => { const m = e.data; if (m.type === 'load') load(m); else if (m.type === 'generate') generate(m); else if (m.type === 'stop') stopper?.interrupt(); else if (m.type === 'unload') unload(); };
