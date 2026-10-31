// 음성 worker: Whisper 전사 + (선택) 화자 구분. 모델은 사용 후 main 이 worker 를 종료해 메모리를 반환한다.
import { pipeline, env } from '@huggingface/transformers';
import * as ort from 'onnxruntime-web';
import { ORT_BASE } from './ort-config.mjs';
import { hfHost, hfUrl } from './hf-config.mjs';
import { planChunks, stitchChunks, detectSuspicious } from '../speech/diarize.mjs';
import { diarizeAudio } from '../speech/pipeline.mjs';
const __h = await hfHost(); if (__h) env.remoteHost = __h;
env.allowLocalModels = false; env.backends.onnx.wasm.wasmPaths = ORT_BASE; ort.env.wasm.wasmPaths = ORT_BASE; ort.env.wasm.numThreads = Math.min(4, self.navigator?.hardwareConcurrency || 1);
const post = m => self.postMessage(m); let cancelled = false;
const WHISPER = { tiny: 'onnx-community/whisper-tiny', base: 'onnx-community/whisper-base', small: 'onnx-community/whisper-small' };
const SEG_URL = 'https://huggingface.co/onnx-community/pyannote-segmentation-3.0/resolve/main/onnx/model_int8.onnx';
const EMB_URL = 'https://huggingface.co/OpenVoiceOS/redimnet-b2-vox2-onnx/resolve/main/redimnet_b2_vox2.onnx';
async function cachedBytes(url, label) {
  const c = await caches.open('ns-models'); let r = await c.match(url); const real = url; url = await hfUrl(url);
  if (!r) { r = await fetch(url); if (!r.ok) throw new Error(`${label} 다운로드 실패 (${r.status})`); const total = +r.headers.get('content-length') || 0; const reader = r.body.getReader(); const parts = []; let got = 0;
    for (;;) { const { done, value } = await reader.read(); if (done) break; parts.push(value); got += value.length; post({ type: 'progress', stage: label, loaded: got, total }); if (cancelled) throw Object.assign(new Error('취소됨'), { name: 'AbortError' }); }
    const buf = new Uint8Array(got); let o = 0; for (const p of parts) { buf.set(p, o); o += p.length; } await c.put(real, new Response(buf)); return buf; }
  return new Uint8Array(await r.arrayBuffer());
}
const rms = a => { let s = 0; for (let i = 0; i < a.length; i += 8) s += a[i] * a[i]; return Math.sqrt(s / (a.length / 8)); };
async function run({ audio, lang, size, diarize, startChunk = 0, prior = [] }) {
  cancelled = false; const dur = audio.length / 16000; const t0 = performance.now();
  const fileProg = new Map();
  post({ type: 'stage', stage: 'load-asr', text: `Whisper ${size} 불러오는 중` });
  const asr = await pipeline('automatic-speech-recognition', WHISPER[size], { device: 'wasm', dtype: { encoder_model: 'fp32', decoder_model_merged: 'q8' }, progress_callback: p => { if (p.status === 'progress' && p.total) { fileProg.set(p.file, p); let l = 0, t = 0; for (const f of fileProg.values()) { l += f.loaded; t += f.total; } post({ type: 'progress', stage: 'asr-model', loaded: l, total: t }); } } });
  const plan = planChunks(dur); const parts = [...prior]; const skipped = [];
  for (const c of plan) {
    if (c.i < startChunk) continue; if (cancelled) { post({ type: 'cancelled', parts, nextChunk: c.i }); await asr.dispose?.(); return; }
    post({ type: 'stage', stage: 'transcribe', text: `전사 중 ${c.i + 1}/${plan.length}`, done: c.i, total: plan.length });
    const seg = audio.subarray(Math.floor(c.start * 16000), Math.floor(c.end * 16000));
    if (rms(seg) < 0.003) { skipped.push(c.i); parts.push({ i: c.i, offset: c.start, span: c.end - c.start, chunks: [], silent: true }); post({ type: 'part', part: parts.at(-1) }); continue; } // 무음 구간은 전사하지 않는다(환각 방지)
    const r = await asr(seg, { return_timestamps: true, language: lang === 'auto' ? undefined : lang, task: 'transcribe', chunk_length_s: 30 });
    const chunks = (r.chunks || [{ timestamp: [0, c.end - c.start], text: r.text }]).map(x => ({ start: x.timestamp[0] ?? 0, end: x.timestamp[1] ?? (c.end - c.start), text: x.text })).filter(x => x.text.trim());
    parts.push({ i: c.i, offset: c.start, span: c.end - c.start, chunks }); post({ type: 'part', part: parts.at(-1) });
  }
  await asr.dispose?.(); const asrMs = performance.now() - t0;
  let chunks = stitchChunks(parts.map(p => ({ offset: p.offset, span: p.span, chunks: p.chunks })));
  let dz = null;
  if (diarize) {
    post({ type: 'stage', stage: 'load-diar', text: '화자 모델 불러오는 중' });
    const segB = await cachedBytes(SEG_URL, '화자 분할 모델'), embB = await cachedBytes(EMB_URL, '화자 임베딩 모델');
    const so = { executionProviders: ['wasm'] }; const seg = await ort.InferenceSession.create(segB, so), emb = await ort.InferenceSession.create(embB, so);
    const runSeg = async b => (await seg.run({ input_values: new ort.Tensor('float32', b, [1, 1, b.length]) })).logits.data;
    const runEmb = async b => (await emb.run({ audio: new ort.Tensor('float32', b, [1, 1, b.length]) })).embs.data;
    dz = await diarizeAudio(audio, { runSeg, runEmb, onProgress: p => post({ type: 'stage', stage: 'diarize', text: `화자 분석 ${p.stage === 'segmentation' ? '구간' : '임베딩'} ${p.done}/${p.total}`, done: p.done, total: p.total }) });
    await seg.release(); await emb.release();
  }
  post({ type: 'done', chunks, diar: dz, suspicious: detectSuspicious(chunks, { durationSec: dur }), skippedSilent: skipped, stats: { audioSec: +dur.toFixed(1), asrSec: +(asrMs / 1000).toFixed(1), totalSec: +((performance.now() - t0) / 1000).toFixed(1), model: size } });
}
self.onmessage = async ({ data: m }) => { if (m.type === 'cancel') { cancelled = true; return; } try { await run(m); } catch (e) { post({ type: e.name === 'AbortError' ? 'cancelled' : 'error', message: String(e?.message || e) }); } };
