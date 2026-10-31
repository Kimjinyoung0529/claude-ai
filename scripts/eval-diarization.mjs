// 실제 모델(pyannote seg + ReDimNet)을 Node(onnxruntime-node)에서 돌려 합성 다화자 음성으로 평가.
// LibriSpeech dev-clean(영어) 발화를 화자별로 이어 붙여 정답이 있는 '대화'를 만든다. 한국어/실제 대화/겹침 평가가 아니다.
import ort from 'onnxruntime-node'; import { execFileSync } from 'node:child_process'; import { writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { diarizeAudio, evalDiarization } from '../src/speech/pipeline.mjs';
const SEGFILE = process.argv[2] || '.cache/seg.onnx'; mkdirSync('.cache/audio', { recursive: true });
const seg = await ort.InferenceSession.create(SEGFILE), emb = await ort.InferenceSession.create('.cache/redimnet.onnx');
const runSeg = async b => (await seg.run({ input_values: new ort.Tensor('float32', b, [1, 1, b.length]) })).logits.data;
const runEmb = async b => (await emb.run({ audio: new ort.Tensor('float32', b, [1, 1, b.length]) })).embs.data;
async function rows(ds, cfg, split, offsets) { const out = []; for (const o of offsets) { const r = await (await fetch(`https://datasets-server.huggingface.co/rows?dataset=${ds}&config=${cfg}&split=${split}&offset=${o}&length=4`)).json(); out.push(...(r.rows || [])); } return out; }
async function pcm(url, name) { const f = `.cache/audio/${name}.f32`; if (!existsSync(f)) { const a = Buffer.from(await (await fetch(url)).arrayBuffer()); writeFileSync(`.cache/audio/${name}.flac`, a); execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-i', `.cache/audio/${name}.flac`, '-ac', '1', '-ar', '16000', '-f', 'f32le', f]); } const b = (await import('node:fs')).readFileSync(f); return new Float32Array(b.buffer, b.byteOffset, b.length / 4); }
const rs = await rows('openslr/librispeech_asr', 'clean', 'validation', [0, 600, 1300, 2000]);
const by = {}; for (const r of rs) (by[r.row.speaker_id] ||= []).push(r);
const spk = Object.keys(by).slice(0, 3); console.log('LibriSpeech speakers:', spk.map(s => `${s}(${by[s].length}발화)`).join(' '));
const scenarios = [['2화자 교대', [0, 1, 0, 1, 0, 1]], ['3화자 교대', [0, 1, 2, 0, 2, 1, 0]]]; const results = [];
for (const [name, order] of scenarios) {
  const used = {}, ref = [], parts = []; let t = 0; const gap = new Float32Array(8000);
  for (const si of order) { if (si >= spk.length) continue; const list = by[spk[si]]; const r = list[(used[si] = (used[si] ?? -1) + 1) % list.length]; const a = await pcm(r.row.audio[0].src, r.row.id);
    ref.push({ start: t, end: t + a.length / 16000, speaker: si }); parts.push(a, gap); t += a.length / 16000 + 0.5; }
  const audio = new Float32Array(parts.reduce((n, p) => n + p.length, 0)); let o = 0; for (const p of parts) { audio.set(p, o); o += p.length; }
  const t0 = performance.now(); const res = await diarizeAudio(audio, { runSeg, runEmb }); const ms = performance.now() - t0;
  const ev = evalDiarization(ref, res.segments, audio.length / 16000); results.push({ scenario: name, seg: SEGFILE, audioSec: +(audio.length / 16000).toFixed(1), procSec: +(ms / 1000).toFixed(1), ...ev, overlapSec: +res.overlap.reduce((s, x) => s + x.end - x.start, 0).toFixed(1), unclassifiedSec: +res.unclassified.reduce((s, x) => s + x.end - x.start, 0).toFixed(1) });
  console.log(results.at(-1));
}
writeFileSync(`reports/diarization-eval-${SEGFILE.includes('int8') ? 'int8' : 'fp32'}.json`, JSON.stringify({ note: 'LibriSpeech dev-clean(영어) 발화를 이어 붙인 합성 대화. Node onnxruntime-node CPU. 한국어·겹침·잡음·실제 대화 평가 아님.', results }, null, 2));
