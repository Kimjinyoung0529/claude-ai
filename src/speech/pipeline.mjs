// 화자 구분 파이프라인(브라우저 worker 와 Node 평가가 공유). 모델 실행은 주입: runSeg(Float32Array[160000]) → logits, runEmb(Float32Array) → 임베딩.
import { SEG, windowStarts, argmaxFrames, framesToSegments, clusterEmbeddings, mergeSegments, cosine } from './diarize.mjs';
const norm = v => { let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) || 1; return Float32Array.from(v, x => x / s); };
export async function diarizeAudio(audio, { runSeg, runEmb, threshold = 0.55, margin = 2.5, maxEmbSec = 6, onProgress = () => {}, signal } = {}) {
  const sr = SEG.sr, W = SEG.windowSec * sr, dur = audio.length / sr, starts = windowStarts(dur);
  const groups = new Map(); let silentWindows = 0;
  for (let wi = 0; wi < starts.length; wi++) {
    if (signal?.aborted) throw Object.assign(new Error('취소됨'), { name: 'AbortError' });
    const t0 = starts[wi], buf = new Float32Array(W); buf.set(audio.subarray(Math.round(t0 * sr), Math.min(audio.length, Math.round(t0 * sr) + W)));
    const cls = argmaxFrames(await runSeg(buf)); const segs = framesToSegments(cls, t0);
    if (!segs.length) silentWindows++;
    const lo = wi === 0 ? 0 : t0 + margin, hi = wi === starts.length - 1 ? Infinity : t0 + SEG.windowSec - margin; // 윈도 가장자리는 신뢰도가 낮아 중첩 구간 중 중앙부만 채택
    for (const s of segs) { const a = Math.max(s.start, lo), b = Math.min(s.end, hi, dur); if (b - a < 0.2) continue; const c = { ...s, start: a, end: b };
      if (c.kind === 'solo') { const k = wi + ':' + c.local; (groups.get(k) || groups.set(k, { segs: [] }).get(k)).segs.push(c); } else (groups.get('ov') || groups.set('ov', { segs: [], overlap: true }).get('ov')).segs.push(c); }
    onProgress({ stage: 'segmentation', done: wi + 1, total: starts.length });
  }
  // (윈도, 로컬 화자) 단위로 모아 임베딩 → 짧은 총 발화는 미분류
  const solo = [...groups.entries()].filter(([k]) => k !== 'ov').map(([, g]) => g); const embedded = [], skipped = [];
  for (let gi = 0; gi < solo.length; gi++) {
    if (signal?.aborted) throw Object.assign(new Error('취소됨'), { name: 'AbortError' });
    const g = solo[gi], total = g.segs.reduce((s, x) => s + x.end - x.start, 0);
    if (total < SEG.minEmbedSec) { skipped.push(g); continue; }
    const parts = []; let left = maxEmbSec; for (const s of g.segs) { if (left <= 0) break; const len = Math.min(s.end - s.start, left); parts.push(audio.subarray(Math.round(s.start * sr), Math.round((s.start + len) * sr))); left -= len; }
    const cat = new Float32Array(parts.reduce((n, p) => n + p.length, 0)); let o = 0; for (const p of parts) { cat.set(p, o); o += p.length; }
    embedded.push({ g, emb: norm(await runEmb(cat)) }); onProgress({ stage: 'embedding', done: gi + 1, total: solo.length });
  }
  const labels = embedded.length ? clusterEmbeddings(embedded.map(e => e.emb), threshold) : [];
  const segments = []; embedded.forEach((e, i) => e.g.segs.forEach(s => segments.push({ ...s, speaker: labels[i] })));
  // 같은 시간대 중복 제거: 시작 순으로 정렬해 이미 덮인 구간은 자름
  segments.sort((a, b) => a.start - b.start); const merged = mergeSegments(segments);
  const overlap = (groups.get('ov')?.segs || []).map(s => ({ start: s.start, end: s.end }));
  const unclassified = skipped.flatMap(g => g.segs.map(s => ({ start: s.start, end: s.end })));
  return { segments: merged, overlap, unclassified, speakers: new Set(labels).size, durationSec: dur, windows: starts.length, silentWindows, threshold };
}
// 프레임 단위 평가: 최적 매핑(순열) 후 화자 혼동 비율. ref/hyp: [{start,end,speaker}], 0.1초 격자
export function evalDiarization(ref, hyp, dur, step = 0.1) {
  const at = (L, t) => { for (const s of L) if (t >= s.start && t < s.end) return s.speaker; return null; };
  const R = [...new Set(ref.map(s => s.speaker))], H = [...new Set(hyp.map(s => s.speaker))]; const cnt = {};
  let speech = 0, missed = 0, falseAlarm = 0;
  for (let t = 0; t < dur; t += step) { const r = at(ref, t), h = at(hyp, t); if (r != null) { speech++; if (h == null) missed++; else cnt[r + '|' + h] = (cnt[r + '|' + h] || 0) + 1; } else if (h != null) falseAlarm++; }
  let best = 0; const perm = (i, used, sum) => { if (i === R.length) { best = Math.max(best, sum); return; } perm(i + 1, used, sum); for (const h of H) if (!used.has(h)) { used.add(h); perm(i + 1, used, sum + (cnt[R[i] + '|' + h] || 0)); used.delete(h); } };
  perm(0, new Set(), 0); const confusion = speech - missed - best;
  return { speechSec: +(speech * step).toFixed(1), missedPct: +(100 * missed / speech).toFixed(1), falseAlarmPct: +(100 * falseAlarm / speech).toFixed(1), confusionPct: +(100 * confusion / speech).toFixed(1), derPct: +(100 * (missed + falseAlarm + confusion) / speech).toFixed(1), refSpeakers: R.length, hypSpeakers: H.length };
}
