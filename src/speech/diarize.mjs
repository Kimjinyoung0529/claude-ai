// 화자 구분의 순수 로직(모델 추론 제외). 익명 화자 구분이며 실제 신원 식별이 아니다.
// segmentation(pyannote-3.0 powerset) → 단독 발화 구간 → (임베딩) → 전역 cosine 클러스터링 → 전사 정렬.
export const SEG = { windowSec: 10, stepSec: 5, sr: 16000, minEmbedSec: 0.8, silenceCls: 0 };

// powerset 클래스: 0=무음, 1..3=단일 화자(로컬 0..2), 4..6=겹침(두 화자). frames: Float32Array[nFrames*7] logits
export function argmaxFrames(logits, nClasses = 7) {
  const n = logits.length / nClasses, out = new Uint8Array(n);
  for (let i = 0; i < n; i++) { let b = 0, bv = -Infinity; for (let c = 0; c < nClasses; c++) { const v = logits[i * nClasses + c]; if (v > bv) { bv = v; b = c; } } out[i] = b; }
  return out;
}
export const windowStarts = (durSec, w = SEG.windowSec, s = SEG.stepSec) => { const a = []; for (let t = 0; t + w < durSec + s; t += s) { a.push(t); if (t + w >= durSec) break; } return a.length ? a : [0]; };
// 한 윈도의 프레임 분류 → 구간 목록. kind: 'solo'(임베딩 대상) | 'overlap' | (무음은 제외)
export function framesToSegments(cls, winStart, winSec = SEG.windowSec, minFrames = 3) {
  const fd = winSec / cls.length, segs = []; let i = 0;
  while (i < cls.length) { const c = cls[i]; let j = i; while (j < cls.length && cls[j] === c) j++;
    if (c !== SEG.silenceCls && j - i >= minFrames) segs.push({ start: winStart + i * fd, end: winStart + j * fd, kind: c <= 3 ? 'solo' : 'overlap', local: c <= 3 ? c - 1 : null, win: winStart });
    i = j; }
  return segs;
}
export function cosine(a, b) { let d = 0, x = 0, y = 0; for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; x += a[i] * a[i]; y += b[i] * b[i]; } return d / (Math.sqrt(x * y) || 1); }
// 평균 연결 응집 클러스터링. threshold: 이 유사도 이상이면 병합. 반환: 각 입력의 클러스터 번호(등장 순서로 0..)
export function clusterEmbeddings(embs, threshold = 0.6) {
  let cl = embs.map((e, i) => ({ m: [i] }));
  const sim = (A, B) => { let s = 0; for (const i of A.m) for (const j of B.m) s += cosine(embs[i], embs[j]); return s / (A.m.length * B.m.length); };
  for (;;) { let bs = -1, bi = -1, bj = -1; for (let i = 0; i < cl.length; i++) for (let j = i + 1; j < cl.length; j++) { const s = sim(cl[i], cl[j]); if (s > bs) { bs = s; bi = i; bj = j; } }
    if (bs < threshold) break; cl[bi] = { m: [...cl[bi].m, ...cl[bj].m] }; cl.splice(bj, 1); }
  const label = new Array(embs.length); cl.sort((a, b) => Math.min(...a.m) - Math.min(...b.m)).forEach((c, k) => c.m.forEach(i => label[i] = k)); return label;
}
// 겹치는 윈도의 같은 시간대 구간을 병합(같은 화자끼리). 입력은 speaker 가 채워진 solo 구간.
export function mergeSegments(segs, gap = 0.3) {
  const s = [...segs].sort((a, b) => a.start - b.start), out = [];
  for (const x of s) { const l = out[out.length - 1]; if (l && l.speaker === x.speaker && x.start <= l.end + gap) l.end = Math.max(l.end, x.end); else out.push({ ...x }); }
  return out;
}
// 전사 청크 {start,end,text} 에 화자를 붙인다. 최대 겹침 화자, 겹침이 청크의 30% 미만이면 null(미분류).
export function alignSpeakers(chunks, segs, { minRatio = 0.3 } = {}) {
  return chunks.map(c => { const dur = Math.max(1e-3, c.end - c.start), by = {};
    for (const s of segs) { const o = Math.min(c.end, s.end) - Math.max(c.start, s.start); if (o > 0) by[s.speaker] = (by[s.speaker] || 0) + o; }
    const best = Object.entries(by).sort((a, b) => b[1] - a[1])[0];
    return { ...c, speaker: best && best[1] / dur >= minRatio ? Number(best[0]) : null };
  });
}
// ── 화자 수동 편집
export const speakerName = (names, id) => id == null ? '미분류' : (names?.[id] || `화자 ${id + 1}`);
export const renameSpeaker = (names, id, name) => ({ ...names, [id]: name.trim() || undefined });
export const mergeSpeakers = (chunks, from, into) => chunks.map(c => c.speaker === from ? { ...c, speaker: into } : c);
export const reassignChunk = (chunks, index, speaker) => chunks.map((c, i) => i === index ? { ...c, speaker, manual: true } : c);
// ── 환각/반복 전사 탐지. 의심 사유 목록을 반환(자동 삭제하지 않고 사용자 확인을 유도).
const KNOWN = [/구독과 좋아요/, /시청해 주셔서 감사/, /thanks? for watching/i, /subscribe/i, /자막 제공/, /MBC 뉴스/, /^\s*(음+|어+)\s*[.…]*$/];
export function detectSuspicious(chunks, { durationSec } = {}) {
  const flags = []; const norm = t => t.toLowerCase().replace(/[\s.,!?…]+/g, ' ').trim();
  chunks.forEach((c, i) => { const t = norm(c.text); if (!t) return;
    if (i > 0 && norm(chunks[i - 1].text) === t) flags.push({ index: i, reason: '직전 문장과 동일(반복)' });
    const w = t.split(' '); if (w.length >= 6) for (let n = 1; n <= 4; n++) { let rep = 1, max = 1; for (let k = n; k + n <= w.length; k += n) { if (w.slice(k, k + n).join(' ') === w.slice(k - n, k).join(' ')) max = Math.max(max, ++rep); else rep = 1; } if (max >= 4) { flags.push({ index: i, reason: `같은 표현 ${max}회 반복` }); break; } }
    const dur = c.end - c.start; if (dur > 0 && t.replace(/ /g, '').length / dur > 25) flags.push({ index: i, reason: '발화 속도가 비정상적으로 빠름' });
    if (KNOWN.some(r => r.test(c.text))) flags.push({ index: i, reason: '무음/잡음에서 자주 생기는 문구' }); });
  if (durationSec && chunks.length) { const covered = chunks.reduce((s, c) => s + Math.max(0, c.end - c.start), 0); if (covered > durationSec * 1.5) flags.push({ index: -1, reason: '전사 구간이 녹음 길이를 초과' }); }
  return flags;
}
// 긴 녹음 chunk 계획 (Whisper 는 30초 창). 겹침을 두어 경계 단어 손실을 줄이고 재개 가능하게 한다.
export function planChunks(durationSec, { chunkSec = 30, overlapSec = 5 } = {}) {
  const out = []; for (let t = 0, i = 0; t < durationSec; t += chunkSec - overlapSec, i++) { out.push({ i, start: t, end: Math.min(durationSec, t + chunkSec) }); if (t + chunkSec >= durationSec) break; } return out;
}
// 겹치는 청크 전사 병합: 이전 청크 경계(overlap 중간)보다 뒤에서 시작하는 것만 채택.
export function stitchChunks(parts, overlapSec = 5) {
  const out = []; parts.forEach((p, k) => { const cut = k === 0 ? -Infinity : p.offset + overlapSec / 2, end = k === parts.length - 1 ? Infinity : p.offset + p.span - overlapSec / 2;
    for (const c of p.chunks) { const s = p.offset + c.start, e = p.offset + c.end; if (s >= cut && s < end) out.push({ ...c, start: s, end: e }); } }); return out;
}
// 전사 + 화자 → 내보내기
const ts = (t, sep) => { const h = Math.floor(t / 3600), m = Math.floor(t % 3600 / 60), s = Math.floor(t % 60), ms = Math.round((t % 1) * 1000); return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}${sep}${String(ms).padStart(3, '0')}`; };
export function exportTranscript(chunks, fmt, names = {}) {
  const label = c => c.speaker === undefined ? '' : speakerName(names, c.speaker) + ': ';
  if (fmt === 'txt') return chunks.map(c => `[${ts(c.start, '.').slice(0, 8)}] ${label(c)}${c.text.trim()}`).join('\n');
  if (fmt === 'srt') return chunks.map((c, i) => `${i + 1}\n${ts(c.start, ',')} --> ${ts(c.end, ',')}\n${label(c)}${c.text.trim()}\n`).join('\n');
  if (fmt === 'vtt') return 'WEBVTT\n\n' + chunks.map(c => `${ts(c.start, '.')} --> ${ts(c.end, '.')}\n${label(c)}${c.text.trim()}\n`).join('\n');
  throw new Error('지원하지 않는 형식');
}
