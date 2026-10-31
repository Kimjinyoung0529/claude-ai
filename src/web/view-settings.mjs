import { h, toast, dialog, confirmDlg, mb, download, progressBar } from './ui.mjs';
import { RUNTIME_MODELS, mergeStatus, STATUS_LABEL, loadCustom, saveCustom, validateCustom } from '../llm/catalog.mjs';
import { taskKey } from '../accel/scheduler.mjs';
export async function mountSettings(ctx, root) {
  const { caps, store, sched } = ctx; const page = h('div', { class: 'page' }); root.append(page);
  const kv = rows => h('dl', { class: 'kv' }, rows.flatMap(([k, v]) => [h('dt', {}, k), h('dd', {}, v)]));
  const yn = b => h('span', { class: 'chip ' + (b ? 'lime' : '') }, b ? '지원' : '미지원');
  page.append(h('h1', {}, '설정'));
  // ── 기기·가속
  const accel = h('div', { class: 'card' }); page.append(h('h2', {}, '기기와 가속'), accel);
  const calOut = h('div');
  function renderAccel() {
    accel.replaceChildren(kv([['WebGPU', yn(caps.webgpu)], ['FP16', yn(caps.fp16)], ['WASM SIMD', yn(caps.simd)], ['스레드(SharedArrayBuffer)', h('span', {}, yn(caps.threads), caps.threads ? '' : ' · 교차 출처 격리 필요')], ['CPU 코어', String(caps.cores)], ['메모리 힌트', caps.deviceMemoryGB ? caps.deviceMemoryGB + ' GB' : '브라우저가 알려주지 않음'], ['저장소', caps.storageQuota ? `${mb(caps.storageQuota.usage)} 사용 / ${mb(caps.storageQuota.quota)}` : '-'], ['WebNN', h('span', {}, yn(caps.webnn), ' · 실행 어댑터 없음'), ], ['Apple NPU(ANE)', '직접 제어 불가 (브라우저가 노출하지 않음)']]),
      h('p', { class: 'sub', style: { marginTop: '10px' } }, '여러 가속기를 연결해도 성능이 합산되지는 않습니다. 작업마다 한 백엔드를 고르며, 아래 측정은 이 기기에서 같은 입력의 출력이 일치한 경우에만 비교합니다.'),
      h('div', { class: 'row' }, h('button', { class: 'btn', onclick: calibrate }, '커널 측정 (벡터 덧셈·행렬곱)'), h('button', { class: 'btn', onclick: llmBench }, '대화 모델 GPU/CPU 측정')), calOut);
  }
  async function calibrate() {
    calOut.replaceChildren(h('p', { class: 'sub' }, '측정 중…')); const rows = [];
    for (const [kind, mk, label] of [['vec.add', n => ({ a: Float32Array.from({ length: n }, (_, i) => Math.sin(i)), b: Float32Array.from({ length: n }, (_, i) => Math.cos(i)) }), '벡터 덧셈'], ['matmul', n => ({ m: n, n, k: n, a: Float32Array.from({ length: n * n }, (_, i) => (i % 7) / 7), b: Float32Array.from({ length: n * n }, (_, i) => (i % 5) / 5) }), '행렬곱']]) {
      for (const n of kind === 'vec.add' ? [4096, 1 << 20] : [64, 256]) { const r = await sched.calibrate({ kind, size: n * (kind === 'matmul' ? n : 1), input: mk(n) }, { runs: 5 }); rows.push([label, n, r]); }
    }
    calOut.replaceChildren(h('table', { class: 'tbl' }, h('tr', {}, ['작업', '크기', '백엔드 (cold / warm 중앙값 ms)', '판정'].map(x => h('th', {}, x))), ...rows.map(([l, n, r]) => h('tr', {}, h('td', {}, l), h('td', {}, String(n)), h('td', {}, r.backends.map(b => b.skipped ? `${b.id}: 건너뜀(${b.skipped})` : b.error ? `${b.id}: 실패` : `${b.id}: ${b.coldMs.toFixed(2)} / ${b.warmMedianMs.toFixed(3)}${b.verified ? '' : ' (출력 불일치)'}`).join('\n')), h('td', {}, r.comparable ? `${r.winner} 우세 ×${r.speedup}` : '비교 불가')))), h('p', { class: 'sub' }, '이 수치는 위 커널에 한정되며 LLM 속도와 같지 않습니다.'));
    sched.metrics.persist();
  }
  async function llmBench() {
    const m = ctx.llm.model?.model; if (!m) return toast('먼저 대화 모델을 불러오세요', true); if (!caps.webgpu) return toast('이 브라우저는 WebGPU를 지원하지 않아 비교할 수 없습니다', true);
    if (!(await confirmDlg('측정 안내', `${m.label}을(를) GPU와 CPU에서 각각 불러와 같은 문장을 두 번씩 생성합니다. 시간이 걸리고 배터리를 씁니다.`, '시작'))) return;
    const res = []; calOut.replaceChildren(h('p', { class: 'sub' }, '측정 중… 화면을 켜 둔 채 기다려 주세요'));
    for (const dev of ['webgpu', 'wasm']) { try { await ctx.loadModel(m, { device: dev }); const runs = []; for (let i = 0; i < 2; i++) runs.push(await ctx.llm.generate([{ role: 'user', content: '대한민국의 수도를 한 문장으로 알려줘.' }], { temperature: 0, max_new_tokens: 48 })); res.push({ dev, info: ctx.llm.info, runs }); } catch (e) { res.push({ dev, error: e.message }); } }
    const ok = res.filter(r => !r.error); const same = ok.length === 2 && ok[0].runs[1].text.trim() === ok[1].runs[1].text.trim();
    calOut.replaceChildren(h('table', { class: 'tbl' }, h('tr', {}, ['장치', 'dtype', '로드', 'cold TTFT', 'warm tok/s', '출력'].map(x => h('th', {}, x))), ...res.map(r => r.error ? h('tr', {}, h('td', {}, r.dev), h('td', { colspan: 5 }, '실패: ' + r.error.slice(0, 120))) : h('tr', {}, h('td', {}, r.info.backend), h('td', {}, r.info.dtype), h('td', {}, r.info.loadMs + 'ms'), h('td', {}, r.runs[0].stats.ttftMs + 'ms'), h('td', {}, String(r.runs[1].stats.tokPerSec)), h('td', {}, r.runs[1].text.trim().slice(0, 30))))),
      h('p', { class: 'sub' }, ok.length < 2 ? '한쪽만 성공해 비교하지 않습니다.' : same ? '두 장치의 출력이 동일해 속도를 비교할 수 있습니다.' : '두 장치의 출력(정밀도 차이)이 달라 같은 조건 비교로 보지 않습니다. 수치는 참고용입니다.'));
    ok.forEach(r => { if (r.runs[1].stats.tokPerSec) sched.metrics.record(taskKey({ kind: 'llm.generate', model: m.id }), r.dev, { ms: 1000 / r.runs[1].stats.tokPerSec, cold: false, ok: true }); }); sched.metrics.persist();
  }
  renderAccel();
  // ── 모델
  page.append(h('h2', {}, '모델'), h('div', { class: 'card' }, h('b', {}, '지금 실행 가능한 모델'), h('p', { class: 'sub' }, '브라우저에서 내려받아 실행합니다. 처음 한 번만 다운로드합니다.'), ...RUNTIME_MODELS.map(m => h('div', { class: 'row', style: { padding: '6px 0' } }, h('span', { class: 'grow' }, m.label, h('small', { class: 'sub', style: { display: 'block', margin: 0 } }, Object.entries(m.sizes).map(([k, v]) => `${k} ${v}MB`).join(' · '))), h('button', { class: 'btn sm', onclick: async () => { try { await ctx.loadModel(m); toast('준비됨'); } catch (e) { if (e.name !== 'AbortError') toast(e.message.slice(0, 200), true); } } }, '불러오기')))));
  const customBox = h('div', { class: 'card' });
  const renderCustom = () => { const list = loadCustom(localStorage); const id = h('input', { placeholder: '내계정/모델-onnx-q4', autocapitalize: 'off' }), label = h('input', { placeholder: '표시 이름(선택)' }), dtype = h('select', {}, ['q4', 'q8', 'q4f16'].map(d => h('option', {}, d))), size = h('input', { type: 'number', placeholder: '다운로드 크기 MB', inputmode: 'numeric' });
    customBox.replaceChildren(h('b', {}, '직접 변환한 모델 추가'), h('p', { class: 'sub' }, 'scripts/convert-llm 등으로 변환해 본인의 Hugging Face 저장소(공개)에 올린 Transformers.js 호환 ONNX 모델을 등록합니다. 파일은 그 저장소에서 내려받습니다. 브라우저 WASM에서는 int4(q4)가 q8보다 약 4배 느립니다(Qwen3-0.6B 실측).'), id, label, h('div', { class: 'row' }, dtype, size), h('button', { class: 'btn sm pri', style: { marginTop: '8px' }, onclick: () => { try { const m = validateCustom({ id: id.value.trim(), label: label.value, dtype: dtype.value, sizeMB: +size.value }); saveCustom(localStorage, [...list.filter(x => x.id !== m.id), m]); toast('등록했습니다 — 모델 선택 창에 나타납니다'); renderCustom(); } catch (e) { toast(e.message, true); } } }, '등록'),
      ...list.map(m => h('div', { class: 'row', style: { marginTop: '6px' } }, h('code', { class: 'grow' }, m.id), h('button', { class: 'btn sm danger', onclick: () => { saveCustom(localStorage, list.filter(x => x.id !== m.id)); renderCustom(); } }, '삭제')))); };
  renderCustom(); page.append(customBox);
  const cand = h('div'); page.append(h('h2', {}, '후보 모델 현황'), h('p', { class: 'sub' }, '아래 모델은 목록에 있다고 실행되는 것이 아닙니다. 상태가 "실행 가능"인 것만 사용할 수 있습니다.'), cand);
  fetch('./model-status.json').then(r => r.json()).then(js => { for (const g of mergeStatus(js)) cand.append(h('div', { class: 'card' }, h('b', {}, g.title), ...g.items.map(m => h('div', { style: { padding: '8px 0', borderTop: '1px solid var(--line)' } }, h('div', { class: 'row' }, h('code', { class: 'grow' }, m.id), h('span', { class: 'chip ' + (m.status === 'runtime' ? 'lime' : m.status?.startsWith('blocked') ? 'bad' : 'warn') }, STATUS_LABEL[m.status] || m.status)), h('div', { class: 'sub', style: { margin: '2px 0 0' } }, [m.license ? `라이선스 ${m.license}` : null, m.detail].filter(Boolean).join(' · ')))))); }).catch(() => cand.append(h('p', { class: 'sub' }, '현황 파일을 불러오지 못했습니다')));
  // ── 데이터
  page.append(h('h2', {}, '데이터'), h('div', { class: 'card' }, h('p', { class: 'sub' }, '노트·할 일·대화·카드는 이 기기의 브라우저 저장소에만 있습니다. Safari 데이터를 지우면 사라지므로 백업을 권장합니다.'), h('div', { class: 'row' },
    h('button', { class: 'btn', onclick: async () => { const j = await store.exportAll(); j.data.notes.forEach(n => delete n.vec); download(`naegyeot-backup-${new Date().toISOString().slice(0, 10)}.json`, JSON.stringify(j), 'application/json'); toast('백업을 내보냈습니다'); } }, '백업 내보내기'),
    h('button', { class: 'btn', onclick: async () => { const f = await new Promise(r => { const i = h('input', { type: 'file', accept: 'application/json' }); i.onchange = () => r(i.files[0]); i.click(); }); if (!f) return; try { const n = await store.importAll(JSON.parse(await f.text())); toast(`${n}개 항목을 복원했습니다 (더 최신인 항목만)`); } catch (e) { toast('복원 실패: ' + e.message, true); } } }, '백업 복원'),
    h('button', { class: 'btn danger', onclick: async () => { if (await confirmDlg('모델 캐시 삭제', '내려받은 모델 파일을 모두 지웁니다. 다시 사용하려면 재다운로드가 필요합니다.', '삭제', true)) { for (const k of await caches.keys()) await caches.delete(k); toast('삭제했습니다'); } } }, '모델 캐시 삭제'))),
    h('div', { class: 'card' }, h('b', {}, '화면'), h('div', { class: 'row', style: { marginTop: '8px' } }, ['dark', 'light'].map(t => h('button', { class: 'btn sm', onclick: () => { document.documentElement.dataset.theme = t; try { localStorage.setItem('ns.theme', t); } catch {} } }, t === 'dark' ? '어둡게' : '밝게')))),
    h('div', { class: 'card' }, h('b', {}, '개인정보 구분'), h('p', { class: 'sub' }, '로컬: 대화·노트·전사·모델 추론. 네트워크: 모델 파일 다운로드(Hugging Face), 사용자가 연결한 원격 MCP 서버 호출.')));
  return () => {};
}
