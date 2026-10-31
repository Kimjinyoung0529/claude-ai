import { h, toast } from './ui.mjs';
import { detectCapabilities } from '../accel/capabilities.mjs';
import { HybridScheduler, MetricsStore, ModelResidency, taskKey } from '../accel/scheduler.mjs';
import { jsBackend, createWasmSimdBackend, createWebGpuBackend } from '../accel/backends.mjs';
import { Store, idbBackend, memoryBackend } from '../data/store.mjs';
import { createLocalTools } from '../tools/local-tools.mjs';
import { LlmClient } from './llm-client.mjs';
import { modelBar, chooseModelDialog } from './model-ui.mjs';
import { RUNTIME_MODELS, pickDtype, modelBytes } from '../llm/catalog.mjs';

const safe = { get: k => { try { return localStorage.getItem(k); } catch { return null; } }, set: (k, v) => { try { localStorage.setItem(k, v); } catch {} } };
const storage = { getItem: k => safe.get(k), setItem: (k, v) => safe.set(k, v) };
const theme = safe.get('ns.theme'); if (theme) document.documentElement.dataset.theme = theme;

const caps = await detectCapabilities();
const sched = new HybridScheduler({ caps, metrics: new MetricsStore(storage) });
sched.register(jsBackend).register(createWasmSimdBackend()).register(createWebGpuBackend());
let backend; try { backend = idbBackend(); await backend.getAll('notes'); } catch { backend = memoryBackend(); setTimeout(() => toast('저장소를 사용할 수 없어 임시 모드입니다 (비공개 모드?). 닫으면 사라집니다.', true), 500); }
const store = new Store(backend); const localTools = createLocalTools(store); const llm = new LlmClient(caps);
const residency = new ModelResidency({ budgetBytes: (caps.deviceMemoryGB ? caps.deviceMemoryGB * 1e9 * 0.5 : Infinity) });

const ctx = { caps, sched, store, llm, localTools, mcp: new Map(), residency, toast };
ctx.modelBarEl = () => modelBar(ctx);
ctx.chooseModel = () => chooseModelDialog(ctx);
// 기록된 실측으로 장치를 고른다: 같은 모델의 webgpu/wasm warm 표본이 둘 다 있을 때만 빠른 쪽, 아니면 webgpu 우선.
ctx.recommendDevice = model => { if (!caps.webgpu) return 'wasm'; const k = taskKey({ kind: 'llm.generate', model: model.id }); const a = sched.metrics.stats(k, 'webgpu'), b = sched.metrics.stats(k, 'wasm'); return a?.warmN >= 1 && b?.warmN >= 1 && b.warmMs < a.warmMs ? 'wasm' : 'webgpu'; };
ctx.loadModel = async (model, { device } = {}) => {
  const dev = device || ctx.recommendDevice(model); const c = { ...caps, webgpu: dev === 'webgpu' && caps.webgpu };
  llm.caps = c; const dt = pickDtype(model, c);
  await residency.acquire(model.id, { sizeBytes: modelBytes(model, dt), load: () => llm.load(model), unload: () => llm.unload() }).catch(async e => { await residency.release().catch(() => {}); throw e; });
  llm.caps = caps; return llm.info;
};
// 모든 LLM 호출의 입구: 모델 확인 → 생성 → 실측 기록(ms/token). device loss 시 CPU 로 재적재 후 1회 재시도.
ctx.runLLM = async (messages, opts = {}) => {
  if (!llm.ready) { await ctx.chooseModel(); if (!llm.ready) throw Object.assign(new Error('모델을 선택하지 않았습니다'), { name: 'AbortError' }); }
  let r; try { r = await llm.generate(messages, opts); } catch (e) {
    if (e.deviceLost && llm.model) { toast('GPU 연결이 끊겨 CPU로 다시 불러옵니다', true); await ctx.loadModel(llm.model.model, { device: 'wasm' }); r = await llm.generate(messages, opts); } else throw e; }
  const s = r.stats; if (!s.cold && s.tokPerSec) sched.metrics.record(taskKey({ kind: 'llm.generate', model: llm.model.model.id }), s.backend, { ms: 1000 / s.tokPerSec, cold: false, ok: true });
  return r;
};
// 음성/임베딩 전에 대화 모델을 해제해 큰 모델이 동시에 상주하지 않게 한다.
ctx.freeForSpeech = async () => { if (llm.worker) { await residency.release().catch(() => {}); await llm.unload(); } };
let emb = null;
ctx.embedder = async onProgress => {
  if (emb) return emb; await ctx.freeForSpeech(); const w = new Worker('./embed-worker.js', { type: 'module' }); let rid = 0; const pend = new Map();
  await new Promise((res, rej) => { w.onmessage = ({ data: m }) => { if (m.type === 'progress') onProgress?.(m); else if (m.type === 'ready') res(); else if (m.type === 'error' && !m.rid) rej(new Error(m.message)); else if (m.type === 'embedded') pend.get(m.rid)?.res(m.vectors); else if (m.type === 'error') pend.get(m.rid)?.rej(new Error(m.message)); }; w.postMessage({ type: 'load' }); });
  emb = { embed: (texts, kind) => new Promise((res, rej) => { const id = ++rid; pend.set(id, { res, rej }); w.postMessage({ type: 'embed', rid: id, texts, kind }); }), close() { w.terminate(); emb = null; } }; return emb;
};

const NAV = [['chat', '대화', '◉', () => import('./view-chat.mjs').then(m => m.mountChat)], ['notes', '노트', '▤', () => import('./view-notes.mjs').then(m => m.mountNotes)], ['studio', '편집', '✦', () => import('./view-studio.mjs').then(m => m.mountStudio)], ['record', '녹음', '●', () => import('./view-record.mjs').then(m => m.mountRecord)], ['agent', '에이전트', '⌘', () => import('./view-agent.mjs').then(m => m.mountAgent)], ['settings', '설정', '⚙', () => import('./view-settings.mjs').then(m => m.mountSettings)]];
const view = document.getElementById('view'); let cleanup = null, cur = null;
async function go(id) {
  const n = NAV.find(x => x[0] === id) || NAV[0]; cur = n[0]; cleanup?.(); cleanup = null; view.replaceChildren(); view.scrollTo(0, 0); history.replaceState(null, '', '#' + n[0]); renderNav();
  try { const mount = await n[3](); cleanup = await mount(ctx, view); } catch (e) { console.error(e); view.replaceChildren(h('div', { class: 'warn-box' }, '화면을 불러오지 못했습니다: ' + e.message)); }
  view.focus({ preventScroll: true });
}
function renderNav() {
  const mobile = NAV.filter(x => ['chat', 'notes', 'studio', 'record'].includes(x[0]));
  document.getElementById('tabbar').replaceChildren(...mobile.map(([id, l, ic]) => h('button', { class: 'tab', 'aria-current': cur === id ? 'page' : null, onclick: () => go(id) }, h('span', { class: 'ic', 'aria-hidden': 'true' }, ic), l)), h('button', { class: 'tab', 'aria-current': ['agent', 'settings'].includes(cur) ? 'page' : null, onclick: moreMenu }, h('span', { class: 'ic', 'aria-hidden': 'true' }, '⋯'), '더보기'));
  document.getElementById('sidenav').replaceChildren(...NAV.map(([id, l, ic]) => h('button', { class: 'sidebtn', 'aria-current': cur === id ? 'page' : null, onclick: () => go(id) }, h('span', { class: 'ic', 'aria-hidden': 'true' }, ic), l)));
  document.getElementById('sidefoot').textContent = caps.webgpu ? 'GPU 가속 가능' : 'CPU 모드';
}
async function moreMenu() { const { dialog } = await import('./ui.mjs'); const r = await dialog(close => [h('h3', {}, '더보기'), ...NAV.filter(x => ['agent', 'settings'].includes(x[0])).map(([id, l, ic]) => h('button', { class: 'btn', style: { width: '100%', marginBottom: '8px', justifyContent: 'flex-start' }, onclick: () => close(id) }, ic + '  ' + l + (id === 'agent' ? ' (실험적)' : ''))), h('button', { class: 'btn', style: { width: '100%' }, onclick: () => close() }, '닫기')]); if (r) go(r); }
// 키보드가 열렸을 때 하단 바를 숨기고 뷰포트 높이를 맞춘다
if (window.visualViewport) visualViewport.addEventListener('resize', () => { const kb = innerHeight - visualViewport.height > 120; document.getElementById('tabbar').style.display = kb && matchMedia('(max-width:899px)').matches ? 'none' : ''; });
if ('serviceWorker' in navigator && location.protocol !== 'file:') navigator.serviceWorker.register('./sw.js').catch(() => {});
window.__ns = ctx; window.__catalog = { RUNTIME_MODELS }; window.addEventListener('hashchange', () => { const id = location.hash.slice(1); if (id && id !== cur) go(id); });
await go(location.hash.slice(1) || 'chat');
