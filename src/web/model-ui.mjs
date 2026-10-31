import { h, mb, progressBar, toast } from './ui.mjs';
import { RUNTIME_MODELS, pickDtype, pickDevice, modelBytes, memoryBudgetOk } from '../llm/catalog.mjs';
// 상단 모델 상태바: 로드 전/로딩(진행·중단)/준비/오류.
export function modelBar(ctx) {
  const { llm } = ctx; const root = h('div', { class: 'modelbar' }); const pb = progressBar();
  const render = () => {
    root.replaceChildren(); const s = llm.state, i = llm.info;
    if (s === 'empty') root.append(h('div', { class: 'row' }, h('span', { class: 'chip' }, '모델 없음'), h('span', { class: 'sub grow', style: { margin: 0 } }, '대화하려면 모델을 불러오세요'), h('button', { class: 'btn sm pri', onclick: () => ctx.chooseModel() }, '모델 선택')));
    else if (s === 'loading') { const d = llm.lastLoad || {}; root.append(h('div', { class: 'row' }, h('b', {}, llm.model?.model.label || '모델'), h('span', { class: 'sub grow', style: { margin: 0 } }, d.status || ''), h('button', { class: 'btn sm danger', onclick: () => llm.cancelLoad() }, '중단')), pb.el); pb.set(d.total ? d.loaded / d.total : 0, d.total ? `${mb(d.loaded || 0)} / ${mb(d.total)}` : '준비 중…'); }
    else if (s === 'error') root.append(h('div', { class: 'row' }, h('span', { class: 'chip bad' }, '오류'), h('span', { class: 'sub grow', style: { margin: 0 } }, (llm.lastLoad?.message || '모델 오류').slice(0, 140)), h('button', { class: 'btn sm', onclick: () => ctx.chooseModel() }, '다시 선택')));
    else root.append(h('div', { class: 'row' }, h('span', { class: 'chip lime' }, s === 'generating' ? '생성 중' : '준비됨'), h('b', {}, llm.model?.model.label || i?.id), h('span', { class: 'chip lav' }, i?.backend === 'webgpu' ? 'GPU' : 'CPU'), h('span', { class: 'chip' }, i?.dtype), i?.fellBack ? h('span', { class: 'chip warn', title: i.errors?.join('\n') }, 'GPU 실패→CPU') : null, h('span', { class: 'grow' }), h('button', { class: 'btn sm', onclick: () => ctx.chooseModel() }, '변경')));
  };
  llm.addEventListener('state', e => { llm.lastLoad = { ...llm.lastLoad, ...e.detail }; render(); }); render(); return root;
}
export async function chooseModelDialog(ctx) {
  const { dialog } = await import('./ui.mjs'); const { caps, llm } = ctx;
  return dialog(close => [h('h3', {}, '대화 모델 선택'), h('p', { class: 'sub' }, caps.webgpu ? `GPU 가속 가능${caps.fp16 ? ' · FP16 지원' : ''}` : 'GPU 가속 불가 · CPU로 실행됩니다 (느릴 수 있음)'),
    ...RUNTIME_MODELS.map(m => { const dt = pickDtype(m, caps), b = modelBytes(m, dt), mem = memoryBudgetOk(m, dt, caps); const cur = llm.model?.model.id === m.id && llm.ready;
      return h('button', { class: 'card item', style: { display: 'block', width: '100%', textAlign: 'left', cursor: 'pointer' }, onclick: async () => { close(); try { await ctx.loadModel(m); toast(m.label + ' 준비됨'); } catch (e) { if (e.name !== 'AbortError') toast(e.message.slice(0, 200), true); } } },
        h('div', { class: 'row' }, h('b', { class: 'grow' }, m.label), cur ? h('span', { class: 'chip lime' }, '사용 중') : null, m.rec === 'default' ? h('span', { class: 'chip lav' }, '추천') : null),
        h('div', { class: 'sub', style: { margin: '4px 0 0' } }, `${m.note} · 다운로드 ${mb(b)} (${dt}) · ${m.lang}`), mem.ok ? null : h('div', { class: 'chip warn' }, `메모리 부족 가능 (≈${mem.needMB}MB 필요)`), m.heavy ? h('div', { class: 'chip warn' }, '무거움 · iPhone에서 실패할 수 있음') : null); }),
    h('button', { class: 'btn', onclick: () => close() }, '닫기')]);
}
