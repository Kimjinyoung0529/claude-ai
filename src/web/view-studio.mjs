import { h, toast, dialog } from './ui.mjs';
import { TOOLKITS, buildMessages, parseCards, parseTodos, stripThink } from './toolkits.mjs';
export async function mountStudio(ctx, root) {
  const { store } = ctx; const page = h('div', { class: 'page' }, ctx.modelBarEl(), h('h1', {}, '문장 편집'), h('p', { class: 'sub' }, '붙여 넣은 글을 기기 안의 모델로 다듬습니다. 결과는 확인 후 저장하세요.'));
  const grid = h('div', { class: 'grid' }, TOOLKITS.map(k => h('button', { class: 'tile', onclick: () => open(k) }, h('span', { class: 'ic' }, k.icon), h('b', {}, k.name), h('span', {}, k.desc)))); page.append(grid, h('div', { class: 'card', style: { marginTop: '14px' } }, h('h3', {}, 'PDF·이미지에서 글 가져오기'), h('p', { class: 'sub' }, 'PDF는 텍스트를 추출하고, 이미지는 글자 인식(OCR, 한국어+영어)을 합니다. 처음 사용할 때 필요한 데이터를 내려받습니다.'), h('div', { class: 'row' }, h('button', { class: 'btn', onclick: () => importDoc('pdf') }, 'PDF 선택'), h('button', { class: 'btn', onclick: () => importDoc('img') }, '이미지 선택'))));
  root.append(page);
  async function importDoc(kind) {
    const f = await new Promise(r => { const i = h('input', { type: 'file', accept: kind === 'pdf' ? 'application/pdf' : 'image/*' }); i.onchange = () => r(i.files[0]); i.click(); }); if (!f) return;
    toast(kind === 'pdf' ? 'PDF 읽는 중…' : '글자 인식 중… (처음엔 데이터를 내려받습니다)');
    try { const { extractPdf, ocrImage } = await import('./docs.mjs'); const text = kind === 'pdf' ? await extractPdf(f) : await ocrImage(f); if (!text.trim()) return toast('추출된 글이 없습니다 (스캔 PDF는 이미지로 시도하세요)', true);
      open(TOOLKITS[0], text); } catch (e) { toast('가져오기 실패: ' + e.message, true); }
  }
  function open(kit, preset = '') {
    const input = h('textarea', { rows: 6, placeholder: '여기에 글을 붙여 넣으세요' }, preset); const sel = kit.options ? h('select', { 'aria-label': '옵션' }, kit.options.map(o => h('option', {}, o))) : null; const out = h('div', { class: 'card out' }, ''); const run = h('button', { class: 'btn pri grow' }, '실행'); const stats = h('div', { class: 'sub' });
    let ac = null;
    run.onclick = async () => {
      if (ac) { ac.abort(); ctx.llm.stop(); return; } if (!input.value.trim()) return toast('글을 입력하세요', true);
      ac = new AbortController(); run.textContent = '중지'; out.textContent = ''; stats.textContent = '';
      try { const r = await ctx.runLLM(buildMessages(kit, input.value, sel?.value), { signal: ac.signal, max_new_tokens: 600, temperature: 0.3, onToken: t => { out.textContent += t; out.textContent = stripThink(out.textContent); } }); out.textContent = stripThink(r.text); stats.textContent = `${r.stats.backend === 'webgpu' ? 'GPU' : 'CPU'} · ${r.stats.tokens}토큰 · ${(r.stats.totalMs / 1000).toFixed(1)}초`;
        if (kit.id === 'cards') { const cs = parseCards(out.textContent); if (cs.length) out.append(h('div', {}, h('button', { class: 'btn sm pri', onclick: async () => { for (const c of cs) await store.create('cards', { ...c, box: 0 }); toast(`${cs.length}장 저장`); } }, `카드 ${cs.length}장 저장`))); } }
      catch (e) { if (e.name !== 'AbortError') toast(e.message, true); } finally { ac = null; run.textContent = '실행'; }
    };
    dialog(close => [h('div', { class: 'row' }, h('b', { class: 'grow' }, kit.icon + ' ' + kit.name), h('button', { class: 'btn sm', onclick: () => close() }, '닫기')), h('label', {}, '원문'), input, sel ? [h('label', {}, '옵션'), sel] : null, h('div', { class: 'row', style: { margin: '12px 0' } }, run), out, stats,
      h('div', { class: 'row' }, h('button', { class: 'btn sm', onclick: () => navigator.clipboard?.writeText(out.textContent).then(() => toast('복사됨')) }, '복사'), h('button', { class: 'btn sm', onclick: () => { input.value = out.textContent; } }, '결과를 입력으로'), h('button', { class: 'btn sm', onclick: async () => { await store.create('notes', { title: kit.name + ' 결과', body: out.textContent, tags: [] }); toast('노트로 저장됨'); } }, '노트로 저장'))]);
  }
  return () => {};
}
