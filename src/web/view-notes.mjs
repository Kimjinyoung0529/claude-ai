import { h, toast, ago, confirmDlg, dialog, progressBar, mb } from './ui.mjs';
import { TOOLKITS, buildMessages, parseTodos, parseTitleTags, parseCards, stripThink } from './toolkits.mjs';
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
export async function mountNotes(ctx, root) {
  const { store } = ctx; let mode = 'notes'; let q = ''; let sem = false;
  const page = h('div', { class: 'page' }); root.append(page);
  const seg = () => h('div', { class: 'seg', role: 'tablist' }, [['notes', '노트'], ['todos', '할 일'], ['cards', '카드']].map(([k, l]) => h('button', { 'aria-pressed': mode === k, onclick: () => { mode = k; render(); } }, l)));
  async function render() {
    page.replaceChildren(h('h1', {}, mode === 'notes' ? '노트' : mode === 'todos' ? '할 일' : '암기 카드'), seg());
    if (mode === 'notes') await renderNotes(); else if (mode === 'todos') await renderTodos(); else await renderCards();
  }
  async function renderNotes() {
    const all = await store.list('notes'); const box = h('div', { class: 'card list' });
    const search = h('input', { type: 'search', placeholder: '노트 검색', value: q, 'aria-label': '노트 검색', oninput: e => { q = e.target.value; clearTimeout(search._t); search._t = setTimeout(() => fill(), 250); } });
    const semToggle = h('label', { class: 'row', style: { margin: '6px 0 12px' } }, h('input', { type: 'checkbox', checked: sem, onchange: e => { sem = e.target.checked; fill(); } }), h('span', {}, '의미 검색 (처음에 약 118MB 모델을 내려받아 기기에서 계산)'));
    page.append(h('div', { class: 'row' }, h('div', { class: 'grow' }, search), h('button', { class: 'btn pri', onclick: () => edit(null) }, '+ 새 노트')), semToggle, box);
    async function fill() {
      let rows = all, scores = null;
      if (q.trim() && sem) { try { scores = await semanticRank(q, all); rows = scores.map(s => s.note); } catch (e) { toast('의미 검색 실패: ' + e.message, true); rows = all.filter(n => (n.title + n.body).includes(q)); } }
      else if (q.trim()) rows = all.filter(n => (n.title + '\n' + n.body + (n.tags || []).join(' ')).toLowerCase().includes(q.toLowerCase()));
      box.replaceChildren(...(rows.length ? rows.map((n, i) => h('button', { class: 'item', onclick: () => edit(n) }, h('span', { class: 't' }, n.title || '(제목 없음)', h('small', {}, (n.body || '').slice(0, 60) || '내용 없음')), scores ? h('span', { class: 'chip lav' }, scores[i].score.toFixed(2)) : h('small', {}, ago(n.updatedAt)))) : [h('p', { class: 'sub' }, q ? '검색 결과가 없습니다' : '노트가 없습니다. 새 노트를 만들거나 녹음을 저장해 보세요.')]));
    }
    await fill();
  }
  async function semanticRank(query, notes) {
    const pb = progressBar('임베딩 모델 준비 중'); const d = document.getElementById('toasts'); const holder = h('div', { class: 'toast' }, pb.el); d.append(holder);
    try { const emb = await ctx.embedder(p => pb.set(p.total ? p.loaded / p.total : 0, `모델 다운로드 ${mb(p.loaded)} / ${mb(p.total)}`));
      const need = notes.filter(n => !n.vec || n.vecRev !== n.rev); for (let i = 0; i < need.length; i++) { pb.set(i / need.length, `색인 ${i + 1}/${need.length}`); const [v] = await emb.embed([need[i].title + '\n' + need[i].body], 'passage'); need[i].vec = v; need[i].vecRev = need[i].rev; await store.b.put('notes', need[i]); }
      const [qv] = await emb.embed([query], 'query'); return notes.filter(n => n.vec).map(n => ({ note: n, score: dot(qv, n.vec) })).sort((a, b) => b.score - a.score).slice(0, 12); }
    finally { holder.remove(); }
  }
  async function edit(n) {
    let note = n || await store.create('notes', { title: '', body: '', tags: [] }); const isNew = !n;
    const title = h('input', { value: note.title, placeholder: '제목' }), body = h('textarea', { rows: 10, placeholder: '내용을 입력하세요' }, note.body), tags = h('input', { value: (note.tags || []).join(' '), placeholder: '태그 (공백으로 구분)' }); const out = h('div', { class: 'out sub' });
    const save = async () => { note = await store.update('notes', note.id, { title: title.value, body: body.value, tags: tags.value.split(/\s+/).filter(Boolean) }); };
    const ai = (kit, post) => h('button', { class: 'btn sm lav', onclick: async () => { if (!body.value.trim()) return toast('내용이 비어 있습니다', true); out.textContent = '생성 중…'; try { const r = await ctx.runLLM(buildMessages(kit, body.value), { onToken: () => {}, max_new_tokens: 400 }); const text = stripThink(r.text); post(text); } catch (e) { out.textContent = ''; if (e.name !== 'AbortError') toast(e.message, true); } } }, kit.name);
    const kit = id => TOOLKITS.find(k => k.id === id);
    await dialog(close => [h('div', { class: 'row' }, h('b', { class: 'grow' }, isNew ? '새 노트' : '노트 편집'), h('button', { class: 'btn sm', onclick: async () => { await save(); close(); } }, '저장 후 닫기')), h('label', {}, '제목'), title, h('label', {}, '내용'), body, h('label', {}, '태그'), tags,
      h('div', { class: 'row', style: { marginTop: '10px' } }, ai(kit('summary'), t => { out.replaceChildren(h('div', {}, t), h('button', { class: 'btn sm', onclick: () => { body.value += '\n\n## 요약\n' + t; } }, '본문에 추가')); }),
        ai(kit('title'), t => { const r = parseTitleTags(t); out.replaceChildren(...r.titles.map(x => h('button', { class: 'btn sm', onclick: () => { title.value = x; } }, x)), r.tags.length ? h('button', { class: 'btn sm', onclick: () => { tags.value = r.tags.join(' '); } }, '#' + r.tags.join(' #')) : null); }),
        ai(kit('todos'), t => { const items = parseTodos(t); out.replaceChildren(items.length ? h('div', {}, items.map(x => h('div', {}, '☐ ' + x)), h('button', { class: 'btn sm pri', onclick: async () => { for (const x of items) await store.create('todos', { text: x, done: false, due: null, from: note.id }); toast(`${items.length}개 추가됨`); } }, '할 일로 추가')) : '추출된 할 일이 없습니다'); })),
      out, h('div', { class: 'row', style: { marginTop: '14px' } }, isNew ? null : h('button', { class: 'btn danger', onclick: async () => { if (await confirmDlg('노트 삭제', '이 노트를 삭제할까요?', '삭제', true)) { await store.remove('notes', note.id); close(); } } }, '삭제'))]);
    if (isNew && !title.value && !body.value) { await store.remove('notes', note.id).catch(() => {}); } else await save().catch(() => {}); render();
  }
  async function renderTodos() {
    const all = await store.list('todos'); const input = h('input', { placeholder: '할 일 추가', enterkeyhint: 'done', onkeydown: async e => { if (e.key === 'Enter' && input.value.trim()) { await store.create('todos', { text: input.value.trim(), done: false, due: null }); render(); } } });
    const open = all.filter(t => !t.done), done = all.filter(t => t.done); const row = t => h('div', { class: 'row item' }, h('input', { type: 'checkbox', checked: t.done, 'aria-label': t.text, onchange: async e => { await store.update('todos', t.id, { done: e.target.checked }); render(); } }), h('span', { class: 't', style: { textDecoration: t.done ? 'line-through' : '', color: t.done ? 'var(--tx2)' : '' } }, t.text), h('button', { class: 'btn sm danger', 'aria-label': '삭제', onclick: async () => { await store.remove('todos', t.id); render(); } }, '✕'));
    page.append(input, h('div', { class: 'card list', style: { marginTop: '12px' } }, open.length ? open.map(row) : h('p', { class: 'sub' }, '남은 할 일이 없습니다')), done.length ? h('details', {}, h('summary', { class: 'sub' }, `완료 ${done.length}`), h('div', { class: 'card list' }, done.map(row))) : null);
  }
  async function renderCards() {
    const all = await store.list('cards'); const due = all.filter(c => !c.next || c.next <= Date.now());
    page.append(h('p', { class: 'sub' }, `카드 ${all.length}장 · 복습 대상 ${due.length}장. 카드는 '문장 편집 → 암기 카드'에서 만듭니다.`), due.length ? h('button', { class: 'btn pri', onclick: () => review(due) }, '복습 시작') : null, h('div', { class: 'card list', style: { marginTop: '12px' } }, all.map(c => h('div', { class: 'row item' }, h('span', { class: 't' }, c.q, h('small', {}, c.a)), h('button', { class: 'btn sm danger', onclick: async () => { await store.remove('cards', c.id); render(); } }, '✕')))));
  }
  async function review(cards) {
    for (const c of cards) { const r = await dialog(close => [h('small', { class: 'sub' }, '질문'), h('h3', {}, c.q), h('details', {}, h('summary', { class: 'btn' }, '답 보기'), h('p', {}, c.a)), h('div', { class: 'row', style: { marginTop: '12px' } }, h('button', { class: 'btn grow', onclick: () => close('again') }, '모름'), h('button', { class: 'btn grow pri', onclick: () => close('good') }, '앎'), h('button', { class: 'btn', onclick: () => close('stop') }, '종료'))]);
      if (r === 'stop' || !r) break; const days = r === 'good' ? Math.min(30, (c.box || 0) * 2 + 1) : 0; await store.update('cards', c.id, { box: r === 'good' ? (c.box || 0) + 1 : 0, next: Date.now() + days * 864e5 }); }
    render();
  }
  await render(); return () => {};
}
