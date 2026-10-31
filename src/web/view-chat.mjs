import { h, toast, ago, confirmDlg } from './ui.mjs';
import { modelBar } from './model-ui.mjs';
import { stripThink } from './toolkits.mjs';
const SYS = '당신은 사용자의 개인 AI 비서입니다. 한국어로 간결하고 정확하게 답하세요. 모르면 모른다고 말하세요.';
export async function mountChat(ctx, root) {
  const { store, llm } = ctx; let conv = null; let abort = null; let thinking = false;
  const list = h('div', { class: 'msgs', 'aria-live': 'polite' }); const input = h('textarea', { rows: 1, placeholder: '메시지를 입력하세요', 'aria-label': '메시지', enterkeyhint: 'send' });
  const send = h('button', { class: 'btn pri', onclick: () => doSend() }, '보내기'); const head = h('div', { class: 'row' });
  const page = h('div', { class: 'page chat' }, modelBar(ctx), head, list, h('div', { class: 'composer' }, input, send));
  root.append(page);
  async function open(id) { conv = id ? await store.get('conversations', id) : null; renderAll(); }
  async function newConv() { conv = await store.create('conversations', { title: '새 대화', messages: [] }); renderAll(); }
  async function save() { conv = await store.update('conversations', conv.id, { messages: conv.messages, title: conv.title }); }
  function renderHead() {
    head.replaceChildren(h('b', { class: 'grow' }, conv?.title || '대화'), h('label', { class: 'row', style: { margin: 0 } }, h('input', { type: 'checkbox', checked: thinking, onchange: e => thinking = e.target.checked }), h('span', {}, '깊게 생각')), h('button', { class: 'btn sm', onclick: pick }, '기록'), h('button', { class: 'btn sm', onclick: newConv }, '새 대화'));
  }
  async function pick() {
    const all = await store.list('conversations'); const { dialog } = await import('./ui.mjs');
    const r = await dialog(close => [h('h3', {}, '대화 기록'), all.length ? h('div', { class: 'list' }, all.map(c => h('div', { class: 'row' }, h('button', { class: 'item grow', onclick: () => close({ open: c.id }) }, h('span', { class: 't' }, c.title, h('small', {}, `${c.messages.length}개 · ${ago(c.updatedAt)}`))), h('button', { class: 'btn sm danger', 'aria-label': '삭제', onclick: () => close({ del: c.id }) }, '삭제')))) : h('p', { class: 'sub' }, '아직 대화가 없습니다'), h('button', { class: 'btn', onclick: () => close() }, '닫기')]);
    if (r?.open) await open(r.open); if (r?.del) { await store.remove('conversations', r.del); if (conv?.id === r.del) conv = null; renderAll(); }
  }
  function bubble(m, i) {
    const think = m.role === 'assistant' ? (m.content.match(/<think>([\s\S]*?)(<\/think>|$)/)?.[1] || '') : ''; const body = m.role === 'assistant' ? stripThink(m.content) : m.content;
    const meta = m.stats ? `${m.stats.backend === 'webgpu' ? 'GPU' : 'CPU'} · ${m.stats.tokens}토큰 · 첫 토큰 ${m.stats.ttftMs}ms${m.stats.tokPerSec ? ` · ${m.stats.tokPerSec} tok/s` : ''}${m.stats.cold ? ' · 첫 생성(cold)' : ''}${m.stats.stopped ? ' · 중단됨' : ''}` : '';
    const acts = h('div', { class: 'row', style: { marginTop: '6px' } },
      h('button', { class: 'btn sm', onclick: () => navigator.clipboard?.writeText(body).then(() => toast('복사됨')) }, '복사'),
      h('button', { class: 'btn sm', onclick: () => branchAt(i) }, '여기서 분기'), m.role === 'assistant' && i === conv.messages.length - 1 ? h('button', { class: 'btn sm', onclick: regenerate }, '다시 생성') : null);
    return h('div', { class: 'msg ' + m.role }, think.trim() ? h('details', {}, h('summary', {}, '생각 과정'), think.trim()) : null, body || (m.role === 'assistant' ? '…' : ''), meta ? h('div', { class: 'meta' }, meta) : null, acts);
  }
  async function branchAt(i) { const c = await store.create('conversations', { title: conv.title + ' (분기)', messages: conv.messages.slice(0, i + 1), parent: conv.id }); conv = c; renderAll(); toast('분기된 대화를 만들었습니다'); }
  function renderList() { list.replaceChildren(...(conv?.messages || []).map(bubble)); if (!conv?.messages.length) list.append(h('p', { class: 'sub' }, '무엇이든 물어보세요. 대화는 이 기기 안에만 저장됩니다.')); page.parentElement.scrollTo({ top: 1e9 }); }
  function renderAll() { renderHead(); renderList(); }
  async function generate() {
    if (!llm.ready) { const ok = await ctx.chooseModel(); if (!llm.ready) return; }
    abort = new AbortController(); send.textContent = '중지'; send.onclick = () => { abort.abort(); llm.stop(); };
    const msgs = [{ role: 'system', content: SYS }, ...conv.messages.map(m => ({ role: m.role, content: m.role === 'assistant' ? stripThink(m.content) : m.content }))];
    const a = { role: 'assistant', content: '' }; conv.messages.push(a); renderList(); const el = list.lastChild;
    try { const r = await llm.generate(msgs, { thinking, signal: abort.signal, max_new_tokens: thinking ? 1024 : 512, onToken: t => { a.content += t; const live = h('div', { class: 'msg assistant' }, stripThink(a.content) || '…'); el.replaceWith(live); } }); a.content = r.text; a.stats = r.stats; }
    catch (e) { a.content ||= ''; toast('생성 실패: ' + e.message, true); if (e.deviceLost) toast('GPU 연결이 끊겼습니다. 모델을 다시 불러오세요.', true); }
    finally { send.textContent = '보내기'; send.onclick = () => doSend(); if (!a.content) conv.messages.pop(); await save(); renderAll(); }
  }
  async function regenerate() { conv.messages.pop(); await generate(); }
  async function doSend() {
    const text = input.value.trim(); if (!text || llm.state === 'generating') return; if (!conv) await newConv();
    input.value = ''; conv.messages.push({ role: 'user', content: text }); if (conv.messages.length === 1) conv.title = text.slice(0, 30); await save(); await generate();
  }
  input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && matchMedia('(min-width:900px)').matches) { e.preventDefault(); doSend(); } });
  const all = await store.list('conversations'); await open(all[0]?.id); renderAll();
  return () => { abort?.abort(); };
}
