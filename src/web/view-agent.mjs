import { h, toast, dialog } from './ui.mjs';
import { runAgent } from '../agent/harness.mjs';
import { McpConnection } from '../mcp/connections.mjs';
import { stripThink } from './toolkits.mjs';
export async function mountAgent(ctx, root) {
  const mcp = ctx.mcp; let ac = null; const enabled = new Set(ctx.localTools.filter(t => t.risk === 'read').map(t => t.name));
  const page = h('div', { class: 'page' }); root.append(page);
  const goal = h('textarea', { rows: 3, placeholder: '예: 최근 노트를 검색해서 할 일을 정리해줘' }); const log = h('div', { class: 'card list' }); const result = h('div'); const run = h('button', { class: 'btn pri grow' }, '실행');
  const mcpBox = h('div');
  const allTools = () => [...ctx.localTools, ...[...mcp.values()].flatMap(c => c.asAgentTools())];
  function renderMcp() {
    mcpBox.replaceChildren(h('div', { class: 'row' }, h('b', { class: 'grow' }, '외부 MCP 서버'), h('button', { class: 'btn sm', onclick: connectDlg }, '+ 연결')), h('p', { class: 'sub' }, '원격 서버는 HTTPS 필요(localhost는 HTTP 가능). 서버가 브라우저 CORS를 허용해야 합니다. 호출할 때마다 승인을 받으며, 그때 입력한 내용이 해당 서버로 전송됩니다.'),
      ...[...mcp.entries()].map(([k, c]) => h('div', { class: 'card' }, h('div', { class: 'row' }, h('b', { class: 'grow' }, k), h('span', { class: 'chip warn' }, '원격 · 기기 밖 전송'), h('button', { class: 'btn sm danger', onclick: async () => { await c.disconnect(); mcp.delete(k); renderMcp(); renderTools(); } }, '해제')), h('div', { class: 'sub' }, c.url.href + ` · 도구 ${c.tools.length}개`))));
  }
  async function connectDlg() {
    const r = await dialog(close => { const url = h('input', { placeholder: 'https://example.com/mcp', inputmode: 'url', autocapitalize: 'off' }), tok = h('input', { type: 'password', placeholder: 'Bearer 토큰 (선택, 메모리에만 보관)', autocomplete: 'off' }), name = h('input', { placeholder: '이름 (영문)', value: 'server' + (mcp.size + 1) });
      return [h('h3', {}, 'MCP 서버 연결'), h('label', {}, '주소'), url, h('label', {}, '이름'), name, h('label', {}, '토큰'), tok, h('p', { class: 'sub' }, 'OAuth 로그인과 stdio 서버는 브라우저에서 지원하지 않습니다.'), h('div', { class: 'row' }, h('button', { class: 'btn grow', onclick: () => close() }, '취소'), h('button', { class: 'btn pri grow', onclick: () => close({ url: url.value.trim(), token: tok.value.trim() || null, name: name.value.replace(/\W/g, '') || 'server' }) }, '연결'))]; });
    if (!r?.url) return; try { const c = new McpConnection({ url: r.url, token: r.token, name: r.name }); toast('연결 중…'); await c.connect(); mcp.set(r.name, c); renderMcp(); renderTools(); toast(`연결됨 · 도구 ${c.tools.length}개`); } catch (e) { toast(e.message, true); }
  }
  const toolsBox = h('div', { class: 'card' });
  function renderTools() { toolsBox.replaceChildren(h('b', {}, '에이전트가 쓸 수 있는 도구'), h('p', { class: 'sub' }, '읽기 도구는 기본 켜짐, 수정·삭제·원격 도구는 실행 전에 매번 승인을 묻습니다.'), h('div', { class: 'row' }, allTools().map(t => h('label', { class: 'chip ' + (t.risk === 'read' ? '' : t.risk === 'destructive' ? 'bad' : 'warn'), style: { margin: 0, cursor: 'pointer' } }, h('input', { type: 'checkbox', checked: enabled.has(t.name), style: { width: '16px', height: '16px', marginRight: '4px' }, onchange: e => e.target.checked ? enabled.add(t.name) : enabled.delete(t.name) }), t.name)))); }
  const approve = req => dialog(close => [h('h3', {}, '도구 실행 승인'), h('div', { class: 'row' }, h('b', {}, req.tool), h('span', { class: 'chip ' + (req.risk === 'destructive' ? 'bad' : 'warn') }, req.risk === 'destructive' ? '삭제' : '수정/외부'), req.tool.startsWith('mcp_') ? h('span', { class: 'chip warn' }, '원격 전송') : null), h('p', { class: 'sub' }, req.description), h('pre', {}, JSON.stringify(req.args, null, 2)), h('div', { class: 'row' }, h('button', { class: 'btn grow', onclick: () => close(false) }, '거절'), h('button', { class: 'btn pri grow', onclick: () => close(true) }, '승인'))]).then(v => !!v);
  const label = { step: '단계', model: '모델 출력', invalid: '형식 오류', approval: '승인 요청', denied: '거절됨', call: '도구 호출', result: '도구 결과', end: '종료' };
  run.onclick = async () => {
    if (ac) { ac.abort(); ctx.llm.stop(); return; } if (!goal.value.trim()) return toast('목표를 입력하세요', true);
    if (!ctx.llm.ready) { await ctx.chooseModel(); if (!ctx.llm.ready) return; }
    ac = new AbortController(); run.textContent = '중지'; log.replaceChildren(); result.replaceChildren();
    const tools = allTools().filter(t => enabled.has(t.name));
    try { const r = await runAgent({ goal: goal.value, tools, signal: ac.signal, maxSteps: 8, timeoutMs: 300000, approve,
      llm: async (messages, { signal }) => (await ctx.runLLM(messages, { signal, max_new_tokens: 400, temperature: 0.2, thinking: false })).text,
      onEvent: e => { if (e.type === 'step' || e.type === 'end') return; log.append(h('div', { class: 'item', style: { display: 'block', cursor: 'default' } }, h('span', { class: 'chip ' + (e.type === 'invalid' || e.type === 'denied' ? 'bad' : e.type === 'call' ? 'lav' : '') }, label[e.type] || e.type), ' ', h('code', {}, e.tool || ''), h('pre', {}, e.raw ?? e.body ?? e.error ?? (e.args ? JSON.stringify(e.args) : '')))); } });
      result.replaceChildren(h('div', { class: 'card' }, h('div', { class: 'row' }, h('b', {}, '결과'), h('span', { class: 'chip ' + (r.status === 'done' ? 'lime' : 'warn') }, r.status), h('span', { class: 'chip' }, r.steps + '단계')), h('div', { class: 'out' }, stripThink(r.answer || '')))); }
    catch (e) { toast(e.message, true); } finally { ac = null; run.textContent = '실행'; }
  };
  page.append(ctx.modelBarEl(), h('div', { class: 'row' }, h('h1', { class: 'grow' }, '에이전트'), h('span', { class: 'chip warn' }, '실험적')), h('p', { class: 'sub' }, '작은 로컬 모델이 도구를 골라 쓰는 실험 기능입니다. 형식을 틀리거나 엉뚱한 도구를 고를 수 있어 모든 수정은 승인을 거칩니다. 도구가 돌려준 글은 명령으로 취급하지 않습니다.'), goal, h('div', { class: 'row', style: { margin: '10px 0' } }, run), result, log, h('h2', {}, '도구'), toolsBox, h('h2', {}, '연결'), mcpBox);
  renderTools(); renderMcp(); return () => { ac?.abort(); };
}
