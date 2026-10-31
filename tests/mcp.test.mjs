import test from 'node:test'; import assert from 'node:assert/strict';
import { McpConnection, validateMcpUrl } from '../src/mcp/connections.mjs';
import { runAgent } from '../src/agent/harness.mjs';
import { startDemoMcp } from '../scripts/demo-mcp-server.mjs';

test('URL 정책: https 와 localhost http 만 허용', () => {
  for (const ok of ['https://a.example/mcp', 'http://localhost:3000/mcp', 'http://127.0.0.1/mcp']) validateMcpUrl(ok);
  for (const bad of ['http://evil.example/mcp', 'ftp://x', 'file:///etc/passwd', 'javascript:alert(1)', 'notaurl']) assert.throws(() => validateMcpUrl(bad), bad);
});
test('실제 MCP 서버: 연결·도구 목록·호출·원격 스키마 검증·오류·해제', async () => {
  const s = await startDemoMcp(); const c = new McpConnection({ url: s.url, name: 'demo', timeoutMs: 5000 });
  try {
    const tools = await c.connect(); assert.deepEqual(tools.map(t => t.name).sort(), ['add', 'echo', 'fail', 'slow']); assert.equal(c.state, 'connected');
    assert.equal(await c.call('echo', { text: '안녕' }), 'echo:안녕'); assert.equal(await c.call('add', { a: 2, b: 3 }), '5');
    await assert.rejects(c.call('add', { a: 'x', b: 1 }), /스키마 검증/); await assert.rejects(c.call('fail', {}), /intentional failure/); await assert.rejects(c.call('nope', {}), /없는 원격 도구/);
  } finally { await c.disconnect(); await s.close(); }
  assert.equal(c.state, 'idle'); await assert.rejects(c.call('echo', { text: 'x' }), /연결되어 있지 않음/);
});
test('타임아웃과 AbortSignal 취소', async () => {
  const s = await startDemoMcp(); const c = new McpConnection({ url: s.url, name: 'demo', timeoutMs: 400 });
  try { await c.connect(); await assert.rejects(c.call('slow', {}), /time|timeout/i);
    const ac = new AbortController(); const p = c.call('slow', {}, { signal: ac.signal }); setTimeout(() => ac.abort(), 50); await assert.rejects(p); }
  finally { await c.disconnect(); await s.close(); }
});
test('Bearer 토큰: 없으면 401 안내, 있으면 연결', async () => {
  const s = await startDemoMcp({ token: 'sekret' });
  try { await assert.rejects(new McpConnection({ url: s.url, timeoutMs: 3000 }).connect(), /401|인증|연결 실패/);
    const c = new McpConnection({ url: s.url, token: 'sekret', timeoutMs: 3000 }); assert.equal((await c.connect()).length, 4); await c.disconnect(); }
  finally { await s.close(); }
});
test('연결 실패는 사용자에게 설명 가능한 오류', async () => { await assert.rejects(new McpConnection({ url: 'http://127.0.0.1:1/mcp', timeoutMs: 1500 }).connect(), /연결 실패|시간 초과/); });
test('에이전트 harness 가 원격 MCP 도구를 승인 후 실제 호출', async () => {
  const s = await startDemoMcp(); const c = new McpConnection({ url: s.url, name: 'demo', timeoutMs: 5000 });
  try { await c.connect(); const tools = c.asAgentTools(); assert.ok(tools.every(t => t.risk === 'write' && t.remote));
    const outs = ['{"action":"tool","tool":"mcp_demo_add","args":{"a":20,"b":22}}', '{"action":"final","answer":"42"}']; let i = 0; let approvals = 0;
    const r = await runAgent({ goal: '20+22', tools, llm: async () => outs[i++], approve: async () => (approvals++, true) });
    assert.equal(r.status, 'done'); assert.equal(approvals, 1); assert.ok(r.log.find(e => e.type === 'result').body.includes('42')); }
  finally { await c.disconnect(); await s.close(); }
});
