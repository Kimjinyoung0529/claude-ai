// 외부 MCP 서버(Streamable HTTP) 연결. HTTPS 또는 localhost HTTP 만 허용. Bearer 토큰은 메모리에만 둔다.
// 한계: 브라우저에서는 서버가 CORS 를 허용해야 한다. stdio 서버는 브라우저에서 불가. OAuth 흐름은 구현하지 않았다(Bearer 토큰 직접 입력).
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import Ajv from 'ajv';

export function validateMcpUrl(u) {
  let url; try { url = new URL(u); } catch { throw new Error('올바른 URL이 아닙니다'); }
  const local = ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname);
  if (url.protocol === 'https:' || (url.protocol === 'http:' && local)) return url;
  throw new Error('HTTPS 또는 localhost HTTP 만 허용됩니다 (원격 서버는 HTTPS 필요)');
}
const ajv = new Ajv({ allErrors: true, strict: false });
export class McpConnection {
  constructor({ url, token = null, timeoutMs = 20000, name = 'server' }) { this.url = validateMcpUrl(url); this.token = token; this.timeoutMs = timeoutMs; this.name = name; this.client = null; this.tools = []; this.state = 'idle'; }
  async connect() {
    this.state = 'connecting';
    const headers = this.token ? { Authorization: `Bearer ${this.token}` } : undefined;
    const transport = new StreamableHTTPClientTransport(this.url, { requestInit: headers ? { headers } : undefined });
    const client = new Client({ name: 'naegyeot-night-signal', version: '0.1.0' }, { capabilities: {} });
    try { await client.connect(transport, { timeout: this.timeoutMs }); const r = await client.listTools(undefined, { timeout: this.timeoutMs }); this.tools = r.tools; }
    catch (e) { this.state = 'error'; try { await client.close(); } catch {} throw new Error(explainConnectError(e, this.url)); }
    this.client = client; this.state = 'connected'; return this.tools;
  }
  // 에이전트 harness 용 도구 형태로 변환. 원격 도구는 항상 write 위험도(승인 필요)로 취급한다.
  asAgentTools() {
    return this.tools.map(t => { const schema = { additionalProperties: true, ...(t.inputSchema || { type: 'object' }) }; const check = ajv.compile(schema);
      return { name: `mcp_${this.name}_${t.name}`.replace(/[^\w]/g, '_'), risk: t.annotations?.readOnlyHint ? 'write' : 'write', remote: true, description: `[원격 MCP: ${this.url.host}] ${t.description || t.name}`, schema, run: (args, { signal } = {}) => this.call(t.name, args, { signal, check }) }; });
  }
  async call(name, args, { signal, check } = {}) {
    if (this.state !== 'connected') throw new Error('MCP 서버에 연결되어 있지 않음');
    const t = this.tools.find(x => x.name === name); if (!t) throw new Error('없는 원격 도구: ' + name);
    const v = check || ajv.compile(t.inputSchema || { type: 'object' }); if (!v(args || {})) throw new Error('원격 스키마 검증 실패: ' + ajv.errorsText(v.errors));
    const res = await this.client.callTool({ name, arguments: args || {} }, undefined, { timeout: this.timeoutMs, signal });
    const text = (res.content || []).map(c => c.type === 'text' ? c.text : `[${c.type}]`).join('\n');
    if (res.isError) throw new Error('원격 도구 오류: ' + text); return text;
  }
  async disconnect() { const c = this.client; this.client = null; this.tools = []; this.state = 'idle'; try { await c?.close(); } catch {} }
}
export function explainConnectError(e, url) {
  const m = String(e?.message || e);
  if (/Failed to fetch|NetworkError|fetch failed|CORS/i.test(m)) return `연결 실패: 서버에 닿지 않거나 브라우저 CORS가 막았을 수 있습니다 (서버가 Access-Control-Allow-Origin, Mcp-Session-Id 노출을 허용해야 함). ${m}`;
  if (/401|Unauthorized/i.test(m)) return '인증 필요(401): Bearer 토큰을 입력하세요. OAuth 로그인 흐름은 지원하지 않습니다.';
  if (/timed? ?out|timeout/i.test(m)) return `시간 초과: ${url} 응답이 없습니다`;
  return '연결 실패: ' + m;
}
