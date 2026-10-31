// 로컬 데모 MCP 서버(Streamable HTTP, stateless, CORS 허용). 테스트/브라우저 검증용.
import http from 'node:http';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import { z } from 'zod';
export function startDemoMcp({ port = 0, token = null } = {}) {
  const mk = () => { const s = new McpServer({ name: 'demo', version: '1.0.0' });
    s.registerTool('echo', { description: '입력 문자열을 그대로 돌려준다', inputSchema: { text: z.string() } }, async ({ text }) => ({ content: [{ type: 'text', text: `echo:${text}` }] }));
    s.registerTool('add', { description: '두 수를 더한다', inputSchema: { a: z.number(), b: z.number() }, annotations: { readOnlyHint: true } }, async ({ a, b }) => ({ content: [{ type: 'text', text: String(a + b) }] }));
    s.registerTool('fail', { description: '항상 실패', inputSchema: {} }, async () => ({ isError: true, content: [{ type: 'text', text: 'intentional failure' }] }));
    s.registerTool('slow', { description: '오래 걸림', inputSchema: {} }, async () => { await new Promise(r => setTimeout(r, 3000)); return { content: [{ type: 'text', text: 'late' }] }; });
    return s; };
  const srv = http.createServer(async (req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*'); res.setHeader('Access-Control-Allow-Headers', '*'); res.setHeader('Access-Control-Allow-Methods', 'GET,POST,DELETE,OPTIONS'); res.setHeader('Access-Control-Expose-Headers', 'Mcp-Session-Id');
    if (req.method === 'OPTIONS') { res.writeHead(204).end(); return; }
    if (token && req.headers.authorization !== `Bearer ${token}`) { res.writeHead(401, { 'content-type': 'application/json' }).end('{"error":"unauthorized"}'); return; }
    if (req.method !== 'POST') { res.writeHead(405).end(); return; }
    let body = ''; for await (const c of req) body += c;
    const server = mk(); const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined, enableJsonResponse: true });
    res.on('close', () => { transport.close(); server.close(); });
    await server.connect(transport); await transport.handleRequest(req, res, JSON.parse(body));
  });
  return new Promise(r => srv.listen(port, '127.0.0.1', () => r({ url: `http://127.0.0.1:${srv.address().port}/mcp`, close: () => new Promise(c => srv.close(c)) })));
}
if (import.meta.url === `file://${process.argv[1]}`) { const s = await startDemoMcp({ port: 8787 }); console.log('demo MCP:', s.url); }
