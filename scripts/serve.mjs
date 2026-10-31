// 정적 서버. 교차 출처 격리(COOP/COEP)를 켜서 WASM 스레드를 쓸 수 있게 한다. localhost 는 보안 컨텍스트라 PWA/마이크 가능.
import http from 'node:http'; import { readFileSync, existsSync, statSync } from 'node:fs'; import { extname, join, normalize } from 'node:path';
const T = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.webmanifest': 'application/manifest+json', '.svg': 'image/svg+xml', '.wasm': 'application/wasm', '.png': 'image/png' };
export function serve({ port = 5173, root = 'dist' } = {}) {
  const s = http.createServer((req, res) => { let p = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname)).replace(/^(\.\.[/\\])+/, ''); let f = join(root, p); if (existsSync(f) && statSync(f).isDirectory()) f = join(f, 'index.html');
    if (!existsSync(f)) { res.writeHead(404).end('not found'); return; }
    res.writeHead(200, { 'content-type': T[extname(f)] || 'application/octet-stream', 'cross-origin-opener-policy': 'same-origin', 'cross-origin-embedder-policy': 'require-corp', 'cross-origin-resource-policy': 'same-origin', 'cache-control': 'no-cache' }).end(readFileSync(f)); });
  return new Promise(r => s.listen(port, '127.0.0.1', () => r({ url: `http://127.0.0.1:${s.address().port}/`, close: () => new Promise(c => s.close(c)) })));
}
if (import.meta.url === `file://${process.argv[1]}`) { const s = await serve({ port: +process.env.PORT || 5173 }); console.log('serving', s.url); }
