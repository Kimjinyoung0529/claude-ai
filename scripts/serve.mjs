// 정적 서버. 교차 출처 격리(COOP/COEP)를 켜서 WASM 스레드를 쓸 수 있게 한다. localhost 는 보안 컨텍스트라 PWA/마이크 가능.
import http from 'node:http'; import { createHash } from 'node:crypto'; import { createReadStream, writeFileSync, mkdirSync, renameSync } from 'node:fs'; import { Readable } from 'node:stream'; import { pipeline } from 'node:stream/promises'; import { createWriteStream } from 'node:fs'; import { readFileSync, existsSync, statSync } from 'node:fs'; import { extname, join, normalize } from 'node:path';
const T = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.webmanifest': 'application/manifest+json', '.svg': 'image/svg+xml', '.wasm': 'application/wasm', '.png': 'image/png' };
export function serve({ port = 5173, root = 'dist', hfMirror = false } = {}) {
  const s = http.createServer(async (req, res) => {
    if (hfMirror && req.url === '/hf-host.txt') { res.writeHead(200, { 'content-type': 'text/plain', 'cross-origin-resource-policy': 'same-origin' }).end(`http://${req.headers.host}/__hf/`); return; }
    if (hfMirror && req.url.startsWith('/__hf/')) { await mirror(req, res); return; }
    let p = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname)).replace(/^(\.\.[/\\])+/, ''); let f = join(root, p); if (existsSync(f) && statSync(f).isDirectory()) f = join(f, 'index.html');
    if (!existsSync(f)) { res.writeHead(404).end('not found'); return; }
    res.writeHead(200, { 'content-type': T[extname(f)] || 'application/octet-stream', 'cross-origin-opener-policy': 'same-origin', 'cross-origin-embedder-policy': 'require-corp', 'cross-origin-resource-policy': 'same-origin', 'cache-control': 'no-cache' }).end(readFileSync(f)); });
  return new Promise(r => s.listen(port, '127.0.0.1', () => r({ url: `http://127.0.0.1:${s.address().port}/`, close: () => new Promise(c => s.close(c)) })));
}
if (import.meta.url === `file://${process.argv[1]}`) { const s = await serve({ port: +process.env.PORT || 5173 }); console.log('serving', s.url); }

// 테스트 전용 HF 미러: 디스크 캐시 후 스트리밍(TLS 검증 유지, 큰 파일도 처리)
async function mirror(req, res) {
  const path = req.url.slice('/__hf/'.length); const url = 'https://huggingface.co/' + path; const key = '.cache/hf/m-' + createHash('sha1').update(url).digest('hex'); mkdirSync('.cache/hf', { recursive: true });
  const H = { 'access-control-allow-origin': '*', 'cross-origin-resource-policy': 'cross-origin' };
  try {
    if (!existsSync(key + '.json')) { let r; for (let i = 0; i < 4; i++) { try { r = await fetch(url, { redirect: 'follow' }); break; } catch (e) { if (i === 3) throw e; await new Promise(x => setTimeout(x, 1500 * (i + 1))); } }
      if (!r.ok) { res.writeHead(r.status, H).end(); return; } await pipeline(Readable.fromWeb(r.body), createWriteStream(key + '.part')); renameSync(key + '.part', key); writeFileSync(key + '.json', JSON.stringify({ ct: r.headers.get('content-type') })); }
    const ct = JSON.parse(readFileSync(key + '.json')).ct || 'application/octet-stream'; res.writeHead(200, { ...H, 'content-type': ct, 'content-length': statSync(key).size }); createReadStream(key).pipe(res);
  } catch (e) { res.writeHead(502, H).end(String(e)); }
}
