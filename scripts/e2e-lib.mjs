import { createRequire } from 'node:module'; import { createHash } from 'node:crypto'; import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
const require = createRequire(import.meta.url); const { chromium } = require('playwright');
import { serve } from './serve.mjs';
mkdirSync('.cache/hf', { recursive: true });
// HF 요청은 Node(TLS 검증 유지)가 대신 가져와 디스크에 캐시한다.
async function hfRoute(route) {
  const url = route.request().url(); const key = '.cache/hf/' + createHash('sha1').update(url).digest('hex'); const meta = key + '.json';
  try {
    let body, ct;
    if (existsSync(meta)) { body = readFileSync(key); ct = JSON.parse(readFileSync(meta)).ct; }
    else { let r; for (let i = 0; i < 4; i++) { try { r = await fetch(url, { redirect: 'follow' }); break; } catch (e) { if (i === 3) throw e; await new Promise(x => setTimeout(x, 2000 * (i + 1))); } } if (!r.ok) return route.fulfill({ status: r.status, headers: { 'access-control-allow-origin': '*', 'cross-origin-resource-policy': 'cross-origin' }, body: '' }); body = Buffer.from(await r.arrayBuffer()); ct = r.headers.get('content-type') || 'application/octet-stream'; writeFileSync(key, body); writeFileSync(meta, JSON.stringify({ ct, url })); }
    await route.fulfill({ status: 200, ...(body.length > 1e8 ? { path: key } : { body }), headers: { 'content-type': ct, 'access-control-allow-origin': '*', 'access-control-expose-headers': '*', 'cross-origin-resource-policy': 'cross-origin', 'content-length': String(body.length) } });
  } catch (e) { await route.fulfill({ status: 502, body: String(e) }); }
}
export async function start({ viewport = { width: 393, height: 852 }, args = [], mic = false } = {}) {
  const srv = await serve({ port: 0, hfMirror: true }); const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium', args: ['--headless=new', '--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan', ...(mic ? ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'] : []), ...args] });
  const ctx = await browser.newContext({ viewport, deviceScaleFactor: 2, permissions: mic ? ['microphone'] : [] });
  await ctx.route(/^https:\/\/(huggingface\.co|cdn-lfs[^/]*\.huggingface\.co|[^/]*\.hf\.co|cdn\.jsdelivr\.net)\//, hfRoute); // 작은 파일 직접 요청용
  const page = await ctx.newPage(); const errors = []; page.on('pageerror', e => errors.push('pageerror: ' + e.message)); page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  return { srv, browser, ctx, page, errors, async close() { await browser.close(); await srv.close(); } };
}
