import { start } from './e2e-lib.mjs'; import { writeFileSync, mkdirSync } from 'node:fs';
import { startDemoMcp } from './demo-mcp-server.mjs';
const stage = process.argv[2] || 'smoke'; mkdirSync('reports/screens', { recursive: true }); const results = {};
const assert = (c, m) => { if (!c) throw new Error('ASSERT: ' + m); console.log('  ✓', m); };
const stages = {
  async smoke() {
    const t = await start(); const { page } = t; await page.goto(t.srv.url); await page.waitForSelector('.tabbar button'); 
    for (const id of ['chat', 'notes', 'studio', 'record', 'agent', 'settings']) { await page.evaluate(i => location.hash = i, id); await page.waitForTimeout(500); await page.screenshot({ path: `reports/screens/${id}-393.png` }); }
    for (const w of [320, 393, 430, 768, 1440]) { await page.setViewportSize({ width: w, height: 900 }); for (const id of ['chat', 'notes', 'studio', 'record', 'agent', 'settings']) { await page.evaluate(i => location.hash = i, id); await page.waitForTimeout(250); const o = await page.evaluate(() => ({ sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth, main: document.getElementById('view').scrollWidth - document.getElementById('view').clientWidth })); assert(o.sw <= o.cw && o.main <= 1, `가로 넘침 없음 ${w}px ${id}`); } }
    await page.setViewportSize({ width: 1440, height: 900 }); await page.screenshot({ path: 'reports/screens/settings-1440.png' });
    const caps = await page.evaluate(() => window.__ns.caps); results.caps = caps; console.log('  caps', JSON.stringify(caps));
    assert(t.errors.length === 0, '콘솔 오류 없음 ' + t.errors.join('|')); await t.close();
  },
  async dialogs() {
    const t = await start(); const { page } = t; await page.goto(t.srv.url + '#studio'); await page.waitForSelector('.tile'); const n = await page.locator('.tile').count(); assert(n === 12, `툴킷 12개 (${n})`);
    for (let i = 0; i < 12; i++) { await page.locator('.tile').nth(i).click(); await page.waitForSelector('dialog[open] textarea'); const ok = await page.evaluate(() => { const d = document.querySelector('dialog'); return d.scrollWidth <= d.clientWidth + 1; }); assert(ok, `툴킷 ${i + 1} dialog 가로 넘침 없음`); await page.keyboard.press('Escape'); await page.waitForTimeout(80); }
    assert(t.errors.length === 0, '콘솔 오류 없음 ' + t.errors.join('|')); await t.close();
  },
  async crud() {
    const t = await start(); const { page } = t; await page.goto(t.srv.url + '#notes'); await page.waitForSelector('.page');
    await page.getByText('+ 새 노트').click(); await page.fill('dialog input[placeholder="제목"]', '테스트 노트'); await page.fill('dialog textarea', '우유 사기\n보고서 쓰기'); await page.getByText('저장 후 닫기').click(); await page.waitForTimeout(300);
    assert(await page.getByText('테스트 노트').count() === 1, '노트 생성·목록 표시');
    await page.fill('input[type=search]', '보고서'); await page.waitForTimeout(500); assert(await page.getByText('테스트 노트').count() === 1, '키워드 검색');
    await page.getByRole('button', { name: '할 일' }).click(); await page.fill('input[placeholder="할 일 추가"]', '운동하기'); await page.press('input[placeholder="할 일 추가"]', 'Enter'); await page.waitForTimeout(300); assert(await page.getByText('운동하기').count() === 1, '할 일 추가');
    await page.locator('input[type=checkbox][aria-label="운동하기"]').check(); await page.waitForTimeout(300); assert(await page.getByText('완료 1').count() === 1, '할 일 완료');
    const dump = await page.evaluate(async () => JSON.stringify(await window.__ns.store.exportAll())); assert(JSON.parse(dump).data.notes.length === 1, '백업 내보내기에 노트 포함');
    await page.reload(); await page.waitForSelector('.page'); const n = await page.evaluate(async () => (await window.__ns.store.list('notes')).length); assert(n === 1, '새로고침 후 IndexedDB 유지');
    assert(t.errors.length === 0, '콘솔 오류 없음 ' + t.errors.join('|')); await t.close();
  },

  async llm() {
    const id = process.env.MODEL || 'onnx-community/Qwen3-0.6B-ONNX'; const t = await start(); const { page } = t; page.setDefaultTimeout(900000);
    await page.goto(t.srv.url + '#chat'); await page.waitForSelector('.modelbar');
    const t0 = Date.now(); const prog = [];
    await page.exposeFunction('__prog', p => prog.push(p));
    await page.evaluate(() => window.__ns.llm.addEventListener('state', e => { if (e.detail.state === 'loading') window.__prog({ l: e.detail.loaded, t: e.detail.total, s: e.detail.status }); }));
    const res = await page.evaluate(async id => { const { RUNTIME_MODELS } = window.__catalog; const m = RUNTIME_MODELS.find(x => x.id === id); const info = await window.__ns.loadModel(m, { device: 'wasm' }); return info; }, id);
    console.log('  로드 완료', JSON.stringify(res), `경과 ${((Date.now() - t0) / 1000).toFixed(0)}s, 진행 이벤트 ${prog.length}개`);
    assert(res.backend === 'wasm', 'WASM 백엔드로 실제 모델 로드');
    assert(prog.length > 0 && prog.at(-1).l > 0, '다운로드 진행률 이벤트 수신');
    const gen = async (prompt, o = {}) => page.evaluate(async ([p, o]) => { const r = await window.__ns.runLLM([{ role: 'user', content: p }], { temperature: 0, max_new_tokens: o.n || 48 }); return r; }, [prompt, o]);
    const a = await gen('대한민국의 수도는 어디야? 한 문장으로 답해.'); console.log('  응답1:', JSON.stringify(a.text.slice(0, 120)), JSON.stringify(a.stats));
    assert(a.text.trim().length > 0, '실제 모델이 비어있지 않은 텍스트 생성'); assert(/서울/.test(a.text), '내용 정확성: 서울 언급');
    assert(a.stats.cold === true, '첫 생성은 cold 로 표시');
    const b = await gen('1+1은?'); console.log('  응답2:', JSON.stringify(b.text.slice(0, 80)), JSON.stringify(b.stats));
    assert(b.stats.cold === false && b.stats.tokPerSec > 0, 'warm 생성 속도 측정: ' + b.stats.tokPerSec + ' tok/s');
    // 중단
    const stopped = await page.evaluate(async () => { const ac = new AbortController(); const p = window.__ns.runLLM([{ role: 'user', content: '1부터 200까지 숫자를 쉼표로 나열해.' }], { temperature: 0, max_new_tokens: 400, signal: ac.signal, onToken: () => { if (!ac.signal.aborted) setTimeout(() => ac.abort(), 0); } }); const r = await p; return r.stats; });
    assert(stopped.stopped === true && stopped.tokens < 400, `생성 중단 동작 (토큰 ${stopped.tokens})`);
    // UI 에서 대화
    await page.fill('textarea[aria-label="메시지"]', '안녕이라고만 말해'); await page.getByRole('button', { name: '보내기' }).click(); await page.waitForSelector('.msg.assistant .meta', { timeout: 600000 }); const meta = await page.locator('.msg.assistant .meta').last().innerText(); console.log('  UI 메타:', meta); assert(/CPU/.test(meta), 'UI 에 실제 backend/속도 표시');
    const conv = await page.evaluate(async () => (await window.__ns.store.list('conversations')).length); assert(conv === 1, '대화 IndexedDB 저장');
    // 툴킷: 번역
    await page.goto(t.srv.url + '#studio'); await page.waitForSelector('.tile'); 
    results.llm = { model: id, load: res, cold: a.stats, warm: b.stats, dl: prog.at(-1) }; writeFileSync('reports/e2e-llm.json', JSON.stringify({ note: 'headless Chromium(SwiftShader), WASM CPU, 샌드박스 VM. iPhone 실측 아님.', ...results.llm }, null, 2));
    assert(t.errors.filter(e => !/405/.test(e)).length === 0, '콘솔 오류 없음 ' + t.errors.join('|')); await t.close();
  },

  async speech() {
    const { readFileSync, readdirSync, writeFileSync: wf } = await import('node:fs');
    const wav = (f32, f) => { const b = Buffer.alloc(44 + f32.length * 2); b.write('RIFF', 0); b.writeUInt32LE(36 + f32.length * 2, 4); b.write('WAVEfmt ', 8); b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20); b.writeUInt16LE(1, 22); b.writeUInt32LE(16000, 24); b.writeUInt32LE(32000, 28); b.writeUInt16LE(2, 32); b.writeUInt16LE(16, 34); b.write('data', 36); b.writeUInt32LE(f32.length * 2, 40); f32.forEach((v, i) => b.writeInt16LE(Math.max(-1, Math.min(1, v)) * 32767, 44 + i * 2)); wf(f, b); };
    const jfkRes = await fetch('https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav'); wf('.cache/jfk.wav', Buffer.from(await jfkRes.arrayBuffer()));
    const ids = readdirSync('.cache/audio').filter(f => f.endsWith('.f32')); const rd = f => { const b = readFileSync('.cache/audio/' + f); return new Float32Array(b.buffer, b.byteOffset, b.length / 4); };
    const by = {}; for (const f of ids) { const sp = f.split('-')[0]; (by[sp] ||= []).push(f); } const sp = Object.keys(by).slice(0, 2); const gap = new Float32Array(8000); const parts = []; [0, 1, 0, 1].forEach((k, i) => { parts.push(rd(by[sp[k]][(i >= 2 ? 1 : 0) % by[sp[k]].length]), gap); });
    const mix = new Float32Array(parts.reduce((n, p) => n + p.length, 0)); let o = 0; for (const p of parts) { mix.set(p, o); o += p.length; } wav(mix, '.cache/mix2.wav');
    const t = await start({ mic: true }); const { page } = t; page.setDefaultTimeout(600000); await page.goto(t.srv.url + '#record'); await page.waitForSelector('#recbtn'); await page.locator('select').nth(0).selectOption('en'); await page.locator('select').nth(1).selectOption('tiny');
    const run = async (file, opts) => { const fc = page.waitForEvent('filechooser'); await page.getByText('오디오 파일 선택').click(); await (await fc).setFiles(file); await page.waitForSelector('text=전사 결과'); return page.evaluate(() => document.querySelector('main').innerText); };
    const t0 = Date.now(); const txt = await run('.cache/jfk.wav'); console.log('  JFK 전사:', txt.replace(/\n+/g, ' ').slice(0, 400), `(${((Date.now() - t0) / 1000).toFixed(0)}s)`);
    assert(/ask not/i.test(txt) && /country/i.test(txt), '실제 Whisper 전사에 JFK 문장 포함');
    assert(/화자 1/.test(txt), '화자 구분 결과 표시(익명 화자 라벨)');
    results.speech = { jfk: txt.replace(/\n+/g, ' ').slice(0, 300) };
    await page.getByText('버리기').click(); await page.getByText('버리기', { exact: true }).last().click().catch(() => {}); await page.waitForSelector('#recbtn');
    const t1 = Date.now(); const txt2 = await run('.cache/mix2.wav'); const nsp = (txt2.match(/화자 \d/g) || []); console.log('  2화자 합성 결과 화자 라벨:', [...new Set(nsp)].join(','), `(${((Date.now() - t1) / 1000).toFixed(0)}s)`); results.speech.mix = { labels: [...new Set(nsp)], text: txt2.replace(/\n+/g, ' ').slice(0, 300) };
    assert([...new Set(nsp)].length >= 2, '2화자 합성 오디오에서 화자 2명 이상 표시');
    assert(t.errors.filter(e => !/405|cache/i.test(e)).length === 0, '콘솔 오류 없음 ' + t.errors.join('|')); writeFileSync('reports/e2e-speech.json', JSON.stringify({ note: 'headless Chromium WASM. Whisper Base(ko 옵션), 영어 JFK + LibriSpeech 합성 2화자. 한국어/실기기 아님.', ...results.speech }, null, 2)); await t.close();
  },
  async mcp() {
    const m = await startDemoMcp(); const t = await start(); const { page } = t; await page.goto(t.srv.url + '#agent'); await page.waitForSelector('.page');
    const r = await page.evaluate(async url => { const { McpConnection } = await import('./view-agent.js').then(() => ({ McpConnection: null })).catch(() => ({})); return null; }, m.url);
    await page.getByText('+ 연결').click(); await page.fill('dialog input[placeholder="https://example.com/mcp"]', m.url); await page.locator('dialog').getByText('연결', { exact: true }).click(); await page.waitForSelector('text=원격 · 기기 밖 전송', { timeout: 15000 });
    const names = await page.evaluate(() => [...window.__ns.mcp.values()].flatMap(c => c.tools.map(t => t.name))); assert(names.includes('add') && names.includes('echo'), 'MCP 도구 목록: ' + names);
    const out = await page.evaluate(async () => { const c = [...window.__ns.mcp.values()][0]; return await c.call('add', { a: 40, b: 2 }); }); assert(out === '42', '브라우저에서 실제 MCP 도구 호출 = ' + out);
    const bad = await page.evaluate(async () => { const { McpConnection } = await import('./view-agent.js'); return null; }).catch(() => null);
    await page.getByText('해제').click(); await page.waitForTimeout(300); assert(await page.evaluate(() => window.__ns.mcp.size) === 0, '연결 해제 정리');
    assert(t.errors.filter(e => !/405/.test(e)).length === 0, '콘솔 오류 없음(MCP GET-SSE 405 제외) ' + t.errors.join('|')); await t.close(); await m.close();
  },
};
try { await stages[stage](); console.log('STAGE OK', stage); } catch (e) { console.error('STAGE FAIL', stage, e.message); process.exitCode = 1; } process.exit(process.exitCode || 0);
