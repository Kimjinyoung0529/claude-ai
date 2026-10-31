import { start } from './e2e-lib.mjs';
const t = await start(); t.page.setDefaultTimeout(120000); await t.page.goto(t.srv.url); await t.page.waitForFunction(() => window.__ns);
t.page.on('console', m => console.log('C:', m.text().slice(0, 200)));
const r = await t.page.evaluate(async () => { await window.__ns.loadModel(window.__catalog.RUNTIME_MODELS[0], { device: 'wasm' }); const ac = new AbortController(); let n = 0; const t0 = performance.now(); let abortedAt = null;
  const p = window.__ns.runLLM([{ role: 'user', content: '1부터 200까지 숫자를 쉼표로 나열해.' }], { temperature: 0, max_new_tokens: 120, signal: ac.signal, onToken: () => { n++; if (n === 3 && !abortedAt) { abortedAt = performance.now() - t0; ac.abort(); } } }); const res = await p; return { stats: res.stats, abortedAt, n, text: res.text.slice(0, 60) }; });
console.log(JSON.stringify(r)); await t.close(); process.exit(0);
