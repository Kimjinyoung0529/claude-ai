import { start } from './e2e-lib.mjs';
const t = await start(); await t.page.goto(t.srv.url); await t.page.waitForFunction(() => window.__ns);
t.page.on('request', r => console.log('REQ', r.url().slice(0, 120))); t.page.on('requestfailed', r => console.log('FAIL', r.url().slice(0, 100), r.failure()?.errorText)); t.page.on('console', m => console.log('C:', m.text().slice(0, 300))); t.page.on('worker', w => { console.log('worker', w.url()); w.on('console', m => console.log('W:', m.text().slice(0, 300))); });
t.page.evaluate(async () => { window.__r = 'pending'; try { window.__r = await window.__ns.loadModel(window.__catalog.RUNTIME_MODELS[0], { device: 'wasm' }); } catch (e) { window.__r = 'ERR ' + e.message; } });
await new Promise(r => setTimeout(r, 8000));
console.log(await t.page.evaluate(() => JSON.stringify({ state: window.__ns.llm.state, r: String(window.__ns.llm.worker), last: window.__ns.llm.lastLoad, res: window.__ns.residency.log, rr: window.__r })));
console.log(t.errors); await t.close(); process.exit(0);
