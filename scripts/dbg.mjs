import { start } from './e2e-lib.mjs';
const t = await start(); t.page.setDefaultTimeout(300000); await t.page.goto(t.srv.url); await t.page.waitForFunction(() => window.__ns);
const r = await t.page.evaluate(async () => { try { await window.__ns.loadModel({ id: 'local/midm', label: 'x', sizes: { q4: 1460 } }, { device: 'wasm' }); return 'ok'; } catch (e) { return e.message; } });
console.log(r); await t.close(); process.exit(0);
