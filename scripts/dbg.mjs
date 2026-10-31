import { start } from './e2e-lib.mjs';
const t = await start(); console.log('started', t.srv.url);
t.page.on('console', m => console.log('C:', m.text().slice(0, 200)));
try { await t.page.goto(t.srv.url, { timeout: 15000, waitUntil: 'domcontentloaded' }); console.log('loaded'); await t.page.waitForTimeout(3000); console.log(await t.page.evaluate(() => document.body.innerText.slice(0, 200))); } catch (e) { console.log('ERR', e.message); }
console.log(t.errors); await t.close();
