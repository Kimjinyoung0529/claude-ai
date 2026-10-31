import { start } from './e2e-lib.mjs';
const t = await start(); t.page.setDefaultTimeout(600000); await t.page.goto(t.srv.url); await t.page.waitForFunction(() => window.__ns);
const r = await t.page.evaluate(async () => { const m = { id: 'onnx-community/Qwen3-0.6B-ONNX', label: 'q4 test', sizes: { q4: 919 } }; await window.__ns.loadModel(m, { device: 'wasm' }); const g = async () => (await window.__ns.runLLM([{ role: 'user', content: '파이썬이 뭐야? 두 문장으로.' }], { temperature: 0, max_new_tokens: 16 })); await g(); const x = await g(); return { dtype: window.__ns.llm.info.dtype, s: x.stats, text: x.text.slice(0, 60) }; });
console.log(JSON.stringify(r)); await t.close(); process.exit(0);
