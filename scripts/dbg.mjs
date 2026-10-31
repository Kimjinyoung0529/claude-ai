import { start } from './e2e-lib.mjs';
const t = await start(); await t.page.goto(t.srv.url);
const r = await t.page.evaluate(async () => { try { const x = await fetch('https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX/resolve/main/onnx/model_quantized.onnx'); const b = await x.arrayBuffer(); return { s: x.status, n: b.byteLength, head: x.status > 299 ? new TextDecoder().decode(b.slice(0, 300)) : '' }; } catch (e) { return String(e); } });
console.log(r); await t.close(); process.exit(0);
