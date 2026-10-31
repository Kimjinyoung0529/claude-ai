import { build } from 'esbuild'; import { cpSync, mkdirSync, rmSync, writeFileSync, existsSync, readFileSync } from 'node:fs';
rmSync('dist', { recursive: true, force: true }); mkdirSync('dist/ort', { recursive: true });
const web = 'src/web/';
const entries = { app: 'app.mjs', 'view-chat': 'view-chat.mjs', 'view-notes': 'view-notes.mjs', 'view-studio': 'view-studio.mjs', 'view-record': 'view-record.mjs', 'view-agent': 'view-agent.mjs', 'view-settings': 'view-settings.mjs', docs: 'docs.mjs', 'llm-worker': 'llm-worker.mjs', 'embed-worker': 'embed-worker.mjs', 'speech-worker': 'speech-worker.mjs' };
// 공유 모듈 중복을 피하려면 splitting 이 필요하지만 worker 는 별도 번들. 화면 모듈은 app 과 함께 splitting.
await build({ entryPoints: Object.fromEntries(['app', 'view-chat', 'view-notes', 'view-studio', 'view-record', 'view-agent', 'view-settings', 'docs'].map(k => [k, web + entries[k]])), bundle: true, splitting: true, format: 'esm', outdir: 'dist', platform: 'browser', target: 'es2022', minify: true, sourcemap: false, logLevel: 'warning', loader: { '.wasm': 'file' }, external: [], define: { 'process.env.NODE_ENV': '"production"' }, conditions: ['browser'] });
await build({ entryPoints: Object.fromEntries(['llm-worker', 'embed-worker', 'speech-worker'].map(k => [k, web + entries[k]])), bundle: true, format: 'esm', outdir: 'dist', platform: 'browser', target: 'es2022', minify: true, logLevel: 'warning', conditions: ['browser'], external: ['node:*'] });
for (const f of ['index.html', 'styles.css', 'sw.js', 'manifest.webmanifest', 'icon.svg', 'icon-180.png', 'icon-192.png', 'icon-512.png']) cpSync(web + f, 'dist/' + f);
const od = 'node_modules/onnxruntime-web/dist/'; for (const f of ['ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) cpSync(od + f, 'dist/ort/' + f);
cpSync('node_modules/pdfjs-dist/build/pdf.worker.min.mjs', 'dist/pdf.worker.js');
mkdirSync('dist/tess', { recursive: true }); cpSync('node_modules/tesseract.js/dist/worker.min.js', 'dist/tess/worker.min.js'); for (const f of ['tesseract-core-simd-lstm.wasm.js', 'tesseract-core-simd-lstm.wasm', 'tesseract-core-lstm.wasm.js', 'tesseract-core-lstm.wasm']) if (existsSync('node_modules/tesseract.js-core/' + f)) cpSync('node_modules/tesseract.js-core/' + f, 'dist/tess/' + f);
if (existsSync('reports/model-status.json')) cpSync('reports/model-status.json', 'dist/model-status.json');
console.log('build ok');
