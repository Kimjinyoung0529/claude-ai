import { createRequire } from 'node:module'; import { readFileSync, writeFileSync } from 'node:fs'; const { chromium } = createRequire(import.meta.url)('playwright');
const svg = readFileSync('src/web/icon.svg', 'utf8'); const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium', args: ['--headless=new', '--no-sandbox'] });
for (const s of [180, 192, 512]) { const p = await b.newPage({ viewport: { width: s, height: s } }); await p.setContent(`<body style="margin:0">${svg.replace('<svg ', `<svg width="${s}" height="${s}" `)}</body>`); writeFileSync(`src/web/icon-${s}.png`, await p.screenshot({ omitBackground: false })); }
await b.close();
