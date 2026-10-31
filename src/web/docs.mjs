// PDF 텍스트 추출(PDF.js) 과 OCR(Tesseract kor+eng). 큰 라이브러리라 필요할 때만 불러온다.
export async function extractPdf(file) {
  const pdfjs = await import('pdfjs-dist/build/pdf.mjs'); pdfjs.GlobalWorkerOptions.workerSrc = new URL('./pdf.worker.js', self.location.href).href;
  const doc = await pdfjs.getDocument({ data: new Uint8Array(await file.arrayBuffer()) }).promise; let out = '';
  for (let i = 1; i <= Math.min(doc.numPages, 60); i++) { const p = await doc.getPage(i); const c = await p.getTextContent(); out += c.items.map(x => x.str + (x.hasEOL ? '\n' : '')).join('') + '\n\n'; }
  return out;
}
export async function ocrImage(file) {
  const T = await import('tesseract.js'); const w = await T.createWorker(['kor', 'eng'], 1, { workerPath: new URL('./tess/worker.min.js', self.location.href).href, corePath: new URL('./tess/', self.location.href).href }); const r = await w.recognize(file); await w.terminate(); return r.data.text;
}
