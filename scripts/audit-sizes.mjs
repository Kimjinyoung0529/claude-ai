// ONNX 파일별 실제 크기(바이트)를 조회해 reports/model-sizes.json 에 기록.
import { writeFileSync } from 'node:fs';
import { CANDIDATES } from './audit-models.mjs';
const ids = [...CANDIDATES.runtime, 'yasserrmd/glm5.1-distill-onnx', ...CANDIDATES.speech];
const out = {};
for (const id of ids) {
  const files = [];
  for (const p of ['onnx', '']) {
    const r = await fetch(`https://huggingface.co/api/models/${id}/tree/main/${p}`); if (!r.ok) continue;
    for (const f of await r.json()) if (f.type === 'file' && /\.onnx(_data)?(_\d+)?$/.test(f.path)) files.push({ path: f.path, size: f.lfs?.size ?? f.size });
  }
  out[id] = files; console.log(id, files.map(f => `${f.path.replace('onnx/','')}:${(f.size/1e6).toFixed(0)}MB`).join(' '));
}
writeFileSync('reports/model-sizes.json', JSON.stringify(out, null, 2));
