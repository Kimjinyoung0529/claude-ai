// 실제 연산 백엔드: 순수 JS(참조), WASM SIMD(손수 작성 모듈), WebGPU 컴퓨트(파이프라인 캐시 + 버퍼 풀).
// kinds: 'vec.add', 'matmul'. LLM 추론 자체를 대체하지 않는다 — 스케줄러 계약과 측정 방법을 실제 커널로 검증하기 위한 백엔드.
import { DeviceLostError } from './scheduler.mjs';

export const jsBackend = {
  id: 'js', kinds: ['vec.add', 'matmul'], priority: 90,
  supports: () => ({ ok: true }),
  async run(t) {
    if (t.kind === 'vec.add') { const { a, b } = t.input; const o = new Float32Array(a.length); for (let i = 0; i < a.length; i++) o[i] = a[i] + b[i]; return o; }
    const { a, b, m, n, k } = t.input; const o = new Float32Array(m * n);
    for (let i = 0; i < m; i++) for (let p = 0; p < k; p++) { const v = a[i * k + p]; for (let j = 0; j < n; j++) o[i * n + j] += v * b[p * n + j]; }
    return o;
  },
};

// WASM 바이너리를 섹션 단위로 조립한다: add(a, b, out, n4) — f32x4 덧셈 루프, 메모리는 import.
const sec = (id, bytes) => [id, bytes.length, ...bytes]; // 길이 < 128 가정
const WASM_ADD = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 1, 0, 0, 0,
  ...sec(1, [1, 0x60, 4, 0x7f, 0x7f, 0x7f, 0x7f, 0]),                 // type: (i32,i32,i32,i32)->()
  ...sec(2, [1, 3, 0x65, 0x6e, 0x76, 3, 0x6d, 0x65, 0x6d, 2, 0, 1]),   // import env.mem (min 1)
  ...sec(3, [1, 0]),                                                   // func 0 : type 0
  ...sec(7, [1, 3, 0x61, 0x64, 0x64, 0, 0]),                           // export "add"
  ...sec(10, (() => { const body = [1, 1, 0x7f, 0x02, 0x40, 0x03, 0x40,
    0x20, 4, 0x20, 3, 0x4f, 0x0d, 1,
    0x20, 2, 0x20, 4, 0x41, 4, 0x74, 0x6a,
    0x20, 0, 0x20, 4, 0x41, 4, 0x74, 0x6a, 0xfd, 0x00, 4, 0,
    0x20, 1, 0x20, 4, 0x41, 4, 0x74, 0x6a, 0xfd, 0x00, 4, 0,
    0xfd, 0xe4, 0x01, 0xfd, 0x0b, 4, 0,
    0x20, 4, 0x41, 1, 0x6a, 0x21, 4, 0x0c, 0, 0x0b, 0x0b, 0x0b];
    return [1, body.length, ...body]; })()),
]);
export function createWasmSimdBackend() {
  let inst = null, mem = null;
  return {
    id: 'wasm-simd', kinds: ['vec.add'], priority: 50,
    supports: (t, caps) => caps.simd ? { ok: true } : { ok: false, reason: 'WASM SIMD 미지원' },
    async run(t) {
      const { a, b } = t.input; const n = a.length; const n4 = n >> 2; const bytes = n * 4 * 3;
      const pages = Math.ceil(bytes / 65536) + 1;
      if (!mem || mem.buffer.byteLength < pages * 65536) { mem = new WebAssembly.Memory({ initial: pages }); inst = (await WebAssembly.instantiate(WASM_ADD, { env: { mem } })).instance; }
      const f = new Float32Array(mem.buffer); f.set(a, 0); f.set(b, n); inst.exports.add(0, n * 4, n * 8, n4);
      const out = new Float32Array(n); out.set(f.subarray(n * 2, n * 2 + n4 * 4)); for (let i = n4 * 4; i < n; i++) out[i] = a[i] + b[i]; return out;
    },
  };
}

// WebGPU 컴퓨트 백엔드. 파이프라인 캐시·버퍼 풀을 켜고 끌 수 있어 효과를 실제로 비교할 수 있다.
export function createWebGpuBackend({ workgroup = 128, cachePipelines = true, poolBuffers = true, id = 'webgpu' } = {}) {
  let device = null, lost = null; const pipes = new Map(); const pool = new Map(); const stats = { pipelineCompiles: 0, bufferAllocs: 0, bufferReuses: 0 };
  async function dev() {
    if (device && !lost) return device;
    const ad = await navigator.gpu.requestAdapter(); if (!ad) throw new Error('WebGPU adapter 없음');
    device = await ad.requestDevice(); lost = null; pipes.clear(); pool.clear();
    device.lost.then(i => { lost = i; });
    return device;
  }
  const wgsl = (kind, wg) => kind === 'vec.add'
    ? `@group(0)@binding(0) var<storage,read> a:array<f32>; @group(0)@binding(1) var<storage,read> b:array<f32>; @group(0)@binding(2) var<storage,read_write> o:array<f32>;
       @compute @workgroup_size(${wg}) fn main(@builtin(global_invocation_id) g:vec3<u32>){ if(g.x<arrayLength(&o)){ o[g.x]=a[g.x]+b[g.x]; } }`
    : `struct P{m:u32,n:u32,k:u32}; @group(0)@binding(0) var<storage,read> a:array<f32>; @group(0)@binding(1) var<storage,read> b:array<f32>; @group(0)@binding(2) var<storage,read_write> o:array<f32>; @group(0)@binding(3) var<uniform> p:P;
       @compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) g:vec3<u32>){ if(g.x>=p.n||g.y>=p.m){return;} var s=0.0; for(var i=0u;i<p.k;i++){ s+=a[g.y*p.k+i]*b[i*p.n+g.x]; } o[g.y*p.n+g.x]=s; }`;
  function pipeline(d, kind) {
    const k = kind + workgroup; if (cachePipelines && pipes.has(k)) return pipes.get(k);
    stats.pipelineCompiles++; const p = d.createComputePipeline({ layout: 'auto', compute: { module: d.createShaderModule({ code: wgsl(kind, workgroup) }), entryPoint: 'main' } });
    if (cachePipelines) pipes.set(k, p); return p;
  }
  function buf(d, size, usage) {
    size = Math.max(16, (size + 3) & ~3); const k = size + ':' + usage;
    if (poolBuffers) { const l = pool.get(k); if (l?.length) { stats.bufferReuses++; return l.pop(); } }
    stats.bufferAllocs++; return d.createBuffer({ size, usage });
  }
  const release = (b, usage) => { if (poolBuffers) { const k = b.size + ':' + usage; (pool.get(k) || pool.set(k, []).get(k)).push(b); } else b.destroy(); };
  const S = 128 | 4 | 8, R = 128 | 4 | 1; // STORAGE|COPY_DST|COPY_SRC ; STORAGE|COPY_SRC|MAP? -> 아래 별도
  return {
    id, kinds: ['vec.add', 'matmul'], priority: 10, stats,
    supports: (t, caps) => caps.webgpu ? { ok: true } : { ok: false, reason: 'WebGPU 미지원' },
    async run(t) {
      let d; try { d = await dev(); } catch (e) { throw e; }
      if (lost) throw new DeviceLostError(lost.message);
      const kind = t.kind; const { a, b } = t.input; const outLen = kind === 'vec.add' ? a.length : t.input.m * t.input.n;
      const U_IN = 0x80 | 0x08, U_OUT = 0x80 | 0x04, U_RD = 0x08 | 0x01;
      const ba = buf(d, a.byteLength, U_IN), bb = buf(d, b.byteLength, U_IN), bo = buf(d, outLen * 4, U_OUT), rd = buf(d, outLen * 4, U_RD);
      d.queue.writeBuffer(ba, 0, a); d.queue.writeBuffer(bb, 0, b);
      const p = pipeline(d, kind); const entries = [{ binding: 0, resource: { buffer: ba } }, { binding: 1, resource: { buffer: bb } }, { binding: 2, resource: { buffer: bo } }];
      let ub = null;
      if (kind === 'matmul') { ub = d.createBuffer({ size: 16, usage: 0x40 | 0x08 }); d.queue.writeBuffer(ub, 0, new Uint32Array([t.input.m, t.input.n, t.input.k, 0])); entries.push({ binding: 3, resource: { buffer: ub } }); }
      const enc = d.createCommandEncoder(); const pass = enc.beginComputePass(); pass.setPipeline(p);
      pass.setBindGroup(0, d.createBindGroup({ layout: p.getBindGroupLayout(0), entries }));
      if (kind === 'vec.add') pass.dispatchWorkgroups(Math.ceil(outLen / workgroup)); else pass.dispatchWorkgroups(Math.ceil(t.input.n / 8), Math.ceil(t.input.m / 8));
      pass.end(); enc.copyBufferToBuffer(bo, 0, rd, 0, outLen * 4); d.queue.submit([enc.finish()]);
      await rd.mapAsync(1); const out = new Float32Array(rd.getMappedRange().slice(0, outLen * 4)); rd.unmap();
      release(ba, U_IN); release(bb, U_IN); release(bo, U_OUT); release(rd, U_RD); ub?.destroy();
      return out;
    },
    async reset() { try { device?.destroy(); } catch {} device = null; pipes.clear(); pool.clear(); },
  };
}
