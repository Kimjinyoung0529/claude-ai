// 실행 환경 감지. env 를 주입할 수 있어 Node 테스트 가능.
export async function detectCapabilities(env = globalThis) {
  const nav = env.navigator || {};
  const caps = {
    webgpu: false, fp16: false, adapter: null, limits: null,
    wasm: typeof env.WebAssembly !== 'undefined', simd: false, threads: false,
    crossOriginIsolated: !!env.crossOriginIsolated, cores: nav.hardwareConcurrency || 1,
    deviceMemoryGB: nav.deviceMemory || null, webnn: !!nav.ml, indexedDB: !!env.indexedDB,
    storageQuota: null,
  };
  try {
    if (nav.gpu) {
      const a = await nav.gpu.requestAdapter();
      if (a) {
        caps.webgpu = true; caps.fp16 = a.features?.has?.('shader-f16') || false;
        caps.limits = { maxBufferSize: a.limits?.maxBufferSize, maxStorageBufferBindingSize: a.limits?.maxStorageBufferBindingSize, maxComputeWorkgroupSizeX: a.limits?.maxComputeWorkgroupSizeX, maxComputeInvocationsPerWorkgroup: a.limits?.maxComputeInvocationsPerWorkgroup };
        caps.adapter = a.info ? { vendor: a.info.vendor, architecture: a.info.architecture } : null;
      }
    }
  } catch {}
  try {
    // 최소 SIMD 모듈(v128.const + drop)
    caps.simd = caps.wasm && env.WebAssembly.validate(new Uint8Array([0,97,115,109,1,0,0,0,1,5,1,96,0,1,123,3,2,1,0,10,10,1,8,0,65,0,253,15,253,98,11]));
    caps.threads = caps.wasm && caps.crossOriginIsolated && typeof env.SharedArrayBuffer !== 'undefined';
  } catch {}
  try { const e = await nav.storage?.estimate?.(); if (e) caps.storageQuota = { usage: e.usage, quota: e.quota }; } catch {}
  return caps;
}
