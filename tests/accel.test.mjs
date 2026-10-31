import test from 'node:test'; import assert from 'node:assert/strict';
import { HybridScheduler, MetricsStore, ModelResidency, taskKey, AbortError, DeviceLostError } from '../src/accel/scheduler.mjs';
import { jsBackend, createWasmSimdBackend } from '../src/accel/backends.mjs';
import { detectCapabilities } from '../src/accel/capabilities.mjs';

const mk = (id, over = {}) => ({ id, kinds: ['x'], priority: 10, supports: () => ({ ok: true }), run: async () => 1, ...over });
const T = (o = {}) => ({ kind: 'x', input: 1, size: 100, ...o });

test('backend 계약 위반은 등록 거부', () => { assert.throws(() => new HybridScheduler().register({ id: 'a' })); });
test('supports=false 인 백엔드는 건너뛰고 reason 이 plan 에 남는다', async () => {
  const s = new HybridScheduler(); s.register(mk('gpu', { supports: () => ({ ok: false, reason: 'no gpu' }) })).register(mk('cpu', { priority: 50 }));
  const p = s.plan(T()); assert.deepEqual(p.order, ['cpu']); assert.equal(p.rows.find(r => r.id === 'gpu').reason, 'no gpu');
});
test('실패/검증 실패 시 다음 백엔드로 대체, attempts 기록', async () => {
  const s = new HybridScheduler(); s.register(mk('a', { run: async () => { throw new Error('boom'); } })).register(mk('b', { priority: 20, run: async () => 'bad' })).register(mk('c', { priority: 30, run: async () => 'ok' }));
  const r = await s.run(T({ validate: v => v === 'ok' || 'nope' })); assert.equal(r.backend, 'c'); assert.equal(r.attempts.length, 2);
});
test('모두 실패하면 attempts 포함 오류', async () => {
  const s = new HybridScheduler(); s.register(mk('a', { run: async () => { throw new Error('boom'); } }));
  await assert.rejects(s.run(T()), e => e.attempts.length === 1);
});
test('같은 resource 의 작업은 직렬화된다', async () => {
  let active = 0, max = 0; const s = new HybridScheduler();
  s.register(mk('a', { run: async () => { active++; max = Math.max(max, active); await new Promise(r => setTimeout(r, 15)); active--; return 1; } }));
  await Promise.all([s.run(T({ resource: 'llm' })), s.run(T({ resource: 'llm' })), s.run(T({ resource: 'llm' }))]); assert.equal(max, 1);
});
test('취소: 진행 중 abort 는 AbortError, 대체 실행하지 않는다', async () => {
  const s = new HybridScheduler(); let ran2 = false; const ac = new AbortController();
  s.register(mk('a', { run: () => new Promise(() => {}) })).register(mk('b', { priority: 20, run: async () => { ran2 = true; return 1; } }));
  const p = s.run(T(), { signal: ac.signal }); setTimeout(() => ac.abort(), 10); await assert.rejects(p, AbortError); assert.equal(ran2, false);
});
test('cold/warm 분리 기록, 표본 부족하면 priority 로 선택(measured 아님)', async () => {
  let t = 0; const s = new HybridScheduler({ now: () => t });
  s.register(mk('a', { run: async () => { t += s._cost; return 1; } })); s._cost = 100; await s.run(T()); s._cost = 10; await s.run(T()); await s.run(T());
  const st = s.metrics.stats(taskKey(T()), 'a'); assert.equal(st.coldMs, 100); assert.equal(st.warmMs, 10); assert.equal(st.warmN, 2);
  assert.match(s.plan(T()).why, /priority/);
});
test('실측: 두 백엔드 모두 warm 표본 충분하면 빠른 쪽을 선택', async () => {
  let t = 0; const s = new HybridScheduler({ now: () => t, minSamples: 3 });
  s.register(mk('fast', { priority: 90, run: async () => { t += 5; return 1; } })).register(mk('slow', { priority: 10, run: async () => { t += 50; return 1; } }));
  await s.calibrate(T(), { runs: 3 }); const p = s.plan(T()); assert.equal(p.why, 'measured'); assert.equal(p.order[0], 'fast');
});
test('calibrate: 출력이 다른 백엔드는 비교에서 제외하고 속도비를 주장하지 않는다', async () => {
  const s = new HybridScheduler(); s.register(mk('a', { run: async () => 1 })).register(mk('b', { priority: 20, run: async () => 2 }));
  const r = await s.calibrate(T(), { runs: 2 }); assert.equal(r.comparable, false); assert.equal(r.speedup, undefined);
});
test('device loss 는 해당 백엔드를 쿨다운하고 대체 실행', async () => {
  let t = 0; const s = new HybridScheduler({ now: () => t, cooldownMs: 1000 }); let resets = 0;
  s.register(mk('gpu', { run: async () => { throw new DeviceLostError(); }, reset: async () => { resets++; } })).register(mk('cpu', { priority: 50 }));
  assert.equal((await s.run(T())).backend, 'cpu'); assert.equal(resets, 1); assert.deepEqual(s.plan(T()).order, ['cpu']);
  t = 2000; assert.deepEqual(s.plan(T()).order, ['gpu', 'cpu']);
});
test('ModelResidency: 새 모델 적재 전에 기존 모델 해제, 동시 적재 없음, 예산 초과 거부', async () => {
  const r = new ModelResidency({ budgetBytes: 1000 }); const live = new Set(); let peak = 0;
  const mkm = id => ({ sizeBytes: 500, load: async () => { await new Promise(r => setTimeout(r, 5)); live.add(id); peak = Math.max(peak, live.size); return id; }, unload: async h => { live.delete(h); } });
  await Promise.all([r.acquire('A', mkm('A')), r.acquire('B', mkm('B')), r.acquire('C', mkm('C'))]);
  assert.equal(peak, 1); assert.deepEqual([...live], ['C']); await assert.rejects(r.acquire('D', { ...mkm('D'), sizeBytes: 5000 }), /예산/);
});
test('WASM SIMD 커널 결과가 JS 참조와 일치 (홀수 길이 포함)', async () => {
  const caps = await detectCapabilities(); assert.equal(caps.simd, true);
  const s = new HybridScheduler({ caps }); s.register(jsBackend).register(createWasmSimdBackend());
  for (const n of [1, 3, 4, 1000, 4099]) {
    const a = Float32Array.from({ length: n }, (_, i) => i * 0.5), b = Float32Array.from({ length: n }, (_, i) => 1 - i);
    const r = await s.calibrate({ kind: 'vec.add', input: { a, b }, size: n }, { runs: 2 }); assert.ok(r.backends.every(x => x.verified && !x.error), JSON.stringify(r));
  }
});
test('capabilities: 빈 환경에서도 예외 없이 false 반환', async () => { const c = await detectCapabilities({}); assert.equal(c.webgpu, false); assert.equal(c.wasm, false); });
