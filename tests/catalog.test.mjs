import test from 'node:test'; import assert from 'node:assert/strict';
import { RUNTIME_MODELS, pickDtype, mergeStatus, memoryBudgetOk, slowOnCpu } from '../src/llm/catalog.mjs';
test('CPU 는 q8 우선(실측: q4 대비 4배 빠름), GPU+FP16 은 q4f16', () => {
  const m = RUNTIME_MODELS.find(x => x.id.includes('Qwen3-0.6B')); assert.equal(pickDtype(m, {}), 'q8'); assert.equal(pickDtype(m, { webgpu: true, fp16: true }), 'q4f16'); assert.equal(pickDtype(m, { webgpu: true, fp16: false }), 'q8');
});
test('q8 이 2.2GB 를 넘거나 없으면 가장 작은 비FP16 으로 대체, FP16 은 CPU 에서 제외', () => {
  assert.equal(pickDtype({ sizes: { q4: 2702, q4f16: 2436, q8: 4870 } }, {}), 'q4'); assert.equal(pickDtype({ sizes: { fp16: 100, q4: 500 } }, {}), 'q4'); assert.equal(slowOnCpu(null, 'q4'), true); assert.equal(slowOnCpu(null, 'q8'), false);
});
test('메모리 힌트가 없으면(Safari) 판단 불가를 숨기지 않는다', () => { assert.equal(memoryBudgetOk(RUNTIME_MODELS[0], 'q8', {}).unknown, true); assert.equal(memoryBudgetOk(RUNTIME_MODELS[5], 'q4', { deviceMemoryGB: 4 }).ok, false); });
test('후보 모델 상태 병합: 상태 파일에 없으면 unverified, 14개 후보 유지', () => { const g = mergeStatus({ models: [{ id: 'ibm-granite/granite-4.2-3b', status: 'needs-export' }] }); assert.equal(g.flatMap(x => x.items).length, 14); assert.equal(g.flatMap(x => x.items).find(x => x.id === 'LiquidAI/LFM2.5-2.6B').status, 'unverified'); });
import { validateCustom, loadCustom, saveCustom } from '../src/llm/catalog.mjs';
test('사용자 지정 모델: 입력 검증과 저장소 라운드트립, 손상된 저장값 무시', () => {
  assert.throws(() => validateCustom({ id: '../x', dtype: 'q4', sizeMB: 1 })); assert.throws(() => validateCustom({ id: 'a/b', dtype: 'zz', sizeMB: 1 })); assert.throws(() => validateCustom({ id: 'a/b', dtype: 'q4', sizeMB: 0 }));
  const st = { v: null, getItem() { return this.v; }, setItem(k, v) { this.v = v; } }; saveCustom(st, [validateCustom({ id: 'me/midm-q4', dtype: 'q4', sizeMB: 1930 })]); assert.equal(loadCustom(st)[0].custom, true); st.v = '{broken'; assert.deepEqual(loadCustom(st), []); st.v = JSON.stringify([{ id: 'x y/z', sizes: { q4: 1 } }]); assert.deepEqual(loadCustom(st), []);
});
