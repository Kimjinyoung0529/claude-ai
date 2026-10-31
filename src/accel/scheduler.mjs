// HybridScheduler: 작업별로 사용 가능한 백엔드를 고르고, 직렬 실행·취소·출력 검증·대체 실행·측정 기록을 담당.
// 여러 백엔드의 성능을 합산하지 않는다. 한 작업은 한 백엔드에서 실행된다.
//
// Backend 계약:
//   { id, kinds: string[], priority: number (작을수록 우선),
//     supports(task, caps) -> {ok:boolean, reason?:string},
//     run(task, ctx:{signal, caps}) -> Promise<result>,
//     dispose?() }
// Task: { kind, model?, dtype?, size?: number(입력 크기), input, validate?(result)->true|string, resource?: string }
export const RUNTIME_VERSION = '1';
export class AbortError extends Error { constructor(m = '작업이 취소되었습니다') { super(m); this.name = 'AbortError'; } }
export class DeviceLostError extends Error { constructor(m = 'GPU 장치를 잃었습니다') { super(m); this.name = 'DeviceLostError'; } }

export function sizeBucket(n) { if (!n) return 's0'; return 's' + Math.ceil(Math.log2(n + 1)); }
export function taskKey(t) { return [t.kind, t.model || '-', t.dtype || '-', sizeBucket(t.size), RUNTIME_VERSION].join('|'); }
const median = a => { const s = [...a].sort((x, y) => x - y); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };

export class MetricsStore {
  constructor(storage = null) { this.map = new Map(); this.storage = storage; if (storage) { try { for (const [k, v] of Object.entries(JSON.parse(storage.getItem('ns.metrics') || '{}'))) this.map.set(k, v); } catch {} } }
  _e(key, be) { const k = key + '#' + be; if (!this.map.has(k)) this.map.set(k, { cold: [], warm: [], failures: 0, verified: 0, lastError: null }); return this.map.get(k); }
  record(key, be, { ms, cold, ok, error }) {
    const e = this._e(key, be);
    if (!ok) { e.failures++; e.lastError = String(error); } else { (cold ? e.cold : e.warm).push(ms); e.verified++; if (e.warm.length > 20) e.warm.shift(); }
    this.persist();
  }
  stats(key, be) { const e = this.map.get(key + '#' + be); if (!e) return null; return { coldMs: e.cold.length ? median(e.cold) : null, warmMs: e.warm.length ? median(e.warm) : null, warmN: e.warm.length, failures: e.failures, verified: e.verified, lastError: e.lastError }; }
  persist() { try { this.storage?.setItem('ns.metrics', JSON.stringify(Object.fromEntries(this.map))); } catch {} }
  clear() { this.map.clear(); this.persist(); }
}

export class HybridScheduler {
  constructor({ caps, metrics = new MetricsStore(), minSamples = 3, cooldownMs = 30000, now = () => performance.now() } = {}) {
    this.caps = caps || {}; this.metrics = metrics; this.minSamples = minSamples; this.cooldownMs = cooldownMs; this.now = now;
    this.backends = new Map(); this.queues = new Map(); this.unhealthy = new Map(); this.warmed = new Set(); this.history = [];
  }
  register(b) {
    for (const f of ['id', 'kinds', 'supports', 'run']) if (b[f] == null) throw new Error(`backend 계약 위반: ${f} 누락`);
    if (!Array.isArray(b.kinds) || typeof b.supports !== 'function' || typeof b.run !== 'function') throw new Error('backend 계약 위반: 타입');
    this.backends.set(b.id, { priority: 100, ...b }); return this;
  }
  unregister(id) { this.backends.delete(id); }
  healthy(id) { const t = this.unhealthy.get(id); if (t == null) return true; if (this.now() - t > this.cooldownMs) { this.unhealthy.delete(id); return true; } return false; }
  // 후보 계산 + 선택 이유. 실측은 같은 key에서 warm 표본이 minSamples 이상일 때만 사용한다.
  plan(task) {
    const key = taskKey(task); const rows = [];
    for (const b of this.backends.values()) {
      if (!b.kinds.includes(task.kind)) continue;
      let s; try { s = b.supports(task, this.caps) || { ok: false }; } catch (e) { s = { ok: false, reason: String(e) }; }
      const st = this.metrics.stats(key, b.id);
      rows.push({ id: b.id, priority: b.priority, supported: !!s.ok, reason: s.reason || null, healthy: this.healthy(b.id), stats: st, measured: !!(st && st.warmN >= this.minSamples) });
    }
    const usable = rows.filter(r => r.supported && r.healthy);
    const measured = usable.filter(r => r.measured);
    let order, why;
    if (measured.length >= 2) { order = [...measured].sort((a, b) => a.stats.warmMs - b.stats.warmMs).concat(usable.filter(r => !r.measured).sort((a, b) => a.priority - b.priority)); why = 'measured'; }
    else { order = [...usable].sort((a, b) => a.priority - b.priority); why = measured.length === 1 ? 'priority(비교 측정 1개뿐)' : 'priority(측정 없음)'; }
    return { key, order: order.map(r => r.id), why, rows };
  }
  _serial(res, fn) {
    const prev = this.queues.get(res) || Promise.resolve();
    const p = prev.catch(() => {}).then(fn); this.queues.set(res, p.catch(() => {})); return p;
  }
  async run(task, { signal } = {}) {
    const plan = this.plan(task);
    if (!plan.order.length) { const e = new Error(`실행 가능한 백엔드 없음 (${task.kind}): ` + plan.rows.map(r => `${r.id}:${r.reason || (r.healthy ? 'unsupported' : 'cooldown')}`).join(', ')); e.plan = plan; throw e; }
    return this._serial(task.resource || 'default', async () => {
      const attempts = [];
      for (const id of plan.order) {
        if (signal?.aborted) throw new AbortError();
        const b = this.backends.get(id); const wk = plan.key + '#' + id; const cold = !this.warmed.has(wk); const t0 = this.now();
        try {
          const result = await this._race(b.run(task, { signal, caps: this.caps }), signal);
          if (task.validate) { const v = task.validate(result); if (v !== true) throw new Error('출력 검증 실패: ' + (v || 'invalid')); }
          const ms = this.now() - t0; this.warmed.add(wk); this.metrics.record(plan.key, id, { ms, cold, ok: true });
          const info = { backend: id, ms, cold, why: plan.why, attempts, order: plan.order };
          this.history.push({ key: plan.key, ...info }); if (this.history.length > 100) this.history.shift();
          return { result, ...info };
        } catch (err) {
          if (err.name === 'AbortError' || signal?.aborted) throw err instanceof AbortError ? err : new AbortError();
          attempts.push({ backend: id, error: String(err.message || err) });
          this.metrics.record(plan.key, id, { ok: false, error: err.message });
          if (err.name === 'DeviceLostError') this.unhealthy.set(id, this.now());
          try { await b.reset?.(err); } catch {}
        }
      }
      const e = new Error('모든 백엔드 실패: ' + attempts.map(a => `${a.backend}(${a.error})`).join('; ')); e.attempts = attempts; throw e;
    });
  }
  _race(p, signal) {
    if (!signal) return p;
    return new Promise((res, rej) => { const on = () => rej(new AbortError()); if (signal.aborted) return on(); signal.addEventListener('abort', on, { once: true }); p.then(v => { signal.removeEventListener('abort', on); res(v); }, e => { signal.removeEventListener('abort', on); rej(e); }); });
  }
  // 동일 입력으로 각 백엔드를 반복 실행. 출력이 reference(첫 지원 백엔드)와 compare 를 통과한 백엔드만 '비교 가능'으로 표시.
  async calibrate(task, { runs = 5, compare = defaultCompare, ids } = {}) {
    const key = taskKey(task); const report = { key, runs, backends: [], comparable: false, winner: null };
    let ref = null;
    for (const b of this.backends.values()) {
      if (!b.kinds.includes(task.kind) || (ids && !ids.includes(b.id))) continue;
      const s = b.supports(task, this.caps); if (!s?.ok) { report.backends.push({ id: b.id, skipped: s?.reason || 'unsupported' }); continue; }
      const row = { id: b.id, coldMs: null, warmMs: [], verified: false, error: null };
      try {
        for (let i = 0; i < runs + 1; i++) {
          const t0 = this.now(); const out = await b.run(task, { caps: this.caps }); const ms = this.now() - t0;
          if (task.validate && task.validate(out) !== true) throw new Error('출력 검증 실패');
          if (i === 0) { row.coldMs = ms; if (ref == null) ref = out; row.verified = b.id === report.backends.find(x => x.verified)?.id || compare(ref, out); } else row.warmMs.push(ms);
          this.metrics.record(key, b.id, { ms, cold: i === 0, ok: true }); this.warmed.add(key + '#' + b.id);
        }
        row.warmMedianMs = median(row.warmMs);
      } catch (e) { row.error = String(e.message || e); }
      report.backends.push(row);
    }
    const ok = report.backends.filter(r => r.verified && !r.error);
    report.comparable = ok.length >= 2;
    if (report.comparable) { const w = [...ok].sort((a, b) => a.warmMedianMs - b.warmMedianMs); report.winner = w[0].id; report.speedup = +(w[w.length - 1].warmMedianMs / w[0].warmMedianMs).toFixed(2); report.note = '동일 입력·동일 조건에서 출력이 일치한 백엔드만 비교한 warm 중앙값'; }
    else report.note = '출력이 일치하는 백엔드가 2개 미만이라 비교 수치를 제공하지 않음';
    return report;
  }
}
export function defaultCompare(a, b, tol = 1e-4) {
  if (typeof a === 'number' && typeof b === 'number') return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a));
  if (ArrayBuffer.isView(a) && ArrayBuffer.isView(b)) { if (a.length !== b.length) return false; for (let i = 0; i < a.length; i++) if (!(Math.abs(a[i] - b[i]) <= tol * Math.max(1, Math.abs(a[i])))) return false; return true; }
  return JSON.stringify(a) === JSON.stringify(b);
}

// 큰 모델 동시 적재 방지: 한 번에 하나만 상주. 새 모델 적재 전에 기존 모델을 해제한다.
export class ModelResidency {
  constructor({ budgetBytes = Infinity } = {}) { this.current = null; this.budget = budgetBytes; this.lock = Promise.resolve(); this.log = []; }
  acquire(id, { sizeBytes = 0, load, unload }) {
    const run = async () => {
      if (sizeBytes > this.budget) throw new Error(`메모리 예산 초과: ${id} ≈ ${(sizeBytes / 1e6) | 0}MB > ${(this.budget / 1e6) | 0}MB`);
      if (this.current?.id === id) return this.current.handle;
      if (this.current) { this.log.push('unload ' + this.current.id); await this.current.unload(); this.current = null; }
      this.log.push('load ' + id); const handle = await load();
      this.current = { id, handle, unload: async () => unload(handle), sizeBytes }; return handle;
    };
    const p = this.lock.then(run); this.lock = p.catch(() => {}); return p;
  }
  async release() { if (this.current) { await this.current.unload(); this.current = null; } }
}
