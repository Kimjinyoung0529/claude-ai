// 로컬 저장소. 백엔드(IndexedDB / 메모리)를 교체 가능. 모든 레코드는 {id, rev, createdAt, updatedAt} 를 가진다.
export const STORES = ['notes', 'todos', 'files', 'conversations', 'cards', 'dict'];
export function memoryBackend() {
  const d = Object.fromEntries(STORES.map(s => [s, new Map()]));
  return { async getAll(s) { return [...d[s].values()].map(v => structuredClone(v)); }, async get(s, id) { const v = d[s].get(id); return v ? structuredClone(v) : undefined; }, async put(s, v) { d[s].set(v.id, structuredClone(v)); }, async del(s, id) { d[s].delete(id); }, async clear() { STORES.forEach(s => d[s].clear()); } };
}
export function idbBackend(name = 'naegyeot') {
  const open = () => new Promise((res, rej) => { const r = indexedDB.open(name, 1); r.onupgradeneeded = () => STORES.forEach(s => r.result.createObjectStore(s, { keyPath: 'id' })); r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); });
  let dbp; const db = () => dbp ||= open();
  const tx = async (s, mode, fn) => { const d = await db(); return new Promise((res, rej) => { const t = d.transaction(s, mode); const q = fn(t.objectStore(s)); t.oncomplete = () => res(q?.result); t.onerror = () => rej(t.error); t.onabort = () => rej(t.error); }); };
  return { getAll: s => tx(s, 'readonly', o => o.getAll()), get: (s, id) => tx(s, 'readonly', o => o.get(id)), put: (s, v) => tx(s, 'readwrite', o => o.put(v)), del: (s, id) => tx(s, 'readwrite', o => o.delete(id)), async clear() { for (const s of STORES) await tx(s, 'readwrite', o => o.clear()); } };
}
export class ConflictError extends Error { constructor(m) { super(m); this.name = 'ConflictError'; } }
const uid = () => (globalThis.crypto?.randomUUID?.() || Math.random().toString(36).slice(2) + Date.now().toString(36));
export class Store {
  constructor(backend = memoryBackend(), now = () => Date.now()) { this.b = backend; this.now = now; this.listeners = new Set(); }
  on(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  _emit(s, type, rec) { this.listeners.forEach(f => f({ store: s, type, rec })); }
  list(s) { return this.b.getAll(s).then(a => a.sort((x, y) => y.updatedAt - x.updatedAt)); }
  get(s, id) { return this.b.get(s, id); }
  async create(s, data) { const t = this.now(); const rec = { ...data, id: data.id || uid(), rev: 1, createdAt: t, updatedAt: t }; await this.b.put(s, rec); this._emit(s, 'create', rec); return rec; }
  // expectedRev 가 주어지면 낙관적 동시성 검사를 한다 (에이전트 도구의 충돌 검사).
  async update(s, id, patch, expectedRev) {
    const cur = await this.b.get(s, id); if (!cur) throw new Error(`${s}/${id} 없음`);
    if (expectedRev != null && cur.rev !== expectedRev) throw new ConflictError(`충돌: 읽은 rev=${expectedRev} 이후 현재 rev=${cur.rev}로 변경됨. 다시 읽어 주세요.`);
    const rec = { ...cur, ...patch, id, rev: cur.rev + 1, createdAt: cur.createdAt, updatedAt: this.now() }; await this.b.put(s, rec); this._emit(s, 'update', rec); return rec;
  }
  async remove(s, id, expectedRev) {
    const cur = await this.b.get(s, id); if (!cur) throw new Error(`${s}/${id} 없음`);
    if (expectedRev != null && cur.rev !== expectedRev) throw new ConflictError(`충돌: 현재 rev=${cur.rev}`);
    await this.b.del(s, id); this._emit(s, 'delete', cur); return cur;
  }
  async exportAll() { const out = { app: 'naegyeot-night-signal', version: 1, exportedAt: new Date(this.now()).toISOString(), data: {} }; for (const s of STORES) out.data[s] = await this.b.getAll(s); return out; }
  async importAll(json, { replace = false } = {}) {
    if (json?.app !== 'naegyeot-night-signal' || !json.data) throw new Error('내곁 백업 파일이 아닙니다');
    for (const s of Object.keys(json.data)) if (!STORES.includes(s)) throw new Error('알 수 없는 저장소: ' + s);
    if (replace) await this.b.clear(); let n = 0;
    for (const s of STORES) for (const r of json.data[s] || []) { if (!r?.id) continue; const cur = replace ? null : await this.b.get(s, r.id); if (!cur || cur.updatedAt < r.updatedAt) { await this.b.put(s, r); n++; } }
    return n;
  }
}
