// 대화 모델 worker 클라이언트. 한 번에 모델 하나만 상주(교체 시 이전 모델 해제).
import { pickDtype, pickDevice, modelBytes } from '../llm/catalog.mjs';
export class LlmClient extends EventTarget {
  constructor(caps) { super(); this.caps = caps; this.worker = null; this.model = null; this.info = null; this.state = 'empty'; this.pending = new Map(); this.rid = 0; this.loading = null; }
  _set(state, extra = {}) { this.state = state; this.dispatchEvent(new CustomEvent('state', { detail: { state, ...extra } })); }
  _spawn() { if (this.worker) this.worker.terminate(); this.worker = new Worker('./llm-worker.js', { type: 'module' }); this.worker.onmessage = e => this._on(e.data); this.worker.onerror = e => { this._fail(new Error('worker 오류: ' + (e.message || 'unknown'))); }; }
  _fail(err) { for (const p of this.pending.values()) p.reject(err); this.pending.clear(); if (this.loading) { this.loading.reject(err); this.loading = null; } this._set('error', { message: err.message }); }
  _on(m) {
    if (m.type === 'progress') this._set('loading', { loaded: m.loaded, total: m.total, status: '다운로드 중' });
    else if (m.type === 'status') this._set('loading', { status: m.text });
    else if (m.type === 'ready') { this.info = m.info; this._set('ready', { info: m.info }); this.loading?.resolve(m.info); this.loading = null; }
    else if (m.type === 'unloaded') { this.info = null; this.model = null; this._set('empty'); }
    else if (m.type === 'token') this.pending.get(m.rid)?.onToken?.(m.text);
    else if (m.type === 'done') { const p = this.pending.get(m.rid); this.pending.delete(m.rid); this._set('ready', { info: this.info }); p?.resolve({ text: m.text, stats: m.stats }); }
    else if (m.type === 'error') {
      const err = Object.assign(new Error(m.message), { deviceLost: m.deviceLost });
      if (m.phase === 'load') { this.loading?.reject(err); this.loading = null; this.model = null; this._set('error', { message: m.message, phase: 'load' }); }
      else { const p = this.pending.get(m.rid); this.pending.delete(m.rid); this._set(this.info ? 'ready' : 'error', { info: this.info, message: m.message }); p?.reject(err); }
    }
  }
  load(model) {
    const dtype = pickDtype(model, this.caps), device = pickDevice(this.caps);
    const wasmDt = pickDtype(model, { ...this.caps, fp16: false, webgpu: false });
    this.model = { model, dtype }; this._spawn(); this._set('loading', { loaded: 0, total: modelBytes(model, dtype), status: '시작' });
    return new Promise((resolve, reject) => { this.loading = { resolve, reject }; this.worker.postMessage({ type: 'load', id: model.id, device, dtype, fallbackDtype: wasmDt }); });
  }
  cancelLoad() { this.worker?.terminate(); this.worker = null; if (this.loading) { this.loading.reject(Object.assign(new Error('불러오기를 중단했습니다'), { name: 'AbortError' })); this.loading = null; } this.model = null; this.info = null; this._set('empty'); }
  generate(messages, { onToken, signal, max_new_tokens = 512, temperature = 0.7, thinking = false } = {}) {
    if (this.state !== 'ready') return Promise.reject(new Error('모델이 준비되지 않았습니다'));
    const rid = ++this.rid; this._set('generating', { info: this.info });
    return new Promise((resolve, reject) => {
      this.pending.set(rid, { resolve, reject, onToken });
      signal?.addEventListener('abort', () => this.worker?.postMessage({ type: 'stop' }), { once: true });
      this.worker.postMessage({ type: 'generate', rid, messages, max_new_tokens, temperature, thinking });
    });
  }
  stop() { this.worker?.postMessage({ type: 'stop' }); }
  async unload() { if (!this.worker) return; this.worker.postMessage({ type: 'unload' }); await new Promise(r => setTimeout(r, 50)); this.worker.terminate(); this.worker = null; this.model = null; this.info = null; this._set('empty'); }
  get ready() { return this.state === 'ready' || this.state === 'generating'; }
}
