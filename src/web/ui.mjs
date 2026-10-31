export function h(tag, props = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) { if (v == null || v === false) continue; if (k === 'class') el.className = v; else if (k.startsWith('on')) el.addEventListener(k.slice(2).toLowerCase(), v); else if (k === 'html') el.innerHTML = v; else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v); else if (v === true) el.setAttribute(k, ''); else el.setAttribute(k, v); }
  for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid.nodeType ? kid : document.createTextNode(kid));
  return el;
}
export function toast(msg, err = false) { const t = h('div', { class: 'toast' + (err ? ' err' : '') }, msg); document.getElementById('toasts').append(t); setTimeout(() => t.remove(), err ? 6000 : 2800); }
export const mb = n => n >= 1e9 ? (n / 1e9).toFixed(2) + ' GB' : (n / 1e6).toFixed(n < 1e7 ? 1 : 0) + ' MB';
export const fmtTime = s => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
export const ago = t => { const s = (Date.now() - t) / 1000; return s < 60 ? '방금' : s < 3600 ? `${s / 60 | 0}분 전` : s < 86400 ? `${s / 3600 | 0}시간 전` : new Date(t).toLocaleDateString('ko-KR'); };
export function dialog(build) { // build(close) → Node. Promise 는 close(value) 로 해결
  const d = document.getElementById('dlg'); return new Promise(res => { d.replaceChildren(); const close = v => { d.close(); res(v); }; d.append(h('div', { class: 'in' }, build(close))); d.onclose = () => res(undefined); d.showModal(); });
}
export const confirmDlg = (title, body, ok = '확인', danger = false) => dialog(close => [h('h3', {}, title), h('p', { class: 'sub' }, body), h('div', { class: 'row' }, h('button', { class: 'btn grow', onclick: () => close(false) }, '취소'), h('button', { class: 'btn grow ' + (danger ? 'danger' : 'pri'), onclick: () => close(true) }, ok))]);
export function download(name, text, type = 'text/plain') { const a = h('a', { href: URL.createObjectURL(new Blob([text], { type })), download: name }); document.body.append(a); a.click(); a.remove(); }
export function progressBar(label) { const bar = h('i'); const txt = h('span', { class: 'sub' }, label || ''); const el = h('div', {}, h('div', { class: 'bar' }, bar), txt); return { el, set(p, t) { bar.style.width = (Math.max(0, Math.min(1, p)) * 100).toFixed(1) + '%'; if (t != null) txt.textContent = t; } }; }
// replaceChildren/append 에 null·false 가 문자열 "null" 로 찍히지 않도록 걸러낸다.
for (const m of ['replaceChildren', 'append']) { const o = Element.prototype[m]; Element.prototype[m] = function (...k) { return o.apply(this, k.filter(x => x != null && x !== false)); }; }
