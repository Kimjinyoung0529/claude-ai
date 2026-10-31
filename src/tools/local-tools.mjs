// 에이전트/AI가 쓰는 로컬 도구 29개. risk: read | write | destructive. write/destructive 는 승인 정책 대상.
// 파일은 기기 전체 파일시스템이 아니라 앱 내부 텍스트 파일(IndexedDB)이다.
import { calculate } from './calc.mjs';
const S = (props, required = []) => ({ type: 'object', properties: props, required, additionalProperties: false });
const str = d => ({ type: 'string', description: d }), int = d => ({ type: 'integer', description: d });
const rev = { type: 'integer', description: '읽을 때 받은 rev. 다르면 충돌 오류.' };
const snip = (t, q) => { const i = t.toLowerCase().indexOf(q.toLowerCase()); return i < 0 ? t.slice(0, 80) : t.slice(Math.max(0, i - 30), i + q.length + 50); };
const pub = r => r && ({ id: r.id, rev: r.rev, ...r });

export function createLocalTools(store, { now = () => new Date() } = {}) {
  const T = [];
  const def = (name, risk, description, schema, run) => T.push({ name, risk, description, schema, run });
  // ── 노트 (6)
  def('note_create', 'write', '새 노트를 만든다.', S({ title: str('제목'), body: str('본문'), tags: { type: 'array', items: { type: 'string' } } }, ['title']), async a => pub(await store.create('notes', { title: a.title, body: a.body || '', tags: a.tags || [] })));
  def('note_read', 'read', '노트 한 개를 읽는다(rev 포함).', S({ id: str('노트 id') }, ['id']), async a => { const n = await store.get('notes', a.id); if (!n) throw new Error('노트 없음'); return n; });
  def('note_list', 'read', '최근 노트 목록(제목·id·rev).', S({ limit: int('최대 개수') }), async a => (await store.list('notes')).slice(0, a.limit || 20).map(n => ({ id: n.id, rev: n.rev, title: n.title, tags: n.tags, updatedAt: n.updatedAt })));
  def('note_search', 'read', '제목·본문·태그에 문자열이 포함된 노트를 찾는다.', S({ query: str('검색어') }, ['query']), async a => (await store.list('notes')).filter(n => (n.title + '\n' + n.body + '\n' + (n.tags || []).join(' ')).toLowerCase().includes(a.query.toLowerCase())).slice(0, 20).map(n => ({ id: n.id, rev: n.rev, title: n.title, snippet: snip(n.body, a.query) })));
  def('note_update', 'write', '노트를 수정한다. expected_rev 필수(충돌 검사).', S({ id: str('id'), expected_rev: rev, title: str('새 제목'), body: str('새 본문'), tags: { type: 'array', items: { type: 'string' } } }, ['id', 'expected_rev']), async a => { const p = {}; for (const k of ['title', 'body', 'tags']) if (a[k] !== undefined) p[k] = a[k]; return pub(await store.update('notes', a.id, p, a.expected_rev)); });
  def('note_delete', 'destructive', '노트를 삭제한다. expected_rev 필수.', S({ id: str('id'), expected_rev: rev }, ['id', 'expected_rev']), async a => { await store.remove('notes', a.id, a.expected_rev); return { deleted: a.id }; });
  // ── 할 일 (6)
  def('todo_create', 'write', '할 일을 만든다.', S({ text: str('내용'), due: str('마감(ISO 날짜, 선택)') }, ['text']), async a => pub(await store.create('todos', { text: a.text, due: a.due || null, done: false })));
  def('todo_list', 'read', '할 일 목록. status: open|done|all', S({ status: { type: 'string', enum: ['open', 'done', 'all'] } }), async a => (await store.list('todos')).filter(t => (a.status || 'open') === 'all' || (a.status || 'open') === (t.done ? 'done' : 'open')).map(t => ({ id: t.id, rev: t.rev, text: t.text, done: t.done, due: t.due })));
  def('todo_complete', 'write', '할 일을 완료 처리.', S({ id: str('id') }, ['id']), async a => pub(await store.update('todos', a.id, { done: true })));
  def('todo_reopen', 'write', '완료한 할 일을 다시 연다.', S({ id: str('id') }, ['id']), async a => pub(await store.update('todos', a.id, { done: false })));
  def('todo_rename', 'write', '할 일 이름을 바꾼다.', S({ id: str('id'), text: str('새 내용') }, ['id', 'text']), async a => pub(await store.update('todos', a.id, { text: a.text })));
  def('todo_delete', 'destructive', '할 일을 삭제한다.', S({ id: str('id'), expected_rev: rev }, ['id']), async a => { await store.remove('todos', a.id, a.expected_rev); return { deleted: a.id }; });
  // ── 앱 내부 파일 (6)
  const pathOk = p => { if (!/^[\w가-힣.\- /]{1,120}$/.test(p) || p.includes('..') || p.startsWith('/')) throw new Error('잘못된 경로(상대 경로, .. 불가)'); return p; };
  const byPath = async p => (await store.list('files')).find(f => f.path === p);
  def('file_create', 'write', '앱 내부 텍스트 파일을 만든다(기기 파일시스템 아님).', S({ path: str('예: memo/a.txt'), content: str('내용') }, ['path', 'content']), async a => { pathOk(a.path); if (await byPath(a.path)) throw new Error('이미 존재: file_replace 사용'); return pub(await store.create('files', { path: a.path, content: a.content })); });
  def('file_read', 'read', '앱 내부 파일을 읽는다.', S({ path: str('경로') }, ['path']), async a => { const f = await byPath(a.path); if (!f) throw new Error('파일 없음'); return { path: f.path, rev: f.rev, content: f.content }; });
  def('file_list', 'read', '앱 내부 파일 목록.', S({ prefix: str('경로 접두사') }), async a => (await store.list('files')).filter(f => f.path.startsWith(a.prefix || '')).map(f => ({ path: f.path, rev: f.rev, bytes: new TextEncoder().encode(f.content).length })));
  def('file_replace', 'write', '파일 내용을 통째로 교체. expected_rev 필수.', S({ path: str('경로'), content: str('새 내용'), expected_rev: rev }, ['path', 'content', 'expected_rev']), async a => { const f = await byPath(a.path); if (!f) throw new Error('파일 없음'); return pub(await store.update('files', f.id, { content: a.content }, a.expected_rev)); });
  def('file_delete', 'destructive', '파일을 삭제. expected_rev 필수.', S({ path: str('경로'), expected_rev: rev }, ['path', 'expected_rev']), async a => { const f = await byPath(a.path); if (!f) throw new Error('파일 없음'); await store.remove('files', f.id, a.expected_rev); return { deleted: a.path }; });
  def('file_search', 'read', '앱 내부 파일 내용 검색.', S({ query: str('검색어') }, ['query']), async a => (await store.list('files')).filter(f => f.content.toLowerCase().includes(a.query.toLowerCase())).slice(0, 20).map(f => ({ path: f.path, snippet: snip(f.content, a.query) })));
  // ── 텍스트 (5)
  def('text_stats', 'read', '글자·단어·줄 수.', S({ text: str('텍스트') }, ['text']), async a => ({ chars: [...a.text].length, words: (a.text.trim().match(/\S+/g) || []).length, lines: a.text.split('\n').length }));
  def('text_replace', 'read', '문자열 치환 결과를 반환(저장하지 않음).', S({ text: str('원문'), find: str('찾을 문자열'), replace: str('바꿀 문자열') }, ['text', 'find', 'replace']), async a => { if (!a.find) throw new Error('find 가 비어 있음'); const parts = a.text.split(a.find); return { result: parts.join(a.replace), count: parts.length - 1 }; });
  def('text_dedupe_lines', 'read', '중복 줄 제거(순서 유지).', S({ text: str('텍스트') }, ['text']), async a => { const seen = new Set(); const out = a.text.split('\n').filter(l => !seen.has(l) && seen.add(l)); return { result: out.join('\n'), removed: a.text.split('\n').length - out.length }; });
  def('text_extract_links', 'read', 'http(s) 링크 추출.', S({ text: str('텍스트') }, ['text']), async a => [...new Set((a.text.match(/https?:\/\/[^\s<>"')\]]+/g) || []).map(u => u.replace(/[.,;:!?]+$/, '')))]);
  def('text_diff', 'read', '두 텍스트의 줄 단위 차이.', S({ a: str('원본'), b: str('수정본') }, ['a', 'b']), async x => { const A = x.a.split('\n'), B = x.b.split('\n'); const sa = new Set(A), sb = new Set(B); return { removed: A.filter(l => !sb.has(l)), added: B.filter(l => !sa.has(l)) }; });
  // ── 데이터/계산/시간 (6)
  def('json_format', 'read', 'JSON 검증 후 정렬·들여쓰기.', S({ text: str('JSON 문자열'), sort_keys: { type: 'boolean' } }, ['text']), async a => { const sort = v => Array.isArray(v) ? v.map(sort) : v && typeof v === 'object' ? Object.fromEntries(Object.keys(v).sort().map(k => [k, sort(v[k])])) : v; const j = JSON.parse(a.text); return { result: JSON.stringify(a.sort_keys ? sort(j) : j, null, 2) }; });
  def('csv_parse', 'read', 'CSV를 행 배열로(따옴표 지원).', S({ text: str('CSV'), delimiter: str('구분자(기본 ,)') }, ['text']), async a => parseCsv(a.text, a.delimiter || ','));
  def('calculate', 'read', '사칙연산·거듭제곱·함수(sqrt, sin …) 계산.', S({ expression: str('식') }, ['expression']), async a => ({ result: calculate(a.expression) }));
  def('time_now', 'read', '현재 시각(ISO, 지정한 시간대의 현지 표기).', S({ timezone: str('IANA 시간대, 기본 Asia/Seoul') }), async a => { const tz = a.timezone || 'Asia/Seoul'; return { iso: now().toISOString(), local: fmtTz(now(), tz), timezone: tz }; });
  def('time_convert', 'read', '시각을 다른 시간대로 변환.', S({ datetime: str('ISO 8601, 오프셋 포함 권장'), to: str('IANA 시간대') }, ['datetime', 'to']), async a => { const d = new Date(a.datetime); if (isNaN(d)) throw new Error('잘못된 날짜'); return { iso: d.toISOString(), local: fmtTz(d, a.to), timezone: a.to }; });
  def('sha256', 'read', '문자열의 SHA-256 해시(hex).', S({ text: str('텍스트') }, ['text']), async a => ({ hex: [...new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(a.text)))].map(b => b.toString(16).padStart(2, '0')).join('') }));
  return T;
}
function fmtTz(d, tz) { try { return new Intl.DateTimeFormat('ko-KR', { timeZone: tz, dateStyle: 'full', timeStyle: 'long' }).format(d); } catch { throw new Error('알 수 없는 시간대: ' + tz); } }
export function parseCsv(text, delim = ',') {
  const rows = []; let row = [], f = '', q = false;
  for (let i = 0; i < text.length; i++) { const c = text[i];
    if (q) { if (c === '"') { if (text[i + 1] === '"') { f += '"'; i++; } else q = false; } else f += c; }
    else if (c === '"') q = true; else if (c === delim) { row.push(f); f = ''; } else if (c === '\n' || c === '\r') { if (c === '\r' && text[i + 1] === '\n') i++; row.push(f); rows.push(row); row = []; f = ''; } else f += c; }
  if (q) throw new Error('닫히지 않은 따옴표'); if (f !== '' || row.length) { row.push(f); rows.push(row); } return rows;
}
