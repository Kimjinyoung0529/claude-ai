// eval 을 쓰지 않는 안전한 계산기: + - * / % ^ ( ) 함수 상수.
const F = { sqrt: Math.sqrt, abs: Math.abs, sin: Math.sin, cos: Math.cos, tan: Math.tan, ln: Math.log, log: Math.log10, floor: Math.floor, ceil: Math.ceil, round: Math.round, min: Math.min, max: Math.max, pow: Math.pow };
const C = { pi: Math.PI, e: Math.E };
export function calculate(src) {
  const t = String(src).replace(/×/g, '*').replace(/÷/g, '/').match(/\d+\.?\d*(?:e[+-]?\d+)?|\.\d+|[a-z]+|[-+*/%^(),]|\S/gi) || []; let i = 0;
  const peek = () => t[i], next = () => t[i++];
  const expect = c => { if (next() !== c) throw new Error(`'${c}' 필요`); };
  function expr() { let v = term(); while (peek() === '+' || peek() === '-') v = next() === '+' ? v + term() : v - term(); return v; }
  function term() { let v = pow(); while (['*', '/', '%'].includes(peek())) { const o = next(), r = pow(); if (o === '*') v *= r; else if (o === '/') { if (r === 0) throw new Error('0으로 나눔'); v /= r; } else v %= r; } return v; }
  function pow() { const b = unary(); if (peek() === '^') { next(); return Math.pow(b, pow()); } return b; }
  function unary() { if (peek() === '-') { next(); return -unary(); } if (peek() === '+') { next(); return unary(); } return atom(); }
  function atom() {
    const x = next(); if (x == null) throw new Error('식이 끝남');
    if (/^[\d.]/.test(x)) { const n = Number(x); if (Number.isNaN(n)) throw new Error('잘못된 숫자 ' + x); return n; }
    if (x === '(') { const v = expr(); expect(')'); return v; }
    const k = x.toLowerCase();
    if (k in C && peek() !== '(') return C[k];
    if (k in F) { expect('('); const a = []; if (peek() !== ')') { a.push(expr()); while (peek() === ',') { next(); a.push(expr()); } } expect(')'); return F[k](...a); }
    throw new Error('알 수 없는 토큰 ' + x);
  }
  const v = expr(); if (i < t.length) throw new Error('불필요한 토큰 ' + t[i]);
  if (!Number.isFinite(v)) throw new Error('결과가 유한하지 않음'); return v;
}
