// 테스트 환경용: 같은 출처에 hf-host.txt 가 있으면 모델을 그 미러에서 받는다. 배포본에는 이 파일이 없다(404 → 기본 huggingface.co).
export async function hfHost() { try { const r = await fetch(new URL('./hf-host.txt', self.location.href)); if (r.ok) return (await r.text()).trim(); } catch {} return null; }
export async function hfUrl(u) { const h = await hfHost(); return h ? u.replace('https://huggingface.co/', h) : u; }
