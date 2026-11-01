"""텍스트 -> wav.

사용: python -m tts.infer "안녕하세요" -o out.wav [--ckpt release/tts_ft.pt] [--vocoder release/vocoder.pt]
--vocoder 를 안 주면 Griffin-Lim 을 쓴다. 화자 0 = KSS(기본).
"""
import argparse

import librosa
import numpy as np
import soundfile as sf
import torch

from tts import audio
from tts.model import TTS
from tts.text import Vocab


def load_tts(path, vocab):
    s = torch.load(path)
    model = TTS(len(vocab), n_spk=s.get("n_spk", 1))
    # v1 체크포인트에는 피치/에너지 모듈이 없으므로 strict=False (새 임베딩은 0 초기화라 출력이 같다)
    model.load_state_dict({k: v.float() for k, v in s["model"].items()}, strict=False)
    model.g2p = bool(s.get("g2p", False))  # True 면 입력 텍스트에 발음 변환을 적용
    return model.eval(), s["mean"], s["std"]


def load_vocoder(path):
    from tts.vocoder import Generator
    g = Generator()
    g.load_state_dict({k: v.float() for k, v in torch.load(path)["G"].items()})
    return g.eval()


@torch.no_grad()
def synthesize(text, model, mean, std, vocoder=None, speaker=0, speed=1.0, vocab=None,
               dur_jitter=0.0, var_scale=1.0):
    if getattr(model, "g2p", False):
        from tts.g2p_ko import g2p
        text = g2p(text)
    ids = torch.tensor([vocab.encode(text)])
    mel_n = model.infer(ids, torch.tensor([speaker]), speed, dur_jitter, var_scale)[0]  # (T, 80) 정규화 멜
    if vocoder is None:
        return audio.mel_to_wav(mel_n.numpy() * std + mean)
    y = vocoder(mel_n.T[None])[0, 0].numpy()
    y = librosa.effects.deemphasis(y, coef=0.97)
    return (y / (np.abs(y).max() + 1e-8) * 0.9).astype(np.float32)


def _fade(y, n=240):
    y = y.copy()
    n = min(n, len(y) // 2)
    if n > 0:
        r = np.linspace(0, 1, n, dtype=np.float32)
        y[:n] *= r
        y[-n:] *= r[::-1]
    return y


@torch.no_grad()
def synthesize_long(text, model, mean, std, vocoder=None, speaker=0, speed=1.0, vocab=None,
                    max_chars=64, verbose=False, dur_jitter=0.0, var_scale=1.0):
    """숫자/영어 처리 + 긴 글 분할 합성. 조각 사이에 문장부호별 쉼을 넣어 이어 붙인다."""
    from tts import frontend
    norm, warns = frontend.normalize_text(text)
    if warns and verbose:
        print("경고(처리되지 못한 글자):", warns)
    parts = []
    for chunk, pause in frontend.split_text(norm, max_chars):
        if verbose:
            print(f"  [{len(chunk):2d}자 +{pause:.2f}s] {chunk}")
        y = synthesize(chunk, model, mean, std, vocoder, speaker, speed, vocab, dur_jitter, var_scale)
        y, _ = librosa.effects.trim(y, top_db=38)
        parts.append(_fade(y))
        if pause > 0:
            parts.append(np.zeros(int(pause * audio.SR), np.float32))
    if not parts:
        return np.zeros(audio.SR // 10, np.float32)
    y = np.concatenate(parts)
    return (y / (np.abs(y).max() + 1e-8) * 0.9).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("-o", "--out", default="out.wav")
    ap.add_argument("--ckpt", default="release/tts_ft.pt")
    ap.add_argument("--vocoder", default="")
    ap.add_argument("--vocab", default="data/proc/vocab.json")
    ap.add_argument("--speaker", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--raw", action="store_true", help="숫자/영어/분할 처리를 끄고 그대로 합성")
    ap.add_argument("--max_chars", type=int, default=64)
    ap.add_argument("--dur_jitter", type=float, default=0.0, help="박자 변화(로그정규 잡음 표준편차, 예 0.08)")
    ap.add_argument("--var_scale", type=float, default=1.0, help="억양 강조 배율(예 1.2). v2 모델에서만 의미 있음")
    a = ap.parse_args()

    vocab = Vocab.load(a.vocab)
    model, mean, std = load_tts(a.ckpt, vocab)
    voc = load_vocoder(a.vocoder) if a.vocoder else None
    if a.raw:
        y = synthesize(a.text, model, mean, std, voc, a.speaker, a.speed, vocab,
                       a.dur_jitter, a.var_scale)
    else:
        y = synthesize_long(a.text, model, mean, std, voc, a.speaker, a.speed, vocab,
                            a.max_chars, verbose=True, dur_jitter=a.dur_jitter, var_scale=a.var_scale)
    sf.write(a.out, y, audio.SR)
    print("saved", a.out, f"{len(y) / audio.SR:.2f}s")


if __name__ == "__main__":
    main()
