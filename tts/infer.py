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
    model.load_state_dict({k: v.float() for k, v in s["model"].items()})
    return model.eval(), s["mean"], s["std"]


def load_vocoder(path):
    from tts.vocoder import Generator
    g = Generator()
    g.load_state_dict({k: v.float() for k, v in torch.load(path)["G"].items()})
    return g.eval()


@torch.no_grad()
def synthesize(text, model, mean, std, vocoder=None, speaker=0, speed=1.0, vocab=None):
    ids = torch.tensor([vocab.encode(text)])
    mel_n = model.infer(ids, torch.tensor([speaker]), speed)[0]  # (T, 80) 정규화 멜
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
                    max_chars=64, verbose=False):
    """숫자/영어 처리 + 긴 글 분할 합성. 조각 사이에 문장부호별 쉼을 넣어 이어 붙인다."""
    from tts import frontend
    norm, warns = frontend.normalize_text(text)
    if warns and verbose:
        print("경고(처리되지 못한 글자):", warns)
    parts = []
    for chunk, pause in frontend.split_text(norm, max_chars):
        if verbose:
            print(f"  [{len(chunk):2d}자 +{pause:.2f}s] {chunk}")
        y = synthesize(chunk, model, mean, std, vocoder, speaker, speed, vocab)
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
    a = ap.parse_args()

    vocab = Vocab.load(a.vocab)
    model, mean, std = load_tts(a.ckpt, vocab)
    voc = load_vocoder(a.vocoder) if a.vocoder else None
    if a.raw:
        y = synthesize(a.text, model, mean, std, voc, a.speaker, a.speed, vocab)
    else:
        y = synthesize_long(a.text, model, mean, std, voc, a.speaker, a.speed, vocab,
                            a.max_chars, verbose=True)
    sf.write(a.out, y, audio.SR)
    print("saved", a.out, f"{len(y) / audio.SR:.2f}s")


if __name__ == "__main__":
    main()
