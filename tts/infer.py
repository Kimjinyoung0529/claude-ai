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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("-o", "--out", default="out.wav")
    ap.add_argument("--ckpt", default="release/tts_ft.pt")
    ap.add_argument("--vocoder", default="")
    ap.add_argument("--vocab", default="data/proc/vocab.json")
    ap.add_argument("--speaker", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0)
    a = ap.parse_args()

    vocab = Vocab.load(a.vocab)
    model, mean, std = load_tts(a.ckpt, vocab)
    voc = load_vocoder(a.vocoder) if a.vocoder else None
    y = synthesize(a.text, model, mean, std, voc, a.speaker, a.speed, vocab)
    sf.write(a.out, y, audio.SR)
    print("saved", a.out, f"{len(y) / audio.SR:.2f}s")


if __name__ == "__main__":
    main()
