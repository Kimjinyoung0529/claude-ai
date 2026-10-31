"""텍스트 -> wav. 사용: python -m tts.infer "안녕하세요" -o out.wav"""
import argparse

import soundfile as sf
import torch

from tts import audio
from tts.model import TTS
from tts.text import Vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("-o", "--out", default="out.wav")
    ap.add_argument("--ckpt", default="ckpt/last.pt")
    ap.add_argument("--vocab", default="data/proc/vocab.json")
    ap.add_argument("--speed", type=float, default=1.0)
    a = ap.parse_args()

    vocab = Vocab.load(a.vocab)
    s = torch.load(a.ckpt)
    model = TTS(len(vocab))
    model.load_state_dict(s["model"])
    model.eval()
    ids = torch.tensor([vocab.encode(a.text)])
    mel = model.infer(ids, a.speed)[0].numpy() * s["std"] + s["mean"]
    sf.write(a.out, audio.mel_to_wav(mel), audio.SR)
    print("saved", a.out, f"{len(mel) * audio.HOP / audio.SR:.2f}s")


if __name__ == "__main__":
    main()
