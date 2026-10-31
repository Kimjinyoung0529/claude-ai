"""공통 오디오 설정과 멜 스펙트로그램 / Griffin-Lim 변환."""
import numpy as np
import librosa

SR = 16000
N_FFT = 1024
HOP = 256
WIN = 1024
N_MELS = 80
FMIN, FMAX = 0, 8000


def wav_to_mel(y: np.ndarray) -> np.ndarray:
    """float32 wav(SR) -> log-mel (T, N_MELS)."""
    y = librosa.effects.preemphasis(y, coef=0.97)
    m = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, win_length=WIN,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=1.0,
    )
    return np.log(np.clip(m, 1e-5, None)).T.astype(np.float32)


def mel_to_wav(logmel: np.ndarray, n_iter: int = 60) -> np.ndarray:
    """log-mel (T, N_MELS) -> wav via Griffin-Lim."""
    m = np.exp(logmel).T
    y = librosa.feature.inverse.mel_to_audio(
        m, sr=SR, n_fft=N_FFT, hop_length=HOP, win_length=WIN,
        fmin=FMIN, fmax=FMAX, power=1.0, n_iter=n_iter,
    )
    y = librosa.effects.deemphasis(y, coef=0.97)
    y = y / (np.abs(y).max() + 1e-8) * 0.9
    return y.astype(np.float32)
