"""호출어 학습용 증강: 잡음 섞기, 합성 잔향, 마이크 특성(EQ), 이득/클리핑, 배경 음성(babble)."""
import numpy as np

from kws import config as C


def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def mix_snr(sig, noise, snr_db, rng):
    """sig 에 snr_db 로 noise 를 섞는다. noise 는 sig 길이에 맞춰 무작위로 자르거나 반복."""
    n = len(sig)
    if len(noise) < n:
        noise = np.tile(noise, int(np.ceil(n / len(noise))))
    s = rng.integers(0, len(noise) - n + 1)
    noise = noise[s:s + n]
    g = rms(sig) / (rms(noise) * 10 ** (snr_db / 20))
    return sig + g * noise


def synth_rir(rng, sr=C.SR):
    """지수적으로 감쇠하는 잡음으로 만든 잔향 임펄스 응답. RT60 0.1~0.7초."""
    rt60 = rng.uniform(0.1, 0.7)
    n = int(sr * rt60)
    t = np.arange(n) / sr
    h = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    h[0] = 1.0
    h /= np.sqrt(np.sum(h ** 2))
    return h


def reverb(x, rng):
    h = synth_rir(rng)
    # 직접음 비중을 무작위로 섞어 잔향 세기를 조절
    wet = rng.uniform(0.2, 1.0)
    y = np.convolve(x, h)[:len(x)]
    return (1 - wet) * x + wet * y * (rms(x) / (rms(y) + 1e-9))


def random_eq(x, rng):
    """저/고역 차단과 기울기로 마이크·스피커 특성을 흉내."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / C.SR)
    tilt = rng.uniform(-6, 6)                      # dB/옥타브 근사 기울기
    g = 10 ** ((tilt * np.log2(np.maximum(f, 50) / 1000)) / 20)
    lo = rng.uniform(50, 300); hi = rng.uniform(3500, 7800)
    g[f < lo] *= 0.1; g[f > hi] *= 0.1
    return np.fft.irfft(X * g, n=len(x))


def place(word, total, end_offset, rng):
    """word 파형을 길이 total 의 빈 창에, 끝이 창 끝에서 end_offset 샘플 앞에 오도록 놓는다."""
    out = np.zeros(total, np.float32)
    end = total - end_offset
    start = end - len(word)
    if start < 0:
        word = word[-start:]; start = 0
    out[start:start + len(word)] = word[:max(0, end - start)]
    return out


def make_positive(word_i16, bg_pool, noise_pool, rng, total=C.CLIP):
    """양성 하나: 단어를 창 끝 근처에 놓고, 앞에 배경 음성을 깔고, 잔향/EQ/잡음을 적용."""
    w = word_i16.astype(np.float32) / 32768
    w = w / (np.max(np.abs(w)) + 1e-6) * rng.uniform(0.2, 0.9)
    if rng.random() < 0.5:
        w = reverb(w, rng)
    if rng.random() < 0.5:
        w = random_eq(w, rng)
    x = place(w, total, int(rng.uniform(0.02, 0.35) * C.SR), rng)
    # 단어 앞 맥락: 다른 사람 말소리가 단어 앞쪽에 깔린 상황 (단어와 겹치지 않는 앞부분만)
    if rng.random() < 0.4 and len(bg_pool):
        bg = bg_pool[rng.integers(len(bg_pool))].astype(np.float32) / 32768
        bg = bg / (np.max(np.abs(bg)) + 1e-6) * rng.uniform(0.1, 0.5)
        mask = np.ones(total, np.float32)
        nz = np.nonzero(x)[0]
        if len(nz):
            mask[max(0, nz[0] - 800):] = 0
        x = x + bg * mask
    return finish(x, noise_pool, rng, snr=(-2, 20))


def make_negative(win_i16, noise_pool, rng, total=C.CLIP):
    x = win_i16.astype(np.float32) / 32768
    x = x / (np.max(np.abs(x)) + 1e-6) * rng.uniform(0.1, 0.9)
    if rng.random() < 0.4:
        x = reverb(x, rng)
    if rng.random() < 0.4:
        x = random_eq(x, rng)
    return finish(x, noise_pool, rng, snr=(-2, 20))


def make_background(noise_pool, bg_pool, rng, total=C.CLIP):
    """아무도 말하지 않는 창: 잡음/무음/먼 대화."""
    r = rng.random()
    if r < 0.15:
        x = rng.standard_normal(total).astype(np.float32) * rng.uniform(1e-4, 5e-3)
    elif r < 0.65 and len(noise_pool):
        n = noise_pool[rng.integers(len(noise_pool))]
        x = n if len(n) >= total else np.tile(n, int(np.ceil(total / len(n))))
        s = rng.integers(0, len(x) - total + 1)
        x = x[s:s + total].astype(np.float32) / 32768 * rng.uniform(0.05, 0.9)
    else:
        parts = [bg_pool[rng.integers(len(bg_pool))].astype(np.float32) / 32768 for _ in range(rng.integers(2, 5))]
        x = np.sum(parts, axis=0) / len(parts) * rng.uniform(0.05, 0.6)
    return finish(x, noise_pool, rng, snr=(5, 30), p_noise=0.3)


def finish(x, noise_pool, rng, snr, p_noise=0.85):
    if len(noise_pool) and rng.random() < p_noise:
        n = noise_pool[rng.integers(len(noise_pool))].astype(np.float32) / 32768
        x = mix_snr(x, n, rng.uniform(*snr), rng)
    if rng.random() < 0.15:                                    # 클리핑
        x = np.clip(x * rng.uniform(1.5, 4), -1, 1)
    x = x * rng.uniform(0.5, 1.5)
    peak = np.max(np.abs(x)) + 1e-9
    if peak > 1:
        x = x / peak
    return x.astype(np.float32)
