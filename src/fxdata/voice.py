"""Performance features from a dry guitar signal, and a simple source-filter "voice" renderer.

생성형 모델의 정답(타깃)을 우리가 직접 만들기 위한 프로토타입이다.
기타 DI에서 음높이/세기를 뽑고(`extract_performance`), 그 값으로 모음 소리를 합성한다(`render_voice`).
라이선스 걸리는 보컬 데이터나 합성기 없이 "연주가 보존되는가"를 측정하는 데 쓴다.

한계: 단선율(한 번에 한 음)만 안정적으로 추적한다. 코드(화음)에서는 음높이 추정이 불안정하다.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from scipy.signal import lfilter

F0_SR = 16000          # 음높이 추정은 16 kHz로 내려서 한다 (기타 기본음은 충분히 포함)
F0_HOP = 80            # 16 kHz에서 80샘플 = 5 ms
FMIN, FMAX = 70.0, 1100.0

# 모음별 포먼트 (중심 주파수 Hz, 대역폭 Hz). 성인 남성 평균 근처의 대략적인 값.
VOWELS = {
    "a": [(800, 80), (1200, 90), (2800, 120), (3600, 150)],
    "o": [(500, 70), (900, 80), (2600, 120), (3400, 150)],
    "e": [(500, 70), (1800, 90), (2600, 120), (3400, 150)],
    "i": [(300, 60), (2300, 90), (3000, 120), (3600, 150)],
    "u": [(320, 60), (800, 80), (2500, 120), (3400, 150)],
}


@dataclass
class Performance:
    f0: np.ndarray        # (frames,) Hz, 무성 구간은 0
    voiced: np.ndarray    # (frames,) bool
    loud_db: np.ndarray   # (frames,) dBFS (RMS)
    hop_s: float          # 프레임 간격(초)


def extract_performance(x: np.ndarray, sr: int) -> Performance:
    y = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=F0_SR)
    f0, voiced, _ = librosa.pyin(
        y, fmin=FMIN, fmax=FMAX, sr=F0_SR, frame_length=2048, hop_length=F0_HOP, fill_na=0.0
    )
    rms = librosa.feature.rms(y=y, frame_length=640, hop_length=F0_HOP, center=True)[0]
    n = min(len(f0), len(rms))
    loud = 20.0 * np.log10(rms[:n] + 1e-6)
    return Performance(f0[:n].astype(np.float32), voiced[:n].astype(bool), loud.astype(np.float32),
                       F0_HOP / F0_SR)


def _to_samples(frames: np.ndarray, hop_s: float, sr: int, n: int, kind: str = "linear") -> np.ndarray:
    t_frames = np.arange(len(frames)) * hop_s
    t = np.arange(n) / sr
    if kind == "hold":
        idx = np.clip(np.searchsorted(t_frames, t, side="right") - 1, 0, len(frames) - 1)
        return frames[idx]
    return np.interp(t, t_frames, frames)


def _glottal(phase: np.ndarray) -> np.ndarray:
    """위상(0~1 반복)에서 성문 펄스 파형을 만든다 (Rosenberg식, 미분해서 방사 특성 반영)."""
    p = phase % 1.0
    open_q = 0.6
    g = np.where(
        p < open_q,
        0.5 * (1.0 - np.cos(np.pi * p / open_q)),
        np.cos(np.pi * (p - open_q) / (2.0 * (1.0 - open_q))),
    )
    return np.diff(g, prepend=g[0])


def _formants(x: np.ndarray, sr: int, vowel: str) -> np.ndarray:
    y = np.zeros_like(x)
    for freq, bw in VOWELS[vowel]:
        r = np.exp(-np.pi * bw / sr)
        theta = 2.0 * np.pi * freq / sr
        a = [1.0, -2.0 * r * np.cos(theta), r * r]
        y += lfilter([1.0 - r], a, x)
    return y


def render_voice(perf: Performance, sr: int, n: int, vowel: str = "a",
                 vibrato_hz: float = 5.5, vibrato_cents: float = 12.0,
                 breath: float = 0.04, seed: int = 0) -> np.ndarray:
    """f0/세기 곡선으로 모음 소리를 합성해 길이 n의 모노 신호를 반환한다."""
    f0 = _to_samples(perf.f0, perf.hop_s, sr, n, "hold")
    voiced = _to_samples(perf.voiced.astype(np.float32), perf.hop_s, sr, n)
    loud = _to_samples(perf.loud_db, perf.hop_s, sr, n)

    t = np.arange(n) / sr
    f0 = f0 * 2.0 ** ((vibrato_cents / 1200.0) * np.sin(2 * np.pi * vibrato_hz * t))
    phase = np.cumsum(np.where(f0 > 0, f0, 0.0)) / sr
    src = _glottal(phase)
    rng = np.random.default_rng(seed)
    src = src + breath * rng.standard_normal(n) * np.abs(src).mean()

    y = _formants(src, sr, vowel)
    # 세기: 기타의 RMS(dB) 곡선을 부드럽게 따라가고, 무성 구간은 조용하게
    env = 10.0 ** ((loud - loud.max()) / 20.0)
    gate = np.clip(voiced, 0.0, 1.0)
    k = max(1, int(0.01 * sr))
    gate = np.convolve(gate, np.ones(k) / k, mode="same")
    y = y * env * gate
    peak = float(np.max(np.abs(y)) + 1e-9)
    return (y * (0.5 / peak)).astype(np.float32)


def pitch_error_cents(ref: Performance, out: Performance) -> dict[str, float]:
    """두 연주 사이의 음높이 차이(센트). 둘 다 유성인 프레임만 비교한다."""
    n = min(len(ref.f0), len(out.f0))
    both = ref.voiced[:n] & out.voiced[:n] & (ref.f0[:n] > 0) & (out.f0[:n] > 0)
    if both.sum() == 0:
        return {"frames": 0, "median_abs": float("nan"), "p90_abs": float("nan"), "octave_err_frac": float("nan")}
    cents = 1200.0 * np.log2(out.f0[:n][both] / ref.f0[:n][both])
    return {
        "frames": int(both.sum()),
        "median_abs": float(np.median(np.abs(cents))),
        "p90_abs": float(np.percentile(np.abs(cents), 90)),
        "octave_err_frac": float(np.mean(np.abs(cents) > 600)),
    }
