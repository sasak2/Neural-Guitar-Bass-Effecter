import numpy as np
import pytest

pytest.importorskip("librosa")

from fxdata.voice import extract_performance, pitch_error_cents, render_voice  # noqa: E402

SR = 48000


def _notes(freqs, seg=0.8):
    out = []
    for f in freqs:
        t = np.arange(int(seg * SR)) / SR
        x = sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 6)) * np.exp(-1.5 * t)
        out.append(x)
    return (np.concatenate(out) * 0.3).astype(np.float32)


def test_voice_preserves_pitch_and_loudness_shape():
    x = _notes([110.0, 146.83, 196.0, 261.63])
    perf = extract_performance(x, SR)
    # 추출한 음높이가 실제 음과 가까워야 한다 (구간 중앙값)
    mid = [perf.f0[int((i * 0.8 + 0.4) / perf.hop_s)] for i in range(4)]
    for got, want in zip(mid, [110.0, 146.83, 196.0, 261.63]):
        assert abs(1200 * np.log2(got / want)) < 25

    y = render_voice(perf, SR, len(x), vibrato_cents=0.0)
    assert y.shape == x.shape and np.isfinite(y).all() and abs(y).max() <= 0.51
    err = pitch_error_cents(perf, extract_performance(y, SR))
    assert err["median_abs"] < 15 and err["octave_err_frac"] < 0.1


def test_silence_renders_silence():
    x = np.zeros(SR, dtype=np.float32)
    y = render_voice(extract_performance(x, SR), SR, len(x))
    assert np.isfinite(y).all()
