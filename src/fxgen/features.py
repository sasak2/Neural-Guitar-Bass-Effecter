"""Performance features (what the player did), aligned to mel frames.

프레임마다 4개 값: [음높이, 유성 여부, 주기성, 세기].
- 음높이: 55 Hz(A1) 기준 반음 수를 PITCH_SPAN으로 나눈 값. 무성 프레임은 0.
- 주기성: 음높이 추정기가 "음이 뚜렷하다"고 본 정도(0~1). 숨소리/잡음/어택 구분에 쓰인다.
- 세기: 파일 안에서 큰 소리(95 백분위) 대비 상대 dB를 0~약 1.2로 옮긴 값.
        절대 음량을 버리는 이유는 녹음 레벨이 데이터셋마다(기타 DI와 보컬 마이크) 다르기 때문이다.

한계: 음높이 추정이 단선율용이라 코드(화음)에서는 믿을 수 없다.
"""

from __future__ import annotations

import librosa
import numpy as np
import torch

from .mel import HOP, SR, frame_rms_db

F0_SR = 16000
F0_HOP = 160                 # 10 ms
FMIN, FMAX = 60.0, 1100.0
PITCH_REF_HZ = 55.0
PITCH_SPAN = 60.0            # 반음. 55 Hz ~ 1760 Hz 를 0~1로
LOUD_RANGE_DB = 60.0
VOICED_PERIODICITY = 0.4
VOICED_MIN_REL_DB = -50.0
N_FEAT = 4


def _f0_crepe(y16: np.ndarray, device: str, model: str) -> tuple[np.ndarray, np.ndarray]:
    import torchcrepe

    audio = torch.from_numpy(y16)[None]
    f0, per = torchcrepe.predict(
        audio, F0_SR, F0_HOP, FMIN, FMAX, model, decoder=torchcrepe.decode.viterbi,
        return_periodicity=True, batch_size=1024, device=device, pad=True,
    )
    per = torchcrepe.filter.median(per, 3)
    f0 = torchcrepe.filter.mean(f0, 3)
    return f0[0].cpu().numpy(), per[0].cpu().numpy()


def _f0_pyin(y16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f0, _, prob = librosa.pyin(y16, fmin=FMIN, fmax=FMAX, sr=F0_SR, frame_length=1024,
                               hop_length=F0_HOP, fill_na=None)
    return f0.astype(np.float32), prob.astype(np.float32)


def extract_features(wav: np.ndarray, device: str = "cpu", method: str = "crepe",
                     crepe_model: str = "full") -> np.ndarray:
    """wav: 44.1 kHz 모노 -> (len(wav) // HOP, 4) float32."""
    n = len(wav) // HOP
    y16 = librosa.resample(wav.astype(np.float32), orig_sr=SR, target_sr=F0_SR)
    if method == "crepe":
        f0, per = _f0_crepe(y16, device, crepe_model)
    elif method == "pyin":
        f0, per = _f0_pyin(y16)
    else:
        raise ValueError(f"unknown f0 method: {method}")

    # 음높이 프레임 i의 중심은 i*10ms, 멜 프레임 j의 중심은 (j+0.5)*HOP/SR 이다. 그 시각으로 보간한다.
    t_src = np.arange(len(f0)) * (F0_HOP / F0_SR)
    t_dst = (np.arange(n) + 0.5) * (HOP / SR)
    logf0 = np.interp(t_dst, t_src, np.log(np.clip(f0, 1.0, None)))
    per = np.interp(t_dst, t_src, per).astype(np.float32)

    db = frame_rms_db(wav)[:n]
    rel = db - np.percentile(db, 95)
    voiced = (per > VOICED_PERIODICITY) & (rel > VOICED_MIN_REL_DB)

    pitch = 12.0 * (logf0 - np.log(PITCH_REF_HZ)) / np.log(2.0) / PITCH_SPAN
    feat = np.zeros((n, N_FEAT), dtype=np.float32)
    feat[:, 0] = np.where(voiced, np.clip(pitch, 0.0, 1.2), 0.0)
    feat[:, 1] = voiced
    feat[:, 2] = per
    feat[:, 3] = np.clip(1.0 + rel / LOUD_RANGE_DB, 0.0, 1.2)
    return feat


def transpose(feat: np.ndarray, semitones: float) -> np.ndarray:
    out = feat.copy()
    v = out[:, 1] > 0.5
    out[v, 0] = np.clip(out[v, 0] + semitones / PITCH_SPAN, 0.0, 1.2)
    return out


def feat_f0_hz(feat: np.ndarray) -> np.ndarray:
    """특징에서 Hz 단위 음높이를 되돌린다 (무성은 0)."""
    f0 = PITCH_REF_HZ * 2.0 ** (feat[:, 0] * PITCH_SPAN / 12.0)
    return np.where(feat[:, 1] > 0.5, f0, 0.0).astype(np.float32)


def pitch_error_cents(ref: np.ndarray, out: np.ndarray) -> dict[str, float]:
    """두 특징 배열의 음높이 차이(센트). 둘 다 유성인 프레임만 비교한다."""
    n = min(len(ref), len(out))
    a, b = feat_f0_hz(ref[:n]), feat_f0_hz(out[:n])
    both = (a > 0) & (b > 0)
    res = {"frames": int(both.sum()),
           "voiced_agree": float(np.mean((a > 0) == (b > 0))) if n else float("nan")}
    if both.sum() == 0:
        return {**res, "median_abs": float("nan"), "p90_abs": float("nan"), "octave_err_frac": float("nan")}
    cents = 1200.0 * np.log2(b[both] / a[both])
    return {**res,
            "median_abs": float(np.median(np.abs(cents))),
            "p90_abs": float(np.percentile(np.abs(cents), 90)),
            "octave_err_frac": float(np.mean(np.abs(cents) > 600))}
