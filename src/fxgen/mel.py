"""Log-mel spectrogram that matches BigVGAN v2 (`nvidia/bigvgan_v2_44khz_128band_512x`).

보코더가 학습할 때 본 멜과 수치가 같아야 소리가 제대로 난다. 그래서 설정과 계산 순서를
BigVGAN의 meldataset.mel_spectrogram 과 동일하게 맞췄다 (config.json: 44.1 kHz, n_fft 2048,
hop 512, 멜 128개, fmin 0, fmax 없음). 프레임 j는 샘플 [j*hop - 768, j*hop + 1280)을 본다.
"""

from __future__ import annotations

import librosa
import numpy as np
import torch
from torch import nn

SR = 44100
N_FFT = 2048
HOP = 512
WIN = 2048
N_MELS = 128
FMIN = 0.0
FMAX = None
LOG_FLOOR = 1e-5

# 모델 입력용 정규화 상수. VocalSet과 Guitar-TECHS 표본의 log-mel 평균 -5.7, 표준편차 2.4 에서 정했다.
MEL_MEAN = -5.7
MEL_STD = 2.5


class LogMel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        basis = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
        self.register_buffer("basis", torch.from_numpy(basis).float(), persistent=False)
        self.register_buffer("window", torch.hann_window(WIN), persistent=False)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, T) in [-1, 1] -> (B, N_MELS, T // HOP) natural-log mel."""
        pad = (N_FFT - HOP) // 2
        x = torch.nn.functional.pad(wav[:, None], (pad, pad), mode="reflect")[:, 0]
        spec = torch.stft(x, N_FFT, hop_length=HOP, win_length=WIN, window=self.window,
                          center=False, normalized=False, onesided=True, return_complex=True)
        mag = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
        return torch.log(torch.clamp(self.basis @ mag, min=LOG_FLOOR))


def normalize(mel: torch.Tensor) -> torch.Tensor:
    return (mel - MEL_MEAN) / MEL_STD


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return x * MEL_STD + MEL_MEAN


def frame_rms_db(wav: np.ndarray) -> np.ndarray:
    """멜 프레임과 같은 위치/길이의 창으로 잰 RMS (dBFS). 반환 길이는 len(wav) // HOP."""
    pad = (N_FFT - HOP) // 2
    x = np.pad(wav.astype(np.float32), (pad, pad), mode="reflect")
    n = len(wav) // HOP
    csum = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    starts = np.arange(n) * HOP
    power = (csum[starts + WIN] - csum[starts]) / WIN
    return (10.0 * np.log10(power + 1e-10)).astype(np.float32)
