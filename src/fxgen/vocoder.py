"""Mel -> waveform. BigVGAN v2 (MIT, 사전학습) 을 쓰고, 테스트용으로 Griffin-Lim을 둔다.

BigVGAN은 pip 패키지가 아니라서 코드를 GitHub에서, 가중치를 Hugging Face에서 받아 온다.
`from_pretrained`는 huggingface_hub 버전에 따라 깨지는 경우가 있어 config/가중치를 직접 읽는다.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

from .mel import FMIN, HOP, N_FFT, SR, WIN

BIGVGAN_REPO = "https://github.com/NVIDIA/BigVGAN"
BIGVGAN_MODEL = "nvidia/bigvgan_v2_44khz_128band_512x"


def load_bigvgan(device: str = "cpu", cache: Path | None = None):
    cache = Path(cache or os.environ.get("FXGEN_CACHE", Path.home() / ".cache" / "fxgen"))
    code = cache / "BigVGAN"
    if not (code / "bigvgan.py").exists():
        cache.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1", BIGVGAN_REPO, str(code)], check=True)
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    import bigvgan  # noqa: E402  (BigVGAN 레포의 모듈)
    from huggingface_hub import hf_hub_download

    cfg = json.loads(Path(hf_hub_download(BIGVGAN_MODEL, "config.json")).read_text())
    assert (cfg["sampling_rate"], cfg["n_fft"], cfg["hop_size"], cfg["num_mels"]) == (SR, N_FFT, HOP, 128), cfg
    model = bigvgan.BigVGAN(bigvgan.AttrDict(cfg), use_cuda_kernel=False)
    state = torch.load(hf_hub_download(BIGVGAN_MODEL, "bigvgan_generator.pt"), map_location="cpu",
                       weights_only=True)
    model.load_state_dict(state["generator"])
    model.remove_weight_norm()
    return model.eval().to(device)


@torch.no_grad()
def bigvgan_decode(voc, mel: torch.Tensor, chunk: int = 512, overlap: int = 16) -> np.ndarray:
    """mel: (T, 128) log-mel -> 파형. 긴 입력은 메모리를 아끼려고 나눠서 돌리고 겹친 부분을 교차 페이드한다."""
    dev = next(voc.parameters()).device
    mel = mel.to(dev).float()
    t_len = mel.shape[0]
    out = np.zeros(t_len * HOP, dtype=np.float32)
    start = 0
    while start < t_len:
        end = min(t_len, start + chunk)
        wav = voc(mel[start:end].T[None])[0, 0].float().cpu().numpy()
        a = start * HOP
        if start > 0:
            n = overlap * HOP
            ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
            out[a : a + n] = out[a : a + n] * (1.0 - ramp) + wav[:n] * ramp
            out[a + n : a + len(wav)] = wav[n:]
        else:
            out[a : a + len(wav)] = wav
        if end == t_len:
            break
        start = end - overlap
    return out


def griffin_lim_decode(mel: torch.Tensor, n_iter: int = 32) -> np.ndarray:
    """보코더 없이 대충 듣기/테스트용. 음질은 나쁘다. 멜 필터의 유사역행렬로 선형 스펙트럼을 어림한다."""
    basis = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=mel.shape[-1], fmin=FMIN)
    mag = np.maximum(np.linalg.pinv(basis) @ np.exp(mel.T.cpu().numpy().astype(np.float32)), 1e-5)
    wav = librosa.griffinlim(mag, n_iter=n_iter, hop_length=HOP, win_length=WIN, n_fft=N_FFT, center=True)
    # center=True 는 프레임 j의 중심을 j*HOP 으로 본다. 우리 멜은 j*HOP + HOP/2 라서 그만큼 민다.
    out = np.zeros(mel.shape[0] * HOP, np.float32)
    n = min(len(wav), len(out) - HOP // 2)
    out[HOP // 2 : HOP // 2 + n] = wav[:n]
    return out
