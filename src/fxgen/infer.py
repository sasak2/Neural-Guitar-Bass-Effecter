"""Convert a performance (e.g. guitar DI) into the timbre of a reference clip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .features import extract_features, pitch_error_cents, transpose
from .flow import sample
from .mel import HOP, LogMel, denormalize, normalize
from .model import FlowNet
from .prep import load_audio


def load_model(ckpt: Path, device: str) -> tuple[FlowNet, dict]:
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model = FlowNet(**ck["hparams"]).to(device)
    model.load_state_dict(ck["ema"])
    return model.eval(), ck


@torch.no_grad()
def reference_timbre(model: FlowNet, ref_wav: np.ndarray, device: str, max_frames: int = 1024) -> torch.Tensor:
    mel = normalize(LogMel().to(device)(torch.from_numpy(ref_wav)[None].to(device)))[0].T[None]
    mel = mel[:, :max_frames]
    return model.timbre(mel, torch.ones(mel.shape[:2], dtype=torch.bool, device=device))


@torch.no_grad()
def generate_mel(model: FlowNet, feat: np.ndarray, timbre: torch.Tensor, device: str,
                 window: int = 384, overlap: int = 96, steps: int = 32, cfg: float = 2.0,
                 seed: int = 0) -> torch.Tensor:
    """(T, 4) 특징 -> (T, 128) log-mel. 학습 길이(window)만큼씩 만들고, 앞 구간의 끝 overlap 프레임을
    고정한 채 다음 구간을 이어서 생성한다 (구간 경계에서 발음/음색이 튀는 것을 줄인다)."""
    g = torch.Generator(device=device).manual_seed(seed)
    f = torch.from_numpy(feat).to(device)
    t_len = f.shape[0]
    out = torch.zeros(t_len, 128, device=device)
    start = 0
    while True:
        end = min(t_len, start + window)
        fw = f[start:end][None]
        known = known_mask = None
        if start > 0:
            known = torch.zeros(1, end - start, 128, device=device)
            known[0, :overlap] = out[start : start + overlap]
            known_mask = torch.zeros(1, end - start, dtype=torch.bool, device=device)
            known_mask[0, :overlap] = True
        x = sample(model, fw, timbre, steps=steps, cfg=cfg, known=known, known_mask=known_mask, generator=g)
        out[start:end] = x[0]
        if end == t_len:
            break
        start = end - overlap
    return denormalize(out)


def convert(ckpt: Path, input_wav: Path, ref_wav: Path, device: str = "cpu", vocoder: str = "bigvgan",
            transpose_semitones: float = 0.0, steps: int = 32, cfg: float = 2.0, seed: int = 0,
            f0_method: str = "crepe", crepe_model: str = "full", check_pitch: bool = True) -> dict:
    """반환: {"wav", "mel", "feat", "pitch"(입력 대비 출력의 음높이 오차)}"""
    from . import vocoder as V

    model, _ = load_model(ckpt, device)
    src = load_audio(Path(input_wav))
    feat = extract_features(src, device=device, method=f0_method, crepe_model=crepe_model)
    if transpose_semitones:
        feat = transpose(feat, transpose_semitones)
    timbre = reference_timbre(model, load_audio(Path(ref_wav)), device)
    mel = generate_mel(model, feat, timbre, device, steps=steps, cfg=cfg, seed=seed)
    if vocoder == "bigvgan":
        wav = V.bigvgan_decode(V.load_bigvgan(device), mel)
    else:
        wav = V.griffin_lim_decode(mel)
    wav = wav[: len(feat) * HOP]
    pitch = None
    if check_pitch:
        out_feat = extract_features(wav, device=device, method=f0_method, crepe_model=crepe_model)
        pitch = pitch_error_cents(feat, out_feat)
    return {"wav": wav, "mel": mel.cpu().numpy(), "feat": feat, "pitch": pitch}
